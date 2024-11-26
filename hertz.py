from __future__ import annotations
from functools import partial
from contextlib import nullcontext
from typing import List, Tuple, Optional, Dict, Union, MutableMapping
import math


import torch
import torch.distributed as dist
from torch.amp import autocast

from einops import rearrange, pack, unpack

from utils import default, maybe

from tinygrad import Tensor, nn, dtypes
from tinygrad.dtype import DTYPES_DICT
import numpy as np # needed for np.cumprod()

from ioblocks import FSQ as torch_FSQ
from model import LatentQuantizer as torch_LatentQuantizer
from model import GaussianMixtureIOLayer as torch_GaussianMixtureIOLayer
from model import TransformerVAE as torch_TransformerVAE
from transformer import Stack as torch_Stack

from icecream import install, ic
install()

CACHE_FILL_VALUE = -1
TWO_SPEAKER = False # Checkpoints for single-speaker and two-speaker models
USE_PURE_AUDIO_ABLATION = False # We trained a base model with no text initialization at all. Toggle this to enable it.
assert not (USE_PURE_AUDIO_ABLATION and TWO_SPEAKER) # We only have a single-speaker version of this model.

configs = {
    'test': { # NOTE: 'float64' unavailable on METAL so removed from `allowed_dtypes`
        'compressor': dict(levels=[8,8,8,8,8], dim=4, num_codebooks=1, keep_num_codebooks_dim=None, scale=None, allowed_dtypes=['float32', 'bfloat16'], channel_first=False, projection_has_bias=True, return_indices=True, force_quantization_f32=True, use_rms=False),
        'latent_quantizer': dict(dim=4, ff_dim=8192, input_dim=32),
        'gaussian_mixture': dict(latent_dim=32, dim=4096, num_components=8,),
        'vae_stack': dict(layers=8, dim=4096, seq_len=8192, n_head=16, ff_dim=11008, kv_heads=16, eps=1e-5, theta=10_000),
        'vae': dict(plex_layer=None, plex_roll=1, split=TWO_SPEAKER)
    },
    'base': {
        'compressor': dict(levels=[8,8,8,8,8], dim=2048, num_codebooks=1, keep_num_codebooks_dim=None, scale=None, allowed_dtypes=['float32', 'float64', 'bfloat16'], channel_first=False, projection_has_bias=True, return_indices=True, force_quantization_f32=True, use_rms=False),
        'latent_quantizer': dict(dim=2048, ff_dim=8192, input_dim=32),
        'gaussian_mixture': dict(latent_dim=32, dim=4096, num_components=8,),
        'vae_stack': dict(layers=8, dim=4096, seq_len=8192, n_head=16, ff_dim=11008, kv_heads=16, eps=1e-5, theta=10_000),
        'vae': dict(plex_layer=None, plex_roll=1, split=TWO_SPEAKER)
        },
}

# **** basic building blocks ****

# like tinygrad's LayerNorm but with only .weight, not .bias
class Norm:
    def __init__(self, dim: int, eps: float = 1e-5,) -> None:
        self.eps, self.weight = eps, Tensor.ones((dim,)) # self.weight used to be torch.nn.Parameter()
    def __call__(self, x: Tensor) -> Tensor:
        # LayerNorm.bias in torch = LayerNorm.elementwise_affine in tinygrad
        # LayerNorm.weight exists in torch but not in tinygrad

        # return nn.LayerNorm(x, (self.weight.shape[0],), weight=self.weight, bias=None, eps=self.eps)
        return x.layernorm(eps=self.eps) * self.weight

class FFNN:
    def __init__(self, dim: int, expand_dim: Optional[int] = None):
        self.expand_dim = expand_dim if expand_dim is not None else 256 * ((int(2 * 4 * dim / 3) + 256 - 1) // 256)
        self.gateup_proj = nn.Linear(dim, 2*self.expand_dim)
        self.down_proj = nn.Linear(self.expand_dim, dim)

    def __call__(self, x: Tensor) -> Tensor:
        gate, up = self.gateup_proj(x).chunk(2, dim=-1)
        return self.down_proj(up * gate.silu())

# **** GQA *****

def get_cache_len(cache: Optional[Tensor]) -> int:
    """
    cache: (batch, seq_len, 2, kv_heads, head_dim)
    """
    if cache is None:
        return 0
    nonzeros = T.any(cache.flatten(2) != CACHE_FILL_VALUE, dim=-1)
    length = nonzeros.sum(dim=-1).int()
    assert T.all(length == length[0])
    return length[0]

class GQA:
    def __init__(self, dim: int, n_head: int, shape_rotator: ShapeRotator, kv_heads: Optional[int] = None, eps: float = 1e-5, causal: bool = True,):
        self.n_heads = n_head
        self.kv_heads = default(kv_heads, n_head)
        self.head_dim = dim // n_head
        self.causal = causal
        self.proj_qkv = nn.Linear(dim, self.head_dim*(n_head+2*self.kv_heads))

        self.norm_q = Norm(self.head_dim*n_head, eps=eps)
        self.norm_k = Norm(self.head_dim*self.kv_heads, eps=eps)
        self.attn_out = nn.Linear(dim, dim)
        self.shape_rotator = shape_rotator

    def _sdpa(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        k = k.repeat_interleave(self.n_heads // self.kv_heads, dim=2)
        v = v.repeat_interleave(self.n_heads // self.kv_heads, dim=2)
        x = Tensor.scaled_dot_product_attention(
            q.transpose(1, 2),
            k.transpose(1, 2),
            v.transpose(1, 2),
            is_causal=False if (q.size(1) != k.size(1)) else self.causal,
        )
        x = x.transpose(1, 2).contiguous()
        return x

    def _attend(self, q: Tensor, k: Tensor, v: Tensor, kv_cache: Optional[Tensor] = None,):
        cache_len = get_cache_len(kv_cache)
        q, k = self.shape_rotator.rotate(q, k, offset=cache_len)
        if kv_cache is not None:
            k = T.cat([kv_cache[:, :cache_len, 0], k], dim=1)
            v = T.cat([kv_cache[:, :cache_len, 1], v], dim=1)
            kv_cache[:, :k.size(1), 0] = k
            kv_cache[:, :v.size(1), 1] = v
        x = self._sdpa(q, k, v)
        return self.attn_out(rearrange(x, 'b s h d -> b s (h d)'))

    def _project(self, x):
        full_q, full_k, full_v = self.proj_qkv(x).chunk(3, dim=-1)
        normed_full_q = self.norm_q(full_q).cast(full_q.dtype)
        normed_full_k = self.norm_k(full_k).cast(full_k.dtype)

        q = rearrange(normed_full_q, 'b s (h d) -> b s h d', h=self.n_heads)
        k = rearrange(normed_full_k, 'b s (h d) -> b s h d', h=self.kv_heads)
        v = rearrange(full_v, 'b s (h d) -> b s h d', h=self.kv_heads)
        return q, k, v

    def __call__(self,x: Tensor, kv: Optional[Tensor] = None,):
        # x.shape=(B, S, D), kv.shape=(B, S, H, D)
        q, k, v = self._project(x)
        return self._attend(q, k, v, kv_cache=kv)

# **** pre-norm ops ####

class PreNormAttn:
    def __init__(self, dim: int, n_head: int, shape_rotator: ShapeRotator, kv_heads: Optional[int] = None, eps: float = 1e-5, causal: bool = True,):
        self.attn_norm = Norm(dim, eps=eps)
        self.attn = GQA(dim, n_head, shape_rotator, kv_heads, eps=eps, causal=causal)

    def __call__(self, x: Tensor, kv: Optional[Tensor] = None) -> Tensor:
        return x + self.attn(self.attn_norm(x), kv) # x.shape=(B, S, D), kv.shape=(B, S, H, D)

class PreNormFFNN:
    def __init__(self,dim: int, ff_dim: int, eps: float = 1e-5,):
        self.ffnn_norm = Norm(dim, eps=eps)
        self.ffnn = FFNN(dim, ff_dim)

    def __call__(self, x: Tensor) -> Tensor:
        return x + self.ffnn(self.ffnn_norm(x))


class GaussianMixtureIOLayer:

    def __init__(self, latent_dim: int, dim: int, num_components):
        self.latent_dim = latent_dim
        self.num_components = num_components
        self.input_projection = nn.Linear(latent_dim, dim)
        self.fc_loc = nn.Linear(dim, num_components * latent_dim)
        self.fc_scale = nn.Linear(dim, num_components * latent_dim)
        self.fc_weight = nn.Linear(dim, num_components)

    def _square_plus(self, x: Tensor):
        return (x + (x.square() + 4).sqrt()) / 2

    def input(self, sampled_latents: Tensor) -> Tensor:
        """Pre-sampled latents Tensor (B, L, Z) -> float tensor (B, L, D)"""
        hidden = self.input_projection(sampled_latents)
        return hidden

    def output(self, h: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """float tensor (B, L, D) -> Tuple of locs, scales, and weights"""
        batch_size, seq_len, _ = h.shape

        locs = self.fc_loc(h).view(batch_size, seq_len, self.num_components, self.latent_dim)
        scales = Tensor.clamp(self._square_plus(self.fc_scale(h)), min=1e-6).view(batch_size, seq_len, self.num_components, self.latent_dim)
        weights = self.fc_weight(h).view(batch_size, seq_len, self.num_components)

        return (locs, scales, weights)

    def loss(self, data, dataHat):
        locs, scales, weights = dataHat
        log_probs = -0.5 * T.sum(
            (data.unsqueeze(-2) - locs).pow(2) / scales.pow(2) +
            2 * T.log(scales) +
            T.log(Tensor(2 * T.pi)),
            dim=-1
        )
        log_weights = F.log_softmax(weights, dim=-1)
        return -T.logsumexp(log_weights + log_probs, dim=-1)

    def temp_sample(self, orig_pdist, categorical_temp: float = 1, gaussian_temp: float = 1):
        locs, scales, weights = orig_pdist
        component_samples = locs + scales * gaussian_temp * Tensor.randn_like(scales)
        mixture_samples = F.gumbel_softmax(weights / categorical_temp, hard=True)
        sampled = (component_samples * mixture_samples.unsqueeze(-1)).sum(dim=-2)
        return sampled


class GPTOutput:
    def __init__(self, dim, vocab_size):
        super().__init__()
        self.output = nn.Linear(dim, vocab_size, bias=False)

    def __call__(self, x):
        return self.output(x)


# helper functions

def pack_one(t, pattern):
    return pack([t], pattern)

def unpack_one(t, ps, pattern):
    return unpack(t, ps, pattern)[0]

def first(l):
    return l[0]

def round_up_multiple(num, mult):
    return math.ceil(num / mult) * mult

def get_code_utilization(codes, codebook_size, get_global=False):
    if get_global and dist.is_initialized():
        world_size = dist.get_world_size()
    else:
        world_size = 1

    if world_size > 1:
        gathered_tokens = [T.zeros_like(codes) for _ in range(world_size)]
        dist.all_gather(gathered_tokens, codes)
        gathered_tokens = T.cat(gathered_tokens, dim=0)
    else:
        gathered_tokens = codes
    unique_tokens = len(T.unique(gathered_tokens))
    code_utilization = unique_tokens / min(gathered_tokens.numel(), codebook_size)
    return code_utilization

# tensor helpers

def round_ste(z: Tensor) -> Tensor:
    """Round with straight through gradients."""
    zhat = z.round()
    return z + (zhat - z).detach()





# main class
# lucidrains fsq
class FSQ:
    @property
    def needs_float32_params(self):
        return True

    def __init__(
        self,
        levels: List[int],
        dim: int | None = None,
        num_codebooks: int = 1,
        keep_num_codebooks_dim: bool | None = None,
        scale: float | None = None,
        allowed_dtypes: Tuple[str, ...] = ('float32', 'float64'),
        channel_first: bool = False,
        projection_has_bias: bool = True,
        return_indices: bool = True,
        force_quantization_f32: bool = True,
        use_rms: bool = False,
        ):

        self.scale = scale
        self.num_codebooks = num_codebooks
        self.channel_first = channel_first
        self.force_quantization_f32 = force_quantization_f32
        self.latent_loss = None

        self._levels = Tensor(levels, dtype=dtypes.int32) # was register_buffer
        self._basis = Tensor(np.array([1] + levels[:-1]).cumprod(axis=0), dtype=dtypes.int32) # was register_buffer # need numpy for cumprod

        self.codebook_dim = len(self._levels)
        self.dim = dim if dim is not None else self.codebook_dim * self.num_codebooks
        self.effective_codebook_dim = self.codebook_dim * num_codebooks
        self.allowed_dtypes = [DTYPES_DICT[dtype_str] for dtype_str in allowed_dtypes]

        keep_num_codebooks_dim = default(keep_num_codebooks_dim, num_codebooks > 1)
        assert not (num_codebooks > 1 and not keep_num_codebooks_dim)
        self.keep_num_codebooks_dim = keep_num_codebooks_dim

        self.has_projections = self.dim != self.effective_codebook_dim
        self.project_in = nn.Linear(self.dim, self.effective_codebook_dim, bias=projection_has_bias) if self.has_projections else nn.Identity()
        self.project_out = nn.Linear(self.effective_codebook_dim, self.dim, bias=projection_has_bias) if self.has_projections else nn.Identity()

        self.return_indices = return_indices
        if return_indices:
            self.codebook_size = self._levels.prod().item()
            self.implicit_codebook = self._indices_to_codes(Tensor.arange(self.codebook_size)) # was register_buffer


    def latent_metric(self, codes, get_global=False):
        return {'code_util_estimate': get_code_utilization(codes, self.codebook_size, get_global)}

    def repr_from_latent(self, latent):
        return self.indices_to_codes(latent)

    def bound(self, z, eps: float = 1e-3):
        """ Bound `z`, an array of shape (..., d). """
        half_l = (self._levels - 1) * (1 + eps) / 2
        offset = (Tensor(self._levels.numpy() % 2) == 0).where(0.5, 0.0) # tinygrad does not support %
        shift = (offset / half_l).atanh()
        return (z + shift).tanh() * half_l - offset

    def quantize(self, z):
        """ Quantizes z, returns quantized zhat, same shape as z. """
        quantized = round_ste(self.bound(z))
        half_width = self._levels // 2 # Renormalize to [-1, 1].
        return quantized / half_width

    def _scale_and_shift(self, zhat_normalized):
        half_width = self._levels // 2
        return (zhat_normalized * half_width) + half_width

    def _scale_and_shift_inverse(self, zhat):
        half_width = self._levels // 2
        return (zhat - half_width) / half_width

    def _indices_to_codes(self, indices: Tensor):
        level_indices = self.indices_to_level_indices(indices)
        codes = self._scale_and_shift_inverse(level_indices)
        return codes

    def codes_to_indices(self, zhat):
        """ Converts a `code` to an index in the codebook. """
        assert zhat.shape[-1] == self.codebook_dim
        zhat = self._scale_and_shift(zhat)
        return (zhat * self._basis).sum(axis=-1, acc_dtype=dtypes.int32)

    def indices_to_level_indices(self, indices: Tensor):
        """ Converts indices to indices at each level, perhaps needed for a transformer with factorized embeddings """
        indices = rearrange(indices, '... -> ... 1')
        codes_non_centered = Tensor((indices // self._basis).numpy() % self._levels.numpy()) # % not supported in tinygrad
        return codes_non_centered

    def indices_to_codes(self, indices):
        """ Inverse of `codes_to_indices`. """
        assert indices is not None

        is_img_or_video = indices.ndim >= (3 + int(self.keep_num_codebooks_dim))

        codes = self._indices_to_codes(indices)

        if self.keep_num_codebooks_dim:
            codes = rearrange(codes, '... c d -> ... (c d)')

        codes = self.project_out(codes)

        if is_img_or_video or self.channel_first:
            codes = rearrange(codes, 'b ... d -> b d ...')

        return codes

    def __call__(self, z: Tensor, return_codes=False):
        # b=batch, n=seq_len, d=dim, c=codebook_dim

        # reshape z
        is_img_or_video = z.ndim >= 4
        need_move_channel_last = is_img_or_video or self.channel_first
        if need_move_channel_last:
            z = rearrange(z, 'b d ... -> b ... d')
            z, ps = pack_one(z, 'b * d')
        assert z.shape[-1] == self.dim, f'expected dimension of {self.dim} but found dimension of {z.shape[-1]}'

        z = self.project_in(z)
        z = rearrange(z, 'b n (c d) -> b n c d', c=self.num_codebooks)

        # maybe force quantization
        force_f32 = self.force_quantization_f32
        quantization_context = partial(autocast, device_type='cuda', enabled=False) if force_f32 else nullcontext
        with quantization_context():
            orig_dtype = z.dtype
            if force_f32 and orig_dtype not in self.allowed_dtypes:
                z = z.float()
            codes = self.quantize(z)
            indices = self.codes_to_indices(codes) if self.return_indices else None
            codes = rearrange(codes, 'b n c d -> b n (c d)').cast(orig_dtype)

        # project out
        if return_codes:
            return codes, indices
        out = self.project_out(codes)

        # reshape outputs
        if need_move_channel_last:
            out = unpack_one(out, ps, 'b * d')
            out = rearrange(out, 'b ... d -> b d ...')
            indices = maybe(unpack_one)(indices, ps, 'b * c')
        if not self.keep_num_codebooks_dim and self.return_indices:
            indices = maybe(rearrange)(indices, '... 1 -> ...')

        return out, indices



class LatentQuantizer:

    def __init__(self, dim: int, ff_dim: int, input_dim: Optional[int], compressor: FSQ, from_pretrained: Optional[Tuple[str, str]] = None):

        self.compressor = compressor
        self.input = nn.Linear(input_dim, dim) if input_dim is not None else nn.Identity()
        self.ffnn = FFNN(dim, ff_dim)

        if from_pretrained is not None: self.load_state_dict(load_ckpt(*from_pretrained))

    @Tensor.test() # with torch.no_grad()
    def __call__(self, x: Tensor, return_latent=False, known_latent=None):
        if known_latent is not None:
            return self.compressor.indices_to_codes(known_latent)
        x, tokens = self.compressor(self.ffnn(self.input(x))) # initially, x.shape=(B, S, D)
        return (x, tokens) if return_latent else x



def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return Tensor.cat(-x2, x1, dim=-1)


def apply_rotary_pos_emb(x, cos, sin, offset: int = 0):
    error_msg = f"Offset and/or input sequence is too large,\\n offset: {offset}, seq_len: {x.shape[1]}, max: {cos.shape[1]}"
    assert (cos.shape[1] >= offset + x.shape[1]), error_msg

    cos_out = cos[:, offset : offset + x.shape[1], :, :]
    sin_out = sin[:, offset : offset + x.shape[1], :, :]
    return (x * cos_out) + (rotate_half(x) * sin_out)


# Adapted from https://github.com/foundation-model-stack/foundation-model-stack
class ShapeRotator:
    def __init__(
        self,
        dim: int,
        end: int,
        theta: float = 10_000,
    ):
        self.dim = dim
        self.ratio = theta
        self.cached_freqs: MutableMapping[int, MutableMapping[int, torch.Tensor]] = {}
        self.max_seq_len_cached: MutableMapping[int, int] = {}
        self.ntk_scaling = False
        self.max_seq_len = end

    def compute_freqs_cis(self, device, max_seq_len=None):
        alpha = 1
        dev_idx = device.index
        max_seq_len = default(max_seq_len, self.max_seq_len)

        if dev_idx not in self.cached_freqs: self.cached_freqs[dev_idx] = {}
        if dev_idx not in self.max_seq_len_cached: self.max_seq_len_cached[dev_idx] = 0
        if self.max_seq_len_cached[dev_idx] > 0: return 1
        max_seq_len = max(max_seq_len, self.max_seq_len)

        if (
            1 in self.cached_freqs[dev_idx]
            and max_seq_len <= self.max_seq_len_cached[dev_idx]
        ):
            return 1

        ratio = self.ratio
        dim = self.dim

        freqs = 1.0 / (ratio ** (Tensor.arange(0, dim, 2, device=device).float() / dim))

        t = Tensor.arange(max_seq_len, device=device, dtype=freqs.dtype)
        freqs = Tensor.einsum("i,j->ij", t, freqs)
        emb = Tensor.cat(freqs, freqs, dim=-1).to(device)

        cos_to_cache = emb.cos()[None, :, None, :]
        sin_to_cache = emb.sin()[None, :, None, :]

        self.max_seq_len_cached[dev_idx] = max_seq_len
        self.cached_freqs[dev_idx][alpha] = Tensor.stack(cos_to_cache, sin_to_cache, dim=-1)

        return alpha

    def rotate(
        self,
        q: Tensor,
        k: Tensor,
        offset: int = 0,
    ) -> Tuple[Tensor, Tensor]:
        """
        Args
        ----
        q : torch.Tensor
            Embedded query tensor, expected size is B x S x H x Eh
        k : torch.Tensor
            Embedded query tensor, expected size is B x S x H x Eh
        """
        assert len(q.size()) == 4
        assert len(k.size()) == 4

        seq_len = self.max_seq_len
        alpha = self.compute_freqs_cis(q.device, seq_len)
        freqs = self.cached_freqs[q.device.index][alpha]

        freqs = freqs.float()  # 1 L D/2 2 2
        q_out = apply_rotary_pos_emb(q, freqs[..., 0], freqs[..., 1], offset=offset).cast(q.dtype)
        k_out = apply_rotary_pos_emb(k, freqs[..., 0], freqs[..., 1], offset=offset).cast(k.dtype)

        # return q_out.view_as(q), k_out.view_as(k) # old, delete when done
        return q_out.reshape(q.shape), k_out.reshape(k.shape)


class Block:
    def __init__(self,
            dim: int,
            layer_id: int = 0,
            n_head: int = 16,
            kv_heads: Optional[int] = None,
            ff_dim: Optional[int] = None,
            eps: float = 1e-5,
            causal: bool = True,
            shape_rotator: Optional[ShapeRotator] = None,
            ):
        self.attn = PreNormAttn(dim, n_head, shape_rotator, kv_heads, eps=eps, causal=causal)
        self.ffnn = PreNormFFNN(dim, ff_dim, eps=eps)
        self.dim = dim
        self.layer_id = layer_id
        self.head_dim = dim // n_head
        self.expand_dim = self.ffnn.ffnn.expand_dim
        self.reset_parameters()

    def reset_parameters(self):
        # `trunc_normal_` not supported in pytorch so replaced with `normal`
        std = 1.0 / math.sqrt(self.dim)
        self.ffnn.ffnn.gateup_proj.weight = Tensor.normal(self.ffnn.ffnn.gateup_proj.weight.shape, std=std)
        self.attn.attn.proj_qkv.weight = Tensor.normal(self.attn.attn.proj_qkv.weight.shape, std=std)
        self.attn.attn.attn_out.weight = Tensor.normal(self.attn.attn.attn_out.weight.shape, std=std)
        # torch.nn.init.trunc_normal_(self.ffnn.ffnn.gateup_proj.weight, std=std, a=-3 * std, b=3 * std)
        # torch.nn.init.trunc_normal_(self.attn.attn.proj_qkv.weight, std=std, a=-3 * std, b=3 * std)
        # torch.nn.init.trunc_normal_(self.attn.attn.attn_out.weight, std=std, a=-3 * std, b=3 * std)

        xstd = 1.0 / math.sqrt(self.expand_dim)
        self.ffnn.ffnn.down_proj.weight = Tensor.normal(self.ffnn.ffnn.down_proj.weight.shape, std=xstd)
        # torch.nn.init.trunc_normal_(self.ffnn.ffnn.down_proj.weight, std=xstd, a=-3 * xstd, b=3 * xstd)

    def __call__(self, x: Tensor, kv: Optional[Tensor] = None) -> Tensor:
        h = self.attn(x, kv) # x.shape=(B, S, D), kv.shape=(B, S, H, D)
        out = self.ffnn(h)
        return out


class Stack:

    def __init__(
        self,
        layers: int,
        dim: int,
        seq_len: int,
        n_head: int = 32,
        ff_dim: int = None,
        kv_heads: int = None,
        eps: float = 1e-5,
        theta: Union[int, float] = 10_000,
        causal: bool = True,
        from_pretrained: Optional[Tuple[str, int]] = None,
        ):

        kv_heads = kv_heads or n_head
        head_dim = dim // n_head
        cache_shape = [layers, seq_len, 2, kv_heads, head_dim]
        self.cache_shape = cache_shape
        self.cache = [None] * layers
        self.shape_rotator = ShapeRotator(dim//n_head, seq_len, theta=theta)
        self.layers = [
            Block(dim=dim, layer_id=l, n_head=n_head, kv_heads=kv_heads, ff_dim=ff_dim, eps=eps, causal=causal, shape_rotator=self.shape_rotator)
            for l in range(layers)
        ]

        if from_pretrained:
            checkpoint = load_ckpt(from_pretrained)
            self.load_state_dict(checkpoint)

    def init_cache(self, bsize, device, dtype, length:int=None):
        if self.cache_shape is None:
            return
        cache_shape = self.cache_shape.copy()
        cache_shape[1] = length or cache_shape[1]
        self.cache = T.full((bsize, *cache_shape), CACHE_FILL_VALUE, device=device, dtype=dtype).transpose(0, 1)

    def deinit_cache(self):
        self.cache = [None] * len(self.cache)

    def __call__(self, x: Tensor) -> Tensor:
        for l, layer in enumerate(self.layers):
            x = layer(x, kv=self.cache[l])
        return x



# class VAE:

#     def __init__(
#         self,
#         io_config: Optional[Dict] = None,
#         stack_config: Optional[Dict] = None,
#         quantizer_config: Optional[Dict] = None,
#         plex_layer: Optional[int] = None,
#         plex_roll: int = 1,
#         split: bool = True,
#         from_pretrained: Optional[Tuple[str, str]] = None,
#         ):

#         self.io = GaussianMixtureIOLayer(**io_config)
#         self.stack = stack_config
#         self.plex_layer = stack_config['layers']//2
#         self.plex_roll = plex_roll
#         self.plex_dim = quantizer_config['dim']
#         self.split = split

#         assert self.plex_dim is not None and stack_config['dim'] is not None, f'One of the following are None: self.plex_dim: {self.plex_dim}, stack_config.dim: {stack_config["dim"]}'
#         self.plex_projection = nn.Linear(self.plex_dim, stack_config['dim'])
#         self.out_norm = Norm(stack_config['dim'])

#         if self.split:
#             self.io2 = io_config()
#             self.plex_projection2 = nn.Linear(self.plex_dim, stack_config['dim'])
#             self.io2.fc_loc = None
#             self.io2.fc_scale = None
#             self.io2.fc_weight = None

#         kv_heads = stack_config['kv_heads'] or stack_config['n_head']
#         head_dim = stack_config['dim'] // stack_config['n_head']
#         self.cache_num_layers = stack_config['layers'] + ((stack_config['layers'] - self.plex_layer) if split else 0)
#         cache_shape = [self.cache_num_layers, stack_config['seq_len'], 2, kv_heads, head_dim]
#         self.cache_shape = cache_shape
#         self.cache = [None] * self.cache_num_layers

#         if from_pretrained is not None:
#             result = self.load_state_dict(checkpoint, strict=False)
#             checkpoint = load_ckpt(*from_pretrained)
#         else:
#             io_config is not None and stack_config is not None and quantizer_config is not None

#         # @Tensor.test on all functions with self.quantizer
#         # Does tinygrad support .eval()? I didn't do anything for this?
#         # self.quantizer = quantizer_config().eval()
#         # self.quantizer.requires_grad = False

#     @Tensor.test
#     def quantize(self, x):
#         if self.split:
#             x1, x2 = x.chunk(2, dim=-1)
#             with T.autocast(device_type='cuda', dtype=T.bfloat16):
#                 quantized1 = self.quantizer(x1)
#                 quantized2 = self.quantizer(x2)
#             return quantized1, quantized2
#         else:
#             with T.autocast(device_type='cuda', dtype=T.bfloat16):
#                 return self.quantizer(x)

#     @Tensor.test
#     def untokenize(self, token_data):
#         return self.quantizer(None, known_latent=token_data)

#     def init_cache(self, bsize, device, dtype, length:int=None):
#         cache_shape = self.cache_shape.copy()
#         cache_shape[1] = length or cache_shape[1]
#         self.cache = T.full((bsize, *cache_shape), CACHE_FILL_VALUE, device=device, dtype=dtype).transpose(0, 1)

#     def deinit_cache(self):
#         self.cache = [None] * self.cache_num_layers

#     @Tensor.test
#     def forward(self, data, next_tokens: Optional[Tuple[T.Tensor, T.Tensor]] = None, temps: Optional[Tuple[float, Tuple[float, float]]] = None):
#         if self.split:
#             x1, x2 = data.chunk(2, dim=-1)
#             x = self.io.input(x1) + self.io2.input(x2)
#         else:
#             x = self.io.input(data)

#         cache_idx = 0
#         for l, layer in enumerate(self.stack.layers):
#             if l == self.plex_layer:
#                 if self.split:
#                     plex1, plex2 = self.quantize(data)
#                     plex1 = T.roll(plex1, -self.plex_roll, dims=1)
#                     plex2 = T.roll(plex2, -self.plex_roll, dims=1)
#                     if exists(next_tokens):
#                         plex1[:, -1:] = self.untokenize(next_tokens[0])
#                         plex2[:, -1:] = self.untokenize(next_tokens[1])
#                     x1 = x + self.plex_projection(plex1)
#                     x2 = x + self.plex_projection2(plex2)
#                 else:
#                     plex = self.quantize(data)
#                     plex = T.roll(plex, -self.plex_roll, dims=1)
#                     if exists(next_tokens):
#                         plex[:, -1:] = self.untokenize(next_tokens)
#                     x = x + self.plex_projection(plex)

#             if l < self.plex_layer:
#                 x = layer(x, kv=self.cache[l])
#             else:
#                 if self.split:
#                     x1 = layer(x1, kv=self.cache[self.plex_layer + cache_idx])
#                     cache_idx += 1
#                     x2 = layer(x2, kv=self.cache[self.plex_layer + cache_idx])
#                     cache_idx += 1
#                 else:
#                     x = layer(x, kv=self.cache[l])

#         with T.autocast(device_type='cuda', dtype=T.bfloat16):
#             if self.split:
#                 x1, x2 = self.out_norm(x1), self.out_norm(x2)
#                 out1, out2 = self.io.output(x1), self.io.output(x2)
#             else:
#                 x = self.out_norm(x)
#                 out = self.io.output(x)

#         if isnt(temps):
#             if self.split:
#                 return out1, out2
#             else:
#                 return out
#         else:
#             if self.split:
#                 next_data1 = self.io.temp_sample(out1, temps)[:, -1:, :]
#                 next_data2 = self.io2.temp_sample(out2, temps)[:, -1:, :]
#                 next_data = T.cat([next_data1, next_data2], dim=-1)
#                 return next_data
#             else:
#                 next_data = self.io.temp_sample(out, temps)[:, -1:, :]
#                 return next_data


# **** testing ****

def override_linear(y: torch.nn.Linear, x: nn.Linear) -> torch.nn.Linear:
    y.weight = torch.nn.Parameter(torch.Tensor(x.weight.detach().numpy()))
    y.bias = torch.nn.Parameter(torch.Tensor(x.bias.detach().numpy()))
    return y

def compare_fsq(config: Dict):

    # init x of shape (b,d,n,c)
    x = np.zeros((1, config['compressor']['dim'], 1, 1), dtype='float32') # METAL does not support numpy default of float64
    x_torch = torch.Tensor(x)
    x_tiny = Tensor(x)
    np.testing.assert_equal(x_tiny.numpy(), x_torch.numpy())

    # init fsq
    fsq_torch = torch_FSQ(torch_FSQ.Config(**config['compressor']))
    fsq_tiny = FSQ(**config['compressor'])

    # init fsq_torch with weights from fsq_tiny
    fsq_torch.project_in = override_linear(fsq_torch.project_in, fsq_tiny.project_in)
    fsq_torch.project_out = override_linear(fsq_torch.project_out, fsq_tiny.project_out)
    np.testing.assert_equal(fsq_torch.project_in.weight.detach().numpy(), fsq_tiny.project_in.weight.numpy())
    np.testing.assert_equal(fsq_torch.project_out.weight.detach().numpy(), fsq_tiny.project_out.weight.numpy())

    # forward pass
    y_torch, idx_torch = fsq_torch.forward(x_torch)
    y_tiny, idx_tiny = fsq_tiny(x_tiny)
    np.testing.assert_allclose(y_torch.detach().numpy(), y_tiny.numpy(), rtol=1e-6)
    np.testing.assert_allclose(idx_torch.detach().numpy(), idx_tiny.numpy(), rtol=1e-6)

def compare_lq(config: Dict):
    # init x (B,N,D)
    # METAL does not support numpy default of float64 so use float32
    x = np.zeros((1, 1, config['latent_quantizer']['input_dim']), dtype='float32')
    x_torch = torch.Tensor(x)
    x_tiny = Tensor(x)
    np.testing.assert_equal(x_tiny.numpy(), x_torch.numpy())

    # init latent quantizer
    lq_config_torch = torch_LatentQuantizer.Config(**config['latent_quantizer'] | {'compressor_config': torch_FSQ.Config(**config['compressor'])})
    lq_torch = torch_LatentQuantizer(lq_config_torch)
    lq_config_tiny = config['latent_quantizer'] | {'compressor':  FSQ(**config['compressor'])}
    lq_tiny = LatentQuantizer(**lq_config_tiny)

    # init lq_torch with weights from lq_tiny
    lq_torch.input = override_linear(lq_torch.input, lq_tiny.input)
    np.testing.assert_equal(lq_torch.input.weight.detach().numpy(), lq_tiny.input.weight.numpy())
    np.testing.assert_equal(lq_torch.input.bias.detach().numpy(), lq_tiny.input.bias.numpy())

    lq_torch.ffnn.gateup_proj = override_linear(lq_torch.ffnn.gateup_proj, lq_tiny.ffnn.gateup_proj)
    np.testing.assert_equal(lq_torch.ffnn.gateup_proj.weight.detach().numpy(), lq_tiny.ffnn.gateup_proj.weight.numpy())
    np.testing.assert_equal(lq_torch.ffnn.gateup_proj.bias.detach().numpy(), lq_tiny.ffnn.gateup_proj.bias.numpy())

    lq_torch.ffnn.down_proj = override_linear(lq_torch.ffnn.down_proj, lq_tiny.ffnn.down_proj)
    np.testing.assert_equal(lq_torch.ffnn.down_proj.weight.detach().numpy(), lq_tiny.ffnn.down_proj.weight.numpy())
    np.testing.assert_equal(lq_torch.ffnn.down_proj.bias.detach().numpy(), lq_tiny.ffnn.down_proj.bias.numpy())

    lq_torch.compressor.project_in = override_linear(lq_torch.compressor.project_in, lq_tiny.compressor.project_in)
    np.testing.assert_equal(lq_torch.compressor.project_in.weight.detach().numpy(), lq_tiny.compressor.project_in.weight.numpy())
    np.testing.assert_equal(lq_torch.compressor.project_in.bias.detach().numpy(), lq_tiny.compressor.project_in.bias.numpy())

    lq_torch.compressor.project_out = override_linear(lq_torch.compressor.project_out, lq_tiny.compressor.project_out)
    np.testing.assert_equal(lq_torch.compressor.project_out.weight.detach().numpy(), lq_tiny.compressor.project_out.weight.numpy())
    np.testing.assert_equal(lq_torch.compressor.project_out.bias.detach().numpy(), lq_tiny.compressor.project_out.bias.numpy())

    # forward pass
    y_torch = lq_torch(x_torch)
    y_tiny = lq_tiny(x_tiny)
    np.testing.assert_allclose(y_torch.detach().numpy(), y_tiny.numpy(), rtol=1e-6)


def compare_gm(config: Dict):
    gm_tiny = GaussianMixtureIOLayer(**config['gaussian_mixture'])

def compare_stack(config: Dict):

    # init x.shape=(B, S, D)
    x = np.ones((1, 1, config['vae_stack']['dim']), dtype='float32')
    x_torch = torch.Tensor(x)
    x_tiny = Tensor(x)
    np.testing.assert_equal(x_tiny.numpy(), x_torch.numpy())

    # init stack
    stack_tiny = Stack(**config['vae_stack'])
    stack_torch = torch_Stack(torch_Stack.Config(**config['vae_stack']))

    # init stack_torch with weights from stack_tiny
    weight_fxs = [
        lambda l: l.attn.attn_norm,
        lambda l: l.attn.attn.project.qkv,
        lambda l: l.attn.attn.norm_q,
        lambda l: l.attn.attn.norm_v,
        lambda l: l.attn.attn.attn_out,
        lambda l: l.ffnn.ffnn_norm,
        lambda l: l.ffnn.ffnn.gateup_proj,
        lambda l: l.ffnn.ffnn.down_proj,
        ]
    for layer_torch, layer_tiny in zip(stack_torch.layers, stack_tiny.layers):
        for weight_fx in weight_fxs:
            weight = weight_fx(layer_torch)


    # forward pass
    y_tiny = stack_tiny(x_tiny)
    print(f'{y_tiny.numpy()=}')
    y_torch = stack_torch(x_torch)
    print(f'{y_torch.detach().numpy()=}')


def compare_vae(config: Dict):

    config_tiny = config['vae'] | {'io_config': config['gaussian_mixture'], 'stack_config': config['vae_stack'], 'quantizer_config': config['latent_quantizer']}
    vae_tiny = VAE(**config_tiny)

if __name__ == '__main__':

    config = configs['test']
    # compare_fsq(configs['test'])
    # compare_lq(configs['test'])
    # compare_gm(configs['test'])
    compare_stack(configs['test'])
    # compare_vae(configs['test'])
