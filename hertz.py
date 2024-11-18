from __future__ import annotations
from functools import partial
from contextlib import nullcontext
from typing import List, Tuple, Optional
from math import ceil


import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.amp import autocast

from einops import rearrange, pack, unpack

from utils import si_module, exists, default, maybe

from tinygrad import Tensor, nn, dtypes
from tinygrad.dtype import DTYPES_DICT
import numpy as np # needed for np.cumprod()

from ioblocks import FSQ as torch_FSQ
from model import LatentQuantizer as torch_LatentQuantizer

from icecream import install, ic
install()

TWO_SPEAKER = False # Checkpoints for single-speaker and two-speaker models
USE_PURE_AUDIO_ABLATION = False # We trained a base model with no text initialization at all. Toggle this to enable it.
assert not (USE_PURE_AUDIO_ABLATION and TWO_SPEAKER) # We only have a single-speaker version of this model.

configs = {
    'test': { # NOTE: 'float64' unavailable on METAL so removed from `allowed_dtypes`
        'compressor': dict(levels=[8,8,8,8,8], dim=4, num_codebooks=1, keep_num_codebooks_dim=None, scale=None, allowed_dtypes=['float32', 'bfloat16'], channel_first=False, projection_has_bias=True, return_indices=True, force_quantization_f32=True, use_rms=False),
        'latent_quantizer': dict(dim=2048, ff_dim=8192, input_dim=32),
    },
    'base': {
        'compressor': dict(levels=[8,8,8,8,8], dim=2048, num_codebooks=1, keep_num_codebooks_dim=None, scale=None, allowed_dtypes=['float32', 'float64', 'bfloat16'], channel_first=False, projection_has_bias=True, return_indices=True, force_quantization_f32=True, use_rms=False),
        'latent_quantizer': dict(dim=2048, ff_dim=8192, input_dim=32),
        },
}


class GaussianMixtureIOLayer:

    def __init__(self, latent_dim: int, dim: int, num_components):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_components = num_components
        self.input_projection = nn.Linear(latent_dim, dim)

        self.fc_loc = nn.Linear(dim, num_components * latent_dim)
        self.fc_scale = nn.Linear(dim, num_components * latent_dim)
        self.fc_weight = nn.Linear(dim, num_components)

    def _square_plus(self, x: Tensor):
        return (x + Tensor.sqrt(T.square(x) + 4)) / 2

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


    def temp_sample(self, orig_pdist, temp):
        locs, scales, weights = orig_pdist
        if temp is None:
            component_samples = locs + scales * T.randn_like(scales)
            mixture_samples = F.gumbel_softmax(weights, hard=True)
            sampled = (component_samples * mixture_samples.unsqueeze(-1)).sum(dim=-2)
        elif isinstance(temp, tuple):
            assert len(temp) == 2
            categorical_temp, gaussian_temp = temp
            component_samples = locs + scales * gaussian_temp * T.randn_like(scales)
            mixture_samples = F.gumbel_softmax(weights / categorical_temp, hard=True)
            sampled = (component_samples * mixture_samples.unsqueeze(-1)).sum(dim=-2)
        else:
            component_samples = locs + scales * temp * T.randn_like(scales)
            mixture_samples = F.gumbel_softmax(weights / temp, hard=True)
            sampled = (component_samples * mixture_samples.unsqueeze(-1)).sum(dim=-2)
        return sampled


class GPTOutput:
    def __init__(self, dim, vocab_size):
        super().__init__()
        self.output = nn.Linear(dim, vocab_size, bias=False)

    def forward(self, x):
        return self.output(x)


# helper functions

def pack_one(t, pattern):
    return pack([t], pattern)

def unpack_one(t, ps, pattern):
    return unpack(t, ps, pattern)[0]

def first(l):
    return l[0]

def round_up_multiple(num, mult):
    return ceil(num / mult) * mult

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
        assert exists(indices)

        is_img_or_video = indices.ndim >= (3 + int(self.keep_num_codebooks_dim))

        codes = self._indices_to_codes(indices)

        if self.keep_num_codebooks_dim:
            codes = rearrange(codes, '... c d -> ... (c d)')

        codes = self.project_out(codes)

        if is_img_or_video or self.channel_first:
            codes = rearrange(codes, 'b ... d -> b d ...')

        return codes

    def reshape_z(self, z: Tensor) -> Tuple[Tensor, Tensor]:
        is_img_or_video = z.ndim >= 4
        need_move_channel_last = is_img_or_video or self.channel_first

        # standardize image or video into (batch, seq, dimension)
        if need_move_channel_last:
            z = rearrange(z, 'b d ... -> b ... d')
            z, ps = pack_one(z, 'b * d')
        assert z.shape[-1] == self.dim, f'expected dimension of {self.dim} but found dimension of {z.shape[-1]}'
        return z, ps

    def reshape_outputs(self, z: Tensor, ps: Tensor, out: Tensor, indices: Tensor):
        is_img_or_video = z.ndim >= 4
        need_move_channel_last = is_img_or_video or self.channel_first

        if need_move_channel_last:
            out = unpack_one(out, ps, 'b * d')
            out = rearrange(out, 'b ... d -> b d ...')
            indices = maybe(unpack_one)(indices, ps, 'b * c')

        if not self.keep_num_codebooks_dim and self.return_indices:
            indices = maybe(rearrange)(indices, '... 1 -> ...')

        return out, indices

    def __call__(self, z: Tensor, return_codes=False):
        # b=batch, n=seq_len, d=dim, c=codebook_dim
        z, ps = self.reshape_z(z)
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
        out, indices = self.reshape_outputs(z, ps, out, indices)
        return out, indices


class FFNN:
    def __init__(self, dim: int, expand_dim: Optional[int] = None):
        expand_dim = expand_dim if expand_dim is not None else 256 * ((int(2 * 4 * dim / 3) + 256 - 1) // 256)
        self.up_proj = nn.Linear(dim, 2*expand_dim)
        self.down_proj = nn.Linear(expand_dim, dim)

    def __call__(self, x: Tensor):
        gate, up = self.up_proj(x).chunk(2, dim=-1)
        return self.down_proj(up * gate.silu())


class LatentQuantizer:

    def __init__(self, dim: int, compressor_config: Optional[FSQ.Config] = None, ff_dim: Optional[int] = None, input_dim: int = None, from_pretrained: Optional[Tuple[str, str]] = None):

        if exists(from_pretrained):
            checkpoint = load_ckpt(*from_pretrained)
        else:
            assert exists(compressor_config), f'hmm {compressor_config}'

        self.compressor = compressor_config
        self.ffnn = FFNN(dim, ff_dim)
        self.input = nn.Linear(input_dim, dim) if exists(input_dim) else nn.Identity()

        if exists(from_pretrained):
            self.load_state_dict(checkpoint)

    def forward(self, x, return_latent=False, known_latent=None):
        """
        x: (B, S, D)
        """
        Tensor.no_grad = True
        if exists(known_latent):
            return self.compressor.indices_to_codes(known_latent)

        x = self.input(x)
        x = self.ffnn(x)
        x, tokens = self.compressor(x)

        Tensor.no_grad = False

        if return_latent:
            return x, tokens
        return x




if __name__ == '__main__':

    def override_linear(x: Tensor, y: torch.Tensor):
        y.weight = torch.nn.Parameter(torch.Tensor(x.weight.numpy()))
        y.bias = torch.nn.Parameter(torch.Tensor(x.bias.numpy()))
        return y

    # init x
    config = configs['test']
    x = np.zeros((1, config['compressor']['dim'], 1, 1), dtype='float32') # METAL does not support numpy default of float64
    x_torch = torch.Tensor(x)
    x_tiny = Tensor(x)
    np.testing.assert_equal(x_tiny.numpy(), x_torch.numpy())

    def compare_fsq(x_torch: torch.Tensor, x_tiny: Tensor):

        # init models
        fsq_torch = torch_FSQ(torch_FSQ.Config(**config['compressor']))
        fsq_tiny = FSQ(**config['compressor'])

        # init fsq_torch with weights from fsq_tiny
        fsq_torch.project_in = override_linear(fsq_tiny.project_in, fsq_torch.project_in)
        fsq_torch.project_out = override_linear(fsq_tiny.project_out, fsq_torch.project_out)

        # test
        np.testing.assert_equal(fsq_torch.project_in.weight.detach().numpy(), fsq_tiny.project_in.weight.numpy())
        np.testing.assert_equal(fsq_torch.project_out.weight.detach().numpy(), fsq_tiny.project_out.weight.numpy())

        # forward pass
        y_torch, idx_torch = fsq_torch.forward(x_torch)
        y_tiny, idx_tiny = fsq_tiny(x_tiny)
        np.testing.assert_allclose(y_torch.detach().numpy(), y_tiny.numpy(), rtol=1e-6)
        np.testing.assert_allclose(idx_torch.detach().numpy(), idx_tiny.numpy(), rtol=1e-6)


    lq_config = torch_LatentQuantizer.Config(**config['latent_quantizer'] | {'compressor_config': torch_FSQ.Config(**config['compressor'])})
    lq_torch = torch_LatentQuantizer(lq_config)
    lq_tiny = LatentQuantizer({'compressor_config': config['compressor']}, **config['latent_quantizer'])
