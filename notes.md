Setup

```
git clone https://github.com/eitanturok/hertz-dev
git cd hertz-dev
git switch -c tinygrad

mamba create -n hertz
mamba activate hertz
mamba install python=3.10

uv pip install -r  requirements.txt
```

Tinygrad is missing:
1. `Tensor.cumprod()`
2. `Tensor(5) % Tensor(7)` is not supported
