# Parametric Swish: a custom PyTorch C++ operator, built with CMake

A hand-written PyTorch C++ extension implementing **Parametric Swish** — `y = x * sigmoid(beta * x)` — where `beta` is a *learnable* scalar parameter rather than the fixed `beta=1` of standard SiLU/Swish. The op is a `torch::autograd::Function` with a manually-derived forward and backward, registered as a first-class PyTorch operator (`torch.ops.parametric_swish_ops.parametric_swish`) via `TORCH_LIBRARY`, and built with CMake rather than `setup.py`.

This is deliberately not a pybind11-wrapped black box: `TORCH_LIBRARY` registration is the same mechanism PyTorch's own operators use, so the op is visible to `torch.ops.*` directly.

## What's in here

- `csrc/parametric_swish.h` / `.cpp` — the `torch::autograd::Function` (forward + backward), independent of any Python binding.
- `csrc/register_ops.cpp` — `TORCH_LIBRARY` registration, making the op loadable from Python.
- `cpp_tests/smoke_test.cpp` — a standalone executable (no Python) that exercises the op and checks one value against a hand-derived closed form.
- `python/parametric_swish/__init__.py` — loads the built extension and wraps it as an `nn.Module` with a trainable `beta`.
- `CMakeLists.txt` — builds both targets against the `torch` package already installed in the active Python environment (no separate libtorch download needed).

## Building and running

```bash
python -m venv .venv
source .venv/Scripts/activate   # or .venv/bin/activate on macOS/Linux
pip install torch

cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DPython3_EXECUTABLE="$(which python)"
cmake --build build --config Release

./build/smoke_test            # or build/smoke_test.exe on Windows
```
