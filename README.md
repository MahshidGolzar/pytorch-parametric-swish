# Parametric Swish: a custom PyTorch C++ operator, built with CMake

A hand-written PyTorch C++ extension implementing **Parametric Swish** — `y = x * sigmoid(beta * x)` — where `beta` is a *learnable* scalar parameter rather than the fixed `beta=1` of standard SiLU/Swish. The op is a `torch::autograd::Function` with a manually-derived forward and backward, registered as a first-class PyTorch operator (`torch.ops.parametric_swish_ops.parametric_swish`) via `TORCH_LIBRARY`, and built with CMake rather than `setup.py`.

## Why this project exists

This is deliberately *not* a pybind11-wrapped black box. `TORCH_LIBRARY` registration is the same mechanism PyTorch's own operators use, so the op is visible to `torch.ops.*` directly — closer to how a real framework component is exposed than an ad-hoc Python binding would be. The point isn't the activation function itself (parametric Swish variants are well known); it's the same discipline as the other two projects in this portfolio: derive the method by hand, write tests that would catch a wrong derivation, and actually catch one.

## The bug this project's tests caught

The first working version of `backward()` (see git history) shipped with a seeded bug in the gradient with respect to `x`:

```cpp
// d/dx [x * sigmoid(beta*x)] = sigmoid(beta*x) + x * beta * sigmoid(beta*x) * (1 - sigmoid(beta*x))
auto dydx = s + x * s * (1 - s);   // <- missing the beta factor
```

The comment directly above the bug states the correct derivative in words — the kind of drift between a comment and the code beside it that's easy to introduce during a refactor and easy to miss in review, since the code still compiles and still produces plausible-looking numbers. The bug is also invisible at `beta=1` (the missing factor is multiplied by 1), so a quick manual check at default settings would not have caught it.

`tests/test_parametric_swish.py::test_gradcheck_against_x` uses `torch.autograd.gradcheck` (finite-difference numerical differentiation) to compare the analytic backward pass against a numerical one, for randomised `beta != 1`. That test — and `test_gradcheck_against_both_simultaneously` — failed against the seeded version; `test_gradcheck_against_beta` and the forward-value tests passed, which narrowed the bug to the `dy/dx` term specifically before looking at the diff. `cpp_tests/smoke_test.cpp` independently reproduces the same failure against one hand-computed value at `beta=1.5`, for anyone who wants to step through it in a debugger instead of via pytest.

The actual commit history: a scaffold commit (bug present), a commit adding the gradcheck suite (and showing it fail), and a fix commit restoring the missing factor. `git log -p` on this repo shows the real diff and the real failing test output, not a reconstruction.

## What's in here

- `csrc/parametric_swish.h` / `.cpp` — the `torch::autograd::Function` (forward + backward), independent of any Python binding.
- `csrc/register_ops.cpp` — `TORCH_LIBRARY` registration, making the op loadable from Python.
- `cpp_tests/smoke_test.cpp` — a standalone executable (no Python) that exercises the op and checks one value against a hand-derived closed form. Intended to be run under a debugger.
- `python/parametric_swish/__init__.py` — loads the built extension and wraps it as an `nn.Module` with a trainable `beta`.
- `tests/test_parametric_swish.py` — pytest suite: closed-form check at `beta=1`, monotonicity in `beta`, three `gradcheck` tests (w.r.t. `x`, w.r.t. `beta`, and both together), and a training-loop smoke test confirming `beta` actually moves under SGD.
- `CMakeLists.txt` — builds both targets against the `torch` package already installed in the active Python environment (no separate libtorch download needed).

## Known limitations

- CPU only — no CUDA kernel. The autograd `Function` is backend-agnostic ATen code, so a CUDA build would mostly be a matter of building against libtorch's CUDA variant and testing on a GPU-equipped machine, not a rewrite — but it hasn't been done or tested here.
- Registered under a single, unqualified dispatch key rather than per-backend (`CPU`/`CUDA`/`Autograd`) registration, which is the simplified, tutorial-level way to register a custom op. A production operator would typically split the autograd formula from backend-specific kernels.
- `beta` is a single global scalar, not per-channel — extending to a per-channel `beta` (closer to PReLU) would change the backward's reduction (`sum()` would need to reduce over all dimensions except the channel dimension instead of to a single scalar).
- Built and tested on Windows with MSVC Build Tools only; not tested with gcc/clang or on Linux, though nothing in the CMake setup is Windows-specific other than the post-build DLL copy step.

## Building and running

Requires: Python with `torch` installed, CMake, Ninja, and a C++ compiler matching the one PyTorch's wheel was built with (MSVC Build Tools on Windows, which is what this was built and tested with — PyTorch 2.14 requires C++20).

```bash
python -m venv .venv
source .venv/Scripts/activate   # or .venv/bin/activate on macOS/Linux
pip install torch pytest ninja

# Configure and build (Windows: run from a "Developer Command Prompt for VS",
# or after calling vcvarsall.bat, so cl.exe is on PATH)
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DPython3_EXECUTABLE="$(which python)"
cmake --build build --config Release

# Unit tests (Python side)
pytest tests/ -v

# Standalone C++ smoke test / debugger target
./build/smoke_test            # or build/smoke_test.exe on Windows
# e.g. under gdb: gdb --args ./build/smoke_test
```
