from __future__ import annotations

import glob
import os

import torch
from torch import nn


def _load_ops() -> None:
    """Load the compiled parametric_swish_ops extension (built via CMake)."""
    here = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.abspath(os.path.join(here, "..", ".."))
    pattern = os.path.join(repo_root, "build", "**", "parametric_swish_ops.*")
    candidates = glob.glob(pattern, recursive=True)
    candidates = [c for c in candidates if c.endswith((".pyd", ".so", ".dylib"))]
    if not candidates:
        raise ImportError(
            "parametric_swish_ops extension not found under build/. "
            "Build it first: cmake --build build --config Release"
        )
    torch.ops.load_library(candidates[0])


_load_ops()

parametric_swish = torch.ops.parametric_swish_ops.parametric_swish


class ParametricSwish(nn.Module):
    """x * sigmoid(beta * x), with beta a learnable scalar parameter.

    beta is initialised to 1.0, which makes this equivalent to standard
    SiLU/Swish at init; training can move it away from 1.0.
    """

    def __init__(self, beta_init: float = 1.0) -> None:
        super().__init__()
        self.beta = nn.Parameter(torch.tensor(float(beta_init)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return parametric_swish(x, self.beta)
