#pragma once

#include <torch/torch.h>

// Parametric Swish activation: y = x * sigmoid(beta * x)
//
// Unlike plain SiLU/Swish (fixed beta=1), beta here is a learnable scalar
// tensor, so the backward pass must also return a gradient with respect to
// beta, not just x. That's the main reason this exists as a hand-written
// autograd::Function instead of just composing existing ATen ops: it's a
// small, realistic example of a multi-input custom op, which is closer to
// what a real fused framework kernel looks like than a single-input
// activation would be.
torch::Tensor parametric_swish(const torch::Tensor &x, const torch::Tensor &beta);
