from __future__ import annotations

import math

import torch
from torch.autograd import gradcheck

from parametric_swish import ParametricSwish, parametric_swish


def test_forward_matches_closed_form_at_beta_one():
    # beta=1 reduces parametric_swish to standard SiLU/Swish: x * sigmoid(x).
    x = torch.tensor([-2.0, -0.5, 0.0, 0.5, 2.0], dtype=torch.float64)
    beta = torch.tensor(1.0, dtype=torch.float64)

    y = parametric_swish(x, beta)
    expected = x * torch.sigmoid(x)

    assert torch.allclose(y, expected, atol=1e-10)


def test_forward_scales_with_beta():
    # Larger beta pushes sigmoid(beta*x) closer to a step function, so for
    # x > 0 the output should get closer to x (and for x < 0, closer to 0).
    x = torch.tensor([3.0], dtype=torch.float64)
    beta_small = torch.tensor(0.1, dtype=torch.float64)
    beta_large = torch.tensor(10.0, dtype=torch.float64)

    y_small = parametric_swish(x, beta_small).item()
    y_large = parametric_swish(x, beta_large).item()

    assert y_small < y_large < x.item()


def test_gradcheck_against_x():
    # Finite-difference check of the gradient w.r.t. x: this is the test that
    # catches the seeded bug described in the README (a missing beta factor
    # in d/dx), because gradcheck compares the analytic backward() against
    # numerical differentiation rather than against another formula we might
    # have gotten wrong in the same way.
    torch.manual_seed(0)
    x = torch.randn(5, dtype=torch.float64, requires_grad=True)
    beta = torch.tensor(1.7, dtype=torch.float64, requires_grad=False)

    assert gradcheck(lambda x_: parametric_swish(x_, beta), (x,), eps=1e-6, atol=1e-6)


def test_gradcheck_against_beta():
    torch.manual_seed(1)
    x = torch.randn(5, dtype=torch.float64, requires_grad=False)
    beta = torch.tensor(0.8, dtype=torch.float64, requires_grad=True)

    assert gradcheck(lambda b: parametric_swish(x, b), (beta,), eps=1e-6, atol=1e-6)


def test_gradcheck_against_both_simultaneously():
    torch.manual_seed(2)
    x = torch.randn(4, dtype=torch.float64, requires_grad=True)
    beta = torch.tensor(1.3, dtype=torch.float64, requires_grad=True)

    assert gradcheck(parametric_swish, (x, beta), eps=1e-6, atol=1e-6)


def test_module_beta_is_trainable_and_moves_during_training():
    torch.manual_seed(3)
    module = ParametricSwish(beta_init=1.0)
    x = torch.randn(32, 8, dtype=torch.float32)
    target = torch.randn(32, 8, dtype=torch.float32)
    optimizer = torch.optim.SGD(module.parameters(), lr=0.5)

    beta_before = module.beta.item()
    for _ in range(50):
        optimizer.zero_grad()
        loss = torch.nn.functional.mse_loss(module(x), target)
        loss.backward()
        optimizer.step()
    beta_after = module.beta.item()

    assert not math.isclose(beta_before, beta_after, abs_tol=1e-4)
