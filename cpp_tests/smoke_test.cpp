// Standalone C++ exercise of parametric_swish, with no Python involved.
//
// This exists so the op can be stepped through directly in a debugger
// (gdb/lldb/Visual Studio) rather than through several layers of Python ->
// pybind-style dispatch -> C++. Run it under a debugger and break on
// ParametricSwishFunction::backward to see exactly where the bug
// deliberately included in the first commit (see README) produced a
// wrong gradient.
#include <cmath>
#include <cstdio>
#include <cstdlib>

#include "../csrc/parametric_swish.h"

namespace {

bool close_enough(double a, double b, double tol = 1e-5) {
  return std::fabs(a - b) <= tol * (1.0 + std::fabs(b));
}

}  // namespace

int main() {
  auto x = torch::tensor({1.0, -2.0, 0.5}, torch::dtype(torch::kFloat64).requires_grad(true));
  auto beta = torch::tensor(1.5, torch::dtype(torch::kFloat64).requires_grad(true));

  auto y = parametric_swish(x, beta);
  auto loss = y.sum();
  loss.backward();

  std::printf("y       = [%f, %f, %f]\n", y[0].item<double>(), y[1].item<double>(), y[2].item<double>());
  std::printf("grad_x  = [%f, %f, %f]\n",
              x.grad()[0].item<double>(), x.grad()[1].item<double>(), x.grad()[2].item<double>());
  std::printf("grad_beta = %f\n", beta.grad().item<double>());

  // Independently-derived closed-form check for a single element (x=1.0),
  // computed by hand in the README: with beta=1.5, x=1.0,
  // s = sigmoid(1.5) = 0.8175744762...
  // dy/dx = s + beta*x*s*(1-s) = 0.8175744762 + 1.5*1*0.8175744762*0.1824255238
  double s = 1.0 / (1.0 + std::exp(-1.5));
  double expected_dydx_at_1 = s + 1.5 * 1.0 * s * (1 - s);
  double actual_dydx_at_1 = x.grad()[0].item<double>();

  if (!close_enough(actual_dydx_at_1, expected_dydx_at_1)) {
    std::printf(
        "FAIL: grad_x[0] = %f, expected %f (closed-form d/dx at x=1, beta=1.5)\n",
        actual_dydx_at_1, expected_dydx_at_1);
    return 1;
  }

  std::printf("PASS: grad_x[0] matches closed-form derivative.\n");
  return 0;
}
