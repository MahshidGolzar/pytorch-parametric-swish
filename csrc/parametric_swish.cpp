#include "parametric_swish.h"

using torch::Tensor;
using torch::autograd::AutogradContext;
using torch::autograd::Function;
using torch::autograd::tensor_list;

namespace {

class ParametricSwishFunction : public Function<ParametricSwishFunction> {
 public:
  static Tensor forward(AutogradContext *ctx, const Tensor &x, const Tensor &beta) {
    auto s = torch::sigmoid(beta * x);
    auto y = x * s;
    ctx->save_for_backward({x, beta, s});
    return y;
  }

  static tensor_list backward(AutogradContext *ctx, tensor_list grad_outputs) {
    auto saved = ctx->get_saved_variables();
    const auto &x = saved[0];
    const auto &beta = saved[1];
    const auto &s = saved[2];
    const auto &grad_output = grad_outputs[0];

    // d/dx [x * sigmoid(beta*x)] = sigmoid(beta*x) + x * beta * sigmoid(beta*x) * (1 - sigmoid(beta*x))
    auto dydx = s + x * s * (1 - s);
    auto grad_x = grad_output * dydx;

    // d/dbeta [x * sigmoid(beta*x)] = x^2 * sigmoid(beta*x) * (1 - sigmoid(beta*x)), summed since beta is scalar
    auto dydbeta = x * x * s * (1 - s);
    auto grad_beta = (grad_output * dydbeta).sum();

    return {grad_x, grad_beta};
  }
};

}  // namespace

Tensor parametric_swish(const Tensor &x, const Tensor &beta) {
  return ParametricSwishFunction::apply(x, beta);
}
