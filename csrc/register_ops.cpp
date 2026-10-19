#include "parametric_swish.h"

// Registers parametric_swish as a first-class PyTorch operator, the same
// mechanism torch.ops.* uses internally. This is deliberately not a pybind11
// binding: pybind11 would wrap the function as an opaque Python callable,
// whereas TORCH_LIBRARY registration makes the op visible to TorchScript,
// torch.compile, and anything else that walks the dispatcher - closer to how
// a real framework component is exposed.
TORCH_LIBRARY(parametric_swish_ops, m) {
  m.def("parametric_swish(Tensor x, Tensor beta) -> Tensor", parametric_swish);
}
