#pragma once

#include <aml/defs.h>
#include <aml/array.h>
#include <aml/handle.h>

namespace aml {
namespace impl {
namespace gpu {

template <typename T, int Dim, typename Op>
__global__ void eval(T *out,
                     Shape<Dim> size,
                     Shape<Dim> stride,
                     Index numel,
                     Op op) {
  Index tid = blockIdx.x * blockDim.x + threadIdx.x;
  Index tstride = blockDim.x * gridDim.x;
  for (Index i = tid; i < numel; i += tstride) {
    Shape<Dim> idx = impl::shape_index(size, i);
    out[impl::dot(idx, stride)] = op(idx);
  }
}

template <typename T, int Dim, typename Op>
void eval(aml::Handle h, Array<T, Dim> &out, const Op &op) {
  auto tic = h.tic("gpu_eval_" + std::to_string(out.size().numel()),
      [h]{ h.synchronize(); });

  auto dims = launch_dims(h.gpu(), out.size().numel());
  eval<<<dims.first, dims.second>>>(
      out.data(), out.size(), out.stride(), out.size().numel(), op);

  tic.stop();
}

}  // namespace gpu
}  // namespace impl
}  // namespace aml

