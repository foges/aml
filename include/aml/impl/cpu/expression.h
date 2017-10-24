#pragma once

#include <aml/defs.h>
#include <aml/array.h>
#include <aml/handle.h>

namespace aml {
namespace impl {
namespace cpu {

template <typename T, int Dim, typename Op>
void eval(T *out,
          const Shape<0>&,
          const Shape<0>&,
          Shape<Dim> &idx,
          const Op &op) {
  *out = op(idx);
}

template <typename T, int DimS, int DimI, typename Op>
void eval(T *out,
          const Shape<DimS> &stride,
          const Shape<DimS> &size,
          Shape<DimI> &idx,
          const Op &op) {
  for (Index i = 0; i < size.head(); ++i) {
    idx[DimS - 1] = i;
    eval(out + i * stride.head(), stride.tail(), size.tail(), idx, op);
  }
}

template <typename T, int Dim, typename Op>
void eval(Handle h, Array<T, Dim> &out, const Op &op) {
  auto tic = h.tic("cpu_eval_" + std::to_string(out.size().numel()));

  Shape<Dim> idx;
  eval(out.data(), out.stride(), out.size(), idx, op);

  tic.stop();
}

}  // namespace cpu
}  // namespace impl
}  // namespace aml

