#pragma once

#include <array>

#include <aml/array.h>
#include <aml/defs.h>
#include <aml/handle.h>
#include <aml/impl/operations.h>
#include <aml/shape.h>

namespace aml {

template <typename T, int Dim>
void set(Handle h, Array<T, Dim> &out, const T &val) {
  AML_DEVICE_EVAL(out.device(), set(h, out, val));
}

template <typename Tin, typename Tout, int Dim>
void copy(Handle h, const ImmutableArray<Tin, Dim> &in, Array<Tout, Dim> &out) {
  AML_ASSERT(in.size() == out.size(), "Shape mismatch");

  impl::copy(h, in, out);
}

template <typename Tin,
          int DimIn,
          typename Tout,
          int DimOut,
          typename TransformOp,
          typename ReduceOp>
void reduce(Handle h,
            const ImmutableArray<Tin, DimIn> &in,
            Array<Tout, DimOut> &out,
            const std::array<int, DimIn - DimOut> &axis,
            const TransformOp &op_t,
            const ReduceOp &op_r) {
  AML_ASSERT(in.device() == out.device(), "Device mismatch");

  std::array<int, DimOut> axis_nr;
  int idx = 0;
  for (int i = 0; i < DimIn; ++i) {
    bool do_reduce = false;
    for (int j = 0; j < DimIn - DimOut; ++j) {
      if (axis[j] == i) {
        do_reduce = true;
      }
    }
    if (!do_reduce) {
      AML_ASSERT(in.size()[i] == out.size()[idx], "Dimension mismatch");
      axis_nr[idx] = i;
      ++idx;
    }
  }
  AML_ASSERT(idx == DimOut, "Repeat axis found");

  AML_DEVICE_EVAL(in.device(),
      reduce<Tin, DimIn, Tout, DimOut, TransformOp, ReduceOp>(
      h, in, out, axis, axis_nr, op_t, op_r));
}

}  // namespace aml

