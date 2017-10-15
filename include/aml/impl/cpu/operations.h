#pragma once

#include <cstring>

#include <aml/array.h>
#include <aml/defs.h>
#include <aml/immutable_array.h>
#include <aml/shape.h>

namespace aml {
namespace impl {
namespace cpu {

/** SET ***********************************************************************/

template <typename T>
void set(T *out,
         const Shape<0>&,
         const Shape<0>&,
         const T &val) {
  *out = val;
}

template <typename T, int Dim>
void set(T *out,
         const Shape<Dim> &stride,
         const Shape<Dim> &size,
         const T &val) {
  for (Index i = 0; i < size.head(); ++i) {
    set(out + i * stride.head(), stride.tail(), size.tail(), val);
  }
}

template <typename T, int Dim>
void set(aml::Handle, Array<T, Dim> &out, const T &val) {
  set(out.data(), out.stride(), out.size(), val);
}

/** UNARY_OP ******************************************************************/

template <typename Tin, typename Tout, typename Op>
void unary_op(const Tin *in,
              const Shape<0>&,
              Tout *out,
              const Shape<0>&,
              const Shape<0>&,
              const Op &op) {
  *out = op(*in);
}

template <typename Tin, typename Tout, int Dim, typename Op>
void unary_op(const Tin *in,
              const Shape<Dim> &in_stride,
              Tout *out,
              const Shape<Dim> &out_stride,
              const Shape<Dim> &size,
              const Op &op) {
  for (Index i = 0; i < size.head(); ++i) {
    unary_op(
        in + i * in_stride.head(), in_stride.tail(),
        out + i * out_stride.head(), out_stride.tail(), size.tail(), op);
  }
}

template <typename Tin, typename Tout, int Dim, typename Op>
void unary_op(aml::Handle,
              const ImmutableArray<Tin, Dim> &in,
              Array<Tout, Dim> &out,
              const Op &op) {
  unary_op(in.data(), in.stride(), out.data(), out.stride(), in.size(), op);
}

/** BINARY_OP *****************************************************************/

template <typename Tin1, typename Tin2, typename Tout, typename Op>
void binary_op(const Tin1 *in1,
               const Shape<0>&,
               const Tin2 *in2,
               const Shape<0>&,
               Tout *out,
               const Shape<0>&,
               const Shape<0>&,
               const Op &op) {
  *out = op(*in1, *in2);
}

template <typename Tin1, typename Tin2, typename Tout, int Dim, typename Op>
void binary_op(const Tin1 *in1,
               const Shape<Dim> &in1_stride,
               const Tin2 *in2,
               const Shape<Dim> &in2_stride,
               Tout *out,
               const Shape<Dim> &out_stride,
               const Shape<Dim> &size,
               const Op &op) {
  for (Index i = 0; i < size.head(); ++i) {
    binary_op(
        in1 + i * in1_stride.head(), in1_stride.tail(),
        in2 + i * in2_stride.head(), in2_stride.tail(),
        out + i * out_stride.head(), out_stride.tail(), size.tail(), op);
  }
}

template <typename Tin1, typename Tin2, typename Tout, int Dim, typename Op>
void binary_op(aml::Handle,
               const ImmutableArray<Tin1, Dim> &in1,
               const ImmutableArray<Tin2, Dim> &in2,
               Array<Tout, Dim> &out,
               const Op &op) {
  binary_op(in1.data(), in1.stride(), in2.data(), in2.stride(),
      out.data(), out.stride(), in1.size(), op);
}

/** REDUCE ********************************************************************/

template <typename Tin,
          typename Tout,
          typename TransformOp,
          typename ReduceOp>
void reduce(const Tin *in,
            const Shape<0>&,
            const Shape<0>&,
            Tout *out,
            const Shape<0>&,
            const TransformOp &op_t,
            const ReduceOp &op_r) {
  *out = op_r(*out, op_t(*in));
}

template <typename Tin,
          int Dim,
          typename Tout,
          typename TransformOp,
          typename ReduceOp>
void reduce(const Tin *in,
            const Shape<Dim> &stride_in,
            const Shape<Dim> &size_in,
            Tout *out,
            const Shape<Dim> &stride_out,
            const TransformOp &op_t,
            const ReduceOp &op_r) {
  for (Index i = 0; i < size_in.head(); ++i) {
    reduce(in + i * stride_in.head(),
           stride_in.tail(),
           size_in.tail(),
           out + i * stride_out.head(),
           stride_out.tail(),
           op_t,
           op_r);
  }
}

template <typename Tin,
          int DimIn,
          typename Tout,
          int DimOut,
          typename TransformOp,
          typename ReduceOp>
void reduce(aml::Handle,
            const ImmutableArray<Tin, DimIn> &in,
            Array<Tout, DimOut> &out,
            const std::array<int, DimIn - DimOut>&,
            const std::array<int, DimOut> &axis_nr,
            const TransformOp &op_t,
            const ReduceOp &op_r) {

  Shape<DimIn> stride_out;
  for (int i = 0; i < DimOut; ++i) {
    stride_out[axis_nr[i]] = out.stride()[i];
  }

  reduce(
      in.data(), in.stride(), in.size(), out.data(), stride_out, op_t, op_r);
}

/** COPY **********************************************************************/

template <typename T, int Dim>
void copy(aml::Handle h, const ImmutableArray<T, Dim> &in, Array<T, Dim> &out) {
  if (in.is_contiguous() && out.is_contiguous()) {
    std::memcpy(out.data(), in.data(), in.size().numel() * sizeof(T));
  } else {
    cpu::unary_op(h, in, out, [](const T &x){ return x; });
  }
}

template <typename Tin, typename Tout, int Dim>
void copy(aml::Handle h,
          const ImmutableArray<Tin, Dim> &in,
          Array<Tout, Dim> &out) {
  cpu::unary_op(h, in, out, [](const Tin &x){ return static_cast<Tout>(x); });
}

}  // namespace cpu
}  // namespace impl
}  // namespace aml

