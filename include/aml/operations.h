#pragma once

#include <array>

#include <aml/defs.h>
#include <aml/impl/operations.h>

namespace aml {

template <typename T>
void gemm(OP op_a,
          OP op_b,
          T alpha,
          const ImmutableMatrix<T> &a,
          const ImmutableMatrix<T> &b,
          T beta,
          Matrix<T> &c) {
  Index Ma = op_a == NO_TRANS ? a.shape()[0] : a.shape()[1];
  Index Mc = c.shape()[0];
  Index Nb = op_b == NO_TRANS ? b.shape()[1] : b.shape()[0];
  Index Nc = c.shape()[1];
  Index Ka = op_a == NO_TRANS ? a.shape()[1] : a.shape()[0];
  Index Kb = op_b == NO_TRANS ? b.shape()[0] : a.shape()[1];

  AML_ASSERT(Ma == Mc, "Leading dimension of A and C matrix must match");
  AML_ASSERT(Nb == Nc, "Trailing dimension of B and C matrix must match");
  AML_ASSERT(Ka == Kb, "Inner dimension of A and B matrix must match");

  Device device = a.device();

  AML_ASSERT(device == b.device() && device == c.device(), "Device mismatch");

  AML_ASSERT(a.stride()[0] == 1, "Leding dimension of A must be contiguous");
  AML_ASSERT(b.stride()[0] == 1, "Leding dimension of B must be contiguous");
  AML_ASSERT(c.stride()[0] == 1, "Leding dimension of C must be contiguous");

  AML_DEVICE_EVAL(device, gemm(op_a, op_b, Ma, Nb, Ka, alpha, a.data(),
      a.stride()[1], b.data(), b.stride()[1], beta, c.data(), c.stride()[1]));
}

template <typename T, int Dim>
void set(Array<T, Dim> &out, const T &val) {
  AML_DEVICE_EVAL(out.device(), set(out, val));
}

template <typename Tin, typename Tout, int Dim, typename Op>
void unary_op(const ImmutableArray<Tin, Dim> &in,
              Array<Tout, Dim> &out,
              const Op &op) {
  AML_ASSERT(in.device() == out.device(), "Device mismatch");
  AML_ASSERT(in.shape() == out.shape(), "Shape mismatch");

  AML_DEVICE_EVAL(in.device(), unary_op(in, out, op));
}

template <typename Tin1, typename Tin2, typename Tout, int Dim, typename Op>
void binary_op(const ImmutableArray<Tin1, Dim> &in1,
               const ImmutableArray<Tin2, Dim> &in2,
               Array<Tout, Dim> &out,
               const Op &op) {
  Device device = in1.device();
  Shape<Dim> shape = in1.shape();

  AML_ASSERT(device == in2.device() && device == out.device(),
      "Device mismatch");
  AML_ASSERT(shape == in2.shape() && shape == out.shape(), "Shape mismatch");

  AML_DEVICE_EVAL(device, binary_op(in1, in2, out, op));
}

template <typename Tin,
          int DimIn,
          typename Tout,
          int DimOut,
          typename TransformOp,
          typename ReduceOp>
void reduce(const ImmutableArray<Tin, DimIn> &in,
            Array<Tout, DimOut> &out,
            std::array<int, DimIn - DimOut> axis,
            const TransformOp &op_t,
            const ReduceOp &op_r) {
  AML_ASSERT(in.device() == out.device(), "Device mismatch");

  Shape<DimIn> stride;
  int num = 0;
  for (int i = 0; i < DimIn; ++i) {
    bool do_reduce = false;
    for (int j = 0; j < DimIn - DimOut; ++j) {
      if (axis[j] == i) {
        do_reduce = true;
      }
    }
    if (!do_reduce) {
      AML_ASSERT(in.shape()[i] == out.shape()[num], "Dimension mismatch");
      stride[i] = out.stride()[num];
      ++num;
    }
  }
  AML_ASSERT(num == DimOut, "Repeat axis found");

  AML_DEVICE_EVAL(in.device(), reduce(in, out, op_t, op_r, stride));
}

}  // namespace aml

