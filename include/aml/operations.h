#pragma once

#include <aml/defs.h>
#include <aml/impl/operations.h>

namespace aml {

template <typename T>
void gemm(OP op_a,
          OP op_b,
          T alpha,
          const ConstMatrix<T> &a,
          const ConstMatrix<T> &b,
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

template <typename Tin, typename Tout, int Dim, typename Op>
void unary_op(const ConstArray<Tin, Dim> &in,
              Array<Tout, Dim> &out,
              const Op &op) {
  AML_ASSERT(in.device() == out.device(), "Device mismatch");
  AML_ASSERT(in.shape() == out.shape(), "Shape mismatch");

  AML_DEVICE_EVAL(in.device(), unary_op(in, out, op));
}

}  // namespace aml

