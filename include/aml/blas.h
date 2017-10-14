#pragma once

#include <aml/array.h>
#include <aml/defs.h>
#include <aml/device.h>
#include <aml/handle.h>
#include <aml/immutable_array.h>
#include <aml/impl/blas.h>

namespace aml {

template <typename T>
void gemm(Handle h,
          OP op_a,
          OP op_b,
          T alpha,
          const ImmutableMatrix<T> &a,
          const ImmutableMatrix<T> &b,
          T beta,
          Matrix<T> &c) {
  Index Ma = op_a == NO_TRANS ? a.size()[0] : a.size()[1];
  Index Mc = c.size()[0];
  Index Nb = op_b == NO_TRANS ? b.size()[1] : b.size()[0];
  Index Nc = c.size()[1];
  Index Ka = op_a == NO_TRANS ? a.size()[1] : a.size()[0];
  Index Kb = op_b == NO_TRANS ? b.size()[0] : a.size()[1];

  AML_ASSERT(Ma == Mc, "Leading dimension of A and C matrix must match");
  AML_ASSERT(Nb == Nc, "Trailing dimension of B and C matrix must match");
  AML_ASSERT(Ka == Kb, "Inner dimension of A and B matrix must match");

  Device device = a.device();

  AML_ASSERT(device == b.device() && device == c.device(), "Device mismatch");

  AML_ASSERT(a.stride()[0] == 1, "Leding dimension of A must be contiguous");
  AML_ASSERT(b.stride()[0] == 1, "Leding dimension of B must be contiguous");
  AML_ASSERT(c.stride()[0] == 1, "Leding dimension of C must be contiguous");

  AML_DEVICE_EVAL(device, gemm(h, op_a, op_b, Ma, Nb, Ka, alpha, a.data(),
      a.stride()[1], b.data(), b.stride()[1], beta, c.data(), c.stride()[1]));
}

template <typename T>
T nrm2(Handle h, const ImmutableVector<T> &x) {
  AML_ASSERT(x.is_contiguous(), "Vector must be contiguous in nrm2");
  return AML_DEVICE_EVAL(x.device(), nrm2(h, x.size()[0], x.data()));
}

}  // namespace aml

