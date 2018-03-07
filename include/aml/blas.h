#pragma once

#include <aml/array.h>
#include <aml/defs.h>
#include <aml/device.h>
#include <aml/handle.h>
#include <aml/immutable_array.h>
#include <aml/impl/blas.h>

#define AML_CHECK_OP(op) \
    AML_ASSERT(op == 'n' || op == 't', "Unsupported transpose type");

namespace aml {

/** BLAS LEVEL 1 **************************************************************/

template <typename T>
T nrm2(Handle h, const ImmutableVector<T> &x) {
  return AML_DEVICE_EVAL(x.device(), nrm2(h, x.size()[0], x.data()));
}

/** BLAS LEVEL 2 **************************************************************/

template <typename T>
inline void gemv(Handle h,
                 char op,
                 float alpha,
                 const ImmutableMatrix<T> &a,
                 const ImmutableVector<T> &x,
                 float beta,
                 Vector<T> &y) {
  AML_CHECK_OP(op);

  Index Ma = op == 'n' ? a.size()[0] : a.size()[1];
  Index My = y.size()[0];
  Index Na = op == 'n' ? a.size()[1] : a.size()[0];
  Index Nx = x.size()[0];

  AML_ASSERT(Ma == My, "Leading dimension of A and y must match");
  AML_ASSERT(Na == Nx, "Leading dimension of A and y must match");

  Device device = a.device();

  AML_ASSERT(device == x.device() && device == y.device(), "Device mismatch");

  AML_DEVICE_EVAL(device, gemv(h, op, Ma, Na, alpha, a.data(),
      a.stride()[1], x.data(), beta, y.data()));
}

template <typename T>
inline void trsv(Handle h,
                 char op,
                 const ImmutableMatrix<T> &a,
                 Vector<T> &x) {
  AML_CHECK_OP(op);
  AML_ASSERT(a.size()[0] == a.size()[1], "A must be square");

  Index Ma = a.size()[0];
  Index Mx = x.size()[0];

  AML_ASSERT(Ma == Mx, "Leading dimension of A and x must match");

  AML_ASSERT(a.device() == x.device(), "Device mismatch");

  AML_DEVICE_EVAL(a.device(),
      trsv(h, op, Ma, a.data(), a.stride()[1], x.data()));
}

/** BLAS LEVEL 3 **************************************************************/

template <typename T>
void gemm(Handle h,
          char op_a,
          char op_b,
          T alpha,
          const ImmutableMatrix<T> &a,
          const ImmutableMatrix<T> &b,
          T beta,
          Matrix<T> &c) {
  AML_CHECK_OP(op_a);
  AML_CHECK_OP(op_b);

  Index Ma = op_a == 'n' ? a.size()[0] : a.size()[1];
  Index Mc = c.size()[0];
  Index Nb = op_b == 'n' ? b.size()[1] : b.size()[0];
  Index Nc = c.size()[1];
  Index Ka = op_a == 'n' ? a.size()[1] : a.size()[0];
  Index Kb = op_b == 'n' ? b.size()[0] : a.size()[1];

  AML_ASSERT(Ma == Mc, "Leading dimension of A and C must match");
  AML_ASSERT(Nb == Nc, "Trailing dimension of B and C must match");
  AML_ASSERT(Ka == Kb, "Inner dimension of A and B must match");

  Device device = a.device();

  AML_ASSERT(device == b.device() && device == c.device(), "Device mismatch");

  AML_DEVICE_EVAL(device, gemm(h, op_a, op_b, Ma, Nb, Ka, alpha, a.data(),
      a.stride()[1], b.data(), b.stride()[1], beta, c.data(), c.stride()[1]));
}

template <typename T>
void syrk(Handle h,
          char op,
          T alpha,
          const ImmutableMatrix<T> &a,
          T beta,
          Matrix<T> &c) {
  AML_CHECK_OP(op);

  AML_ASSERT(c.size()[0] == c.size()[1], "C must be square");

  Index Na = op == 'n' ? a.size()[0] : a.size()[1];
  Index Nc = c.size()[0];
  Index Ka = op == 'n' ? a.size()[1] : a.size()[0];

  AML_ASSERT(Na == Nc, "Leading dimension of A and C must match");

  AML_ASSERT(a.device() == c.device(), "Device mismatch");

  AML_DEVICE_EVAL(a.device(), syrk(h, op, Na, Ka, alpha, a.data(),
      a.stride()[1], beta, c.data(), c.stride()[1]));
}

template <typename T>
void trsm(Handle h,
          char op,
          T alpha,
          const ImmutableMatrix<T> &a,
          Matrix<T> &b) {
  AML_CHECK_OP(op);

  AML_ASSERT(a.size()[0] == a.size()[1], "A must be square");

  Index Ma = a.size()[0];
  Index Mb = b.size()[0];
  Index Nb = b.size()[1];

  AML_ASSERT(Ma == Mb, "Leading dimension of A and B must match");

  AML_ASSERT(a.device() == b.device(), "Device mismatch");

  AML_DEVICE_EVAL(a.device(), trsm(h, op, Ma, Nb, alpha, a.data(),
      a.stride()[1], b.data(), b.stride()[1]));
}

}  // namespace aml

