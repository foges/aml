#pragma once

#include <cblas.h>

#include <aml/defs.h>

namespace aml {
namespace impl {
namespace cpu {

/** GEMM **********************************************************************/

inline CBLAS_TRANSPOSE convert_op(OP op) {
  return op == NO_TRANS ? CblasNoTrans : CblasTrans;
}

inline void gemm(OP op_a,
                 OP op_b,
                 Index m,
                 Index n,
                 Index k,
                 float alpha,
                 const float *a,
                 Index lda,
                 const float *b,
                 Index ldb,
                 float beta,
                 float *c,
                 Index ldc) {
  cblas_sgemm(CblasColMajor, convert_op(op_a), convert_op(op_b),
      m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

inline void gemm(OP op_a,
                 OP op_b,
                 Index m,
                 Index n,
                 Index k,
                 double alpha,
                 const double *a,
                 Index lda,
                 const double *b,
                 Index ldb,
                 double beta,
                 double *c,
                 Index ldc) {
  cblas_dgemm(CblasColMajor, convert_op(op_a), convert_op(op_b),
      m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

/** UNARY_OP ******************************************************************/

template <typename Tin, typename Tout, typename Op>
void unary_op(const Tin *in,
              Shape<0> in_stride,
              Index in_start,
              Tout *out,
              Shape<0> out_stride,
              Index out_start,
              Shape<0> shape,
              const Op &op) {
  out[out_start] = op(in[in_start]);
}

template <typename Tin, typename Tout, int Dim, typename Op>
void unary_op(const Tin *in,
              Shape<Dim> in_stride,
              Index in_start,
              Tout *out,
              Shape<Dim> out_stride,
              Index out_start,
              Shape<Dim> shape,
              const Op &op) {
  for (Index i = 0; i < shape.head(); ++i) {
    unary_op(
        in, in_stride.tail(), in_start,
        out, out_stride.tail(), out_start,
        shape.tail(), op);

    in_start += in_stride.head();
    out_start += out_stride.head();
  }
}

template <typename Tin, typename Tout, int Dim, typename Op>
void unary_op(const ConstArray<Tin, Dim> &in,
              Array<Tout, Dim> &out,
              const Op &op) {
  unary_op(
      in.data(), in.stride(), 0,
      out.data(), out.stride(), 0,
      in.shape(), op);
}

}  // namespace cpu
}  // namespace impl
}  // namespace aml

