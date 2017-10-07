#pragma once

#include <cblas.h>

#include <aml/defs.h>

namespace aml {
namespace impl {
namespace cpu {

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

}  // namespace cpu
}  // namespace impl
}  // namespace aml

