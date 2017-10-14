#pragma once

#include <cublas_v2.h>

#include <aml/defs.h>
#include <aml/handle.h>

namespace aml {
namespace impl {
namespace gpu {

inline cublasOperation_t convert_op(OP op) {
  return op == NO_TRANS ? CUBLAS_OP_N : CUBLAS_OP_T;
}

inline void gemm(aml::Handle h,
                 OP op_a,
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
  cublasStatus_t stat = cublasSgemm(h.gpu()->cublas(), convert_op(op_a),
      convert_op(op_b), m, n, k, &alpha, a, lda, b, ldb, &beta, c, ldc);
  AML_ASSERT(stat == CUBLAS_STATUS_SUCCESS, "CuBLAS sgemm failed");
}

inline void gemm(aml::Handle h,
                 OP op_a,
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
  cublasStatus_t stat = cublasDgemm(h.gpu()->cublas(), convert_op(op_a),
      convert_op(op_b), m, n, k, &alpha, a, lda, b, ldb, &beta, c, ldc);
  AML_ASSERT(stat == CUBLAS_STATUS_SUCCESS, "CuBLAS dgemm failed");
}

inline float nrm2(aml::Handle h,
                  Index n,
                  const float *x) {
  float result = 0;
  cublasStatus_t stat = cublasSnrm2(h.gpu()->cublas(), n, x, 1, &result);
  AML_ASSERT(stat == CUBLAS_STATUS_SUCCESS, "CuBLAS snrm2 failed");
  return result;
}

inline double nrm2(aml::Handle h,
                   Index n,
                   const double *x) {
  double result = 0;
  cublasStatus_t stat = cublasDnrm2(h.gpu()->cublas(), n, x, 1, &result);
  AML_ASSERT(stat == CUBLAS_STATUS_SUCCESS, "CuBLAS dnrm2 failed");
  return result;
}

}  // namespace gpu
}  // namespace impl
}  // namespace aml

