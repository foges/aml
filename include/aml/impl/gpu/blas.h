#pragma once

#include <cublas_v2.h>

#include <aml/defs.h>
#include <aml/handle.h>

namespace aml {
namespace impl {
namespace gpu {

inline cublasOperation_t convert_op(char op) {
  if (op == 'n') {
    return CUBLAS_OP_N;
  } else {
    return CUBLAS_OP_T;
  }
}

/** BLAS LEVEL 1 **************************************************************/

inline float nrm2(aml::Handle h,
                  Index n,
                  const float *x) {
  AML_ASSERT_INT(n);
  float result = 0;
  cublasStatus_t stat = cublasSnrm2(h.gpu()->cublas(), n, x, 1, &result);
  AML_ASSERT(stat == CUBLAS_STATUS_SUCCESS, "CuBLAS snrm2 failed");
  return result;
}

inline double nrm2(aml::Handle h,
                   Index n,
                   const double *x) {
  AML_ASSERT_INT(n);
  double result = 0;
  cublasStatus_t stat = cublasDnrm2(h.gpu()->cublas(), n, x, 1, &result);
  AML_ASSERT(stat == CUBLAS_STATUS_SUCCESS, "CuBLAS dnrm2 failed");
  return result;
}

/** BLAS LEVEL 2 **************************************************************/

inline void gemv(aml::Handle h,
                 char op,
                 Index m,
                 Index n,
                 float alpha,
                 const float *a,
                 Index lda,
                 const float *x,
                 float beta,
                 float *y) {
  AML_ASSERT_INT(m, n, lda);
  cublasStatus_t stat = cublasSgemv(h.gpu()->cublas(), convert_op(op),
      m, n, &alpha, a, lda, x, 1, &beta, y, 1);
  AML_ASSERT(stat == CUBLAS_STATUS_SUCCESS, "CuBLAS sgemv failed");
}

inline void gemv(aml::Handle h,
                 char op,
                 Index m,
                 Index n,
                 double alpha,
                 const double *a,
                 Index lda,
                 const double *x,
                 double beta,
                 double *y) {
  AML_ASSERT_INT(m, n, lda);
  cublasStatus_t stat = cublasDgemv(h.gpu()->cublas(), convert_op(op),
      m, n, &alpha, a, lda, x, 1, &beta, y, 1);
  AML_ASSERT(stat == CUBLAS_STATUS_SUCCESS, "CuBLAS dgemv failed");
}

inline void trsv(aml::Handle h,
                 char op,
                 Index m,
                 const float *a,
                 Index lda,
                 float *x) {
  AML_ASSERT_INT(m, lda);
  cublasStatus_t stat = cublasStrsv(h.gpu()->cublas(), CUBLAS_FILL_MODE_LOWER,
      convert_op(op), CUBLAS_DIAG_NON_UNIT, m, a, lda, x, 1);
  AML_ASSERT(stat == CUBLAS_STATUS_SUCCESS, "CuBLAS strsv failed");
}

inline void trsv(aml::Handle h,
                 char op,
                 Index m,
                 const double *a,
                 Index lda,
                 double *x) {
  AML_ASSERT_INT(m, lda);
  cublasStatus_t stat = cublasDtrsv(h.gpu()->cublas(), CUBLAS_FILL_MODE_LOWER,
      convert_op(op), CUBLAS_DIAG_NON_UNIT, m, a, lda, x, 1);
  AML_ASSERT(stat == CUBLAS_STATUS_SUCCESS, "CuBLAS dtrsv failed");
}

/** BLAS LEVEL 3 **************************************************************/

inline void gemm(aml::Handle h,
                 char op_a,
                 char op_b,
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
  AML_ASSERT_INT(m, n, k, lda, ldb, ldc);
  cublasStatus_t stat = cublasSgemm(h.gpu()->cublas(), convert_op(op_a),
      convert_op(op_b), m, n, k, &alpha, a, lda, b, ldb, &beta, c, ldc);
  AML_ASSERT(stat == CUBLAS_STATUS_SUCCESS, "CuBLAS sgemm failed");
}

inline void gemm(aml::Handle h,
                 char op_a,
                 char op_b,
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
  AML_ASSERT_INT(m, n, k, lda, ldb, ldc);
  cublasStatus_t stat = cublasDgemm(h.gpu()->cublas(), convert_op(op_a),
      convert_op(op_b), m, n, k, &alpha, a, lda, b, ldb, &beta, c, ldc);
  AML_ASSERT(stat == CUBLAS_STATUS_SUCCESS, "CuBLAS dgemm failed");
}

inline void syrk(aml::Handle h,
                 char op,
                 Index n,
                 Index k,
                 float alpha,
                 const float *a,
                 Index lda,
                 float beta,
                 float *c,
                 Index ldc) {
  AML_ASSERT_INT(n, k, lda, ldc);
  cublasStatus_t stat = cublasSsyrk(h.gpu()->cublas(), CUBLAS_FILL_MODE_LOWER,
      convert_op(op), n, k, &alpha, a, lda, &beta, c, ldc);
  AML_ASSERT(stat == CUBLAS_STATUS_SUCCESS, "CuBLAS ssyrk failed");
}

inline void syrk(aml::Handle h,
                 char op,
                 Index n,
                 Index k,
                 double alpha,
                 const double *a,
                 Index lda,
                 double beta,
                 double *c,
                 Index ldc) {
  AML_ASSERT_INT(n, k, lda, ldc);
  cublasStatus_t stat = cublasDsyrk(h.gpu()->cublas(), CUBLAS_FILL_MODE_LOWER,
      convert_op(op), n, k, &alpha, a, lda, &beta, c, ldc);
  AML_ASSERT(stat == CUBLAS_STATUS_SUCCESS, "CuBLAS dsyrk failed");
}

inline void trsm(aml::Handle h,
                 char op,
                 Index m,
                 Index n,
                 float alpha,
                 const float *a,
                 Index lda,
                 float *b,
                 Index ldb) {
  AML_ASSERT_INT(m, n, lda, ldb);
  cublasStatus_t stat = cublasStrsm(h.gpu()->cublas(),
      CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_LOWER, convert_op(op),
      CUBLAS_DIAG_NON_UNIT, m, n, &alpha, a, lda, b, ldb);
  AML_ASSERT(stat == CUBLAS_STATUS_SUCCESS, "CuBLAS sstrsm failed");
}

inline void trsm(aml::Handle h,
                 char op,
                 Index m,
                 Index n,
                 double alpha,
                 const double *a,
                 Index lda,
                 double *b,
                 Index ldb) {
  AML_ASSERT_INT(m, n, lda, ldb);
  cublasStatus_t stat = cublasDtrsm(h.gpu()->cublas(),
      CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_LOWER, convert_op(op),
      CUBLAS_DIAG_NON_UNIT, m, n, &alpha, a, lda, b, ldb);
  AML_ASSERT(stat == CUBLAS_STATUS_SUCCESS, "CuBLAS dstrsm failed");
}

}  // namespace gpu
}  // namespace impl
}  // namespace aml

