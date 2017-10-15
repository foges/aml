#pragma once

#include <cblas.h>

#include <aml/defs.h>
#include <aml/handle.h>

namespace aml {
namespace impl {
namespace cpu {

inline CBLAS_TRANSPOSE convert_op(char op) {
  if (op == 'n') {
    return CblasNoTrans;
  } else {
    return CblasTrans;
  }
}

/** BLAS LEVEL 1 **************************************************************/

inline float nrm2(aml::Handle,
                  Index n,
                  const float *x) {
  AML_ASSERT_INT(n);
  return cblas_snrm2(n, x, 1);
}

inline double nrm2(aml::Handle,
                   Index n,
                   const double *x) {
  AML_ASSERT_INT(n);
  return cblas_dnrm2(n, x, 1);
}

/** BLAS LEVEL 2 **************************************************************/

inline void gemv(aml::Handle,
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
  cblas_sgemv(CblasColMajor,
      convert_op(op), m, n, alpha, a, lda, x, 1, beta, y, 1);
}

inline void gemv(aml::Handle,
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
  cblas_dgemv(CblasColMajor,
      convert_op(op), m, n, alpha, a, lda, x, 1, beta, y, 1);
}

inline void trsv(aml::Handle,
                 char op,
                 Index m,
                 const float *a,
                 Index lda,
                 float *x) {
  AML_ASSERT_INT(m, lda);
  cblas_strsv(CblasColMajor, CblasLower,
      convert_op(op), CblasNonUnit, m, a, lda, x, 1);
}

inline void trsv(aml::Handle,
                 char op,
                 Index m,
                 const double *a,
                 Index lda,
                 double *x) {
  AML_ASSERT_INT(m, lda);
  cblas_dtrsv(CblasColMajor, CblasLower,
      convert_op(op), CblasNonUnit, m, a, lda, x, 1);
}

/** BLAS LEVEL 3 **************************************************************/

inline void gemm(aml::Handle,
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
  cblas_sgemm(CblasColMajor, convert_op(op_a), convert_op(op_b),
      m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

inline void gemm(aml::Handle,
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
  cblas_dgemm(CblasColMajor, convert_op(op_a), convert_op(op_b),
      m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

inline void syrk(aml::Handle,
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
  cblas_ssyrk(CblasColMajor, CblasLower,
      convert_op(op), n, k, alpha, a, lda, beta, c, ldc);
}

inline void syrk(aml::Handle,
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
  cblas_dsyrk(CblasColMajor, CblasLower,
      convert_op(op), n, k, alpha, a, lda, beta, c, ldc);
}

inline void trsm(aml::Handle,
                 char op,
                 Index m,
                 Index n,
                 float alpha,
                 const float *a,
                 Index lda,
                 float *b,
                 Index ldb) {
  AML_ASSERT_INT(m, n, lda, ldb);
  cblas_strsm(
      CblasColMajor, CblasRight, CblasLower, convert_op(op), CblasNonUnit,
      m, n, alpha, a, lda, b, ldb);
}

inline void trsm(aml::Handle,
                 char op,
                 Index m,
                 Index n,
                 double alpha,
                 const double *a,
                 Index lda,
                 double *b,
                 Index ldb) {
  AML_ASSERT_INT(m, n, lda, ldb);
  cblas_dtrsm(
      CblasColMajor, CblasRight, CblasLower, convert_op(op), CblasNonUnit,
      m, n, alpha, a, lda, b, ldb);
}

}  // namespace cpu
}  // namespace impl
}  // namespace aml

