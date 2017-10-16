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

inline float nrm2(aml::Handle h,
                  Index n,
                  const float *x) {
  AML_ASSERT_INT(n);
  auto tic = h.tic("nrm2s_" + std::to_string(n));
  float res = cblas_snrm2(n, x, 1);
  tic.stop();
  return res;
}

inline double nrm2(aml::Handle h,
                   Index n,
                   const double *x) {
  AML_ASSERT_INT(n);
  auto tic = h.tic("nrm2d_" + std::to_string(n));
  double res = cblas_dnrm2(n, x, 1);
  tic.stop();
  return res;
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
  auto tic = h.tic("gemvs_" + std::to_string(m) + "_" + std::to_string(n));
  cblas_sgemv(CblasColMajor,
      convert_op(op), m, n, alpha, a, lda, x, 1, beta, y, 1);
  tic.stop();
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
  auto tic = h.tic("gemvs_" + std::to_string(m) + "_" + std::to_string(n));
  cblas_dgemv(CblasColMajor,
      convert_op(op), m, n, alpha, a, lda, x, 1, beta, y, 1);
  tic.stop();
}

inline void trsv(aml::Handle h,
                 char op,
                 Index m,
                 const float *a,
                 Index lda,
                 float *x) {
  AML_ASSERT_INT(m, lda);
  auto tic = h.tic("trsvs_" + std::to_string(m));
  cblas_strsv(CblasColMajor, CblasLower,
      convert_op(op), CblasNonUnit, m, a, lda, x, 1);
  tic.stop();
}

inline void trsv(aml::Handle h,
                 char op,
                 Index m,
                 const double *a,
                 Index lda,
                 double *x) {
  AML_ASSERT_INT(m, lda);
  auto tic = h.tic("trsvd_" + std::to_string(m));

  cblas_dtrsv(CblasColMajor, CblasLower,
      convert_op(op), CblasNonUnit, m, a, lda, x, 1);
  tic.stop();
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
  auto tic = h.tic("gemms_"
      + std::to_string(m) + "_" + std::to_string(n) + "_" + std::to_string(k));

  cblas_sgemm(CblasColMajor, convert_op(op_a), convert_op(op_b),
      m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
  tic.stop();
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
  auto tic = h.tic("gemmd_"
      + std::to_string(m) + "_" + std::to_string(n) + "_" + std::to_string(k));

  cblas_dgemm(CblasColMajor, convert_op(op_a), convert_op(op_b),
      m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
  tic.stop();
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
  auto tic = h.tic("syrkd_" + std::to_string(n) + "_" + std::to_string(k));

  cblas_ssyrk(CblasColMajor, CblasLower,
      convert_op(op), n, k, alpha, a, lda, beta, c, ldc);
  tic.stop();
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
  auto tic = h.tic("syrkd_" + std::to_string(n) + "_" + std::to_string(k));

  cblas_dsyrk(CblasColMajor, CblasLower,
      convert_op(op), n, k, alpha, a, lda, beta, c, ldc);
  tic.stop();
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
  auto tic = h.tic("trsms_" + std::to_string(m) + "_" + std::to_string(n));

  cblas_strsm(
      CblasColMajor, CblasRight, CblasLower, convert_op(op), CblasNonUnit,
      m, n, alpha, a, lda, b, ldb);
  tic.stop();
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
  auto tic = h.tic("trsmd_" + std::to_string(m) + "_" + std::to_string(n));

  cblas_dtrsm(
      CblasColMajor, CblasRight, CblasLower, convert_op(op), CblasNonUnit,
      m, n, alpha, a, lda, b, ldb);
  tic.stop();
}

}  // namespace cpu
}  // namespace impl
}  // namespace aml

