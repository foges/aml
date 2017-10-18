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
  auto tic = h.tic("gpu_snrm2_" + std::to_string(n), [h]{ h.synchronize(); });

  float result = 0;
  cublasStatus_t stat = cublasSnrm2(h.gpu()->cublas(), n, x, 1, &result);

  AML_ASSERT(stat == CUBLAS_STATUS_SUCCESS, "CuBLAS snrm2 failed");
  tic.stop();

  return result;
}

inline double nrm2(aml::Handle h,
                   Index n,
                   const double *x) {
  AML_ASSERT_INT(n);
  auto tic = h.tic("gpu_dnrm2_" + std::to_string(n), [h]{ h.synchronize(); });

  double result = 0;
  cublasStatus_t stat = cublasDnrm2(h.gpu()->cublas(), n, x, 1, &result);

  AML_ASSERT(stat == CUBLAS_STATUS_SUCCESS, "CuBLAS dnrm2 failed");
  tic.stop();

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
  auto tic = h.tic("gpu_sgemv_" + std::to_string(m) + "_" + std::to_string(n),
      [h]{ h.synchronize(); });

  cublasStatus_t stat = cublasSgemv(h.gpu()->cublas(), convert_op(op),
      m, n, &alpha, a, lda, x, 1, &beta, y, 1);

  AML_ASSERT(stat == CUBLAS_STATUS_SUCCESS, "CuBLAS sgemv failed");
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
  auto tic = h.tic("gpu_dgemv_" + std::to_string(m) + "_" + std::to_string(n),
      [h]{ h.synchronize(); });

  cublasStatus_t stat = cublasDgemv(h.gpu()->cublas(), convert_op(op),
      m, n, &alpha, a, lda, x, 1, &beta, y, 1);

  AML_ASSERT(stat == CUBLAS_STATUS_SUCCESS, "CuBLAS dgemv failed");
  tic.stop();
}

inline void trsv(aml::Handle h,
                 char op,
                 Index m,
                 const float *a,
                 Index lda,
                 float *x) {
  AML_ASSERT_INT(m, lda);
  auto tic = h.tic("gpu_strsv_" + std::to_string(m), [h]{ h.synchronize(); });

  cublasStatus_t stat = cublasStrsv(h.gpu()->cublas(), CUBLAS_FILL_MODE_LOWER,
      convert_op(op), CUBLAS_DIAG_NON_UNIT, m, a, lda, x, 1);

  AML_ASSERT(stat == CUBLAS_STATUS_SUCCESS, "CuBLAS strsv failed");
  tic.stop();
}

inline void trsv(aml::Handle h,
                 char op,
                 Index m,
                 const double *a,
                 Index lda,
                 double *x) {
  AML_ASSERT_INT(m, lda);
  auto tic = h.tic("gpu_dtrsv_" + std::to_string(m), [h]{ h.synchronize(); });

  cublasStatus_t stat = cublasDtrsv(h.gpu()->cublas(), CUBLAS_FILL_MODE_LOWER,
      convert_op(op), CUBLAS_DIAG_NON_UNIT, m, a, lda, x, 1);

  AML_ASSERT(stat == CUBLAS_STATUS_SUCCESS, "CuBLAS dtrsv failed");
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
  auto tic = h.tic("gpu_sgemm_"
      + std::to_string(m) + "_" + std::to_string(n) + "_" + std::to_string(k),
      [h]{ h.synchronize(); });

  cublasStatus_t stat = cublasSgemm(h.gpu()->cublas(), convert_op(op_a),
      convert_op(op_b), m, n, k, &alpha, a, lda, b, ldb, &beta, c, ldc);

  AML_ASSERT(stat == CUBLAS_STATUS_SUCCESS, "CuBLAS sgemm failed");
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
  auto tic = h.tic("gpu_dgemm_"
      + std::to_string(m) + "_" + std::to_string(n) + "_" + std::to_string(k),
      [h]{ h.synchronize(); });

  cublasStatus_t stat = cublasDgemm(h.gpu()->cublas(), convert_op(op_a),
      convert_op(op_b), m, n, k, &alpha, a, lda, b, ldb, &beta, c, ldc);

  AML_ASSERT(stat == CUBLAS_STATUS_SUCCESS, "CuBLAS dgemm failed");
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
  auto tic = h.tic("gpu_ssyrk_" + std::to_string(n) + "_" + std::to_string(k),
      [h]{ h.synchronize(); });

  cublasStatus_t stat = cublasSsyrk(h.gpu()->cublas(), CUBLAS_FILL_MODE_LOWER,
      convert_op(op), n, k, &alpha, a, lda, &beta, c, ldc);

  AML_ASSERT(stat == CUBLAS_STATUS_SUCCESS, "CuBLAS ssyrk failed");
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
  auto tic = h.tic("gpu_dsyrk_" + std::to_string(n) + "_" + std::to_string(k),
      [h]{ h.synchronize(); });

  cublasStatus_t stat = cublasDsyrk(h.gpu()->cublas(), CUBLAS_FILL_MODE_LOWER,
      convert_op(op), n, k, &alpha, a, lda, &beta, c, ldc);

  AML_ASSERT(stat == CUBLAS_STATUS_SUCCESS, "CuBLAS dsyrk failed");
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
  auto tic = h.tic("gpu_strsm_" + std::to_string(m) + "_" + std::to_string(n),
      [h]{ h.synchronize(); });

  cublasStatus_t stat = cublasStrsm(h.gpu()->cublas(),
      CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_LOWER, convert_op(op),
      CUBLAS_DIAG_NON_UNIT, m, n, &alpha, a, lda, b, ldb);

  AML_ASSERT(stat == CUBLAS_STATUS_SUCCESS, "CuBLAS sstrsm failed");
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
  auto tic = h.tic("gpu_dtrsm_" + std::to_string(m) + "_" + std::to_string(n),
      [h]{ h.synchronize(); });

  cublasStatus_t stat = cublasDtrsm(h.gpu()->cublas(),
      CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_LOWER, convert_op(op),
      CUBLAS_DIAG_NON_UNIT, m, n, &alpha, a, lda, b, ldb);

  AML_ASSERT(stat == CUBLAS_STATUS_SUCCESS, "CuBLAS dstrsm failed");
  tic.stop();
}

}  // namespace gpu
}  // namespace impl
}  // namespace aml

