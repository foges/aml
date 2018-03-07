#pragma once

#include <cublas_v2.h>
#include <cusolverDn.h>

#include <aml/defs.h>
#include <aml/handle.h>

namespace aml {
namespace impl {
namespace gpu {

inline void potrf(aml::Handle h,
                  Index n,
                  float *a,
                  Index lda) {
  AML_ASSERT_INT(n, lda);

  int lwork = 0;
  cusolverStatus_t stat = solverDnSpotrf_bufferSize(h.gpu()->solverdn(),
      CUBLAS_FILL_MODE_LOWER, n, a, lda, &lwork);
  AML_ASSERT(stat == CUSOLVER_STATUS_SUCCESS, "CuSolver spotrf_buffer failed");
  float *work = h.gpu()->workspace(lwork);

  int info = 0;
  stat = cusolverDnSpotrf(h.gpu()->solverdn(), CUBLAS_FILL_MODE_LOWER,
      n, a, lda, work, lwork, &info);

  AML_ASSERT(info >= 0, "CuSolver failed incorrect parameter");
  AML_ASSERT(info <= 0, "CuSolver failed not positive definite");
  AML_ASSERT(stat == CUSOLVER_STATUS_SUCCESS, "CuSolver spotrf failed");
}

inline void potrf(aml::Handle h,
                  Index n,
                  double *a,
                  Index lda) {
  AML_ASSERT_INT(n, lda);

  int lwork = 0;
  cusolverStatus_t stat = solverDnDpotrf_bufferSize(h.gpu()->solverdn(),
      CUBLAS_FILL_MODE_LOWER, n, a, lda, &lwork);
  AML_ASSERT(stat == CUSOLVER_STATUS_SUCCESS, "CuSolver dpotrf_buffer failed");
  double *work = h.gpu()->workspace(lwork);

  int info = 0;
  stat = cusolverDnDpotrf(h.gpu()->solverdn(), CUBLAS_FILL_MODE_LOWER,
      n, a, lda, work, lwork, &info);

  AML_ASSERT(info >= 0, "CuSolver dpotrf failed incorrect parameter");
  AML_ASSERT(info <= 0, "CuSolver dpotrf failed not positive definite");
  AML_ASSERT(stat == CUSOLVER_STATUS_SUCCESS, "CuSolver dpotrf failed");
}

}  // namespace gpu
}  // namespace impl
}  // namespace aml

