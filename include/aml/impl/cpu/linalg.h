#pragma once

#include <clapack.h>

#include <aml/defs.h>
#include <aml/handle.h>

namespace aml {
namespace impl {
namespace cpu {

inline void potrf(aml::Handle,
                  Index n,
                  float *a,
                  Index lda) {
  AML_ASSERT_INT(n, lda);

  int info = 0;
  char uplo = 'L';
  int ni = static_cast<int>(n);
  int ldai = static_cast<int>(lda);
  spotrf_(&uplo, &ni, &a, &ldai, &info);

  AML_ASSERT(info >= 0, "Clapack spotrf failed incorrect parameter");
  AML_ASSERT(info <= 0, "Clapack spotrf failed not positive definite");
}

inline void potrf(aml::Handle,
                  Index n,
                  double *a,
                  Index lda) {
  AML_ASSERT_INT(n, lda);

  int info = 0;
  char uplo = 'L';
  int ni = static_cast<int>(n);
  int ldai = static_cast<int>(lda);
  dpotrf_(&uplo, &ni, &a, &ldai, &info);

  AML_ASSERT(info >= 0, "Clapack dpotrf failed incorrect parameter");
  AML_ASSERT(info <= 0, "Clapack dpotrf failed not positive definite");
}

}  // namespace cpu
}  // namespace impl
}  // namespace aml

