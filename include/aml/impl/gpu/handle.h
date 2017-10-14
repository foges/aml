#pragma once

#include <cublas_v2.h>

#include <aml/defs.h>

namespace aml {
namespace impl {
namespace gpu {

class Handle {
public:
  Handle() {
    cublasStatus_t stat = cublasCreate(&h_cublas);
    AML_ASSERT(stat == CUBLAS_STATUS_SUCCESS, "CuBLAS initialization failed");
  }

  ~Handle() {
    cublasDestroy(h_cublas);
  }

  cublasHandle_t cublas() {
    return h_cublas;
  }

private:
  cublasHandle_t h_cublas;
};

}  // namespace gpu
}  // namespace impl
}  // namespace aml

