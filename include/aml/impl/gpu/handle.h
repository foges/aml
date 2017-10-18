#pragma once

#include <cublas_v2.h>
#include <cusolverDn.h>

#include <aml/defs.h>

namespace aml {
namespace impl {
namespace gpu {

class Handle {
public:
  Handle() : workspace_(nullptr), workspace_size_(0) {
    {
      cudaError_t stat = cudaGetDeviceProperties(&device_properties_, 0);
      AML_ASSERT(stat == cudaSuccess, "Could not get device properties");
    }
    {
      cublasStatus_t stat = cublasCreate(&h_cublas_);
      AML_ASSERT(stat == CUBLAS_STATUS_SUCCESS,
          "CuBLAS initialization failed");
    }
    {
      cusolverStatus_t stat = cusolverDnCreate(&h_cusolverdn_);
      AML_ASSERT(stat == CUSOLVER_STATUS_SUCCESS,
          "CuSolverDn initialization failed");
    }
  }

  ~Handle() {
    this->clear();
    {
      cublasStatus_t stat = cublasDestroy(h_cublas_);
      AML_ASSERT(stat == CUBLAS_STATUS_SUCCESS,
          "CuBLAS destruction failed");
    }
    {
      cusolverStatus_t stat = cusolverDnDestroy(h_cusolverdn_);
      AML_ASSERT(stat == CUSOLVER_STATUS_SUCCESS,
          "CuSolverDn destruction failed");
    }
  }

  void clear() {
    if (workspace_ != nullptr) {
      AML_GPU_CHECK(cudaFree(workspace_));
    }
    workspace_ = nullptr;
    workspace_size_ = 0;
  }

  void synchronize() const {
    // TODO change stream
    cudaStreamSynchronize(nullptr);
  }

  cublasHandle_t cublas() {
    return h_cublas_;
  }

  cusolverDnHandle_t cusolverdn() {
    return h_cusolverdn_;
  }

  template <typename T>
  T* workspace(size_t numel) {
    size_t size = numel * sizeof(T);
    if (workspace_size_ < size) {
      this->clear();
      AML_GPU_CHECK(cudaMalloc(&workspace_, size));
      workspace_size_ = size;
    }
    return static_cast<T*>(workspace_);
  }

  int num_procs() const {
    return device_properties_.multiProcessorCount;
  }

  int shared_mem_per_proc() const {
    return 49152;
  }

  Handle(const Handle&) = delete;
  Handle& operator=(const Handle&) = delete;

private:
  cudaDeviceProp device_properties_;
  cublasHandle_t h_cublas_;
  cusolverDnHandle_t h_cusolverdn_;

  void *workspace_;
  size_t workspace_size_;
};

}  // namespace gpu
}  // namespace impl
}  // namespace aml

