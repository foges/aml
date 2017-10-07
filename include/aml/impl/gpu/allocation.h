#pragma once

#include <aml/defs.h>

namespace aml {
namespace impl {
namespace gpu {

inline void* malloc(size_t size) {
  void *data = nullptr;
  AML_GPU_CHECK(cudaMalloc(&data, size));
  return data;
}

inline void free(void *data) {
  AML_GPU_CHECK(cudaFree(data));
}

}  // namespace gpu
}  // namespace impl
}  // namespace aml

