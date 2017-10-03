#pragma once

#include <aml/defs.h>

namespace aml {
namespace impl {

namespace cpu {

inline void* malloc(size_t size) {
  return std::malloc(size);
}

inline void free(void *data) {
  std::free(data);
}

}  // namespace cpu

#ifdef AML_GPU

namespace gpu {

inline void* malloc(size_t size) {
  void *data = nullptr;
  AML_GPU_CHECK(cudaMalloc(&data, size));
  return data;
}

inline void free(void *data) {
  AML_GPU_ERROR(cudaFree(data));
}

}  // namespace gpu

#endif  // AML_GPU

}  // namespace impl
}  // namespace ml

