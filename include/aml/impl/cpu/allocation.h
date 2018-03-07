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
}  // namespace impl
}  // namespace aml

