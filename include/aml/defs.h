#pragma once

#include <cassert>
#include <cstdint>

#ifdef AML_GPU
#define AML_HOST_DEVICE __host__ __device__
#else
#define AML_HOST_DEVICE
#endif

#define AML_DEBUG_ASSERT(x) assert((x))

namespace aml {
using Index = uint32_t;
}  // namespace aml
