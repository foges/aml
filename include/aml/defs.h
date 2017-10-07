#pragma once

#include <cassert>
#include <cstdint>
#include <cstdio>
#include <sstream>
#include <stdexcept>

// Print and assert
#define AML_STRINGIFY(x) #x
#define AML_TOSTRING(x) AML_STRINGIFY(x)

#define AML_ASSERT_EXCEPTION(exception, statement, message) \
  do { \
    if (!(statement)) { \
      std::stringstream ss; \
      ss << __FILE__ << ":" \
         << __LINE__ << " " \
         << AML_TOSTRING((statement)) << " - " \
         << (message); \
      throw exception(ss.str()); \
    } \
  } while(0)

#define AML_ASSERT(statement, message) \
  AML_ASSERT_EXCEPTION(std::runtime_error, (statement), (message))

// Debug
#ifdef AML_DEBUG

#define AML_DEBUG_ASSERT(statement) assert(statement)
#define AML_DEBUG_PRINTF(...) std::printf(_1, ...)

#else

#define AML_DEBUG_ASSERT(x)
#define AML_DEBUG_PRINTF(x)

#endif  // endif AML_DEBUG

// GPU
#ifdef AML_GPU
#define AML_HOST_DEVICE __host__ __device__

#define AML_GPU_CHECK(statement) \
  do { \
    cudaError_t error = (statement); \
    AML_ASSERT(error == cudaSuccess, cudaGetErrorString(error)); \
  } while(0)

#else

#define AML_HOST_DEVICE

#endif  // endif AML_GPU

// Types
namespace aml {

using Index = uint32_t;

enum OP { TRANS, NO_TRANS };

}  // namespace aml

