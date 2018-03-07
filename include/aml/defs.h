#pragma once

#include <cassert>
#include <cstdint>
#include <cstdio>
#include <sstream>
#include <stdexcept>

// Print and assert
#define AML_STRINGIFY(x) #x
#define AML_TOSTRING(x) AML_STRINGIFY(x)
#define AML_GET_MACRO6(_1, _2, _3, _4, _5, _6, NAME, ...) NAME

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

#define AML_INT_CHECK1(x0) \
  (static_cast<decltype(x0)>(static_cast<int>(x0)) == x0)
#define AML_INT_CHECK2(x0, x1) \
  AML_INT_CHECK1(x0) && AML_INT_CHECK1(x1)
#define AML_INT_CHECK3(x0, x1, x2) \
  AML_INT_CHECK1(x0) && AML_INT_CHECK2(x1, x2)
#define AML_INT_CHECK4(x0, x1, x2, x3) \
  AML_INT_CHECK1(x0) && AML_INT_CHECK3(x1, x2, x3)
#define AML_INT_CHECK5(x0, x1, x2, x3, x4) \
  AML_INT_CHECK1(x0) && AML_INT_CHECK4(x1, x2, x3, x4)
#define AML_INT_CHECK6(x0, x1, x2, x3, x4, x5) \
  AML_INT_CHECK1(x0) && AML_INT_CHECK5(x1, x2, x3, x4, x5)
#define AML_INT_CHECK(...) \
  AML_GET_MACRO6(__VA_ARGS__, \
      AML_INT_CHECK6, AML_INT_CHECK5, AML_INT_CHECK4, \
      AML_INT_CHECK3, AML_INT_CHECK2, AML_INT_CHECK1)(__VA_ARGS__)

#define AML_ASSERT_INT(...) \
  AML_ASSERT(AML_INT_CHECK(__VA_ARGS__), "Integer overflow")

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

#define CUDA_CHECK_ERR() \
  do { \
    cudaError_t error = cudaGetLastError(); \
    if (error != cudaSuccess) { \
      std::cout << __FILE__ << ":" << __LINE__ << ":" \
                << __func__ << "\n" \
                << "ERROR_CUDA: " << cudaGetErrorString(error) \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
  } while (0)

#else

#define AML_HOST_DEVICE

#endif  // endif AML_GPU

namespace aml {

template <bool B, typename T = void>
using enable_if_t = typename std::enable_if<B, T>::type;

}  // namespace aml
