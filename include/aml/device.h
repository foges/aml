#pragma once

namespace aml {

#ifdef AML_GPU
enum Device { CPU, GPU };

#define AML_DEVICE_EVAL(device, ...) \
  [&]() -> decltype(::aml::impl::cpu::__VA_ARGS__) { \
    do { \
      switch ((device)) { \
        case aml::GPU: return (::aml::impl::gpu::__VA_ARGS__); \
        case aml::CPU: return (::aml::impl::cpu::__VA_ARGS__); \
        default: abort(); \
      } \
    } while(0); \
  }()

#else
enum Device { CPU };

#define AML_DEVICE_EVAL(device, ...) \
  [&]() -> decltype(::aml::impl::cpu::__VA_ARGS__) { \
    do { \
      switch ((device)) { \
        case aml::CPU: return (::aml::impl::cpu::__VA_ARGS__); \
        default: abort(); \
      } \
    } while(0); \
  }()

#endif

}  // namespace aml

