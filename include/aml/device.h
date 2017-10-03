#pragma once

namespace aml {

#ifdef AML_GPU
enum Device { CPU, GPU };

#define AML_DEVICE_EVAL(device, function) \
  [&]() -> decltype(::aml::impl::cpu::function) { \
    do { \
      switch ((device)) { \
        case aml::GPU: return (::aml::impl::gpu::function); \
        case aml::CPU: return (::aml::impl::cpu::function); \
        default: abort(); \
      } \
    } while(0); \
  }()

#else
enum Device { CPU };

#define AML_DEVICE_EVAL(device, function) \
  [&]() -> decltype(::aml::impl::cpu::function) { \
    do { \
      switch ((device)) { \
        case aml::CPU: return (::aml::impl::cpu::function); \
        default: abort(); \
      } \
    } while(0); \
  }()

#endif

}  // namespace aml

