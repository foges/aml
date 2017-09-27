#pragma once

namespace aml {

#ifdef AML_GPU
enum Device { CPU, GPU };

#define AML_DEVICE_EVAL(device, function) \
  [&]() -> decltype(cpu::function) { \
    do { \
      switch ((device)) { \
        case aml::GPU: return (gpu::function); \
        case aml::CPU: return (cpu::function); \
        default: abort(); \
      } \
    } while(0); \
  }()

#else
enum Device { CPU };

#define AML_DEVICE_EVAL(device, function) \
  [&]() -> decltype(cpu::function) { \
    do { \
      switch ((device)) { \
        case aml::CPU: return (cpu::function); \
        default: abort(); \
      } \
    } while(0); \
  }()

#endif

}  // namespace aml

