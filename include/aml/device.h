#pragma once

namespace aml {

#ifdef AML_GPU
enum Device { CPU, GPU };

#define AML_DEVICE_EVAL(device, function) \
  do { \
    switch ((device)) { \
      case GPU: gpu::function; break; \
      case CPU: cpu::function; break; \
      default: abort(); \
    } \
  } while (0)

#else
enum Device { CPU };

#define AML_DEVICE_EVAL(device, function) \
  do { \
    switch ((device)) { \
      case CPU: cpu::function; break; \
      default: abort(); \
    } \
  } while (0)
#endif

}  // namespace aml

