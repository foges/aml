#pragma once

#include <chrono>

#include <aml/defs.h>

namespace aml {
namespace impl {

enum Profile {
  // CPU
  PROF_CPU_BLAS_NRM2,
  PROF_CPU_BLAS_GEMV,
  PROF_CPU_BLAS_TRSV,
  PROF_CPU_BLAS_GEMM,
  PROF_CPU_BLAS_SYRK,
  PROF_CPU_BLAS_TRSM,

  PROF_CPU_LINALG_POTRF,

  PROF_CPU_OP_SET,
  PROF_CPU_OP_UNARY,
  PROF_CPU_OP_BINARY,
  PROF_CPU_OP_REDUCE,
  PROF_CPU_OP_COPY,

  // GPU
  PROF_GPU_BLAS_NRM2,
  PROF_GPU_BLAS_GEMV,
  PROF_GPU_BLAS_TRSV,
  PROF_GPU_BLAS_GEMM,
  PROF_GPU_BLAS_SYRK,
  PROF_GPU_BLAS_TRSM,

  PROF_GPU_LINALG_POTRF,

  PROF_GPU_OP_SET,
  PROF_GPU_OP_UNARY,
  PROF_GPU_OP_BINARY,
  PROF_GPU_OP_REDUCE,
  PROF_GPU_OP_COPY_G2C,
  PROF_GPU_OP_COPY_G2G,
  PROF_GPU_OP_COPY_C2G,

  // DON'T TOUCH
  PROF_SIZE
};

inline std::string name(Profile prof) {
  static const char *names[] = {
      // CPU
      "CPU_BLAS_NRM2",
      "CPU_BLAS_GEMV",
      "CPU_BLAS_TRSV",
      "CPU_BLAS_GEMM",
      "CPU_BLAS_SYRK",
      "CPU_BLAS_TRSM",

      "CPU_LINALG_POTRF",

      "CPU_OP_SET",
      "CPU_OP_UNARY",
      "CPU_OP_BINARY",
      "CPU_OP_REDUCE",
      "CPU_OP_COPY",

      // GPU
      "GPU_BLAS_NRM2",
      "GPU_BLAS_GEMV",
      "GPU_BLAS_TRSV",
      "GPU_BLAS_GEMM",
      "GPU_BLAS_SYRK",
      "GPU_BLAS_TRSM",

      "GPU_LINALG_POTRF",

      "GPU_OP_SET",
      "GPU_OP_UNARY",
      "GPU_OP_BINARY",
      "GPU_OP_REDUCE",
      "GPU_OP_COPY_G2C",
      "GPU_OP_COPY_G2G",
      "GPU_OP_COPY_C2G",
  };

  static_assert(sizeof(names) / sizeof(names[0]) == PROF_SIZE,
      "Profiler names mismatch");
  AML_ASSERT(prof < PROF_SIZE, "Profile out of bounds");
  return names[prof];
}

using time_point = std::chrono::time_point<std::chrono::high_resolution_clock>;

class Profiler {
public:
  Profiler() : duration_us_(), start_times_() { }

  void tic(Profile prof) {
    AML_DEBUG_ASSERT(prof < PROF_SIZE);
    start_times_[prof] = std::chrono::high_resolution_clock::now();
  }

  void toc(Profile prof) {
    AML_DEBUG_ASSERT(prof < PROF_SIZE);
    auto elapsed =
        std::chrono::high_resolution_clock::now() - start_times_[prof];
    duration_us_[prof] +=
        std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();
  }

  std::string to_string() const {
    // TODO
    return "";
  }

private:
  //bool has_started_[PROF_SIZE]; // Debug
  uint64_t duration_us_[PROF_SIZE];
  time_point start_times_[PROF_SIZE];
};

}  // namespace impl
}  // namespace aml

