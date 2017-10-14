#pragma once

#include <utility>

#include <aml/defs.h>

namespace aml {
namespace impl {
namespace gpu {

inline std::pair<int, int> launch_dims(Index numel,
                                       int shared_mem_per_thread=0) {
  // TODO
  // cudaDeviceProp deviceProperties;
  // cudaGetDeviceProperties(&deviceProperties, 0);
  // deviceProperties.multiProcessorCount;
  int shared_mem_size = 49152;
  int num_sms = 28;
  int grid_dim = static_cast<int>(std::min<Index>(numel, 32 * num_sms));
  int block_dim =
      static_cast<int>(std::min<Index>(256, (numel + grid_dim - 1) / grid_dim));
  if (shared_mem_per_thread > 0) {
    block_dim = std::min(block_dim, shared_mem_size / shared_mem_per_thread);
  }
  return std::make_pair(grid_dim, block_dim);
}

}  // namespace gpu
}  // namespace impl
}  // namespace aml

