#pragma once

#include <utility>

#include <aml/defs.h>
#include <aml/impl/gpu/handle.h>

namespace aml {
namespace impl {
namespace gpu {

inline std::pair<int, int> launch_dims(const Handle *h, Index numel) {
  int grid_dim = static_cast<int>(std::min<Index>(numel, 32 * h->num_procs()));
  int block_dim =
      static_cast<int>(std::min<Index>(256, (numel + grid_dim - 1) / grid_dim));
  return std::make_pair(grid_dim, block_dim);
}

}  // namespace gpu
}  // namespace impl
}  // namespace aml

