#pragma once

#include <aml/array.h>
#include <aml/defs.h>
#include <aml/handle.h>
#include <aml/immutable_array.h>
#include <aml/impl/gpu/utils.h>
#include <aml/shape.h>

namespace aml {
namespace impl {
namespace gpu {

/** SET ***********************************************************************/

template <typename T, int Dim>
__global__ void set(T *out, Shape<Dim> size, Shape<Dim> stride, T val) {
  Index tid = blockIdx.x * blockDim.x + threadIdx.x;
  Index tstride = blockDim.x * gridDim.x;
  for (Index i = tid; i < size.numel(); i += tstride) {
    Index offset = impl::dot(impl::shape_index(size, i), stride);
    out[offset] = val;
  }
}

template <typename T, int Dim>
void set(aml::Handle h, Array<T, Dim> &out, const T &val) {
  auto dims = launch_dims(h.gpu(), out.size().numel());
  set<<<dims.first, dims.second>>>(out.data(), out.size(), out.stride(), val);
}

/** UNARY_OP ******************************************************************/

template <typename Tin, typename Tout, int Dim, typename Op>
__global__ void unary_op(const Tin *in,
                         Tout *out,
                         Shape<Dim> size,
                         Shape<Dim> stride_in,
                         Shape<Dim> stride_out,
                         Op op) {
  Index tid = blockIdx.x * blockDim.x + threadIdx.x;
  Index tstride = blockDim.x * gridDim.x;
  for (Index i = tid; i < size.numel(); i += tstride) {
    Shape<Dim> idx = impl::shape_index(size, i);
    Index offset_in = impl::dot(idx, stride_in);
    Index offset_out = impl::dot(idx, stride_out);
    out[offset_out] = op(in[offset_in]);
  }
}

template <typename Tin, typename Tout, int Dim, typename Op>
void unary_op(aml::Handle h,
              const ImmutableArray<Tin, Dim> &in,
              Array<Tout, Dim> &out,
              const Op &op) {
  auto dims = launch_dims(h.gpu(), in.size().numel());
  unary_op<<<dims.first, dims.second>>>(
      in.data(), out.data(), in.size(), in.stride(), out.stride(), op);
}

/** BINARY_OP *****************************************************************/

template <typename Tin1, typename Tin2, typename Tout, int Dim, typename Op>
__global__ void binary_op(const Tin1 *in1,
                          const Tin2 *in2,
                          Tout *out,
                          Shape<Dim> size,
                          Shape<Dim> stride_in1,
                          Shape<Dim> stride_in2,
                          Shape<Dim> stride_out,
                          Op op) {
  Index tid = blockIdx.x * blockDim.x + threadIdx.x;
  Index tstride = blockDim.x * gridDim.x;
  for (Index i = tid; i < size.numel(); i += tstride) {
    Shape<Dim> idx = impl::shape_index(size, i);
    Index offset_in1 = impl::dot(idx, stride_in1);
    Index offset_in2 = impl::dot(idx, stride_in2);
    Index offset_out = impl::dot(idx, stride_out);
    out[offset_out] = op(in1[offset_in1], in2[offset_in2]);
  }
}

template <typename Tin1, typename Tin2, typename Tout, int Dim, typename Op>
void binary_op(aml::Handle h,
               const ImmutableArray<Tin1, Dim> &in1,
               const ImmutableArray<Tin2, Dim> &in2,
               Array<Tout, Dim> &out,
               const Op &op) {
  auto dims = launch_dims(h.gpu(), in1.size().numel());
  binary_op<<<dims.first, dims.second>>>(
      in1.data(), in2.data(), out.data(), in1.size(),
      in1.stride(), in2.stride(), out.stride(), op);
}

/** REDUCE ********************************************************************/

template <typename Tin,
          int DimIn,
          typename Tout,
          int DimOut,
          typename TransformOp,
          typename ReduceOp>
__global__ void reduce(const Tin *in,
                       Tout *out,
                       Shape<DimIn> size_in,
                       Shape<DimOut> size_out,
                       Shape<DimIn> stride_in,
                       Shape<DimOut> stride_out,
                       Shape<DimIn - DimOut> axis,
                       Shape<DimOut> axis_nr,
                       Shape<DimIn - DimOut> size_r,
                       const TransformOp op_t,
                       const ReduceOp op_r) {
  extern __shared__ Tout res[];

  Index numel_in = size_in.numel();
  Index numel_out = size_out.numel();
  Index numel_reduce = numel_in / numel_out;

  Index wid = threadIdx.x / 32;
  Index tid = threadIdx.x % 32;
  Index bid = blockIdx.x * blockDim.x / 32 + wid;
  Index bstride = blockDim.x / 32 * gridDim.x;
  for (Index i = bid; i < numel_out; i += bstride) {
    Shape<DimOut> idx_out = impl::shape_index(size_out, i);

    Shape<DimIn> idx_in;
    #pragma unroll
    for (int j = 0; j < DimOut; ++j) {
      idx_in[axis_nr[j]] = idx_out[j];
    }

    Tout val;
    for (Index k = tid; k < numel_reduce; k += 32) {
      Shape<DimIn - DimOut> idx_r = impl::shape_index(size_r, k);
      #pragma unroll
      for (int j = 0; j < DimIn - DimOut; ++j) {
        idx_in[axis[j]] = idx_r[j];
      }
      Index offset_in = impl::dot(idx_in, stride_in);

      if (k == tid) {
        val = op_t(in[offset_in]);
      } else {
        val = op_r(val, op_t(in[offset_in]));
      }
    }

    int sid = 32 * wid + tid;
    res[sid] = val;

    #pragma unroll
    for (int i = 16; i > 0; i /= 2) {
      if (tid < i && tid + i < numel_reduce) {
        res[sid] = op_r(res[sid], res[sid + i]);
      }
    }

    if (tid == 0) {
      Index offset_out = impl::dot(idx_out, stride_out);
      out[offset_out] = res[sid];
    }
  }
}

template <typename Tin,
          int DimIn,
          typename Tout,
          int DimOut,
          typename TransformOp,
          typename ReduceOp>
void reduce(aml::Handle h,
            const ImmutableArray<Tin, DimIn> &in,
            Array<Tout, DimOut> &out,
            const std::array<int, DimIn - DimOut> &axis,
            const std::array<int, DimOut> &axis_nr,
            const TransformOp &op_t,
            const ReduceOp &op_r) {

  Shape<DimOut> axis_nr_shape;
  for (int i = 0; i < DimOut; ++i) {
    axis_nr_shape[i] = axis_nr[i];
  }
  Shape<DimIn - DimOut> axis_shape;
  for (int i = 0; i < DimIn - DimOut; ++i) {
    axis_shape[i] = axis[i];
  }
  Shape<DimIn - DimOut> size_r_shape;
  for (int i = 0; i < DimIn - DimOut; ++i) {
    size_r_shape[i] = in.size()[axis[i]];
  }

  Index numel = out.size().numel();

  int num_proc = h.gpu()->num_procs();
  int sm_per_proc = h.gpu()->shared_mem_per_proc();
  int sm_per_warp = 32 * sizeof(Tout);
  int max_warps_per_proc = sm_per_proc / sm_per_warp;
  AML_ASSERT(max_warps_per_proc >= 1, "Must have at least one warp per proc");
  int max_blocks = num_proc * std::min(32, max_warps_per_proc);;
  int grid_dim = static_cast<int>(std::min<Index>(numel, max_blocks));

  Index numel_per_block = (numel + grid_dim - 1) / grid_dim;
  int max_threads_per_proc = 32 * std::min(8, max_warps_per_proc);
  int block_dim = std::min<int>(max_threads_per_proc, 32 * numel_per_block);

  reduce<<<grid_dim, block_dim, block_dim * sizeof(Tout)>>>(
      in.data(), out.data(), in.size(), out.size(), in.stride(), out.stride(),
      axis_shape, axis_nr_shape, size_r_shape, op_t, op_r);
}

/** COPY **********************************************************************/

template <typename Tin, typename Tout>
struct IdentityFunctor {
  AML_HOST_DEVICE Tout operator()(const Tin &x) { return static_cast<Tout>(x); }
};

template <typename T, int Dim>
void copy(aml::Handle h, const ImmutableArray<T, Dim> &in, Array<T, Dim> &out) {
  if (in.is_contiguous() && out.is_contiguous()) {
    size_t count = in.size().numel() * sizeof(T);
    if (in.device() == aml::CPU && out.device() == aml::GPU) {
      AML_GPU_CHECK(cudaMemcpy(out.data(), in.data(), count,
          cudaMemcpyHostToDevice));
    } else if (in.device() == aml::GPU && out.device() == aml::CPU) {
      AML_GPU_CHECK(cudaMemcpy(out.data(), in.data(), count,
          cudaMemcpyDeviceToHost));
    } else if (in.device() == aml::GPU && out.device() == aml::GPU) {
      AML_GPU_CHECK(cudaMemcpy(out.data(), in.data(), count,
          cudaMemcpyDeviceToDevice));
    }
  } else {
    gpu::unary_op(h, in, out, IdentityFunctor<T, T>());
  }
}

template <typename Tin, typename Tout, int Dim>
void copy(aml::Handle h,
          const ImmutableArray<Tin, Dim> &in,
          Array<Tout, Dim> &out) {
  gpu::unary_op(h, in, out, IdentityFunctor<Tin, Tout>());
}

}  // namespace gpu
}  // namespace impl
}  // namespace aml

