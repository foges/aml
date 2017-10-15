#pragma once

#include <aml/array.h>
#include <aml/handle.h>
#include <aml/immutable_array.h>
#include <aml/impl/cpu/operations.h>

#ifdef AML_GPU
#include <aml/impl/gpu/operations.h>
#endif

namespace aml{
namespace impl {

#ifdef AML_GPU

template <typename Tin, typename Tout, int Dim>
void copy(Handle h,
          const ImmutableArray<Tin, Dim> &in,
          Array<Tout, Dim> &out) {
  if (in.device() == aml::CPU && out.device() == aml::CPU) {
    cpu::copy(h, in, out);
  } else {
    if (in.device() != out.device()) {
        AML_ASSERT(in.is_contiguous() && out.is_contiguous(),
            "Cannot copy non-contiguous arrays between devices");
        AML_ASSERT((std::is_same<Tin, Tout>::value),
            "Cannot copy arrays of different types between devices");
    }

    gpu::copy(h, in, out);
  }
}

#else

template <typename Tin, typename Tout, int Dim>
void copy(Handle h,
          const ImmutableArray<Tin, Dim> &in,
          Array<Tout, Dim> &out) {
  cpu::copy(h, in, out);
}

#endif

}  // namespace impl
}  // namespace aml

