#pragma once

#include <aml/array.h>
#include <aml/immutable_array.h>
#include <aml/impl/cpu/operations.h>

#ifdef AML_GPU
#include <aml/impl/gpu/operations.h>
#endif

namespace aml{
namespace impl {

#ifdef AML_GPU

template <typename Tin, typename Tout, int Dim>
void copy(const ImmutableArray<Tin, Dim> &in, Array<Tout, Dim> &out) {
  if (in.device() == aml::CPU && out.device() == aml::CPU) {
    cpu::copy(in, out);
  } else {
    if (in.device() != out.device()) {
        AML_ASSERT(in.is_contiguous() && out.is_contiguous(),
            "Cannot copy non-contiguous arrays between devices");
        AML_ASSERT((std::is_same<Tin, Tout>::value),
            "Cannot copy arrays of different types between devices");
    }

    gpu::copy(in, out);
  }
}

#else

template <typename Tin, typename Tout, int Dim>
void copy(const ImmutableArray<Tin, Dim> &in, Array<Tout, Dim> &out) {
  cpu::copy(in, out);
}

#endif

}  // namespace impl
}  // namespace aml

