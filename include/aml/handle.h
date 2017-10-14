#pragma once

#ifdef AML_GPU
#include <aml/impl/gpu/handle.h>
#endif

namespace aml {

class Handle {
public:
  Handle() { }

  void init() {
#ifdef AML_GPU
    h_gpu = new impl::gpu::Handle();
#endif
  }

  void destroy() {
#ifdef AML_GPU
    delete h_gpu;
#endif
  }

#ifdef AML_GPU
  impl::gpu::Handle* gpu() {
    return h_gpu;
  }
#endif

private:
#ifdef AML_GPU
  impl::gpu::Handle *h_gpu;
#endif
};

}  // namespace aml

