#pragma once

#include <aml/impl/profiler.h>

#ifdef AML_GPU
#include <aml/impl/gpu/handle.h>
#endif

namespace aml {

class Handle {
public:
  Handle() :
#ifdef AML_GPU
      h_gpu_(nullptr),
#endif
      profiler_(nullptr) { }

  void init(bool enable_profiling=false) {
    profiler_ = new impl::Profiler(enable_profiling);
#ifdef AML_GPU
    h_gpu_ = new impl::gpu::Handle();
#endif
  }

  void destroy() {
#ifdef AML_GPU
    delete h_gpu_;
#endif
  }

  void clear() {
#ifdef AML_GPU
    h_gpu_->clear();
#endif
  }

  void synchronize() const {
#ifdef AML_GPU
    h_gpu_->synchronize();
#endif
  }

  impl::Toc tic(const std::string &name) {
    return profiler_->tic(name);
  }

  impl::Toc tic(const std::string &name, std::function<void()> sync) {
    return profiler_->tic(name, sync);
  }

  void print_profile() const {
    std::cout << profiler_->to_string();
  }

#ifdef AML_GPU
  impl::gpu::Handle* gpu() {
    return h_gpu_;
  }
#endif

private:
#ifdef AML_GPU
  impl::gpu::Handle *h_gpu_;
#endif

  impl::Profiler *profiler_;
};

}  // namespace aml

