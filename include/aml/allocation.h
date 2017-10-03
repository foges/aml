#pragma once

#include <aml/defs.h>
#include <aml/device.h>
#include <aml/impl/allocation.h>

namespace aml {

class Allocation {
public:
  Allocation(Device device, size_t size)
    : device_(device),
      size_(size),
      data_(nullptr) {

    if (size_ > 0) {
      data_ = AML_DEVICE_EVAL(device_, malloc(size_));
      AML_ASSERT(data_ != nullptr, "Failed to allocate memory");
    }
  }

  Allocation() : Allocation(CPU, 0) { }

  ~Allocation() {
    if (data_ != nullptr) {
      AML_DEVICE_EVAL(device_, free(data_));
      size_ = 0;
      data_ = nullptr;
    }
  }

  // TODO: What's the best way to do this?
  Allocation(const Allocation&) = delete;
  Allocation& operator=(const Allocation&) = delete;	

  void* data() { return data_; }

  const void* data() const { return data_; }

  size_t size() const { return size_; }

  Device device() const { return device_; }

private:
  Device device_;
  size_t size_;
  void *data_;
};

}  // namespace aml

