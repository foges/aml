#pragma once

#include <algorithm>
#include <memory>
#include <vector>

#include <aml/allocation.h>
#include <aml/defs.h>
#include <aml/device.h>
#include <aml/immutable_array.h>
#include <aml/shape.h>

namespace aml {

template <typename T, int Dim>
class Array : public ImmutableArray<T, Dim> {
public:
  Array(Device device, const Shape<Dim> &size)
      : allocation_(new Allocation(device, sizeof(T) * size.numel())),
        data_(static_cast<T*>(allocation_->data())),
        size_(size),
        stride_(impl::strides(size)){ }

  Array(const std::shared_ptr<Allocation> &allocation,
        T *data,
        const Shape<Dim> &size,
        const Shape<Dim> &stride)
      : allocation_(allocation),
        data_(data),
        size_(size),
        stride_(stride) {
    AML_ASSERT(Dim == 0 || stride[0] == 1,
        "Array must be contiguous in leading dimension");
  }

  Array() : data_(nullptr), size_(), stride_(impl::strides(size_)) { }

  ~Array() { }

  Device device() const {
    AML_DEBUG_ASSERT(allocation_ != nullptr);
    return allocation_->device();
  }

  std::shared_ptr<const Allocation> allocation() const {
    return allocation_;
  }

  std::shared_ptr<Allocation> allocation() {
    return allocation_;
  }

  const T* data() const {
    return data_;
  }

  T* data() {
    return data_;
  }

  Shape<Dim> size() const {
    return size_;
  }

  Shape<Dim> stride() const {
    return stride_;
  }

  bool is_contiguous() const {
    return stride_ == impl::strides(size_);
  }

private:
  std::shared_ptr<Allocation> allocation_;

  T *data_;

  Shape<Dim> size_;
  Shape<Dim> stride_;
};

template <typename T>
using Vector = Array<T, 1>;

template <typename T>
using Matrix = Array<T, 2>;

template <typename T, int Dim>
Array<T, Dim> make_array(const std::vector<T> data, const Shape<Dim> &size) {
  AML_ASSERT(data.size() == size.numel(), "Number of elements must match");

  Array<T, Dim> array(aml::CPU, size);
  std::copy(data.begin(), data.end(), array.data());

  return array;
}

template <typename T, int Dim>
Array<T, Dim> slice(Array<T, Dim> array, Shape<Dim> begin, Shape<Dim> end) {
  AML_ASSERT(begin <= array.size(), "Cannot slice outside of array");
  AML_ASSERT(end <= array.size(), "Cannot slice outside of array");
  AML_ASSERT(begin <= end, "Slice begin must come before end");

  return Array<T, Dim>(
      array.allocation(),
      array.data() + impl::dot(begin, array.stride()),
      impl::diff(end, begin),
      array.stride());
}

template <int DimNew, typename T, int DimOld>
Array<T, DimNew> reshape(Array<T, DimOld> array, Shape<DimNew> size) {
  AML_ASSERT(array.size().numel() == size.numel(),
      "Reshape must keep the same number of elements");
  AML_ASSERT(array.is_contiguous(),
      "Cannot reshape non-contiguous arrays");

  return Array<T, DimNew>(
      array.allocation(),
      array.data(),
      size,
      impl::strides(size));
}

}  // namespace aml

