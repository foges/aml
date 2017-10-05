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
  Array(Device device, const Shape<Dim> &shape)
      : allocation_(new Allocation(device, sizeof(T) * shape.numel())),
        data_(static_cast<T*>(allocation_->data())),
        shape_(shape),
        stride_(impl::strides(shape)){ }

  Array(const std::shared_ptr<Allocation> &allocation,
        T *data,
        const Shape<Dim> &shape,
        const Shape<Dim> &stride)
      : allocation_(allocation),
        data_(data),
        shape_(shape),
        stride_(stride) { }

  Array() : data_(nullptr), shape_(), stride_(impl::strides(shape_)) { }

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

  Shape<Dim> shape() const {
    return shape_;
  }

  Shape<Dim> stride() const {
    return stride_;
  }

  bool is_contiguous() const {
    return stride_ == impl::strides(shape_);
  }

private:
  std::shared_ptr<Allocation> allocation_;

  T *data_;

  Shape<Dim> shape_;
  Shape<Dim> stride_;
};

template <typename T>
using Vector = Array<T, 1>;

template <typename T>
using Matrix = Array<T, 2>;

template <typename T, int Dim>
Array<T, Dim> make_array(const std::vector<T> data, const Shape<Dim> &shape) {
  AML_ASSERT(data.size() == shape.numel(), "Number of elements must match");

  Array<T, Dim> array(aml::CPU, shape);
  std::copy(data.begin(), data.end(), array.data());

  return array;
}

template <typename T, int Dim>
Array<T, Dim> slice(Array<T, Dim> array, Shape<Dim> begin, Shape<Dim> end) {
  AML_ASSERT(begin <= array.shape(), "Cannot slice outside of array");
  AML_ASSERT(end <= array.shape(), "Cannot slice outside of array");
  AML_ASSERT(begin <= end, "Slice begin must come before end");

  return Array<T, Dim>(
      array.allocation(),
      array.data() + impl::dot(begin, array.stride()),
      impl::diff(end, begin),
      array.stride());
}

template <typename T, int DimOld, int DimNew>
Array<T, DimNew> reshape(Array<T, DimOld> array, Shape<DimNew> shape) {
  AML_ASSERT(array.shape().numel() == shape.numel(),
      "Reshape must keep the same number of elements");
  AML_ASSERT(array.is_contiguous(),
      "Cannot reshape non-contiguous arrays");

  return Array<T, DimNew>(
      array.allocation(),
      array.data(),
      shape,
      impl::strides(shape));
}

}  // namespace aml

