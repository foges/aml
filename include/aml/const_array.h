#pragma once

#include <memory>

#include <aml/allocation.h>
#include <aml/array.h>
#include <aml/defs.h>
#include <aml/device.h>
#include <aml/immutable_array.h>
#include <aml/shape.h>

namespace aml {

template <typename T, int Dim>
class ConstArray : public ImmutableArray<T, Dim> {
public:
  ConstArray(std::shared_ptr<const Allocation> allocation,
             const T *data,
             Shape<Dim> shape,
             Shape<Dim> stride)
      : allocation_(allocation),
        data_(data),
        shape_(shape),
        stride_(stride) { }

  ConstArray(const Array<T, Dim> &array)
      : ConstArray(array.allocation(),
                   array.data(),
                   array.shape(),
                   array.stride()) { }

  ConstArray() : data_(nullptr), shape_(), stride_(impl::strides(shape_)) { }

  ~ConstArray() { }

  Device device() const {
    return allocation_->device();
  }

  std::shared_ptr<const Allocation> allocation() const {
    return allocation_;
  }

  const T* data() const {
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
  std::shared_ptr<const Allocation> allocation_;

  T const *data_;

  Shape<Dim> shape_;
  Shape<Dim> stride_;
};

template <typename T>
using ConstVector = ConstArray<T, 1>;

template <typename T>
using ConstMatrix = ConstArray<T, 2>;

template <typename T, int Dim>
ConstArray<T, Dim> slice(const ImmutableArray<T, Dim> &array,
                         const Shape<Dim> &begin,
                         const Shape<Dim> &end) {
  AML_ASSERT(end <= array.shape(), "Cannot slice outside of array");
  AML_ASSERT(begin <= end, "Slice begin must come before end");

  return ConstArray<T, Dim>(
      array.allocation(),
      array.data() + impl::dot(begin, array.stride()),
      impl::diff(end, begin),
      array.stride());
}

template <typename T, int DimOld, int DimNew>
ConstArray<T, DimNew> reshape(const ImmutableArray<T, DimOld> &array,
                              const Shape<DimNew> &shape) {
  AML_ASSERT(array.shape().numel() == shape.numel(),
      "Reshape must keep the same number of elements");
  AML_ASSERT(array.is_contiguous(),
      "Cannot reshape non-contiguous arrays");

  return ConstArray<T, DimNew>(
      array.allocation(),
      array.data(),
      shape,
      impl::strides(shape));
}

}  // namespace aml

