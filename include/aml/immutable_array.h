#pragma once

#include <memory>

#include <aml/allocation.h>
#include <aml/shape.h>

namespace aml {

template <typename T, int Dim>
class ImmutableArray {
public:
  virtual ~ImmutableArray() { }

  virtual Device device() const = 0;

  virtual std::shared_ptr<const Allocation> allocation() const = 0;

  virtual const T* data() const = 0;

  virtual Shape<Dim> shape() const = 0;

  virtual Shape<Dim> stride() const = 0;

  virtual bool is_contiguous() const = 0;
};

template <typename T>
using ImmutableVector = ImmutableArray<T, 1>;

template <typename T>
using ImmutableMatrix = ImmutableArray<T, 2>;

}  // namespace aml

