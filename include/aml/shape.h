#pragma once

#include <initializer_list>
#include <limits>

#include <aml/impl/shape.h>
#include <aml/defs.h>

namespace aml {

template <int Dim>
struct Shape {
  AML_HOST_DEVICE Shape() : dims_() { }

  template <typename ...Ints>
  AML_HOST_DEVICE Shape(const Ints&&... dims) : dims_{dims...} { }

  AML_HOST_DEVICE Index operator[](int i) const {
    AML_DEBUG_ASSERT(i < Dim);
    return dims_[i];
  }

  AML_HOST_DEVICE Index& operator[](int i) {
    AML_DEBUG_ASSERT(i < Dim);
    return dims_[i];
  }

  AML_HOST_DEVICE int dim() const {
    return Dim;
  }

  AML_HOST_DEVICE Index numel() const {
    Index num = 1;
    for (auto dim : dims_) {
      num *= dim;
    }
    return num;
  }

  AML_HOST_DEVICE bool operator==(const Shape<Dim> &rhs) const {
    for (int i = 0; i < Dim; ++i) {
      if ((*this)[i] != rhs[i]) {
        return false;
      }
    }
    return true;
  }

  AML_HOST_DEVICE bool operator!=(const Shape<Dim> &rhs) const {
    return !(*this == rhs);
  }

  AML_HOST_DEVICE bool operator<=(const Shape<Dim> &rhs) const {
    for (int i = 0; i < Dim; ++i) {
      if (dims_[i] > rhs[i]) {
        return false;
      }
    }
    return true;
  }

  AML_HOST_DEVICE bool operator>(const Shape<Dim> &rhs) const {
    return !(*this <= rhs);
  }

  AML_HOST_DEVICE Index* begin() {
    return dims_;
  }

  AML_HOST_DEVICE const Index* begin() const {
    return dims_;
  }

  AML_HOST_DEVICE Index* end() {
    return dims_ + Dim;
  }

  AML_HOST_DEVICE const Index* end() const {
    return dims_ + Dim;
  }

  AML_HOST_DEVICE Index head() const {
    return dims_[Dim - 1];
  }

  AML_HOST_DEVICE Shape<Dim - 1> tail() const {
    Shape<Dim - 1> s;
    for (int i = 0; i < Dim - 1; ++i) {
      s[i] = dims_[i];
    }
    return s;
  }

  Index dims_[Dim];
};

template <class... T>
Shape<sizeof...(T)> make_shape(T&&... idx) {
  return { std::forward<Index>(idx)... };
}

}  // namespace aml

