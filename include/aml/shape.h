#pragma once

#include <initializer_list>
#include <limits>

#include <aml/defs.h>
#include <aml/types.h>

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
    for (int i = 0; i < Dim; ++i) {
      num *= dims_[i];
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
    #pragma unroll
    for (int i = 0; i < Dim - 1; ++i) {
      s[i] = dims_[i];
    }
    return s;
  }

  Index dims_[Dim == 0 ? 1 : Dim];
};

template <class... T>
Shape<sizeof...(T)> make_shape(T&&... idx) {
  return { std::forward<Index>(idx)... };
}

namespace impl {

template <int Dim>
AML_HOST_DEVICE Shape<Dim> strides(const Shape<Dim> &shape) {
  Shape<Dim> stride;
  Index product = 1;
  #pragma unroll
  for (Index i = 0; i < Dim; ++i) {
    stride[i] = product;
    product *= shape[i];
  }
  return stride;
}

template <>
inline AML_HOST_DEVICE Shape<0> strides(const Shape<0>&) {
  return Shape<0>();
}

template <int Dim>
AML_HOST_DEVICE Index dot(const Shape<Dim> &index, const Shape<Dim> &stride) {
  AML_DEBUG_ASSERT(stride[0] == 1);
  Index product = index[0];
  #pragma unroll
  for (Index i = 1; i < Dim; ++i) {
    product += index[i] * stride[i];
  }
  return product;
}

template <>
inline AML_HOST_DEVICE Index dot(const Shape<0>&, const Shape<0>&) {
  return 0;
}

template <int Dim>
AML_HOST_DEVICE Shape<Dim> diff(const Shape<Dim> &lhs, const Shape<Dim> &rhs) {
  Shape<Dim> shape;
  #pragma unroll
  for (Index i = 0; i < Dim; ++i) {
    shape[i] = lhs[i] - rhs[i];
  }
  return shape;
}

template <>
inline AML_HOST_DEVICE Shape<0> diff(const Shape<0>&, const Shape<0>&) {
  return Shape<0>();
}

template <int Dim>
AML_HOST_DEVICE Shape<Dim> shape_index(const Shape<Dim> &shape, Index index) {
  Shape<Dim> index_shape;
  #pragma unroll
  for (Index i = 0; i < Dim; ++i) {
    index_shape[i] = index % shape[i];
    index /= shape[i];
  }
  return index_shape;
}

template <>
inline AML_HOST_DEVICE Shape<0> shape_index(const Shape<0>&, Index) {
  return Shape<0>();
}

}  // namespace impl
}  // namespace aml

