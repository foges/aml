#pragma once

#include <aml/defs.h>
#include <aml/shape.h>

namespace aml {
namespace impl {

template <int Dim>
Shape<Dim> strides(const Shape<Dim> &shape) {
  Shape<Dim> stride;
  Index product = 1;
  for (Index i = 0; i < Dim; ++i) {
    stride[i] = product;
    product *= shape[i];
  }
  return stride;
}

template <int Dim>
Index dot(const Shape<Dim> &s1, const Shape<Dim> &s2) {
  Index product = 0;
  for (Index i = 0; i < Dim; ++i) {
    product += s1[i] * s2[i];
  }
  return product;
}

template <int Dim>
Shape<Dim> diff(const Shape<Dim> &lhs, const Shape<Dim> &rhs) {
  Shape<Dim> shape;
  for (Index i = 0; i < Dim; ++i) {
    shape[i] = lhs[i] - rhs[i];
  }
  return shape;
}

}  // namespace impl
}  // namespace ml

