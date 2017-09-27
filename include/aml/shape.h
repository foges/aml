#pragma once

#include <initializer_list>

#include <aml/defs.h>

namespace aml {

template <Index Dim>
class Shape {
public:
  AML_HOST_DEVICE Shape() : dims_() { }

  template <typename IndexCastable>
  AML_HOST_DEVICE Shape(std::initializer_list<IndexCastable> dims) {
    Index i = Index(0);
    for (auto dim : dims) {
      dims_[i++] = static_cast<Index>(dim);
    }
  }

  AML_HOST_DEVICE Index operator[](Index i) const {
    AML_DEBUG_ASSERT(i < Dim);
    return dims_[i];
  }

  AML_HOST_DEVICE Index& operator[](Index i) {
    AML_DEBUG_ASSERT(i < Dim);
    return dims_[i];
  }

  Index size() const {
    return Dim;
  }

private:
  Index dims_[Dim];
};

}  // namespace aml

