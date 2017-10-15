#pragma once

#include <aml/array.h>
#include <aml/defs.h>
#include <aml/handle.h>
#include <aml/impl/linalg.h>

namespace aml {

template <typename T>
void potrf(aml::Handle h, Matrix<T> &a) {
  AML_ASSERT(a.size()[0] == a.size()[1], "A must be square");

  AML_DEVICE_EVAL(a.device(), potrf(h, a.size()[0], a.data(), a.stride()[1]));
}

}  // namespace aml

