#pragma once

#include <aml/array.h>
#include <aml/constants.h>
#include <aml/defs.h>
#include <aml/operations.h>
#include <aml/shape.h>

namespace aml {

template <typename T, int Dim>
Array<T, Dim> zeros(Device device, const Shape<Dim> &size) {
  Array<T, Dim> array(device, size);
  set(array, static_cast<T>(0));
  return array;
}

template <typename T, int Dim>
Array<T, Dim> ones(Device device, const Shape<Dim> &size) {
  Array<T, Dim> array(device, size);
  set(array, static_cast<T>(1));
  return array;
}

template <typename T, int Dim>
Array<T, Dim> nans(Device device, const Shape<Dim> &size) {
  Array<T, Dim> array(device, size);
  set(array, nan<T>());
  return array;
}

}

