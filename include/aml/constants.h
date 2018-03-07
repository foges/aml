#pragma once

#include <cmath>

namespace aml {

template <typename T> inline T nan();
template<> inline float nan<float>() { return std::nanf(""); }
template<> inline double nan<double>() { return std::nan(""); }

}  // namespace aml

