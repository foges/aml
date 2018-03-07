# AML - Accelerated Matrix Library

AML is a C++ header-only matrix library that makes it super easy to write
high-performance Linear Algebra code for both CPU and GPU platforms. Its goal is
to provide the full performance of a BLAS and cuBLAS libraries, while making it
as simple to use as higher-level languages and frameworks such as MATLAB and
numpy. Its syntax should feel intutitive for anyone familiar with MATLAB.

At the core of AML is the N-dimensional array class `aml::Array<T, N>`, with the
1D and 2D specializations `aml::Vector<T>` and `aml::Matrix<T>`.

Some reasons to use AML:
 - Switching between CPU and GPU is trivial. Simply instantiate your array on
   the CPU or GPU and AML will take care of making sure your code is executed on
   the right device.
 - Operations can be expressed elegantly in vector form, such as `x + y` or
   `max(x, y)`.
 - It takes care of memory management.

# Usage

AML aims to be as transparent as possible. This means two things

 - Before calling any math-heavy functions, you need to initialize a handle and
   pass it around (the handle primarily carries other CUDA handles)
 - Functions never return new arrays, you always need to initialize it manually

# Example

Vectorized code
```c++
#include <gtest/gtest.h>
#include <assert.h>

#include <aml/aml.h>

TEST(SomeTest, Empty) {
  // Setup
  aml::Handle h;
  h.init();

  auto x = aml::make_array({1, 1});
  auto y = aml::make_array({1, 8});
  auto z = aml::make_array({0, 0});
  aml::eval(h, z, aml::max(x, y));
  assert(z.data()[0] == 1 && z.data()[1] == 8);

  // Teardown
  h.destroy();
}
```
