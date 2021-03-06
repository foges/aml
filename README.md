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
   `max(x, y)`, and will be lazily evaluated.
 - It takes care of memory management.

# Usage

AML aims to be as transparent as possible. This means two things

 - Before calling any math-heavy functions, you need to initialize a handle and
   pass it around (the handle primarily carries other CUDA handles)
 - Functions never return new arrays, you always need to initialize them manually

# Example

Vectorized code
```c++
#include <assert.h>

#include <aml/aml.h>

int main() {
  // Setup
  aml::Handle h;
  h.init();

  auto x = aml::make_array({3, 4});
  auto y = aml::make_array({1, 8});
  auto z = aml::make_array({2, 1});
  aml::eval(h, z, aml::max(x, y) * z);
  assert(z.data()[0] == 6 && z.data()[1] == 8);

  // Teardown
  h.destroy();
}
```
