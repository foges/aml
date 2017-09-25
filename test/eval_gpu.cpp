#include <iostream>

#include <aml/aml.h>

namespace aml {
namespace cpu {
void speak() { std::cout << "cpu" << std::endl; }
}  // namespace cpu

namespace gpu {
void speak() { std::cout << "gpu" << std::endl; }
}  // namespace gpu

void test() {
  AML_DEVICE_EVAL(aml::CPU, speak());
  AML_DEVICE_EVAL(aml::GPU, speak());
}

}  // namespace aml

int main() {
  aml::test();
}
