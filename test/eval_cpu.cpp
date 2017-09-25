#include <iostream>

#include <aml/aml.h>

namespace aml {
namespace cpu {
void speak() { std::cout << "cpu" << std::endl; }
}  // namespace cpu

void test() {
  AML_DEVICE_EVAL(aml::CPU, speak());
}

}  // namespace aml

int main() {
  aml::test();
}
