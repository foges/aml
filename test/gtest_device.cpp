#include <gtest/gtest.h>

#include <aml/aml.h>

namespace cpu {
int number() { return 1; }
int number_return(int x) { return x; }
void no_return() { }
}  // namespace cpu

TEST(DeviceTestCpu, Number) {
  EXPECT_EQ(1, AML_DEVICE_EVAL(aml::CPU, number()));
}

TEST(DeviceTestCpu, NumberArg) {
  EXPECT_EQ(8, AML_DEVICE_EVAL(aml::CPU, number_return(8)));
  EXPECT_EQ(0, AML_DEVICE_EVAL(aml::CPU, number_return(0)));
}

TEST(DeviceTestCpu, NoReturn) {
  AML_DEVICE_EVAL(aml::CPU, no_return());
}

#ifdef AML_GPU
namespace gpu {
int number() { return 2; }
int number_return_neg(int x) { return -x; }
void no_return() { }
}  // namespace gpu

TEST(DeviceTestGpu, Number) {
  EXPECT_EQ(2, AML_DEVICE_EVAL(aml::GPU, number()));
}

TEST(DeviceTestGpu, NumberArg) {
  EXPECT_EQ(3, AML_DEVICE_EVAL(aml::GPU, number_return_neg(-3)));
  EXPECT_EQ(-9, AML_DEVICE_EVAL(aml::GPU, number_return_neg(9)));
}

TEST(DeviceTestGpu, NoReturn) {
  AML_DEVICE_EVAL(aml::GPU, no_return());
}
#endif

