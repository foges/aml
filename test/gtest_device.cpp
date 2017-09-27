#include <string>

#include <gtest/gtest.h>

#include <aml/aml.h>

namespace cpu {
int number() { return 1; }
int number_return(int x) { return x; }
std::string string() { return "1"; }
std::string string_return(std::string x) { return x; }
void no_return() { }
}  // namespace cpu

TEST(DeviceTestCpu, Number) {
  EXPECT_EQ(1, AML_DEVICE_EVAL(aml::CPU, number()));
}

TEST(DeviceTestCpu, NumberVariable) {
  int x = 1;
  EXPECT_EQ(1, AML_DEVICE_EVAL(aml::CPU, number_return(x)));
}

TEST(DeviceTestCpu, NumberArg) {
  EXPECT_EQ(8, AML_DEVICE_EVAL(aml::CPU, number_return(8)));
  EXPECT_EQ(0, AML_DEVICE_EVAL(aml::CPU, number_return(0)));
}

TEST(DeviceTestCpu, String) {
  EXPECT_EQ("1", AML_DEVICE_EVAL(aml::CPU, string()));
}

TEST(DeviceTestCpu, StringVariable) {
  std::string x = "s";
  EXPECT_EQ("s", AML_DEVICE_EVAL(aml::CPU, string_return(x)));
}

TEST(DeviceTestCpu, StringArg) {
  EXPECT_EQ("a", AML_DEVICE_EVAL(aml::CPU, string_return("a")));
  EXPECT_EQ("bb", AML_DEVICE_EVAL(aml::CPU, string_return("bb")));
}

TEST(DeviceTestCpu, NoReturn) {
  AML_DEVICE_EVAL(aml::CPU, no_return());
}

#ifdef AML_GPU
namespace gpu {
int number() { return 2; }
int number_return(int x) { return x; }
std::string string() { return "2"; }
std::string string_return(std::string x) { return x; }
void no_return() { }
}  // namespace gpu

TEST(DeviceTestGpu, Number) {
  EXPECT_EQ(2, AML_DEVICE_EVAL(aml::GPU, number()));
}

TEST(DeviceTestGpu, NumberVariable) {
  int x = 1;
  EXPECT_EQ(1, AML_DEVICE_EVAL(aml::GPU, number_return(x)));
}

TEST(DeviceTestGpu, NumberArg) {
  EXPECT_EQ(3, AML_DEVICE_EVAL(aml::GPU, number_return(3)));
  EXPECT_EQ(9, AML_DEVICE_EVAL(aml::GPU, number_return(9)));
}

TEST(DeviceTestGpu, String) {
  EXPECT_EQ("2", AML_DEVICE_EVAL(aml::GPU, string()));
}

TEST(DeviceTestGpu, StringVariable) {
  std::string x = "s";
  EXPECT_EQ("s", AML_DEVICE_EVAL(aml::GPU, string_return(x)));
}

TEST(DeviceTestGpu, StringArg) {
  EXPECT_EQ("a", AML_DEVICE_EVAL(aml::GPU, string_return("a")));
  EXPECT_EQ("bb", AML_DEVICE_EVAL(aml::GPU, string_return("bb")));
}

TEST(DeviceTestGpu, NoReturn) {
  AML_DEVICE_EVAL(aml::GPU, no_return());
}
#endif

