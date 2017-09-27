#include <gtest/gtest.h>

#include <aml/aml.h>

namespace cpu {
int number() { return 1; }
}  // namespace cpu

TEST(CpuEvalTest, Number) {
  EXPECT_EQ(1, AML_DEVICE_EVAL(aml::CPU, number()));
}

#ifdef AML_GPU
namespace gpu {
int number() { return 2; }
}  // namespace gpu

TEST(GpuEvalTest, Number) {
  EXPECT_EQ(2, AML_DEVICE_EVAL(aml::GPU, number()));
}
#endif

