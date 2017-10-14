#include <gtest/gtest.h>

#include <aml/aml.h>

TEST(AllocationTest, GpuSize0) {
  const aml::Allocation a(aml::GPU, 0);;
  EXPECT_EQ(a.size(), static_cast<size_t>(0));
  EXPECT_EQ(a.data(), nullptr);
  EXPECT_EQ(a.device(), aml::CPU);
}

TEST(AllocationTest, GpuSize2) {
  aml::Allocation a(aml::GPU, 2 * sizeof(int));;

  cudaPointerAttributes attr;
  cudaError_t err = cudaPointerGetAttributes(&attr, a.data());

  EXPECT_EQ(err, cudaSuccess);
  EXPECT_EQ(attr.memoryType, cudaMemoryTypeDevice);
  EXPECT_EQ(a.size(), 2 * sizeof(int));
  EXPECT_EQ(a.device(), aml::GPU);
}

TEST(AllocationTest, GpuFree) {
  aml::Allocation a(aml::GPU, sizeof(int));

  EXPECT_EQ(a.size(), sizeof(int));
  EXPECT_NE(a.data(), nullptr);

  a.~Allocation();

  EXPECT_EQ(a.size(), static_cast<size_t>(0));
  EXPECT_EQ(a.data(), nullptr);
}

