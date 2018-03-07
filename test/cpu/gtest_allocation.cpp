#include <gtest/gtest.h>

#include <aml/aml.h>

const size_t kZeroSize = 0;

TEST(AllocationTest, Empty) {
  aml::Allocation a;
  EXPECT_EQ(a.size(), kZeroSize);
  EXPECT_EQ(a.data(), nullptr);
  EXPECT_EQ(a.device(), aml::CPU);
}

TEST(AllocationTest, EmptyConst) {
  const aml::Allocation a;
  EXPECT_EQ(a.size(), kZeroSize);
  EXPECT_EQ(a.data(), nullptr);
  EXPECT_EQ(a.device(), aml::CPU);
}

TEST(AllocationTest, Size0) {
  const aml::Allocation a(aml::CPU, 0);;
  EXPECT_EQ(a.size(), kZeroSize);
  EXPECT_EQ(a.data(), nullptr);
  EXPECT_EQ(a.device(), aml::CPU);
}

TEST(AllocationTest, Size2) {
  aml::Allocation a(aml::CPU, 2 * sizeof(int));;
  int *data = static_cast<int*>(a.data());
  data[0] = 3;
  data[1] = 4;

  EXPECT_EQ(a.size(), 2 * sizeof(int));
  EXPECT_NE(a.data(), nullptr);
  EXPECT_EQ(a.device(), aml::CPU);
  EXPECT_EQ(data[0], 3);
  EXPECT_EQ(data[1], 4);
}

TEST(AllocationTest, Free) {
  aml::Allocation a(aml::CPU, sizeof(int));

  EXPECT_EQ(a.size(), sizeof(int));
  EXPECT_NE(a.data(), nullptr);

  a.~Allocation();

  EXPECT_EQ(a.size(), kZeroSize);
  EXPECT_EQ(a.data(), nullptr);
}

