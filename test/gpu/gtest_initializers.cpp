#include <cmath>

#include <gtest/gtest.h>

#include <aml/aml.h>

TEST(InitializersTest, GpuZeros) {
  auto s = aml::make_shape(3, 1);
  auto a = aml::zeros<float, 2>(aml::GPU, s);
  auto a_h = aml::Array<float, 2>(aml::CPU, s);
  aml::copy(a, a_h);

  EXPECT_EQ(a.size(), s);
  EXPECT_EQ(a.device(), aml::GPU);
  EXPECT_TRUE(a.is_contiguous());
  EXPECT_EQ(a_h.data()[0], 0.0f);
  EXPECT_EQ(a_h.data()[1], 0.0f);
  EXPECT_EQ(a_h.data()[2], 0.0f);
}

TEST(InitializersTest, GpuOnes) {
  auto s = aml::make_shape(3, 1);
  auto a = aml::ones<float, 2>(aml::CPU, s);
  auto a_h = aml::Array<float, 2>(aml::CPU, s);
  aml::copy(a, a_h);

  EXPECT_EQ(a.size(), s);
  EXPECT_EQ(a.device(), aml::CPU);
  EXPECT_TRUE(a.is_contiguous());
  EXPECT_EQ(a_h.data()[0], 1.0f);
  EXPECT_EQ(a_h.data()[1], 1.0f);
  EXPECT_EQ(a_h.data()[2], 1.0f);
}

TEST(InitializersTest, GpuNansFloat) {
  auto s = aml::make_shape(3, 1);
  auto a = aml::nans<float, 2>(aml::CPU, s);
  auto a_h = aml::Array<float, 2>(aml::CPU, s);
  aml::copy(a, a_h);

  EXPECT_EQ(a.size(), s);
  EXPECT_EQ(a.device(), aml::CPU);
  EXPECT_TRUE(a.is_contiguous());
  EXPECT_TRUE(std::isnan(a_h.data()[0]));
  EXPECT_TRUE(std::isnan(a_h.data()[1]));
  EXPECT_TRUE(std::isnan(a_h.data()[2]));
}

TEST(InitializersTest, GpuNansDouble) {
  auto s = aml::make_shape(3, 1);
  auto a = aml::nans<double, 2>(aml::CPU, s);
  auto a_h = aml::Array<double, 2>(aml::CPU, s);
  aml::copy(a, a_h);

  EXPECT_EQ(a.size(), s);
  EXPECT_EQ(a.device(), aml::CPU);
  EXPECT_TRUE(a.is_contiguous());
  EXPECT_TRUE(std::isnan(a_h.data()[0]));
  EXPECT_TRUE(std::isnan(a_h.data()[1]));
  EXPECT_TRUE(std::isnan(a_h.data()[2]));
}

