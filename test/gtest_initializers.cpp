#include <cmath>

#include <gtest/gtest.h>

#include <aml/aml.h>

TEST(InitializersTest, Zeros) {
  auto s = aml::make_shape(3, 1);
  auto a = aml::zeros<float, 2>(aml::CPU, s);
  EXPECT_EQ(a.shape(), s);
  EXPECT_EQ(a.device(), aml::CPU);
  EXPECT_TRUE(a.is_contiguous());
  EXPECT_EQ(a.data()[0], 0.0f);
  EXPECT_EQ(a.data()[1], 0.0f);
  EXPECT_EQ(a.data()[2], 0.0f);
}

TEST(InitializersTest, Ones) {
  auto s = aml::make_shape(3, 1);
  auto a = aml::ones<float, 2>(aml::CPU, s);
  EXPECT_EQ(a.shape(), s);
  EXPECT_EQ(a.device(), aml::CPU);
  EXPECT_TRUE(a.is_contiguous());
  EXPECT_EQ(a.data()[0], 1.0f);
  EXPECT_EQ(a.data()[1], 1.0f);
  EXPECT_EQ(a.data()[2], 1.0f);
}

TEST(InitializersTest, NansFloat) {
  auto s = aml::make_shape(3, 1);
  auto a = aml::nans<float, 2>(aml::CPU, s);
  EXPECT_EQ(a.shape(), s);
  EXPECT_EQ(a.device(), aml::CPU);
  EXPECT_TRUE(a.is_contiguous());
  EXPECT_TRUE(std::isnan(a.data()[0]));
  EXPECT_TRUE(std::isnan(a.data()[1]));
  EXPECT_TRUE(std::isnan(a.data()[2]));
}

TEST(InitializersTest, NansDouble) {
  auto s = aml::make_shape(3, 1);
  auto a = aml::nans<double, 2>(aml::CPU, s);
  EXPECT_EQ(a.shape(), s);
  EXPECT_EQ(a.device(), aml::CPU);
  EXPECT_TRUE(a.is_contiguous());
  EXPECT_TRUE(std::isnan(a.data()[0]));
  EXPECT_TRUE(std::isnan(a.data()[1]));
  EXPECT_TRUE(std::isnan(a.data()[2]));
}
