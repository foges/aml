#include <cmath>

#include <gtest/gtest.h>

#include <aml/aml.h>

class InitializersTestGpu : public ::testing::Test {
public:
  InitializersTestGpu() {
    h.init();
  }

  ~InitializersTestGpu() {
    h.destroy();
  }

protected:
  aml::Handle h;
};

TEST_F(InitializersTestGpu, Zeros) {
  auto s = aml::make_shape(3, 1);
  auto a = aml::zeros<float, 2>(h, aml::GPU, s);
  auto a_h = aml::Array<float, 2>(aml::CPU, s);
  aml::copy(h, a, a_h);

  EXPECT_EQ(a.size(), s);
  EXPECT_EQ(a.device(), aml::GPU);
  EXPECT_TRUE(a.is_contiguous());
  EXPECT_EQ(a_h.data()[0], 0.0f);
  EXPECT_EQ(a_h.data()[1], 0.0f);
  EXPECT_EQ(a_h.data()[2], 0.0f);
}

TEST_F(InitializersTestGpu, Ones) {
  auto s = aml::make_shape(3, 1);
  auto a = aml::ones<float, 2>(h, aml::CPU, s);
  auto a_h = aml::Array<float, 2>(aml::CPU, s);
  aml::copy(h, a, a_h);

  EXPECT_EQ(a.size(), s);
  EXPECT_EQ(a.device(), aml::CPU);
  EXPECT_TRUE(a.is_contiguous());
  EXPECT_EQ(a_h.data()[0], 1.0f);
  EXPECT_EQ(a_h.data()[1], 1.0f);
  EXPECT_EQ(a_h.data()[2], 1.0f);
}

TEST_F(InitializersTestGpu, NansFloat) {
  auto s = aml::make_shape(3, 1);
  auto a = aml::nans<float, 2>(h, aml::CPU, s);
  auto a_h = aml::Array<float, 2>(aml::CPU, s);
  aml::copy(h, a, a_h);

  EXPECT_EQ(a.size(), s);
  EXPECT_EQ(a.device(), aml::CPU);
  EXPECT_TRUE(a.is_contiguous());
  EXPECT_TRUE(std::isnan(a_h.data()[0]));
  EXPECT_TRUE(std::isnan(a_h.data()[1]));
  EXPECT_TRUE(std::isnan(a_h.data()[2]));
}

TEST_F(InitializersTestGpu, NansDouble) {
  auto s = aml::make_shape(3, 1);
  auto a = aml::nans<double, 2>(h, aml::CPU, s);
  auto a_h = aml::Array<double, 2>(aml::CPU, s);
  aml::copy(h, a, a_h);

  EXPECT_EQ(a.size(), s);
  EXPECT_EQ(a.device(), aml::CPU);
  EXPECT_TRUE(a.is_contiguous());
  EXPECT_TRUE(std::isnan(a_h.data()[0]));
  EXPECT_TRUE(std::isnan(a_h.data()[1]));
  EXPECT_TRUE(std::isnan(a_h.data()[2]));
}

