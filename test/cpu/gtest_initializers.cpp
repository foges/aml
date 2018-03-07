#include <cmath>

#include <gtest/gtest.h>

#include <aml/aml.h>

class InitializersTest : public ::testing::Test {
public:
  InitializersTest() {
    h.init();
  }

  ~InitializersTest() {
    h.destroy();
  }

protected:
  aml::Handle h;
};

TEST_F(InitializersTest, Zeros) {
  auto s = aml::make_shape(3, 1);
  auto a = aml::zeros<float, 2>(h, aml::CPU, s);
  EXPECT_EQ(a.size(), s);
  EXPECT_EQ(a.device(), aml::CPU);
  EXPECT_TRUE(a.is_contiguous());
  EXPECT_EQ(a.data()[0], 0.0f);
  EXPECT_EQ(a.data()[1], 0.0f);
  EXPECT_EQ(a.data()[2], 0.0f);
}

TEST_F(InitializersTest, Ones) {
  auto s = aml::make_shape(3, 1);
  auto a = aml::ones<float, 2>(h, aml::CPU, s);
  EXPECT_EQ(a.size(), s);
  EXPECT_EQ(a.device(), aml::CPU);
  EXPECT_TRUE(a.is_contiguous());
  EXPECT_EQ(a.data()[0], 1.0f);
  EXPECT_EQ(a.data()[1], 1.0f);
  EXPECT_EQ(a.data()[2], 1.0f);
}

TEST_F(InitializersTest, NansFloat) {
  auto s = aml::make_shape(3, 1);
  auto a = aml::nans<float, 2>(h, aml::CPU, s);
  EXPECT_EQ(a.size(), s);
  EXPECT_EQ(a.device(), aml::CPU);
  EXPECT_TRUE(a.is_contiguous());
  EXPECT_TRUE(std::isnan(a.data()[0]));
  EXPECT_TRUE(std::isnan(a.data()[1]));
  EXPECT_TRUE(std::isnan(a.data()[2]));
}

TEST_F(InitializersTest, NansDouble) {
  auto s = aml::make_shape(3, 1);
  auto a = aml::nans<double, 2>(h, aml::CPU, s);
  EXPECT_EQ(a.size(), s);
  EXPECT_EQ(a.device(), aml::CPU);
  EXPECT_TRUE(a.is_contiguous());
  EXPECT_TRUE(std::isnan(a.data()[0]));
  EXPECT_TRUE(std::isnan(a.data()[1]));
  EXPECT_TRUE(std::isnan(a.data()[2]));
}
