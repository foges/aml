#include <cmath>

#include <gtest/gtest.h>

#include <aml/aml.h>

class ExpressionTestGpu : public ::testing::Test {
public:
  ExpressionTestGpu() {
    h.init();
  }

  ~ExpressionTestGpu() {
    h.destroy();
  }

protected:
  aml::Handle h;
};

TEST_F(ExpressionTestGpu, Unary) {
  auto x = aml::Array<double, 4>(aml::GPU, {1, 2, 1, 2});
  auto y = aml::Array<double, 4>(aml::GPU, x.size());
  aml::set(h, x, 1.0);
  auto y_h = aml::Array<double, 4>(aml::CPU, x.size());

  aml::eval(h, y, aml::exp(x));
  aml::copy(h, y, y_h);

  EXPECT_EQ(y_h.data()[0], std::exp(1.0));
  EXPECT_EQ(y_h.data()[1], std::exp(1.0));
  EXPECT_EQ(y_h.data()[2], std::exp(1.0));
  EXPECT_EQ(y_h.data()[3], std::exp(1.0));
}

TEST_F(ExpressionTestGpu, Binary) {
  auto x = aml::Array<double, 4>(aml::GPU, {1, 2, 1, 2});
  auto y = aml::Array<double, 4>(aml::GPU, x.size());
  auto z = aml::Array<double, 4>(aml::GPU, x.size());
  aml::set(h, x, 2.0);
  aml::set(h, y, 1.0);
  auto z_h = aml::Array<double, 4>(aml::CPU, x.size());

  aml::eval(h, z, x - y);
  aml::copy(h, z, z_h);

  EXPECT_EQ(z_h.data()[0], 1.0);
  EXPECT_EQ(z_h.data()[1], 1.0);
  EXPECT_EQ(z_h.data()[2], 1.0);
  EXPECT_EQ(z_h.data()[3], 1.0);
}

