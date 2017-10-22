#include <gtest/gtest.h>

#include <aml/aml.h>

class ExpressionTest : public ::testing::Test {
public:
  ExpressionTest() {
    h.init();
  }

  ~ExpressionTest() {
    h.destroy();
  }

protected:
  aml::Handle h;
};

TEST_F(ExpressionTest, Plus) {
  auto x = aml::ones<int, 2>(h, aml::CPU, {2, 2});
  auto y = aml::ones<int, 2>(h, aml::CPU, {2, 2});
  auto z = aml::ones<int, 2>(h, aml::CPU, {2, 2});

  aml::eval(h, z, x + y);

  EXPECT_EQ(z.data()[0], 2);
  EXPECT_EQ(z.data()[1], 2);
  EXPECT_EQ(z.data()[2], 2);
  EXPECT_EQ(z.data()[3], 2);
}
