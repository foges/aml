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

/** PLUS **********************************************************************/

TEST_F(ExpressionTest, PlusAA) {
  auto x = aml::ones<double, 2>(h, aml::CPU, {1, 2});
  auto y = aml::ones<double, 2>(h, aml::CPU, {1, 2});
  auto z = aml::nans<double, 2>(h, aml::CPU, {1, 2});

  aml::eval(h, z, x + y);

  EXPECT_EQ(z.data()[0], 2);
  EXPECT_EQ(z.data()[1], 2);
}

TEST_F(ExpressionTest, PlusAE) {
  auto x = aml::ones<double, 2>(h, aml::CPU, {1, 2});
  auto y = aml::make_expression(aml::ones<double, 2>(h, aml::CPU, {1, 2}));
  auto z = aml::nans<double, 2>(h, aml::CPU, {1, 2});

  aml::eval(h, z, x + y);

  EXPECT_EQ(z.data()[0], 2);
  EXPECT_EQ(z.data()[1], 2);
}

TEST_F(ExpressionTest, PlusEA) {
  auto x = aml::make_expression(aml::ones<double, 2>(h, aml::CPU, {1, 2}));
  auto y = aml::ones<double, 2>(h, aml::CPU, {1, 2});
  auto z = aml::nans<double, 2>(h, aml::CPU, {1, 2});

  aml::eval(h, z, x + y);

  EXPECT_EQ(z.data()[0], 2);
  EXPECT_EQ(z.data()[1], 2);
}

TEST_F(ExpressionTest, PlusEE) {
  auto x = aml::make_expression(aml::ones<double, 2>(h, aml::CPU, {1, 2}));
  auto y = aml::make_expression(aml::ones<double, 2>(h, aml::CPU, {1, 2}));
  auto z = aml::nans<double, 2>(h, aml::CPU, {1, 2});

  aml::eval(h, z, x + y);

  EXPECT_EQ(z.data()[0], 2);
  EXPECT_EQ(z.data()[1], 2);
}

TEST_F(ExpressionTest, PlusSE) {
  auto x = 1.0;
  auto y = aml::make_expression(aml::ones<double, 2>(h, aml::CPU, {1, 2}));
  auto z = aml::nans<double, 2>(h, aml::CPU, {1, 2});

  aml::eval(h, z, x + y);

  EXPECT_EQ(z.data()[0], 2);
  EXPECT_EQ(z.data()[1], 2);
}

TEST_F(ExpressionTest, PlusES) {
  auto x = aml::make_expression(aml::ones<double, 2>(h, aml::CPU, {1, 2}));
  auto y = 1.0;
  auto z = aml::nans<double, 2>(h, aml::CPU, {1, 2});

  aml::eval(h, z, x + y);

  EXPECT_EQ(z.data()[0], 2);
  EXPECT_EQ(z.data()[1], 2);
}

TEST_F(ExpressionTest, PlusSA) {
  auto x = 1.0;
  auto y = aml::ones<double, 2>(h, aml::CPU, {1, 2});
  auto z = aml::nans<double, 2>(h, aml::CPU, {1, 2});

  aml::eval(h, z, x + y);

  EXPECT_EQ(z.data()[0], 2);
  EXPECT_EQ(z.data()[1], 2);
}

TEST_F(ExpressionTest, PlusAS) {
  auto x = aml::ones<double, 2>(h, aml::CPU, {1, 2});
  auto y = 1.0;
  auto z = aml::nans<double, 2>(h, aml::CPU, {1, 2});

  aml::eval(h, z, x + y);

  EXPECT_EQ(z.data()[0], 2);
  EXPECT_EQ(z.data()[1], 2);
}

/** MINUS *********************************************************************/

TEST_F(ExpressionTest, MinusAA) {
  auto x = aml::zeros<double, 2>(h, aml::CPU, {1, 2});
  auto y = aml::ones<double, 2>(h, aml::CPU, {1, 2});
  auto z = aml::nans<double, 2>(h, aml::CPU, {1, 2});

  aml::eval(h, z, x - y);

  EXPECT_EQ(z.data()[0], -1);
  EXPECT_EQ(z.data()[1], -1);
}

TEST_F(ExpressionTest, MinusAE) {
  auto x = aml::zeros<double, 2>(h, aml::CPU, {1, 2});
  auto y = aml::make_expression(aml::ones<double, 2>(h, aml::CPU, {1, 2}));
  auto z = aml::nans<double, 2>(h, aml::CPU, {1, 2});

  aml::eval(h, z, x - y);

  EXPECT_EQ(z.data()[0], -1);
  EXPECT_EQ(z.data()[1], -1);
}

TEST_F(ExpressionTest, MinusEA) {
  auto x = aml::make_expression(aml::zeros<double, 2>(h, aml::CPU, {1, 2}));
  auto y = aml::ones<double, 2>(h, aml::CPU, {1, 2});
  auto z = aml::nans<double, 2>(h, aml::CPU, {1, 2});

  aml::eval(h, z, x - y);

  EXPECT_EQ(z.data()[0], -1);
  EXPECT_EQ(z.data()[1], -1);
}

TEST_F(ExpressionTest, MinusEE) {
  auto x = aml::make_expression(aml::zeros<double, 2>(h, aml::CPU, {1, 2}));
  auto y = aml::make_expression(aml::ones<double, 2>(h, aml::CPU, {1, 2}));
  auto z = aml::nans<double, 2>(h, aml::CPU, {1, 2});

  aml::eval(h, z, x - y);

  EXPECT_EQ(z.data()[0], -1);
  EXPECT_EQ(z.data()[1], -1);
}

TEST_F(ExpressionTest, MinusSE) {
  auto x = 0.0;
  auto y = aml::make_expression(aml::ones<double, 2>(h, aml::CPU, {1, 2}));
  auto z = aml::nans<double, 2>(h, aml::CPU, {1, 2});

  aml::eval(h, z, x - y);

  EXPECT_EQ(z.data()[0], -1);
  EXPECT_EQ(z.data()[1], -1);
}

TEST_F(ExpressionTest, MinusES) {
  auto x = aml::make_expression(aml::zeros<double, 2>(h, aml::CPU, {1, 2}));
  auto y = 1.0;
  auto z = aml::nans<double, 2>(h, aml::CPU, {1, 2});

  aml::eval(h, z, x - y);

  EXPECT_EQ(z.data()[0], -1);
  EXPECT_EQ(z.data()[1], -1);
}

TEST_F(ExpressionTest, MinusSA) {
  auto x = 0.0;
  auto y = aml::ones<double, 2>(h, aml::CPU, {1, 2});
  auto z = aml::nans<double, 2>(h, aml::CPU, {1, 2});

  aml::eval(h, z, x - y);

  EXPECT_EQ(z.data()[0], -1);
  EXPECT_EQ(z.data()[1], -1);
}

TEST_F(ExpressionTest, MinusAS) {
  auto x = aml::zeros<double, 2>(h, aml::CPU, {1, 2});
  auto y = 1.0;
  auto z = aml::nans<double, 2>(h, aml::CPU, {1, 2});

  aml::eval(h, z, x - y);

  EXPECT_EQ(z.data()[0], -1);
  EXPECT_EQ(z.data()[1], -1);
}

/** MULTIPLY ******************************************************************/

TEST_F(ExpressionTest, MultiplyAA) {
  auto x = aml::zeros<double, 2>(h, aml::CPU, {1, 2});
  auto y = aml::ones<double, 2>(h, aml::CPU, {1, 2});
  auto z = aml::nans<double, 2>(h, aml::CPU, {1, 2});

  aml::eval(h, z, x * y);

  EXPECT_EQ(z.data()[0], 0);
  EXPECT_EQ(z.data()[1], 0);
}

TEST_F(ExpressionTest, MultiplyAE) {
  auto x = aml::zeros<double, 2>(h, aml::CPU, {1, 2});
  auto y = aml::make_expression(aml::ones<double, 2>(h, aml::CPU, {1, 2}));
  auto z = aml::nans<double, 2>(h, aml::CPU, {1, 2});

  aml::eval(h, z, x * y);

  EXPECT_EQ(z.data()[0], 0);
  EXPECT_EQ(z.data()[1], 0);
}

TEST_F(ExpressionTest, MultiplyEA) {
  auto x = aml::make_expression(aml::zeros<double, 2>(h, aml::CPU, {1, 2}));
  auto y = aml::ones<double, 2>(h, aml::CPU, {1, 2});
  auto z = aml::nans<double, 2>(h, aml::CPU, {1, 2});

  aml::eval(h, z, x * y);

  EXPECT_EQ(z.data()[0], 0);
  EXPECT_EQ(z.data()[1], 0);
}

TEST_F(ExpressionTest, MultiplyEE) {
  auto x = aml::make_expression(aml::zeros<double, 2>(h, aml::CPU, {1, 2}));
  auto y = aml::make_expression(aml::ones<double, 2>(h, aml::CPU, {1, 2}));
  auto z = aml::nans<double, 2>(h, aml::CPU, {1, 2});

  aml::eval(h, z, x * y);

  EXPECT_EQ(z.data()[0], 0);
  EXPECT_EQ(z.data()[1], 0);
}

TEST_F(ExpressionTest, MultiplySE) {
  auto x = 0.0;
  auto y = aml::make_expression(aml::ones<double, 2>(h, aml::CPU, {1, 2}));
  auto z = aml::nans<double, 2>(h, aml::CPU, {1, 2});

  aml::eval(h, z, x * y);

  EXPECT_EQ(z.data()[0], 0);
  EXPECT_EQ(z.data()[1], 0);
}

TEST_F(ExpressionTest, MultiplyES) {
  auto x = aml::make_expression(aml::zeros<double, 2>(h, aml::CPU, {1, 2}));
  auto y = 1.0;
  auto z = aml::nans<double, 2>(h, aml::CPU, {1, 2});

  aml::eval(h, z, x * y);

  EXPECT_EQ(z.data()[0], 0);
  EXPECT_EQ(z.data()[1], 0);
}

/** DIVIDE ********************************************************************/

TEST_F(ExpressionTest, DivideAA) {
  auto x = aml::zeros<double, 2>(h, aml::CPU, {1, 2});
  auto y = aml::ones<double, 2>(h, aml::CPU, {1, 2});
  auto z = aml::nans<double, 2>(h, aml::CPU, {1, 2});

  aml::eval(h, z, x / y);

  EXPECT_EQ(z.data()[0], 0);
  EXPECT_EQ(z.data()[1], 0);
}

TEST_F(ExpressionTest, DivideAE) {
  auto x = aml::zeros<double, 2>(h, aml::CPU, {1, 2});
  auto y = aml::make_expression(aml::ones<double, 2>(h, aml::CPU, {1, 2}));
  auto z = aml::nans<double, 2>(h, aml::CPU, {1, 2});

  aml::eval(h, z, x / y);

  EXPECT_EQ(z.data()[0], 0);
  EXPECT_EQ(z.data()[1], 0);
}

TEST_F(ExpressionTest, DivideEA) {
  auto x = aml::make_expression(aml::zeros<double, 2>(h, aml::CPU, {1, 2}));
  auto y = aml::ones<double, 2>(h, aml::CPU, {1, 2});
  auto z = aml::nans<double, 2>(h, aml::CPU, {1, 2});

  aml::eval(h, z, x / y);

  EXPECT_EQ(z.data()[0], 0);
  EXPECT_EQ(z.data()[1], 0);
}

TEST_F(ExpressionTest, DivideEE) {
  auto x = aml::make_expression(aml::zeros<double, 2>(h, aml::CPU, {1, 2}));
  auto y = aml::make_expression(aml::ones<double, 2>(h, aml::CPU, {1, 2}));
  auto z = aml::nans<double, 2>(h, aml::CPU, {1, 2});

  aml::eval(h, z, x / y);

  EXPECT_EQ(z.data()[0], 0);
  EXPECT_EQ(z.data()[1], 0);
}

TEST_F(ExpressionTest, DivideSE) {
  auto x = 0.0;
  auto y = aml::make_expression(aml::ones<double, 2>(h, aml::CPU, {1, 2}));
  auto z = aml::nans<double, 2>(h, aml::CPU, {1, 2});

  aml::eval(h, z, x / y);

  EXPECT_EQ(z.data()[0], 0);
  EXPECT_EQ(z.data()[1], 0);
}

TEST_F(ExpressionTest, DivideES) {
  auto x = aml::make_expression(aml::zeros<double, 2>(h, aml::CPU, {1, 2}));
  auto y = 1.0;
  auto z = aml::nans<double, 2>(h, aml::CPU, {1, 2});

  aml::eval(h, z, x / y);

  EXPECT_EQ(z.data()[0], 0);
  EXPECT_EQ(z.data()[1], 0);
}

TEST_F(ExpressionTest, DivideSA) {
  auto x = 0.0;
  auto y = aml::ones<double, 2>(h, aml::CPU, {1, 2});
  auto z = aml::nans<double, 2>(h, aml::CPU, {1, 2});

  aml::eval(h, z, x / y);

  EXPECT_EQ(z.data()[0], 0);
  EXPECT_EQ(z.data()[1], 0);
}

TEST_F(ExpressionTest, DivideAS) {
  auto x = aml::zeros<double, 2>(h, aml::CPU, {1, 2});
  auto y = 1.0;
  auto z = aml::nans<double, 2>(h, aml::CPU, {1, 2});

  aml::eval(h, z, x / y);

  EXPECT_EQ(z.data()[0], 0);
  EXPECT_EQ(z.data()[1], 0);
}

/** MIN ***********************************************************************/

TEST_F(ExpressionTest, MinAA) {
  auto x = aml::zeros<double, 2>(h, aml::CPU, {1, 2});
  auto y = aml::ones<double, 2>(h, aml::CPU, {1, 2});
  auto z = aml::nans<double, 2>(h, aml::CPU, {1, 2});

  aml::eval(h, z, aml::min(x, y));

  EXPECT_EQ(z.data()[0], 0);
  EXPECT_EQ(z.data()[1], 0);
}

TEST_F(ExpressionTest, MinAE) {
  auto x = aml::zeros<double, 2>(h, aml::CPU, {1, 2});
  auto y = aml::make_expression(aml::ones<double, 2>(h, aml::CPU, {1, 2}));
  auto z = aml::nans<double, 2>(h, aml::CPU, {1, 2});

  aml::eval(h, z, aml::min(x, y));

  EXPECT_EQ(z.data()[0], 0);
  EXPECT_EQ(z.data()[1], 0);
}

TEST_F(ExpressionTest, MinEA) {
  auto x = aml::make_expression(aml::zeros<double, 2>(h, aml::CPU, {1, 2}));
  auto y = aml::ones<double, 2>(h, aml::CPU, {1, 2});
  auto z = aml::nans<double, 2>(h, aml::CPU, {1, 2});

  aml::eval(h, z, aml::min(x, y));

  EXPECT_EQ(z.data()[0], 0);
  EXPECT_EQ(z.data()[1], 0);
}

TEST_F(ExpressionTest, MinEE) {
  auto x = aml::make_expression(aml::zeros<double, 2>(h, aml::CPU, {1, 2}));
  auto y = aml::make_expression(aml::ones<double, 2>(h, aml::CPU, {1, 2}));
  auto z = aml::nans<double, 2>(h, aml::CPU, {1, 2});

  aml::eval(h, z, aml::min(x, y));

  EXPECT_EQ(z.data()[0], 0);
  EXPECT_EQ(z.data()[1], 0);
}

TEST_F(ExpressionTest, MinSE) {
  auto x = 0.0;
  auto y = aml::make_expression(aml::ones<double, 2>(h, aml::CPU, {1, 2}));
  auto z = aml::nans<double, 2>(h, aml::CPU, {1, 2});

  aml::eval(h, z, aml::min(x, y));

  EXPECT_EQ(z.data()[0], 0);
  EXPECT_EQ(z.data()[1], 0);
}

TEST_F(ExpressionTest, MinES) {
  auto x = aml::make_expression(aml::zeros<double, 2>(h, aml::CPU, {1, 2}));
  auto y = 1.0;
  auto z = aml::nans<double, 2>(h, aml::CPU, {1, 2});

  aml::eval(h, z, aml::min(x, y));

  EXPECT_EQ(z.data()[0], 0);
  EXPECT_EQ(z.data()[1], 0);
}

TEST_F(ExpressionTest, MinSA) {
  auto x = 0.0;
  auto y = aml::ones<double, 2>(h, aml::CPU, {1, 2});
  auto z = aml::nans<double, 2>(h, aml::CPU, {1, 2});

  aml::eval(h, z, aml::min(x, y));

  EXPECT_EQ(z.data()[0], 0);
  EXPECT_EQ(z.data()[1], 0);
}

TEST_F(ExpressionTest, MinAS) {
  auto x = aml::zeros<double, 2>(h, aml::CPU, {1, 2});
  auto y = 1.0;
  auto z = aml::nans<double, 2>(h, aml::CPU, {1, 2});

  aml::eval(h, z, aml::min(x, y));

  EXPECT_EQ(z.data()[0], 0);
  EXPECT_EQ(z.data()[1], 0);
}

/** MAX ***********************************************************************/

TEST_F(ExpressionTest, MaxAA) {
  auto x = aml::zeros<double, 2>(h, aml::CPU, {1, 2});
  auto y = aml::ones<double, 2>(h, aml::CPU, {1, 2});
  auto z = aml::nans<double, 2>(h, aml::CPU, {1, 2});

  aml::eval(h, z, aml::max(x, y));

  EXPECT_EQ(z.data()[0], 1);
  EXPECT_EQ(z.data()[1], 1);
}

TEST_F(ExpressionTest, MaxAE) {
  auto x = aml::zeros<double, 2>(h, aml::CPU, {1, 2});
  auto y = aml::make_expression(aml::ones<double, 2>(h, aml::CPU, {1, 2}));
  auto z = aml::nans<double, 2>(h, aml::CPU, {1, 2});

  aml::eval(h, z, aml::max(x, y));

  EXPECT_EQ(z.data()[0], 1);
  EXPECT_EQ(z.data()[1], 1);
}

TEST_F(ExpressionTest, MaxEA) {
  auto x = aml::make_expression(aml::zeros<double, 2>(h, aml::CPU, {1, 2}));
  auto y = aml::ones<double, 2>(h, aml::CPU, {1, 2});
  auto z = aml::nans<double, 2>(h, aml::CPU, {1, 2});

  aml::eval(h, z, aml::max(x, y));

  EXPECT_EQ(z.data()[0], 1);
  EXPECT_EQ(z.data()[1], 1);
}

TEST_F(ExpressionTest, MaxEE) {
  auto x = aml::make_expression(aml::zeros<double, 2>(h, aml::CPU, {1, 2}));
  auto y = aml::make_expression(aml::ones<double, 2>(h, aml::CPU, {1, 2}));
  auto z = aml::nans<double, 2>(h, aml::CPU, {1, 2});

  aml::eval(h, z, aml::max(x, y));

  EXPECT_EQ(z.data()[0], 1);
  EXPECT_EQ(z.data()[1], 1);
}

TEST_F(ExpressionTest, MaxSE) {
  auto x = 0.0;
  auto y = aml::make_expression(aml::ones<double, 2>(h, aml::CPU, {1, 2}));
  auto z = aml::nans<double, 2>(h, aml::CPU, {1, 2});

  aml::eval(h, z, aml::max(x, y));

  EXPECT_EQ(z.data()[0], 1);
  EXPECT_EQ(z.data()[1], 1);
}

TEST_F(ExpressionTest, MaxES) {
  auto x = aml::make_expression(aml::zeros<double, 2>(h, aml::CPU, {1, 2}));
  auto y = 1.0;
  auto z = aml::nans<double, 2>(h, aml::CPU, {1, 2});

  aml::eval(h, z, aml::max(x, y));

  EXPECT_EQ(z.data()[0], 1);
  EXPECT_EQ(z.data()[1], 1);
}

TEST_F(ExpressionTest, MaxSA) {
  auto x = 0.0;
  auto y = aml::ones<double, 2>(h, aml::CPU, {1, 2});
  auto z = aml::nans<double, 2>(h, aml::CPU, {1, 2});

  aml::eval(h, z, aml::max(x, y));

  EXPECT_EQ(z.data()[0], 1);
  EXPECT_EQ(z.data()[1], 1);
}

TEST_F(ExpressionTest, MaxAS) {
  auto x = aml::zeros<double, 2>(h, aml::CPU, {1, 2});
  auto y = 1.0;
  auto z = aml::nans<double, 2>(h, aml::CPU, {1, 2});

  aml::eval(h, z, aml::max(x, y));

  EXPECT_EQ(z.data()[0], 1);
  EXPECT_EQ(z.data()[1], 1);
}

/** POW ***********************************************************************/

TEST_F(ExpressionTest, PowAA) {
  auto x = aml::zeros<double, 2>(h, aml::CPU, {1, 2});
  auto y = aml::ones<double, 2>(h, aml::CPU, {1, 2});
  auto z = aml::nans<double, 2>(h, aml::CPU, {1, 2});

  aml::eval(h, z, aml::pow(x, y));

  EXPECT_EQ(z.data()[0], 0);
  EXPECT_EQ(z.data()[1], 0);
}

TEST_F(ExpressionTest, PowAE) {
  auto x = aml::zeros<double, 2>(h, aml::CPU, {1, 2});
  auto y = aml::make_expression(aml::ones<double, 2>(h, aml::CPU, {1, 2}));
  auto z = aml::nans<double, 2>(h, aml::CPU, {1, 2});

  aml::eval(h, z, aml::pow(x, y));

  EXPECT_EQ(z.data()[0], 0);
  EXPECT_EQ(z.data()[1], 0);
}

TEST_F(ExpressionTest, PowEA) {
  auto x = aml::make_expression(aml::zeros<double, 2>(h, aml::CPU, {1, 2}));
  auto y = aml::ones<double, 2>(h, aml::CPU, {1, 2});
  auto z = aml::nans<double, 2>(h, aml::CPU, {1, 2});

  aml::eval(h, z, aml::pow(x, y));

  EXPECT_EQ(z.data()[0], 0);
  EXPECT_EQ(z.data()[1], 0);
}

TEST_F(ExpressionTest, PowEE) {
  auto x = aml::make_expression(aml::zeros<double, 2>(h, aml::CPU, {1, 2}));
  auto y = aml::make_expression(aml::ones<double, 2>(h, aml::CPU, {1, 2}));
  auto z = aml::nans<double, 2>(h, aml::CPU, {1, 2});

  aml::eval(h, z, aml::pow(x, y));

  EXPECT_EQ(z.data()[0], 0);
  EXPECT_EQ(z.data()[1], 0);
}

TEST_F(ExpressionTest, PowSE) {
  auto x = 0.0;
  auto y = aml::make_expression(aml::ones<double, 2>(h, aml::CPU, {1, 2}));
  auto z = aml::nans<double, 2>(h, aml::CPU, {1, 2});

  aml::eval(h, z, aml::pow(x, y));

  EXPECT_EQ(z.data()[0], 0);
  EXPECT_EQ(z.data()[1], 0);
}

TEST_F(ExpressionTest, PowES) {
  auto x = aml::make_expression(aml::zeros<double, 2>(h, aml::CPU, {1, 2}));
  auto y = 1.0;
  auto z = aml::nans<double, 2>(h, aml::CPU, {1, 2});

  aml::eval(h, z, aml::pow(x, y));

  EXPECT_EQ(z.data()[0], 0);
  EXPECT_EQ(z.data()[1], 0);
}

TEST_F(ExpressionTest, PowSA) {
  auto x = 0.0;
  auto y = aml::ones<double, 2>(h, aml::CPU, {1, 2});
  auto z = aml::nans<double, 2>(h, aml::CPU, {1, 2});

  aml::eval(h, z, aml::pow(x, y));

  EXPECT_EQ(z.data()[0], 0);
  EXPECT_EQ(z.data()[1], 0);
}

TEST_F(ExpressionTest, PowAS) {
  auto x = aml::zeros<double, 2>(h, aml::CPU, {1, 2});
  auto y = 1.0;
  auto z = aml::nans<double, 2>(h, aml::CPU, {1, 2});

  aml::eval(h, z, aml::pow(x, y));

  EXPECT_EQ(z.data()[0], 0);
  EXPECT_EQ(z.data()[1], 0);
}

