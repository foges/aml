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
