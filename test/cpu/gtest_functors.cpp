#include <cmath>

#include <gtest/gtest.h>

#include <aml/aml.h>

TEST(FunctorTest, Abs) {
  EXPECT_EQ(aml::Abs()(-3), 3);
  EXPECT_EQ(aml::Abs()(-0.5f), 0.5f);
  EXPECT_EQ(aml::Abs()(-0.5), 0.5);
}

TEST(FunctorTest, Divide) {
  EXPECT_EQ(aml::Divide()(-0.5f, 0.5f), -1.0f);
  EXPECT_EQ(aml::Divide()(12, 3), 4);
}

TEST(FunctorTest, Exp) {
  EXPECT_EQ(aml::Exp()(-0.5f), std::exp(-0.5f));
  EXPECT_EQ(aml::Exp()(-0.5), std::exp(-0.5));
}

TEST(FunctorTest, Identity) {
  EXPECT_EQ(aml::Identity()(0.5f), 0.5f);
  EXPECT_EQ(aml::Identity()(0.5), 0.5);
  EXPECT_EQ(aml::Identity()(-0.5f), -0.5f);
  EXPECT_EQ(aml::Identity()(-0.5), -0.5);
}

TEST(FunctorTest, Inv) {
  EXPECT_EQ(aml::Inv()(0.5f), 2.0f);
  EXPECT_EQ(aml::Inv()(0.5), 2.0);
}

TEST(FunctorTest, Log) {
  EXPECT_EQ(aml::Log()(0.5f), std::log(0.5f));
  EXPECT_EQ(aml::Log()(0.5), std::log(0.5));
}

TEST(FunctorTest, Max) {
  EXPECT_EQ(aml::Max()(-5, 2), 2);
  EXPECT_EQ(aml::Max()(2, -5), 2);
  EXPECT_EQ(aml::Max()(-0.5f, 0.25f), 0.25f);
  EXPECT_EQ(aml::Max()(0.25f, -0.5f), 0.25f);
  EXPECT_EQ(aml::Max()(-0.5, 0.25), 0.25);
  EXPECT_EQ(aml::Max()(0.25, -0.5), 0.25);
}

TEST(FunctorTest, Min) {
  EXPECT_EQ(aml::Min()(-5, 2), -5);
  EXPECT_EQ(aml::Min()(2, -5), -5);
  EXPECT_EQ(aml::Min()(-0.5f, 0.25f), -0.5f);
  EXPECT_EQ(aml::Min()(0.25f, -0.5f), -0.5f);
  EXPECT_EQ(aml::Min()(-0.5, 0.25), -0.5);
  EXPECT_EQ(aml::Min()(0.25, -0.5), -0.5);
}

TEST(FunctorTest, Minus) {
  EXPECT_EQ(aml::Minus()(2.0f, 2.0f), 0.0f);
  EXPECT_EQ(aml::Minus()(-2.0, -1.5), -0.5);
}

TEST(FunctorTest, Multiply) {
  EXPECT_EQ(aml::Multiply()(2.0f, 2.0f), 4.0f);
  EXPECT_EQ(aml::Multiply()(2.0, -1.5), -3.0);
}

TEST(FunctorTest, Negative) {
  EXPECT_EQ(aml::Negative()(2.0f), -2.0f);
  EXPECT_EQ(aml::Negative()(-1.5), 1.5);
}

TEST(FunctorTest, Plus) {
  EXPECT_EQ(aml::Plus()(2.0f, -2.0f), 0.0f);
  EXPECT_EQ(aml::Plus()(2.0, 1.5), 3.5);
}

TEST(FunctorTest, Pow) {
  EXPECT_EQ(aml::Pow()(2.0f, 0.0f), 1.0f);
  EXPECT_EQ(aml::Pow()(2.0, 1.0), 2.0);
  EXPECT_EQ(aml::Pow()(2.0, -1.0), 0.5);
}

TEST(FunctorTest, Sqrt) {
  EXPECT_EQ(aml::Sqrt()(2.0f), std::sqrt(2.0f));
  EXPECT_EQ(aml::Sqrt()(2.0), std::sqrt(2.0));
}

TEST(FunctorTest, Square) {
  EXPECT_EQ(aml::Square()(-3), 9);
  EXPECT_EQ(aml::Square()(3), 9);
  EXPECT_EQ(aml::Square()(-2.5f), 6.25f);
  EXPECT_EQ(aml::Square()(2.5f), 6.25f);
  EXPECT_EQ(aml::Square()(-2.5), 6.25);
  EXPECT_EQ(aml::Square()(2.5), 6.25);
}

