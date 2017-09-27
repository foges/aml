#include <gtest/gtest.h>

#include <aml/aml.h>

TEST(ShapeTest, DefaultConstructor) {
  aml::Shape<2> s;
  EXPECT_EQ(0, s[0]);
  EXPECT_EQ(0, s[1]);
}

TEST(ShapeTest, InitializerList) {
  aml::Shape<2> s = {3, 4};
  EXPECT_EQ(3, s[0]);
  EXPECT_EQ(4, s[1]);
}

TEST(ShapeTest, InitializerListConstructor) {
  aml::Shape<2> s({3, 4});
  EXPECT_EQ(3, s[0]);
  EXPECT_EQ(4, s[1]);
}

TEST(ShapeTest, MakeShape1) {
  aml::Shape<1> s = aml::make_shape(4);
  EXPECT_EQ(4, s[0]);
}

TEST(ShapeTest, MakeShape2) {
  aml::Shape<2> s = aml::make_shape(3, 4);
  EXPECT_EQ(3, s[0]);
  EXPECT_EQ(4, s[1]);
}

TEST(ShapeTest, MakeShapeAuto1) {
  auto s = aml::make_shape(3);
  EXPECT_EQ(3, s[0]);
}

TEST(ShapeTest, MakeShapeAuto2) {
  auto s = aml::make_shape(3, 4);
  EXPECT_EQ(3, s[0]);
  EXPECT_EQ(4, s[1]);
}

TEST(ShapeTest, Equals) {
  aml::Shape<1> s1 = {4};
  aml::Shape<1> s2 = {4};
  EXPECT_EQ(s1, s2);
}

TEST(ShapeTest, NotEquals) {
  aml::Shape<2> s1 = {4, 3};
  aml::Shape<2> s2 = {4, 4};
  EXPECT_NE(s1, s2);
}

TEST(ShapeTest, Copy) {
  aml::Shape<1> s1 = {4};
  aml::Shape<1> s2 = s1;
  EXPECT_EQ(s1, s2);
}

TEST(ShapeTest, Size1) {
  aml::Shape<1> s = {4};
  EXPECT_EQ(1, s.size());
}

TEST(ShapeTest, Size2) {
  aml::Shape<2> s = {9, 5};
  EXPECT_EQ(2, s.size());
}
