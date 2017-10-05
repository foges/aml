#include <gtest/gtest.h>

#include <aml/aml.h>

#define EXPECT_EQ_IDX(lhs, rhs) EXPECT_EQ((lhs), static_cast<aml::Index>(rhs))

TEST(ShapeTest, DefaultConstructor) {
  aml::Shape<2> s;
  EXPECT_EQ_IDX(s[0], 0);
  EXPECT_EQ_IDX(s[1], 0);
}

TEST(ShapeTest, Constructor) {
  aml::Shape<2> s = aml::Shape<2>(3u, 4u);
  EXPECT_EQ_IDX(s[0], 3);
  EXPECT_EQ_IDX(s[1], 4);
}

TEST(ShapeTest, Assignment) {
  aml::Shape<2> s = aml::make_shape(3, 4);
  EXPECT_EQ_IDX(s[0], 3);
  EXPECT_EQ_IDX(s[1], 4);
  s[1] = 5;
  EXPECT_EQ_IDX(s[1], 5);
}

TEST(ShapeTest, MakeShape1) {
  aml::Shape<1> s = aml::make_shape(4);
  EXPECT_EQ_IDX(s[0], 4);
}

TEST(ShapeTest, MakeShape2) {
  aml::Shape<2> s = aml::make_shape(3, 4);
  EXPECT_EQ_IDX(s[0], 3);
  EXPECT_EQ_IDX(s[1], 4);
}

TEST(ShapeTest, Equals) {
  aml::Shape<1> s1 = aml::make_shape(4);
  aml::Shape<1> s2 = aml::make_shape(4);
  EXPECT_EQ(s1, s2);
}

TEST(ShapeTest, NotEquals) {
  aml::Shape<2> s1 = aml::make_shape(4, 3);
  aml::Shape<2> s2 = aml::make_shape(4, 4);
  EXPECT_NE(s1, s2);
}

TEST(ShapeTest, LessThanEqual) {
  aml::Shape<3> s1 = aml::make_shape(4, 3, 1);
  aml::Shape<3> s2 = aml::make_shape(4, 5, 1);
  EXPECT_LE(s1, s2);
}

TEST(ShapeTest, GreaterThan) {
  aml::Shape<3> s1 = aml::make_shape(4, 5, 2);
  aml::Shape<3> s2 = aml::make_shape(4, 5, 1);
  EXPECT_GT(s1, s2);
}

TEST(ShapeTest, Iterator) {
  aml::Shape<2> s = aml::make_shape(3, 4);
  auto it = s.begin();
  EXPECT_EQ_IDX(*it, 3);
  *it = 7;
  EXPECT_EQ_IDX(*it, 7);
  ++it;
  EXPECT_EQ_IDX(*it, 4);
  *it = 7;
  EXPECT_EQ_IDX(*it, 7);
  ++it;
  EXPECT_EQ(it, s.end());
}

TEST(ShapeTest, IteratorConst) {
  const aml::Shape<2> s = aml::make_shape(3, 4);
  auto it = s.begin();
  EXPECT_EQ_IDX(*it, 3);
  ++it;
  EXPECT_EQ_IDX(*it, 4);
  ++it;
  EXPECT_EQ(it, s.end());
}

TEST(ShapeTest, Copy) {
  aml::Shape<1> s1 = aml::make_shape(4);
  aml::Shape<1> s2 = s1;
  EXPECT_EQ(s1, s2);
}

TEST(ShapeTest, Size1) {
  aml::Shape<1> s = aml::make_shape(4);
  EXPECT_EQ(s.dim(), 1);
}

TEST(ShapeTest, Size2) {
  aml::Shape<2> s = aml::make_shape(9, 5);
  EXPECT_EQ(s.dim(), 2);
}

