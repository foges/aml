#include <gtest/gtest.h>

#include <aml/aml.h>

TEST(ShapeTest, DefaultConstructor) {
  aml::Shape<2> s;
  assert(s[0] == 0);
  assert(s[1] == 0);
}

TEST(ShapeTest, InitializerList) {
  aml::Shape<2> s = {3, 4};
  assert(s[0] == 3);
  assert(s[1] == 4);
}

