#include <gtest/gtest.h>

#include <aml/aml.h>

TEST(ConstArrayTest, DefaultConstructor) {
  aml::ConstArray<float, 2> ca;
  EXPECT_EQ(ca.allocation(), nullptr);
  EXPECT_EQ(ca.data(), nullptr);
  EXPECT_EQ(ca.size(), aml::make_shape(0, 0));
  EXPECT_EQ(ca.stride(), aml::make_shape(1, 0));
  EXPECT_TRUE(ca.is_contiguous());
}

TEST(ConstArrayTest, DefaultConstructor1) {
  aml::Array<float, 2> a;
  aml::ConstArray<float, 2> ca(a);
  EXPECT_EQ(ca.allocation(), nullptr);
  EXPECT_EQ(ca.data(), nullptr);
  EXPECT_EQ(ca.size(), aml::make_shape(0, 0));
  EXPECT_EQ(ca.stride(), aml::make_shape(1, 0));
  EXPECT_TRUE(ca.is_contiguous());
}

TEST(ConstArrayTest, CopyConstructor1) {
  aml::ConstArray<float, 2> a;
  aml::ConstArray<float, 2> ca(a);
  EXPECT_EQ(ca.allocation(), nullptr);
  EXPECT_EQ(ca.data(), nullptr);
  EXPECT_EQ(ca.size(), aml::make_shape(0, 0));
  EXPECT_EQ(ca.stride(), aml::make_shape(1, 0));
  EXPECT_TRUE(ca.is_contiguous());
}

TEST(ConstArrayTest, Constructor1) {
  aml::Array<float, 2> a(aml::CPU, aml::make_shape(3, 4));
  aml::ConstArray<float, 2> ca(a);
  EXPECT_EQ(ca.device(), aml::CPU);
  EXPECT_NE(ca.allocation(), nullptr);
  EXPECT_EQ(ca.allocation()->size(), 3 * 4 * sizeof(float));
  EXPECT_NE(ca.data(), nullptr);
  EXPECT_EQ(ca.size(), aml::make_shape(3, 4));
  EXPECT_EQ(ca.stride(), aml::make_shape(1, 3));
  EXPECT_TRUE(ca.is_contiguous());
}

TEST(ConstArrayTest, Slice) {
  aml::Array<float, 2> a(aml::CPU, aml::make_shape(3, 4));
  aml::ConstArray<float, 2> ca(a);
  aml::ConstArray<float, 2> s =
      aml::slice(ca, aml::make_shape(1, 1), aml::make_shape(3, 4));

  EXPECT_EQ(s.device(), aml::CPU);
  EXPECT_EQ(s.allocation(), a.allocation());
  EXPECT_EQ(s.data(), a.data() + 4);
  EXPECT_EQ(s.size(), aml::make_shape(2, 3));
  EXPECT_EQ(s.stride(), a.stride());
  EXPECT_FALSE(s.is_contiguous());
}

TEST(ConstArrayTest, ContiguousSlice) {
  aml::Array<float, 2> a(aml::CPU, aml::make_shape(3, 4));
  aml::ConstArray<float, 2> ca(a);
  aml::ConstArray<float, 2> s =
      aml::slice(ca, aml::make_shape(0, 2), aml::make_shape(3, 4));

  EXPECT_EQ(s.device(), aml::CPU);
  EXPECT_EQ(s.allocation(), a.allocation());
  EXPECT_EQ(s.data(), a.data() + 6);
  EXPECT_EQ(s.size(), aml::make_shape(3, 2));
  EXPECT_EQ(s.stride(), a.stride());
  EXPECT_TRUE(s.is_contiguous());
}

TEST(ConstArrayTest, Reshape) {
  aml::Array<float, 2> a(aml::CPU, aml::make_shape(3, 4));
  aml::ConstArray<float, 2> ca(a);
  aml::ConstArray<float, 3> r =
      aml::reshape(ca, aml::make_shape(6, 2, 1));

  EXPECT_EQ(r.device(), aml::CPU);
  EXPECT_EQ(r.allocation(), a.allocation());
  EXPECT_EQ(r.data(), a.data());
  EXPECT_EQ(r.size(), aml::make_shape(6, 2, 1));
  EXPECT_EQ(r.stride(), aml::make_shape(1, 6, 12));
  EXPECT_TRUE(r.is_contiguous());
}
