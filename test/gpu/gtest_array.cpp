#include <gtest/gtest.h>

#include <aml/aml.h>

TEST(ArrayTestGpu, Constructor) {
  aml::Array<float, 2> a(aml::GPU, aml::make_shape(3, 4));
  EXPECT_EQ(a.device(), aml::GPU);
  EXPECT_NE(a.allocation(), nullptr);
  EXPECT_EQ(a.allocation()->size(), 3 * 4 * sizeof(float));
  EXPECT_NE(a.data(), nullptr);
  EXPECT_EQ(a.size(), aml::make_shape(3, 4));
  EXPECT_EQ(a.stride(), aml::make_shape(1, 3));
  EXPECT_TRUE(a.is_contiguous());
}

TEST(ArrayTestGpu, Slice) {
  aml::Array<float, 2> a(aml::GPU, aml::make_shape(3, 4));
  aml::Array<float, 2> s =
      aml::slice(a, aml::make_shape(1, 1), aml::make_shape(3, 4));

  EXPECT_EQ(s.device(), aml::GPU);
  EXPECT_EQ(s.allocation(), a.allocation());
  EXPECT_EQ(s.data(), a.data() + 4);
  EXPECT_EQ(s.size(), aml::make_shape(2, 3));
  EXPECT_EQ(s.stride(), a.stride());
  EXPECT_FALSE(s.is_contiguous());
}

TEST(ArrayTestGpu, ContiguousSlice) {
  aml::Array<float, 2> a(aml::GPU, aml::make_shape(3, 4));
  aml::Array<float, 2> s =
      aml::slice(a, aml::make_shape(0, 2), aml::make_shape(3, 4));

  EXPECT_EQ(s.device(), aml::GPU);
  EXPECT_EQ(s.allocation(), a.allocation());
  EXPECT_EQ(s.data(), a.data() + 6);
  EXPECT_EQ(s.size(), aml::make_shape(3, 2));
  EXPECT_EQ(s.stride(), a.stride());
  EXPECT_TRUE(s.is_contiguous());
}

TEST(ArrayTestGpu, Reshape) {
  aml::Array<float, 2> a(aml::GPU, aml::make_shape(3, 4));
  aml::Array<float, 3> r =
      aml::reshape(a, aml::make_shape(6, 2, 1));

  EXPECT_EQ(r.device(), aml::GPU);
  EXPECT_EQ(r.allocation(), a.allocation());
  EXPECT_EQ(r.data(), a.data());
  EXPECT_EQ(r.size(), aml::make_shape(6, 2, 1));
  EXPECT_EQ(r.stride(), aml::make_shape(1, 6, 12));
  EXPECT_TRUE(r.is_contiguous());
}
