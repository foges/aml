#include <gtest/gtest.h>

#include <aml/aml.h>

TEST(ArrayTest, DefaultConstructor) {
  aml::Array<float, 2> a;
  EXPECT_EQ(a.allocation(), nullptr);
  EXPECT_EQ(a.data(), nullptr);
  EXPECT_EQ(a.size(), aml::make_shape(0, 0));
  EXPECT_EQ(a.stride(), aml::make_shape(1, 0));
  EXPECT_TRUE(a.is_contiguous());
}

TEST(ArrayTest, DefaultConstructorConst) {
  const aml::Array<float, 2> a;
  EXPECT_EQ(a.allocation(), nullptr);
  EXPECT_EQ(a.data(), nullptr);
  EXPECT_EQ(a.size(), aml::make_shape(0, 0));
  EXPECT_EQ(a.stride(), aml::make_shape(1, 0));
  EXPECT_TRUE(a.is_contiguous());
}

TEST(ArrayTest, Constructor) {
  aml::Array<float, 2> a(aml::CPU, aml::make_shape(3, 4));
  EXPECT_EQ(a.device(), aml::CPU);
  EXPECT_NE(a.allocation(), nullptr);
  EXPECT_EQ(a.allocation()->size(), 3 * 4 * sizeof(float));
  EXPECT_NE(a.data(), nullptr);
  EXPECT_EQ(a.size(), aml::make_shape(3, 4));
  EXPECT_EQ(a.stride(), aml::make_shape(1, 3));
  EXPECT_TRUE(a.is_contiguous());
}

TEST(ArrayTest, MakeArray) {
  std::vector<double> data = {4.0, 1.3, 1.4, 5.7, 6.9, 7.1};
  aml::Shape<3> size = aml::make_shape(1, 2, 3);
  aml::Array<double, 3> a = aml::make_array(data, size);

  EXPECT_EQ(a.device(), aml::CPU);
  EXPECT_EQ(a.size(), size);
  EXPECT_TRUE(a.is_contiguous());
  for (size_t i = 0; i < data.size(); ++i) {
    EXPECT_EQ(a.data()[i], data[i]);
  }
}

TEST(ArrayTest, Slice) {
  aml::Array<float, 2> a(aml::CPU, aml::make_shape(3, 4));
  aml::Array<float, 2> s =
      aml::slice(a, aml::make_shape(1, 1), aml::make_shape(3, 4));

  EXPECT_EQ(s.device(), aml::CPU);
  EXPECT_EQ(s.allocation(), a.allocation());
  EXPECT_EQ(s.data(), a.data() + 4);
  EXPECT_EQ(s.size(), aml::make_shape(2, 3));
  EXPECT_EQ(s.stride(), a.stride());
  EXPECT_FALSE(s.is_contiguous());
}

TEST(ArrayTest, ContiguousSlice) {
  aml::Array<float, 2> a(aml::CPU, aml::make_shape(3, 4));
  aml::Array<float, 2> s =
      aml::slice(a, aml::make_shape(0, 2), aml::make_shape(3, 4));

  EXPECT_EQ(s.device(), aml::CPU);
  EXPECT_EQ(s.allocation(), a.allocation());
  EXPECT_EQ(s.data(), a.data() + 6);
  EXPECT_EQ(s.size(), aml::make_shape(3, 2));
  EXPECT_EQ(s.stride(), a.stride());
  EXPECT_TRUE(s.is_contiguous());
}

TEST(ArrayTest, Reshape) {
  aml::Array<float, 2> a(aml::CPU, aml::make_shape(3, 4));
  aml::Array<float, 3> r =
      aml::reshape(a, aml::make_shape(6, 2, 1));

  EXPECT_EQ(r.device(), aml::CPU);
  EXPECT_EQ(r.allocation(), a.allocation());
  EXPECT_EQ(r.data(), a.data());
  EXPECT_EQ(r.size(), aml::make_shape(6, 2, 1));
  EXPECT_EQ(r.stride(), aml::make_shape(1, 6, 12));
  EXPECT_TRUE(r.is_contiguous());
}
