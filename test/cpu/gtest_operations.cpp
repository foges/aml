#include <gtest/gtest.h>

#include <aml/aml.h>

TEST(OperationsTest, Set) {
  auto a = aml::Array<int, 4>(aml::CPU, aml::make_shape(1, 2, 1, 2));
  aml::set(a, 9);
  EXPECT_EQ(a.data()[0], 9);
  EXPECT_EQ(a.data()[1], 9);
  EXPECT_EQ(a.data()[2], 9);
  EXPECT_EQ(a.data()[3], 9);
}

TEST(OperationsTest, Copy) {
  std::vector<int> data = {4, 2, 3, -9};
  auto a = aml::make_array(data, aml::make_shape(2, 2));
  auto b = aml::Array<int, 2>(aml::CPU, a.size());
  aml::copy(a, b);
  EXPECT_EQ(b.data()[0], 4);
  EXPECT_EQ(b.data()[1], 2);
  EXPECT_EQ(b.data()[2], 3);
  EXPECT_EQ(b.data()[3], -9);
}

TEST(OperationsTest, CopyTypeCast) {
  std::vector<double> data = {4.0, 2.0, 3.0, -9.0};
  auto a = aml::make_array(data, aml::make_shape(2, 2));
  auto b = aml::Array<int, 2>(aml::CPU, a.size());
  aml::copy(a, b);
  EXPECT_EQ(b.data()[0], 4);
  EXPECT_EQ(b.data()[1], 2);
  EXPECT_EQ(b.data()[2], 3);
  EXPECT_EQ(b.data()[3], -9);
}

TEST(OperationsTest, CopyNonContiguous) {
  std::vector<int> data = {4, 2, 3, -9};
  auto a = aml::make_array(data, aml::make_shape(2, 2));
  auto b = aml::slice(a, aml::make_shape(1, 0), aml::make_shape(2, 2));
  auto c = aml::Array<int, 2>(aml::CPU, b.size());
  aml::copy(b, c);
  EXPECT_EQ(c.data()[0], 2);
  EXPECT_EQ(c.data()[1], -9);
}

TEST(OperationsTest, UnaryOpAbs) {
  std::vector<int> data = {-4, 2, -3, -9};
  auto a = aml::make_array(data, aml::make_shape(1, 2, 1, 2));
  aml::unary_op(a, a, aml::Abs());
  EXPECT_EQ(a.data()[0], 4);
  EXPECT_EQ(a.data()[1], 2);
  EXPECT_EQ(a.data()[2], 3);
  EXPECT_EQ(a.data()[3], 9);
}

TEST(OperationsTest, BinaryOpMax) {
  std::vector<int> data1 = {4, 5, 3, 7};
  std::vector<int> data2 = {8, 4, 1, 9};
  auto in1 = aml::make_array(data1, aml::make_shape(1, 2, 2, 1));
  auto in2 = aml::make_array(data2, aml::make_shape(1, 2, 2, 1));
  auto out = aml::Array<int, 4>(aml::CPU, aml::make_shape(1, 2, 2, 1));
  aml::binary_op(in1, in2, out, aml::Max());
  EXPECT_EQ(out.data()[0], 8);
  EXPECT_EQ(out.data()[1], 5);
  EXPECT_EQ(out.data()[2], 3);
  EXPECT_EQ(out.data()[3], 9);
}

TEST(OperationsTest, BinaryOpMaxSlice) {
  std::vector<int> data1 = {4, 5, 3, 7};
  std::vector<int> data2 = {8, 4, 1, 9};
  auto in1 = aml::make_array(data1, aml::make_shape(1, 2, 2, 1));
  auto in2 = aml::make_array(data2, aml::make_shape(1, 2, 2, 1));
  auto in1s = aml::slice(in1, aml::make_shape(0, 1, 0, 0),
      aml::make_shape(1, 2, 2, 1));
  auto in2s = aml::slice(in2, aml::make_shape(0, 1, 0, 0),
      aml::make_shape(1, 2, 2, 1));
  auto out = aml::Array<int, 4>(aml::CPU, aml::make_shape(1, 1, 2, 1));
  aml::binary_op(in1s, in2s, out, aml::Min());
  EXPECT_EQ(out.data()[0], 4);
  EXPECT_EQ(out.data()[1], 7);
}

TEST(OperationsTest, ReduceMax) {
  std::vector<int> data = {4, -9, -3, 7};
  std::vector<int> res = {0};
  auto in = aml::make_array(data, aml::make_shape(1, 2, 2, 1));
  auto out = aml::make_array(res, aml::Shape<0>());
  aml::reduce(in, out, {{0, 1, 3, 2}}, aml::Identity(), aml::Max());
  EXPECT_EQ(out.data()[0], 7);
}

TEST(OperationsTest, ReduceMaxAbs) {
  std::vector<int> data = {4, -9, -3, 7};
  std::vector<int> res = {0};
  auto in = aml::make_array(data, aml::make_shape(1, 2, 2, 1));
  auto out = aml::make_array(res, aml::Shape<0>());
  aml::reduce(in, out, {{0, 1, 3, 2}}, aml::Abs(), aml::Max());
  EXPECT_EQ(out.data()[0], 9);
}

TEST(OperationsTest, ReduceMaxAbsSlice) {
  std::vector<int> data = {4, -9, -3, 7};
  std::vector<int> res = {0};
  auto in = aml::make_array(data, aml::make_shape(1, 2, 2, 1));
  auto ins = aml::slice(in, aml::make_shape(0, 0, 0, 0),
      aml::make_shape(1, 1, 2, 1));
  auto out = aml::make_array(res, aml::Shape<0>());
  aml::reduce(ins, out, {{3, 1, 0, 2}}, aml::Abs(), aml::Max());
  EXPECT_EQ(out.data()[0], 4);
}

TEST(OperationsTest, ReducePartial1Max) {
  std::vector<int> data = {4, -9, -3, 7};
  std::vector<int> res = {0, 0};
  auto in = aml::make_array(data, aml::make_shape(1, 2, 2, 1));
  auto out = aml::make_array(res, aml::make_shape(2));
  aml::reduce(in, out, {{0, 3, 2}}, aml::Identity(), aml::Max());
  EXPECT_EQ(out.data()[0], 4);
  EXPECT_EQ(out.data()[1], 7);
}

TEST(OperationsTest, ReducePartial2Max) {
  std::vector<int> data = {4, 7, -3, -9};
  std::vector<int> res = {-10, -10};
  auto in = aml::make_array(data, aml::make_shape(1, 2, 2, 1));
  auto out = aml::make_array(res, aml::make_shape(2));
  aml::reduce(in, out, {{0, 3, 1}}, aml::Identity(), aml::Max());
  EXPECT_EQ(out.data()[0], 7);
  EXPECT_EQ(out.data()[1], -3);
}

