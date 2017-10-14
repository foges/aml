#include <gtest/gtest.h>

#include <aml/aml.h>

TEST(OperationsTest, GpuSet) {
  auto a = aml::Array<int, 4>(aml::GPU, aml::make_shape(1, 2, 1, 2));
  aml::set(a, 9);
  auto a_h = aml::Array<int, 4>(aml::CPU, aml::make_shape(1, 2, 1, 2));
  aml::copy(a, a_h);

  EXPECT_EQ(a_h.data()[0], 9);
  EXPECT_EQ(a_h.data()[1], 9);
  EXPECT_EQ(a_h.data()[2], 9);
  EXPECT_EQ(a_h.data()[3], 9);
}

TEST(OperationsTest, GpuCopyCpuGpu) {
  std::vector<int> data = {4, 2, 3, -9};
  auto a = aml::make_array(data, aml::make_shape(2, 2));
  auto b = aml::Array<int, 2>(aml::GPU, a.size());
  aml::copy(a, b);
  auto b_h = aml::Array<int, 2>(aml::CPU, aml::make_shape(2, 2));
  aml::copy(b, b_h);

  EXPECT_EQ(b_h.data()[0], 4);
  EXPECT_EQ(b_h.data()[1], 2);
  EXPECT_EQ(b_h.data()[2], 3);
  EXPECT_EQ(b_h.data()[3], -9);
}

TEST(OperationsTest, GpuCopyGpuGpu) {
  std::vector<int> data = {4, 2, 3, -9};
  auto a = aml::make_array(data, aml::make_shape(2, 2));
  auto b = aml::Array<int, 2>(aml::GPU, a.size());
  auto c = aml::Array<int, 2>(aml::GPU, a.size());
  aml::copy(a, b);
  aml::copy(b, c);
  auto c_h = aml::Array<int, 2>(aml::CPU, aml::make_shape(2, 2));
  aml::copy(c, c_h);

  EXPECT_EQ(c_h.data()[0], 4);
  EXPECT_EQ(c_h.data()[1], 2);
  EXPECT_EQ(c_h.data()[2], 3);
  EXPECT_EQ(c_h.data()[3], -9);
}

TEST(OperationsTest, GpuCopyTypeCast) {
  std::vector<double> data = {4.0, 2.0, 3.0, -9.0};
  auto a = aml::make_array(data, aml::make_shape(2, 2));
  auto b = aml::Array<double, 2>(aml::GPU, a.size());
  auto c = aml::Array<int, 2>(aml::GPU, a.size());
  aml::copy(b, c);
  auto c_h = aml::Array<int, 2>(aml::CPU, aml::make_shape(2, 2));
  aml::copy(c, c_h);

  EXPECT_EQ(c_h.data()[0], 4);
  EXPECT_EQ(c_h.data()[1], 2);
  EXPECT_EQ(c_h.data()[2], 3);
  EXPECT_EQ(c_h.data()[3], -9);
}

TEST(OperationsTest, GpuCopyNonContiguous) {
  std::vector<int> data = {4, 2, 3, -9};
  auto a_h = aml::make_array(data, aml::make_shape(2, 2));
  auto a = aml::Array<int, 2>(aml::GPU, a_h.size());
  aml::copy(a_h, a);
  auto b = aml::slice(a, aml::make_shape(1, 0), aml::make_shape(2, 2));
  auto c = aml::Array<int, 2>(aml::GPU, b.size());
  aml::copy(b, c);
  auto c_h = aml::Array<int, 2>(aml::CPU, aml::make_shape(2, 2));
  aml::copy(c, c_h);

  EXPECT_EQ(c_h.data()[0], 2);
  EXPECT_EQ(c_h.data()[1], -9);
}

TEST(OperationsTest, GpuUnaryOpAbs) {
  std::vector<int> data = {-4, 2, -3, -9};
  auto a_h = aml::make_array(data, aml::make_shape(1, 2, 1, 2));
  auto a = aml::Array<int, 4>(aml::GPU, a_h.size());
  aml::copy(a_h, a);
  aml::unary_op(a, a, aml::Abs());
  aml::copy(a, a_h);
  EXPECT_EQ(a_h.data()[0], 4);
  EXPECT_EQ(a_h.data()[1], 2);
  EXPECT_EQ(a_h.data()[2], 3);
  EXPECT_EQ(a_h.data()[3], 9);
}

TEST(OperationsTest, GpuBinaryOpMax) {
  std::vector<int> data1 = {4, 5, 3, 7};
  std::vector<int> data2 = {8, 4, 1, 9};
  auto in1_h = aml::make_array(data1, aml::make_shape(1, 2, 2, 1));
  auto in2_h = aml::make_array(data2, aml::make_shape(1, 2, 2, 1));
  auto out_h = aml::Array<int, 4>(aml::CPU, aml::make_shape(1, 2, 2, 1));
  auto in1 = aml::Array<int, 4>(aml::GPU, in1_h.size());
  auto in2 = aml::Array<int, 4>(aml::GPU, in2_h.size());
  auto out = aml::Array<int, 4>(aml::GPU, out_h.size());
  aml::copy(in1_h, in1);
  aml::copy(in2_h, in2);
  aml::binary_op(in1, in2, out, aml::Max());
  aml::copy(out, out_h);
  EXPECT_EQ(out_h.data()[0], 8);
  EXPECT_EQ(out_h.data()[1], 5);
  EXPECT_EQ(out_h.data()[2], 3);
  EXPECT_EQ(out_h.data()[3], 9);
}

TEST(OperationsTest, GpuBinaryOpMaxSlice) {
  std::vector<int> data1 = {4, 5, 3, 7};
  std::vector<int> data2 = {8, 4, 1, 9};
  auto in1_h = aml::make_array(data1, aml::make_shape(1, 2, 2, 1));
  auto in2_h = aml::make_array(data2, aml::make_shape(1, 2, 2, 1));
  auto out_h = aml::Array<int, 4>(aml::CPU, aml::make_shape(1, 1, 2, 1));
  auto in1 = aml::Array<int, 4>(aml::GPU, in1_h.size());
  auto in2 = aml::Array<int, 4>(aml::GPU, in2_h.size());
  auto in1s = aml::slice(in1, aml::make_shape(0, 1, 0, 0),
      aml::make_shape(1, 2, 2, 1));
  auto in2s = aml::slice(in2, aml::make_shape(0, 1, 0, 0),
      aml::make_shape(1, 2, 2, 1));
  auto out = aml::Array<int, 4>(aml::GPU, out_h.size());
  aml::copy(in1_h, in1);
  aml::copy(in2_h, in2);
  aml::binary_op(in1s, in2s, out, aml::Min());
  aml::copy(out, out_h);
  EXPECT_EQ(out_h.data()[0], 4);
  EXPECT_EQ(out_h.data()[1], 7);
}

TEST(OperationsTest, GpuReduceMax) {
  std::vector<int> data = {4, -9, -3, 7};
  std::vector<int> res = {0};
  auto in_h = aml::make_array(data, aml::make_shape(1, 2, 2, 1));
  auto in = aml::Array<int, 4>(aml::GPU, in_h.size());
  auto out_h = aml::make_array(res, aml::Shape<0>());
  auto out = aml::Array<int, 0>(aml::GPU, out_h.size());
  aml::copy(in_h, in);
  aml::copy(out_h, out);
  aml::reduce(in, out, {{0, 1, 3, 2}}, aml::Identity(), aml::Max());
  aml::copy(out, out_h);
  EXPECT_EQ(out_h.data()[0], 7);
}

TEST(OperationsTest, GpuReduceMaxAbs) {
  std::vector<int> data = {4, -9, -3, 7};
  std::vector<int> res = {0};
  auto in_h = aml::make_array(data, aml::make_shape(1, 2, 2, 1));
  auto in = aml::Array<int, 4>(aml::GPU, in_h.size());
  auto out_h = aml::make_array(res, aml::Shape<0>());
  auto out = aml::Array<int, 0>(aml::GPU, out_h.size());
  aml::copy(in_h, in);
  aml::copy(out_h, out);
  aml::reduce(in, out, {{0, 1, 3, 2}}, aml::Abs(), aml::Max());
  aml::copy(out, out_h);
  EXPECT_EQ(out_h.data()[0], 9);
}

TEST(OperationsTest, GpuReduceMaxAbsSlice) {
  std::vector<int> data = {4, -9, -3, 7};
  std::vector<int> res = {0};
  auto in_h = aml::make_array(data, aml::make_shape(1, 2, 2, 1));
  auto in = aml::Array<int, 4>(aml::GPU, in_h.size());
  auto ins = aml::slice(in, aml::make_shape(0, 0, 0, 0),
      aml::make_shape(1, 1, 2, 1));
  auto out_h = aml::make_array(res, aml::Shape<0>());
  auto out = aml::Array<int, 0>(aml::GPU, out_h.size());
  aml::copy(in_h, in);
  aml::copy(out_h, out);
  aml::reduce(ins, out, {{3, 1, 0, 2}}, aml::Abs(), aml::Max());
  aml::copy(out, out_h);
  EXPECT_EQ(out_h.data()[0], 4);
}

TEST(OperationsTest, GpuReducePartial1Max) {
  std::vector<int> data = {4, -9, -3, 7};
  std::vector<int> res = {0, 0};
  auto in_h = aml::make_array(data, aml::make_shape(1, 2, 2, 1));
  auto in = aml::Array<int, 4>(aml::GPU, in_h.size());
  auto out_h = aml::make_array(res, aml::make_shape(2));
  auto out = aml::Array<int, 1>(aml::GPU, out_h.size());
  aml::copy(in_h, in);
  aml::copy(out_h, out);
  aml::reduce(in, out, {{0, 3, 2}}, aml::Identity(), aml::Max());
  aml::copy(out, out_h);
  EXPECT_EQ(out.data()[0], 4);
  EXPECT_EQ(out.data()[1], 7);
}

TEST(OperationsTest, GpuReducePartial2Max) {
  std::vector<int> data = {4, 7, -3, -9};
  std::vector<int> res = {-10, -10};
  auto in = aml::make_array(data, aml::make_shape(1, 2, 2, 1));
  auto in_h = aml::Array<int, 4>(aml::GPU, in.size());
  auto out_h = aml::make_array(res, aml::make_shape(2));
  auto out = aml::Array<int, 1>(aml::GPU, out_h.size());
  aml::copy(in_h, in);
  aml::copy(out_h, out);
  aml::reduce(in, out, {{0, 3, 1}}, aml::Identity(), aml::Max());
  aml::copy(out, out_h);
  EXPECT_EQ(out_h.data()[0], 7);
  EXPECT_EQ(out_h.data()[1], -3);
}

