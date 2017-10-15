#include <gtest/gtest.h>

#include <aml/aml.h>

class OperationsTest : public ::testing::Test {
public:
  OperationsTest() {
    h.init();
  }

  ~OperationsTest() {
    h.destroy();
  }

protected:
  aml::Handle h;
};


TEST_F(OperationsTest, Set) {
  auto a = aml::Array<int, 4>(aml::CPU, aml::make_shape(1, 2, 1, 2));
  aml::set(h, a, 9);
  EXPECT_EQ(a.data()[0], 9);
  EXPECT_EQ(a.data()[1], 9);
  EXPECT_EQ(a.data()[2], 9);
  EXPECT_EQ(a.data()[3], 9);
}

TEST_F(OperationsTest, Copy) {
  std::vector<int> data = {4, 2, 3, -9};
  auto a = aml::make_array(data, aml::make_shape(2, 2));
  auto b = aml::Array<int, 2>(aml::CPU, a.size());
  aml::copy(h, a, b);
  EXPECT_EQ(b.data()[0], 4);
  EXPECT_EQ(b.data()[1], 2);
  EXPECT_EQ(b.data()[2], 3);
  EXPECT_EQ(b.data()[3], -9);
}

TEST_F(OperationsTest, CopyTypeCast) {
  std::vector<double> data = {4.0, 2.0, 3.0, -9.0};
  auto a = aml::make_array(data, aml::make_shape(2, 2));
  auto b = aml::Array<int, 2>(aml::CPU, a.size());
  aml::copy(h, a, b);
  EXPECT_EQ(b.data()[0], 4);
  EXPECT_EQ(b.data()[1], 2);
  EXPECT_EQ(b.data()[2], 3);
  EXPECT_EQ(b.data()[3], -9);
}

TEST_F(OperationsTest, CopyNonContiguous) {
  std::vector<int> data = {4, 2, 3, -9};
  auto a = aml::make_array(data, aml::make_shape(2, 2));
  auto b = aml::slice(a, aml::make_shape(1, 0), aml::make_shape(2, 2));
  auto c = aml::Array<int, 2>(aml::CPU, b.size());
  aml::copy(h, b, c);
  EXPECT_EQ(c.data()[0], 2);
  EXPECT_EQ(c.data()[1], -9);
}

TEST_F(OperationsTest, UnaryOpAbs) {
  std::vector<int> data = {-4, 2, -3, -9};
  auto a = aml::make_array(data, aml::make_shape(1, 2, 1, 2));
  aml::unary_op(h, a, a, aml::Abs());
  EXPECT_EQ(a.data()[0], 4);
  EXPECT_EQ(a.data()[1], 2);
  EXPECT_EQ(a.data()[2], 3);
  EXPECT_EQ(a.data()[3], 9);
}

TEST_F(OperationsTest, BinaryOpMax) {
  std::vector<int> data1 = {4, 5, 3, 7};
  std::vector<int> data2 = {8, 4, 1, 9};
  auto in1 = aml::make_array(data1, aml::make_shape(1, 2, 2, 1));
  auto in2 = aml::make_array(data2, aml::make_shape(1, 2, 2, 1));
  auto out = aml::Array<int, 4>(aml::CPU, aml::make_shape(1, 2, 2, 1));
  aml::binary_op(h, in1, in2, out, aml::Max());
  EXPECT_EQ(out.data()[0], 8);
  EXPECT_EQ(out.data()[1], 5);
  EXPECT_EQ(out.data()[2], 3);
  EXPECT_EQ(out.data()[3], 9);
}

TEST_F(OperationsTest, BinaryOpMaxSlice) {
  std::vector<int> data1 = {4, 5, 3, 7};
  std::vector<int> data2 = {8, 4, 1, 9};
  auto in1 = aml::make_array(data1, aml::make_shape(1, 2, 2, 1));
  auto in2 = aml::make_array(data2, aml::make_shape(1, 2, 2, 1));
  auto in1s = aml::slice(in1, aml::make_shape(0, 1, 0, 0),
      aml::make_shape(1, 2, 2, 1));
  auto in2s = aml::slice(in2, aml::make_shape(0, 1, 0, 0),
      aml::make_shape(1, 2, 2, 1));
  auto out = aml::Array<int, 4>(aml::CPU, aml::make_shape(1, 1, 2, 1));
  aml::binary_op(h, in1s, in2s, out, aml::Min());
  EXPECT_EQ(out.data()[0], 4);
  EXPECT_EQ(out.data()[1], 7);
}

TEST_F(OperationsTest, ReduceMax) {
  std::vector<int> data = {4, -9, -3, 7};
  std::vector<int> res = {0};
  auto in = aml::make_array(data, aml::make_shape(1, 2, 2, 1));
  auto out = aml::make_array(res, aml::Shape<0>());
  aml::reduce(h, in, out, {{0, 1, 3, 2}}, aml::Identity(), aml::Max());
  EXPECT_EQ(out.data()[0], 7);
}

TEST_F(OperationsTest, ReduceMaxAbs) {
  std::vector<int> data = {4, -9, -3, 7};
  std::vector<int> res = {0};
  auto in = aml::make_array(data, aml::make_shape(1, 2, 2, 1));
  auto out = aml::make_array(res, aml::Shape<0>());
  aml::reduce(h, in, out, {{0, 1, 3, 2}}, aml::Abs(), aml::Max());
  EXPECT_EQ(out.data()[0], 9);
}

TEST_F(OperationsTest, ReduceMaxAbsSlice) {
  std::vector<int> data = {4, -9, -3, 7};
  std::vector<int> res = {0};
  auto in = aml::make_array(data, aml::make_shape(1, 2, 2, 1));
  auto ins = aml::slice(in, aml::make_shape(0, 0, 0, 0),
      aml::make_shape(1, 1, 2, 1));
  auto out = aml::make_array(res, aml::Shape<0>());
  aml::reduce(h, ins, out, {{3, 1, 0, 2}}, aml::Abs(), aml::Max());
  EXPECT_EQ(out.data()[0], 4);
}

TEST_F(OperationsTest, ReducePartial1Max) {
  std::vector<int> data = {4, -9, -3, 7};
  std::vector<int> res = {0, 0};
  auto in = aml::make_array(data, aml::make_shape(1, 2, 2, 1));
  auto out = aml::make_array(res, aml::make_shape(2));
  aml::reduce(h, in, out, {{0, 3, 2}}, aml::Identity(), aml::Max());
  EXPECT_EQ(out.data()[0], 4);
  EXPECT_EQ(out.data()[1], 7);
}

TEST_F(OperationsTest, ReducePartial2Max) {
  std::vector<int> data = {4, 7, -3, -9};
  std::vector<int> res = {-10, -10};
  auto in = aml::make_array(data, aml::make_shape(1, 2, 2, 1));
  auto out = aml::make_array(res, aml::make_shape(2));
  aml::reduce(h, in, out, {{0, 3, 1}}, aml::Identity(), aml::Max());
  EXPECT_EQ(out.data()[0], 7);
  EXPECT_EQ(out.data()[1], -3);
}

TEST_F(OperationsTest, ReducePartialSliceSumLarge) {
  size_t M = 50;
  size_t N = 2 * M;
  std::vector<int> data(N * M);
  std::vector<int> res0(M, 0);
  std::vector<int> res1(N, 0);
  std::vector<int> res0s(M / 2, 0);
  std::vector<int> res1s(N / 2, 0);
  for (size_t j = 0; j < N; ++j) {
    for (size_t i = 0; i < M; ++i) {
      data[i + j * M] = (i % 2 == 0 ? 1 : -1) * (j + 1);
    }
  }
  auto in = aml::make_array(data, aml::make_shape(1, M, 1, N));
  auto out0 = aml::make_array(res0, aml::make_shape(M));
  auto out1 = aml::make_array(res1, aml::make_shape(N));

  auto ins = aml::slice(in, aml::make_shape(0, 0, 0, 0),
      aml::make_shape(1, M / 2, 1, N / 2));
  auto out0s = aml::make_array(res0s, aml::make_shape(M / 2));
  auto out1s = aml::make_array(res1s, aml::make_shape(N / 2));

  aml::reduce(h, in, out0, {{0, 3, 2}}, aml::Abs(), aml::Plus());
  aml::reduce(h, in, out1, {{0, 1, 2}}, aml::Abs(), aml::Plus());
  aml::reduce(h, ins, out0s, {{0, 3, 2}}, aml::Abs(), aml::Plus());
  aml::reduce(h, ins, out1s, {{0, 1, 2}}, aml::Abs(), aml::Plus());

  for (size_t i = 0; i < M; ++i) {
    ASSERT_EQ(out0.data()[i], N * (N + 1) / 2);
  }
  for (size_t j = 0; j < N; ++j) {
    ASSERT_EQ(out1.data()[j], (j + 1) * M);
  }
  for (size_t i = 0; i < M / 2; ++i) {
    ASSERT_EQ(out0s.data()[i], N  / 2 * (N / 2 + 1) / 2);
  }
  for (size_t j = 0; j < N / 2; ++j) {
    ASSERT_EQ(out1s.data()[j], (j + 1) * M / 2);
  }
}

