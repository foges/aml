#include <gtest/gtest.h>

#include <aml/aml.h>

class OperationsTestGpu : public ::testing::Test {
public:
  OperationsTestGpu() {
    h.init();
  }

  ~OperationsTestGpu() {
    h.destroy();
  }

protected:
  aml::Handle h;
};

TEST_F(OperationsTestGpu, Set) {
  auto a = aml::Array<int, 4>(aml::GPU, aml::make_shape(1, 2, 1, 2));
  aml::set(h, a, 9);
  auto a_h = aml::Array<int, 4>(aml::CPU, aml::make_shape(1, 2, 1, 2));
  aml::copy(h, a, a_h);

  EXPECT_EQ(a_h.data()[0], 9);
  EXPECT_EQ(a_h.data()[1], 9);
  EXPECT_EQ(a_h.data()[2], 9);
  EXPECT_EQ(a_h.data()[3], 9);
}

TEST_F(OperationsTestGpu, CopyCpu) {
  std::vector<int> data = {4, 2, 3, -9};
  auto a = aml::make_array(data, aml::make_shape(2, 2));
  auto b = aml::Array<int, 2>(aml::GPU, a.size());
  aml::copy(h, a, b);
  auto b_h = aml::Array<int, 2>(aml::CPU, aml::make_shape(2, 2));
  aml::copy(h, b, b_h);

  EXPECT_EQ(b_h.data()[0], 4);
  EXPECT_EQ(b_h.data()[1], 2);
  EXPECT_EQ(b_h.data()[2], 3);
  EXPECT_EQ(b_h.data()[3], -9);
}

TEST_F(OperationsTestGpu, Copy) {
  std::vector<int> data = {4, 2, 3, -9};
  auto a = aml::make_array(data, aml::make_shape(2, 2));
  auto b = aml::Array<int, 2>(aml::GPU, a.size());
  auto c = aml::Array<int, 2>(aml::GPU, a.size());
  aml::copy(h, a, b);
  aml::copy(h, b, c);
  auto c_h = aml::Array<int, 2>(aml::CPU, aml::make_shape(2, 2));
  aml::copy(h, c, c_h);

  EXPECT_EQ(c_h.data()[0], 4);
  EXPECT_EQ(c_h.data()[1], 2);
  EXPECT_EQ(c_h.data()[2], 3);
  EXPECT_EQ(c_h.data()[3], -9);
}

TEST_F(OperationsTestGpu, CopyTypeCast) {
  std::vector<double> data = {4.0, 2.0, 3.0, -9.0};
  auto a = aml::make_array(data, aml::make_shape(2, 2));
  auto b = aml::Array<double, 2>(aml::GPU, a.size());
  auto c = aml::Array<int, 2>(aml::GPU, a.size());
  aml::copy(h, a, b);
  aml::copy(h, b, c);
  auto c_h = aml::Array<int, 2>(aml::CPU, aml::make_shape(2, 2));
  aml::copy(h, c, c_h);

  EXPECT_EQ(c_h.data()[0], 4);
  EXPECT_EQ(c_h.data()[1], 2);
  EXPECT_EQ(c_h.data()[2], 3);
  EXPECT_EQ(c_h.data()[3], -9);
}

TEST_F(OperationsTestGpu, CopyNonContiguous) {
  std::vector<int> data = {4, 2, 3, -9};
  auto a_h = aml::make_array(data, aml::make_shape(2, 2));
  auto a = aml::Array<int, 2>(aml::GPU, a_h.size());
  aml::copy(h, a_h, a);
  auto b = aml::slice(a, aml::make_shape(1, 0), aml::make_shape(2, 2));
  auto c = aml::Array<int, 2>(aml::GPU, b.size());
  aml::copy(h, b, c);
  auto c_h = aml::Array<int, 2>(aml::CPU, c.size());
  aml::copy(h, c, c_h);

  EXPECT_EQ(c_h.data()[0], 2);
  EXPECT_EQ(c_h.data()[1], -9);
}

TEST_F(OperationsTestGpu, ReduceMax) {
  std::vector<int> data = {4, -9, -3, 7};
  std::vector<int> res = {0};
  auto in_h = aml::make_array(data, aml::make_shape(1, 2, 2, 1));
  auto in = aml::Array<int, 4>(aml::GPU, in_h.size());
  auto out_h = aml::make_array(res, aml::Shape<0>());
  auto out = aml::Array<int, 0>(aml::GPU, out_h.size());
  aml::copy(h, in_h, in);
  aml::copy(h, out_h, out);
  aml::reduce(h, in, out, {{0, 1, 3, 2}}, aml::Identity(), aml::Max());
  aml::copy(h, out, out_h);
  EXPECT_EQ(out_h.data()[0], 7);
}

TEST_F(OperationsTestGpu, ReduceMaxAbs) {
  std::vector<int> data = {4, -9, -3, 7};
  std::vector<int> res = {0};
  auto in_h = aml::make_array(data, aml::make_shape(1, 2, 2, 1));
  auto in = aml::Array<int, 4>(aml::GPU, in_h.size());
  auto out_h = aml::make_array(res, aml::Shape<0>());
  auto out = aml::Array<int, 0>(aml::GPU, out_h.size());
  aml::copy(h, in_h, in);
  aml::copy(h, out_h, out);
  aml::reduce(h, in, out, {{0, 1, 3, 2}}, aml::Abs(), aml::Max());
  aml::copy(h, out, out_h);
  EXPECT_EQ(out_h.data()[0], 9);
}

TEST_F(OperationsTestGpu, ReduceMaxAbsSlice) {
  std::vector<int> data = {4, -9, -3, 7};
  std::vector<int> res = {0};
  auto in_h = aml::make_array(data, aml::make_shape(1, 2, 2, 1));
  auto in = aml::Array<int, 4>(aml::GPU, in_h.size());
  auto ins = aml::slice(in, aml::make_shape(0, 0, 0, 0),
      aml::make_shape(1, 1, 2, 1));
  auto out_h = aml::make_array(res, aml::Shape<0>());
  auto out = aml::Array<int, 0>(aml::GPU, out_h.size());
  aml::copy(h, in_h, in);
  aml::copy(h, out_h, out);
  aml::reduce(h, ins, out, {{3, 1, 0, 2}}, aml::Abs(), aml::Max());
  aml::copy(h, out, out_h);
  EXPECT_EQ(out_h.data()[0], 4);
}

TEST_F(OperationsTestGpu, ReducePartial1Max) {
  std::vector<int> data = {4, -9, -3, 7};
  std::vector<int> res = {0, 0};
  auto in_h = aml::make_array(data, aml::make_shape(1, 2, 2, 1));
  auto in = aml::Array<int, 4>(aml::GPU, in_h.size());
  auto out_h = aml::make_array(res, aml::make_shape(2));
  auto out = aml::Array<int, 1>(aml::GPU, out_h.size());
  aml::copy(h, in_h, in);
  aml::copy(h, out_h, out);
  aml::reduce(h, in, out, {{0, 3, 2}}, aml::Identity(), aml::Max());
  aml::copy(h, out, out_h);
  EXPECT_EQ(out_h.data()[0], 4);
  EXPECT_EQ(out_h.data()[1], 7);
}

TEST_F(OperationsTestGpu, ReducePartial2Max) {
  std::vector<int> data = {4, 7, -3, -9};
  std::vector<int> res = {-10, -10};
  auto in_h = aml::make_array(data, aml::make_shape(1, 2, 2, 1));
  auto in = aml::Array<int, 4>(aml::GPU, in_h.size());
  auto out_h = aml::make_array(res, aml::make_shape(2));
  auto out = aml::Array<int, 1>(aml::GPU, out_h.size());
  aml::copy(h, in_h, in);
  aml::copy(h, out_h, out);
  aml::reduce(h, in, out, {{0, 3, 1}}, aml::Identity(), aml::Max());
  aml::copy(h, out, out_h);
  EXPECT_EQ(out_h.data()[0], 7);
  EXPECT_EQ(out_h.data()[1], -3);
}

TEST_F(OperationsTestGpu, ReducePartialSliceSumLarge) {
  size_t M = 52;
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
  auto in_h = aml::make_array(data, aml::make_shape(1, M, 1, N));
  auto in = aml::Array<int, 4>(aml::GPU, in_h.size());
  auto out0_h = aml::make_array(res0, aml::make_shape(M));
  auto out0 = aml::Array<int, 1>(aml::GPU, out0_h.size());
  auto out1_h = aml::make_array(res1, aml::make_shape(N));
  auto out1 = aml::Array<int, 1>(aml::GPU, out1_h.size());

  auto ins = aml::slice(in, aml::make_shape(0, 0, 0, 0),
      aml::make_shape(1, M / 2, 1, N / 2));
  auto out0s_h = aml::make_array(res0s, aml::make_shape(M / 2));
  auto out0s = aml::Array<int, 1>(aml::GPU, out0s_h.size());
  auto out1s_h = aml::make_array(res1s, aml::make_shape(N / 2));
  auto out1s = aml::Array<int, 1>(aml::GPU, out1s_h.size());

  aml::copy(h, in_h, in);
  aml::copy(h, out0_h, out0);
  aml::copy(h, out1_h, out1);
  aml::copy(h, out0s_h, out0s);
  aml::copy(h, out1s_h, out1s);

  aml::reduce(h, in, out0, {{0, 3, 2}}, aml::Abs(), aml::Plus());
  aml::reduce(h, in, out1, {{0, 1, 2}}, aml::Abs(), aml::Plus());
  aml::reduce(h, ins, out0s, {{0, 3, 2}}, aml::Abs(), aml::Plus());
  aml::reduce(h, ins, out1s, {{0, 1, 2}}, aml::Abs(), aml::Plus());

  aml::copy(h, out0, out0_h);
  aml::copy(h, out1, out1_h);
  aml::copy(h, out0s, out0s_h);
  aml::copy(h, out1s, out1s_h);

  for (size_t i = 0; i < M; ++i) {
    ASSERT_EQ(out0_h.data()[i], N * (N + 1) / 2);
  }
  for (size_t j = 0; j < N; ++j) {
    ASSERT_EQ(out1_h.data()[j], (j + 1) * M);
  }
  for (size_t i = 0; i < M / 2; ++i) {
    ASSERT_EQ(out0s_h.data()[i], N / 2 * (N / 2 + 1) / 2);
  }
  for (size_t j = 0; j < N / 2; ++j) {
    ASSERT_EQ(out1s_h.data()[j], (j + 1) * M / 2);
  }
}

