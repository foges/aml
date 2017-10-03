#include <vector>

#include <gtest/gtest.h>

#include <aml/aml.h>

std::vector<float> Af   = {-1.0f,  2.0f,  3.0f,  -5.0f};
std::vector<float> Bf   = { 1.0f, -1.0f, -2.0f,   2.0f};
std::vector<float> CNNf = {-4.0f,  7.0f,  8.0f, -14.0f};
std::vector<float> CTNf = {-3.0f,  8.0f,  6.0f, -16.0f};
std::vector<float> CNTf = {-7.0f, 12.0f,  7.0f, -12.0f};
std::vector<float> CTTf = {-5.0f, 13.0f,  5.0f, -13.0f};

class GemmTestFloat : public ::testing::Test {
public:
  GemmTestFloat() {
    A = aml::make_array(Af, aml::make_shape(2, 2));
    B = aml::make_array(Bf, aml::make_shape(2, 2));
  }

protected:
  virtual void SetUp() {
    std::vector<float> data = {0.0f, 0.0f, 0.0f, 0.0f};
    C = aml::make_array(data, aml::make_shape(2, 2));
  }

  aml::ConstMatrix<float> A;
  aml::ConstMatrix<float> B;
  aml::Matrix<float> C;
};

TEST_F(GemmTestFloat, NN) {
  aml::gemm(aml::NO_TRANS, aml::NO_TRANS, 1.0f, A, B, 0.0f, C);
  for (size_t i = 0; i < Af.size(); ++i) {
    EXPECT_EQ(C.data()[i], CNNf[i]);
  }
}

TEST_F(GemmTestFloat, TN) {
  aml::gemm(aml::TRANS, aml::NO_TRANS, 1.0f, A, B, 0.0f, C);
  for (size_t i = 0; i < Af.size(); ++i) {
    EXPECT_EQ(C.data()[i], CTNf[i]);
  }
}

TEST_F(GemmTestFloat, NT) {
  aml::gemm(aml::NO_TRANS, aml::TRANS, 1.0f, A, B, 0.0f, C);
  for (size_t i = 0; i < Af.size(); ++i) {
    EXPECT_EQ(C.data()[i], CNTf[i]);
  }
}

TEST_F(GemmTestFloat, TT) {
  aml::gemm(aml::TRANS, aml::TRANS, 1.0f, A, B, 0.0f, C);
  for (size_t i = 0; i < Af.size(); ++i) {
    EXPECT_EQ(C.data()[i], CTTf[i]);
  }
}

std::vector<double> Ad   = {-1.0,  2.0,  3.0,  -5.0};
std::vector<double> Bd   = { 1.0, -1.0, -2.0,   2.0};
std::vector<double> CNNd = {-4.0,  7.0,  8.0, -14.0};
std::vector<double> CTNd = {-3.0,  8.0,  6.0, -16.0};
std::vector<double> CNTd = {-7.0, 12.0,  7.0, -12.0};
std::vector<double> CTTd = {-5.0, 13.0,  5.0, -13.0};

class GemmTestDouble : public ::testing::Test {
public:
  GemmTestDouble() {
    A = aml::make_array(Ad, aml::make_shape(2, 2));
    B = aml::make_array(Bd, aml::make_shape(2, 2));
  }

protected:
  virtual void SetUp() {
    std::vector<double> data = {0.0, 0.0, 0.0, 0.0};
    C = aml::make_array(data, aml::make_shape(2, 2));
  }

  aml::ConstMatrix<double> A;
  aml::ConstMatrix<double> B;
  aml::Matrix<double> C;
};

TEST_F(GemmTestDouble, NN) {
  aml::gemm(aml::NO_TRANS, aml::NO_TRANS, 1.0, A, B, 0.0, C);
  for (size_t i = 0; i < Af.size(); ++i) {
    EXPECT_EQ(C.data()[i], CNNd[i]);
  }
}

TEST_F(GemmTestDouble, TN) {
  aml::gemm(aml::TRANS, aml::NO_TRANS, 1.0, A, B, 0.0, C);
  for (size_t i = 0; i < Af.size(); ++i) {
    EXPECT_EQ(C.data()[i], CTNd[i]);
  }
}

TEST_F(GemmTestDouble, NT) {
  aml::gemm(aml::NO_TRANS, aml::TRANS, 1.0, A, B, 0.0, C);
  for (size_t i = 0; i < Af.size(); ++i) {
    EXPECT_EQ(C.data()[i], CNTd[i]);
  }
}

TEST_F(GemmTestDouble, TT) {
  aml::gemm(aml::TRANS, aml::TRANS, 1.0, A, B, 0.0, C);
  for (size_t i = 0; i < Af.size(); ++i) {
    EXPECT_EQ(C.data()[i], CTTd[i]);
  }
}

// TODO:
// - Larger non-square tests
// - GPU tests
// - alpha, beta != 1, 0
// - stridede

TEST(OperationsTest, UnaryOpAbs) {
  std::vector<int> data = {-4, 2, -3, -9};
  auto a = aml::make_array(data, aml::make_shape(1, 2, 1, 2));
  aml::unary_op(aml::make_const(a), a, aml::Abs());
  EXPECT_EQ(a.data()[0], 4);
  EXPECT_EQ(a.data()[1], 2);
  EXPECT_EQ(a.data()[2], 3);
  EXPECT_EQ(a.data()[3], 9);
}

TEST(OperationsTest, BinaryOpMax) {
  std::vector<int> data1 = {4, 5, 3, 9};
  std::vector<int> data2 = {8, 4, 1, 7};
  auto in1 = aml::make_array(data1, aml::make_shape(1, 2, 2, 1));
  auto in2 = aml::make_array(data2, aml::make_shape(1, 2, 2, 1));
  auto out = aml::Array<int, 4>(aml::CPU, aml::make_shape(1, 2, 2, 1));
  aml::binary_op(aml::make_const(in1), aml::make_const(in2), out, aml::Max());
  EXPECT_EQ(out.data()[0], 8);
  EXPECT_EQ(out.data()[1], 5);
  EXPECT_EQ(out.data()[2], 3);
  EXPECT_EQ(out.data()[3], 9);
}
