#include <vector>

#include <gtest/gtest.h>

#include <aml/aml.h>

std::vector<float> Afl  = {-1.0f,  2.0f,  3.0f,  -5.0f};
std::vector<float> Bfl  = { 1.0f, -1.0f, -2.0f,   2.0f};

std::vector<double> Ado  = {-1.0,  2.0,  3.0,  -5.0};
std::vector<double> Bdo  = { 1.0, -1.0, -2.0,   2.0};

std::vector<double> CNN = {-4.0,  7.0,  8.0, -14.0};
std::vector<double> CTN = {-3.0,  8.0,  6.0, -16.0};
std::vector<double> CNT = {-7.0, 12.0,  7.0, -12.0};
std::vector<double> CTT = {-5.0, 13.0,  5.0, -13.0};

class GemmTest : public ::testing::Test {
public:
  GemmTest() {
    size = 4;
    Af = aml::make_array(Afl, aml::make_shape(2, 2));
    Bf = aml::make_array(Bfl, aml::make_shape(2, 2));
    Ad = aml::make_array(Ado, aml::make_shape(2, 2));
    Bd = aml::make_array(Bdo, aml::make_shape(2, 2));
  }

protected:
  virtual void SetUp() {
    std::vector<float> data_f = {0.0f, 0.0f, 0.0f, 0.0f};
    std::vector<double> data_d = {0.0, 0.0, 0.0, 0.0};
    Cf = aml::make_array(data_f, aml::make_shape(2, 2));
    Cd = aml::make_array(data_d, aml::make_shape(2, 2));
  }

  size_t size;
  aml::ConstMatrix<float> Af;
  aml::ConstMatrix<float> Bf;
  aml::Matrix<float> Cf;
  aml::ConstMatrix<double> Ad;
  aml::ConstMatrix<double> Bd;
  aml::Matrix<double> Cd;
};

TEST_F(GemmTest, FloatNN) {
  aml::gemm(aml::NO_TRANS, aml::NO_TRANS, 1.0f, Af, Bf, 0.0f, Cf);
  for (size_t i = 0; i < size; ++i) {
    EXPECT_EQ(Cf.data()[i], CNN[i]);
  }
}

TEST_F(GemmTest, FloatTN) {
  aml::gemm(aml::TRANS, aml::NO_TRANS, 1.0f, Af, Bf, 0.0f, Cf);
  for (size_t i = 0; i < size; ++i) {
    EXPECT_EQ(Cf.data()[i], CTN[i]);
  }
}

TEST_F(GemmTest, FloatNT) {
  aml::gemm(aml::NO_TRANS, aml::TRANS, 1.0f, Af, Bf, 0.0f, Cf);
  for (size_t i = 0; i < size; ++i) {
    EXPECT_EQ(Cf.data()[i], CNT[i]);
  }
}

TEST_F(GemmTest, FloatTT) {
  aml::gemm(aml::TRANS, aml::TRANS, 1.0f, Af, Bf, 0.0f, Cf);
  for (size_t i = 0; i < size; ++i) {
    EXPECT_EQ(Cf.data()[i], CTT[i]);
  }
}

TEST_F(GemmTest, DoubleNN) {
  aml::gemm(aml::NO_TRANS, aml::NO_TRANS, 1.0, Ad, Bd, 0.0, Cd);
  for (size_t i = 0; i < size; ++i) {
    EXPECT_EQ(Cd.data()[i], CNN[i]);
  }
}

TEST_F(GemmTest, DoubleTN) {
  aml::gemm(aml::TRANS, aml::NO_TRANS, 1.0, Ad, Bd, 0.0, Cd);
  for (size_t i = 0; i < size; ++i) {
    EXPECT_EQ(Cd.data()[i], CTN[i]);
  }
}

TEST_F(GemmTest, DoubleNT) {
  aml::gemm(aml::NO_TRANS, aml::TRANS, 1.0, Ad, Bd, 0.0, Cd);
  for (size_t i = 0; i < size; ++i) {
    EXPECT_EQ(Cd.data()[i], CNT[i]);
  }
}

TEST_F(GemmTest, DoubleTT) {
  aml::gemm(aml::TRANS, aml::TRANS, 1.0, Ad, Bd, 0.0, Cd);
  for (size_t i = 0; i < size; ++i) {
    EXPECT_EQ(Cd.data()[i], CTT[i]);
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
  aml::binary_op(aml::make_const(in1s), aml::make_const(in2s), out, aml::Max());
  EXPECT_EQ(out.data()[0], 5);
  EXPECT_EQ(out.data()[1], 9);
}
