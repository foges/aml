#include <cmath>
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

class BlasTest : public ::testing::Test {
public:
  BlasTest() {
    h.init();
    size = 4;
    Af = aml::make_array(Afl, aml::make_shape(2, 2));
    Bf = aml::make_array(Bfl, aml::make_shape(2, 2));
    Ad = aml::make_array(Ado, aml::make_shape(2, 2));
    Bd = aml::make_array(Bdo, aml::make_shape(2, 2));
  }

  ~BlasTest() {
    h.destroy();
  }

protected:
  virtual void SetUp() {
    std::vector<float> data_f = {0.0f, 0.0f, 0.0f, 0.0f};
    std::vector<double> data_d = {0.0, 0.0, 0.0, 0.0};
    Cf = aml::make_array(data_f, aml::make_shape(2, 2));
    Cd = aml::make_array(data_d, aml::make_shape(2, 2));
  }

  aml::Handle h;

  size_t size;
  aml::ConstMatrix<float> Af;
  aml::ConstMatrix<float> Bf;
  aml::Matrix<float> Cf;
  aml::ConstMatrix<double> Ad;
  aml::ConstMatrix<double> Bd;
  aml::Matrix<double> Cd;
};

TEST_F(BlasTest, GemmFloatNN) {
  aml::gemm(h, 'n', 'n', 1.0f, Af, Bf, 0.0f, Cf);
  for (size_t i = 0; i < size; ++i) {
    EXPECT_EQ(Cf.data()[i], CNN[i]);
  }
}

TEST_F(BlasTest, GemmFloatTN) {
  aml::gemm(h, 't', 'n', 1.0f, Af, Bf, 0.0f, Cf);
  for (size_t i = 0; i < size; ++i) {
    EXPECT_EQ(Cf.data()[i], CTN[i]);
  }
}

TEST_F(BlasTest, GemmFloatNT) {
  aml::gemm(h, 'n', 't', 1.0f, Af, Bf, 0.0f, Cf);
  for (size_t i = 0; i < size; ++i) {
    EXPECT_EQ(Cf.data()[i], CNT[i]);
  }
}

TEST_F(BlasTest, GemmFloatTT) {
  aml::gemm(h, 't', 't', 1.0f, Af, Bf, 0.0f, Cf);
  for (size_t i = 0; i < size; ++i) {
    EXPECT_EQ(Cf.data()[i], CTT[i]);
  }
}

TEST_F(BlasTest, GemmDoubleNN) {
  aml::gemm(h, 'n', 'n', 1.0, Ad, Bd, 0.0, Cd);
  for (size_t i = 0; i < size; ++i) {
    EXPECT_EQ(Cd.data()[i], CNN[i]);
  }
}

TEST_F(BlasTest, GemmDoubleTN) {
  aml::gemm(h, 't', 'n', 1.0, Ad, Bd, 0.0, Cd);
  for (size_t i = 0; i < size; ++i) {
    EXPECT_EQ(Cd.data()[i], CTN[i]);
  }
}

TEST_F(BlasTest, GemmDoubleNT) {
  aml::gemm(h, 'n', 't', 1.0, Ad, Bd, 0.0, Cd);
  for (size_t i = 0; i < size; ++i) {
    EXPECT_EQ(Cd.data()[i], CNT[i]);
  }
}

TEST_F(BlasTest, GemmDoubleTT) {
  aml::gemm(h, 't', 't', 1.0, Ad, Bd, 0.0, Cd);
  for (size_t i = 0; i < size; ++i) {
    EXPECT_EQ(Cd.data()[i], CTT[i]);
  }
}

TEST_F(BlasTest, Norm2Float) {
  auto x = aml::reshape(Af, aml::make_shape(4));
  float nrm2 = aml::nrm2(h, x);
  EXPECT_EQ(nrm2, std::sqrt(39.0f));
}

TEST_F(BlasTest, Norm2Double) {
  auto x = aml::reshape(Ad, aml::make_shape(4));
  double nrm2 = aml::nrm2(h, x);
  EXPECT_EQ(nrm2, std::sqrt(39.0));
}

// TODO:
// - Larger non-square
// - alpha, beta != 1, 0
// - strided

