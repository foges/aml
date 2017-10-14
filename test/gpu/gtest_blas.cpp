#include <vector>

#include <gtest/gtest.h>

#include <aml/aml.h>

std::vector<float> Afl_h  = {-1.0f,  2.0f,  3.0f,  -5.0f};
std::vector<float> Bfl_h  = { 1.0f, -1.0f, -2.0f,   2.0f};

std::vector<double> Ado_h  = {-1.0,  2.0,  3.0,  -5.0};
std::vector<double> Bdo_h  = { 1.0, -1.0, -2.0,   2.0};

std::vector<double> CNN_h = {-4.0,  7.0,  8.0, -14.0};
std::vector<double> CTN_h = {-3.0,  8.0,  6.0, -16.0};
std::vector<double> CNT_h = {-7.0, 12.0,  7.0, -12.0};
std::vector<double> CTT_h = {-5.0, 13.0,  5.0, -13.0};

class BlasTest : public ::testing::Test {
public:
  BlasTest() {
    h.init();
    size = 4;
    auto Af_h = aml::make_array(Afl_h, aml::make_shape(2, 2));
    auto Bf_h = aml::make_array(Bfl_h, aml::make_shape(2, 2));
    auto Ad_h = aml::make_array(Ado_h, aml::make_shape(2, 2));
    auto Bd_h = aml::make_array(Bdo_h, aml::make_shape(2, 2));
    Af = aml::Array<float, 2>(aml::GPU, aml::make_shape(2, 2));
    Bf = aml::Array<float, 2>(aml::GPU, aml::make_shape(2, 2));
    Ad = aml::Array<double, 2>(aml::GPU, aml::make_shape(2, 2));
    Bd = aml::Array<double, 2>(aml::GPU, aml::make_shape(2, 2));
    aml::copy(Af_h, Af);
    aml::copy(Bf_h, Bf);
    aml::copy(Ad_h, Ad);
    aml::copy(Bd_h, Bd);
  }

  ~BlasTest() {
    h.destroy();
  }

protected:
  virtual void SetUp() {
    std::vector<float> data_f = {0.0f, 0.0f, 0.0f, 0.0f};
    std::vector<double> data_d = {0.0, 0.0, 0.0, 0.0};
    Cf_h = aml::make_array(data_f, aml::make_shape(2, 2));
    Cd_h = aml::make_array(data_d, aml::make_shape(2, 2));
    Cf = aml::Array<float, 2>(aml::GPU, aml::make_shape(2, 2));
    Cd = aml::Array<double, 2>(aml::GPU, aml::make_shape(2, 2));
    aml::copy(Cf_h, Cf);
    aml::copy(Cd_h, Cd);
  }

  aml::Handle h;

  size_t size;
  aml::Matrix<float> Af;
  aml::Matrix<float> Bf;
  aml::Matrix<float> Cf;
  aml::Matrix<float> Cf_h;
  aml::Matrix<double> Ad;
  aml::Matrix<double> Bd;
  aml::Matrix<double> Cd;
  aml::Matrix<double> Cd_h;
};

TEST_F(BlasTest, GpuGemmFloatNN) {
  aml::gemm(h, aml::NO_TRANS, aml::NO_TRANS, 1.0f, Af, Bf, 0.0f, Cf);
  for (size_t i = 0; i < size; ++i) {
    EXPECT_EQ(Cf.data()[i], CNN_h[i]);
  }
}

TEST_F(BlasTest, GpuGemmFloatTN) {
  aml::gemm(h, aml::TRANS, aml::NO_TRANS, 1.0f, Af, Bf, 0.0f, Cf);
  aml::copy(Cf, Cf_h);
  for (size_t i = 0; i < size; ++i) {
    EXPECT_EQ(Cf_h.data()[i], CTN_h[i]);
  }
}

TEST_F(BlasTest, GpuGemmFloatNT) {
  aml::gemm(h, aml::NO_TRANS, aml::TRANS, 1.0f, Af, Bf, 0.0f, Cf);
  aml::copy(Cf, Cf_h);
  for (size_t i = 0; i < size; ++i) {
    EXPECT_EQ(Cf_h.data()[i], CNT_h[i]);
  }
}

TEST_F(BlasTest, GpuGemmFloatTT) {
  aml::gemm(h, aml::TRANS, aml::TRANS, 1.0f, Af, Bf, 0.0f, Cf);
  aml::copy(Cf, Cf_h);
  for (size_t i = 0; i < size; ++i) {
    EXPECT_EQ(Cf_h.data()[i], CTT_h[i]);
  }
}

TEST_F(BlasTest, GpuGemmDoubleNN) {
  aml::gemm(h, aml::NO_TRANS, aml::NO_TRANS, 1.0, Ad, Bd, 0.0, Cd);
  aml::copy(Cd, Cd_h);
  for (size_t i = 0; i < size; ++i) {
    EXPECT_EQ(Cd_h.data()[i], CNN_h[i]);
  }
}

TEST_F(BlasTest, GpuGemmDoubleTN) {
  aml::gemm(h, aml::TRANS, aml::NO_TRANS, 1.0, Ad, Bd, 0.0, Cd);
  aml::copy(Cd, Cd_h);
  for (size_t i = 0; i < size; ++i) {
    EXPECT_EQ(Cd_h.data()[i], CTN_h[i]);
  }
}

TEST_F(BlasTest, GpuGemmDoubleNT) {
  aml::gemm(h, aml::NO_TRANS, aml::TRANS, 1.0, Ad, Bd, 0.0, Cd);
  aml::copy(Cd, Cd_h);
  for (size_t i = 0; i < size; ++i) {
    EXPECT_EQ(Cd_h.data()[i], CNT_h[i]);
  }
}

TEST_F(BlasTest, GpuGemmDoubleTT) {
  aml::gemm(h, aml::TRANS, aml::TRANS, 1.0, Ad, Bd, 0.0, Cd);
  aml::copy(Cd, Cd_h);
  for (size_t i = 0; i < size; ++i) {
    EXPECT_EQ(Cd_h.data()[i], CTT_h[i]);
  }
}

TEST_F(BlasTest, GpuNorm2Float) {
  auto x = aml::reshape(Af, aml::make_shape(4));
  float nrm2 = aml::nrm2(h, x);
  EXPECT_EQ(nrm2, std::sqrtf(39.0f));
}

TEST_F(BlasTest, GpuNorm2Double) {
  auto x = aml::reshape(Ad, aml::make_shape(4));
  double nrm2 = aml::nrm2(h, x);
  EXPECT_EQ(nrm2, std::sqrt(39.0));
}

