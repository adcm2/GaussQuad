#include <gtest/gtest.h>

#include <GaussQuad/All>
#include <complex>
#include <concepts>
#include <limits>

// set some data.
constexpr int n = 10;

template <std::floating_point Float>
constexpr Float exact = static_cast<Float>(2.0) / static_cast<Float>(3.0);

template <std::floating_point Float>
constexpr auto eps = 100 * std::numeric_limits<Float>::epsilon();

template <std::floating_point Float>
auto RealFunction = [](Float x) -> Float { return x * x; };

template <std::floating_point Float>
auto ComplexFunction = [](Float x) -> std::complex<Float> {
  return std::complex<Float>{x * x, x * x};
};

template <std::floating_point Float>
int TestGauss(int n) {
  using Complex = std::complex<Float>;
  auto q = GaussQuad::GaussLegendreQuadrature1D<Float>(n);
  Float error1 = std::abs(q.Integrate(RealFunction<Float>) - exact<Float>);
  Float error2 = std::abs(q.Integrate(ComplexFunction<Float>) -
                          Complex{exact<Float>, exact<Float>});

  return (error1 < eps<Float> and error2 < eps<Float>) ? 0 : 1;
}

template <std::floating_point Float>
int TestRadau(int n) {
  using Complex = std::complex<Float>;
  auto q = GaussQuad::GaussRadauLegendreQuadrature1D<Float>(n);
  Float error1 = std::abs(q.Integrate(RealFunction<Float>) - exact<Float>);
  Float error2 = std::abs(q.Integrate(ComplexFunction<Float>) -
                          Complex{exact<Float>, exact<Float>});

  return (error1 < eps<Float> and error2 < eps<Float>) ? 0 : 1;
}

template <std::floating_point Float>
int TestLobatto(int n) {
  using Complex = std::complex<Float>;
  auto q = GaussQuad::GaussLobattoLegendreQuadrature1D<Float>(n);
  Float error1 = std::abs(q.Integrate(RealFunction<Float>) - exact<Float>);
  Float error2 = std::abs(q.Integrate(ComplexFunction<Float>) -
                          Complex{exact<Float>, exact<Float>});

  return (error1 < eps<Float> and error2 < eps<Float>) ? 0 : 1;
}

// Set the tests.
TEST(Gauss, Float) {
  int i = TestGauss<float>(n);
  EXPECT_EQ(i, 0);
}

TEST(Gauss, Double) {
  int i = TestGauss<double>(n);
  EXPECT_EQ(i, 0);
}

TEST(Gauss, LongDouble) {
  int i = TestGauss<long double>(n);
  EXPECT_EQ(i, 0);
}

TEST(Radau, Float) {
  int i = TestRadau<float>(n);
  EXPECT_EQ(i, 0);
}

TEST(Radau, Double) {
  int i = TestRadau<double>(n);
  EXPECT_EQ(i, 0);
}

TEST(Radau, LongDouble) {
  int i = TestRadau<long double>(n);
  EXPECT_EQ(i, 0);
}

TEST(Lobatto, Float) {
  int i = TestLobatto<float>(n);
  EXPECT_EQ(i, 0);
}

TEST(Lobatto, Double) {
  int i = TestLobatto<double>(n);
  EXPECT_EQ(i, 0);
}

TEST(Lobatto, LongDouble) {
  int i = TestLobatto<long double>(n);
  EXPECT_EQ(i, 0);
}
