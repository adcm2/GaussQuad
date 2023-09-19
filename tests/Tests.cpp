#include <gtest/gtest.h>

#include <GaussQuad/All>
#include <Interp/Polynomial>
#include <complex>
#include <concepts>
#include <limits>
#include <random>

template <std::floating_point Float>
constexpr auto eps = 2000 * std::numeric_limits<Float>::epsilon();

template <std::floating_point Float, bool Complex = false>
int TestGauss(int n) {
  auto q = GaussQuad::GaussLegendreQuadrature1D<Float>(n);
  int m = 2 * n - 1;
  if constexpr (Complex) {
    auto p = Interp::Polynomial1D<std::complex<Float>>::Random(m);
    Float error = std::abs(q.Integrate(p) - p.Integrate(-1, 1));
    return (error < eps<Float>) ? 0 : 1;
  } else {
    auto p = Interp::Polynomial1D<Float>::Random(m);
    Float error = std::abs(q.Integrate(p) - p.Integrate(-1, 1));
    return (error < eps<Float>) ? 0 : 1;
  }
}

template <std::floating_point Float, bool Complex = false>
int TestRadau(int n) {
  auto q = GaussQuad::GaussRadauLegendreQuadrature1D<Float>(n);
  int m = 2 * n - 3;
  if constexpr (Complex) {
    auto p = Interp::Polynomial1D<std::complex<Float>>::Random(m);
    Float error = std::abs(q.Integrate(p) - p.Integrate(-1, 1));
    return (error < eps<Float>) ? 0 : 1;
  } else {
    auto p = Interp::Polynomial1D<Float>::Random(m);
    Float error = std::abs(q.Integrate(p) - p.Integrate(-1, 1));
    return (error < eps<Float>) ? 0 : 1;
  }
}

template <std::floating_point Float, bool Complex = false>
int TestLobatto(int n) {
  auto q = GaussQuad::GaussLobattoLegendreQuadrature1D<Float>(n);
  int m = 2 * n - 3;
  if constexpr (Complex) {
    auto p = Interp::Polynomial1D<std::complex<Float>>::Random(m);
    Float error = std::abs(q.Integrate(p) - p.Integrate(-1, 1));
    return (error < eps<Float>) ? 0 : 1;
  } else {
    auto p = Interp::Polynomial1D<Float>::Random(m);
    Float error = std::abs(q.Integrate(p) - p.Integrate(-1, 1));
    return (error < eps<Float>) ? 0 : 1;
  }
}

int RandomDegree() {
  std::random_device rd{};
  std::mt19937_64 gen{rd()};
  std::uniform_int_distribution d{5, 200};
  return d(gen);
}

// Set the tests.

TEST(Gauss, RealDouble) {
  int n = RandomDegree();
  int i = TestGauss<double>(n);
  EXPECT_EQ(i, 0);
}

TEST(Gauss, ComplexDouble) {
  int n = RandomDegree();
  int i = TestGauss<double, true>(n);
  EXPECT_EQ(i, 0);
}

TEST(Gauss, RealLongDouble) {
  int n = RandomDegree();
  int i = TestGauss<long double>(n);
  EXPECT_EQ(i, 0);
}

TEST(Gauss, ComplexLongDouble) {
  int n = RandomDegree();
  int i = TestGauss<long double, true>(n);
  EXPECT_EQ(i, 0);
}

TEST(Radau, RealDouble) {
  int n = RandomDegree();
  int i = TestRadau<double>(n);
  EXPECT_EQ(i, 0);
}

TEST(Radau, ComplexDouble) {
  int n = RandomDegree();
  int i = TestRadau<double, true>(n);
  EXPECT_EQ(i, 0);
}

TEST(Radau, RealLongDouble) {
  int n = RandomDegree();
  int i = TestRadau<long double>(n);
  EXPECT_EQ(i, 0);
}

TEST(Radau, ComplexLongDouble) {
  int n = RandomDegree();
  int i = TestRadau<long double, true>(n);
  EXPECT_EQ(i, 0);
}

TEST(Lobatto, RealDouble) {
  int n = RandomDegree();
  int i = TestLobatto<double>(n);
  EXPECT_EQ(i, 0);
}

TEST(Lobatto, ComplexDouble) {
  int n = RandomDegree();
  int i = TestLobatto<double, true>(n);
  EXPECT_EQ(i, 0);
}

TEST(Lobatto, RealLongDouble) {
  int n = RandomDegree();
  int i = TestLobatto<long double>(n);
  EXPECT_EQ(i, 0);
}

TEST(Lobatto, ComplexLongDouble) {
  int n = RandomDegree();
  int i = TestLobatto<long double, true>(n);
  EXPECT_EQ(i, 0);
}
