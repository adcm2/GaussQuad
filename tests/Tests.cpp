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

// Set the testing function.
template <std::floating_point Float, typename QuadratureType>
int TestQuad(int n) {
  using Complex = std::complex<Float>;
  GaussQuad::Quadrature<Float, QuadratureType> q(n);
  Float error1 = std::abs(q.integrate(RealFunction<Float>) - exact<Float>);
  Float error2 = std::abs(q.integrate(ComplexFunction<Float>) -
                          Complex{exact<Float>, exact<Float>});

  return (error1 < eps<Float> and error2 < eps<Float>) ? 0 : 1;
}

// Set the tests.
TEST(Gauss, Float) {
  int i = TestQuad<float, GaussQuad::None>(n);
  EXPECT_EQ(i, 0);
}

TEST(Gauss, Double) {
  int i = TestQuad<double, GaussQuad::None>(n);
  EXPECT_EQ(i, 0);
}

TEST(Gauss, LongDouble) {
  int i = TestQuad<long double, GaussQuad::None>(n);
  EXPECT_EQ(i, 0);
}

TEST(Radau, Float) {
  int i = TestQuad<float, GaussQuad::Radau>(n);
  EXPECT_EQ(i, 0);
}

TEST(Radau, Double) {
  int i = TestQuad<double, GaussQuad::Radau>(n);
  EXPECT_EQ(i, 0);
}

TEST(Radau, LongDouble) {
  int i = TestQuad<long double, GaussQuad::Radau>(n);
  EXPECT_EQ(i, 0);
}

TEST(Lobatto, Float) {
  int i = TestQuad<float, GaussQuad::Lobatto>(n);
  EXPECT_EQ(i, 0);
}

TEST(Lobatto, Double) {
  int i = TestQuad<double, GaussQuad::Lobatto>(n);
  EXPECT_EQ(i, 0);
}

TEST(Lobatto, LongDouble) {
  int i = TestQuad<long double, GaussQuad::Lobatto>(n);
  EXPECT_EQ(i, 0);
}
