#ifndef GAUSS_QUAD_ORTHOGONAL_POLYNOMIAL_GUARD_H
#define GAUSS_QUAD_ORTHOGONAL_POLYNOMIAL_GUARD_H

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <concepts>
#include <numbers>
#include <utility>
#include <vector>

namespace GaussQuad {

template <std::floating_point Float>
class JacobiPolynomial {
 public:
  // Constructor.
  JacobiPolynomial(Float alpha, Float beta) : alpha{alpha}, beta{beta} {}

  // Basic data functions.
  constexpr Float X1() const { return -1; }
  constexpr Float X2() const { return 1; }
  Float Mu() const {
    using std::exp;
    using std::lgamma;
    using std::log;
    constexpr Float ln2 = std::numbers::ln2_v<Float>;
    return exp(lgamma(alpha + 1) + lgamma(beta + 1) + ln2 * (alpha + beta + 1) -
               lgamma(alpha + beta + 1) - log(alpha + beta + 1));
  }

  // Recursion coefficient functions.
  Float A1(int n) const {
    return 2 * (n + 1) * (n + alpha + beta + 1) * (2 * n + alpha + beta);
  }
  Float A2(int n) const {
    return (2 * n + alpha + beta + 1) * (alpha * alpha - beta * beta);
  }
  Float A3(int n) const {
    Float tmp = 2 * n + alpha + beta;
    return tmp * (tmp + 1) * (tmp + 2);
  }
  Float A4(int n) const {
    return 2 * (n + alpha) * (n + beta) * (2 * n + alpha + beta + 2);
  }

  // Matrix coefficients for Goulb and Welsch method.
  Float D(int n) const {
    Float num = beta * beta - alpha * alpha;
    Float den = (2 * n + alpha + beta - 2) * (2 * n + alpha + beta);
    return num != 0 ? num / den : 0.0;
  }
  Float E(int n) const {
    Float num = 4 * n * (n + alpha) * (n + beta) * (n + alpha + beta);
    Float den = (2 * n + alpha + beta - 1) * pow((2 * n + alpha + beta), 2) *
                (2 * n + alpha + beta + 1);
    return std::sqrt(num / den);
  }

  Float operator()(int n, Float x) { return Value(n, x); }

  // Evaluation of derivatices via recursion.
  Float Derivative(int n, Float x) {
    switch (n) {
      case 0:
        return 0;
      default:
        Float tmp = 2 * n + alpha + beta;
        Float b1 = tmp * (1 - x * x);
        Float b2 = n * (alpha - beta - tmp * x);
        Float b3 = 2 * (n + alpha) * (n + beta);
        return (b2 * Value(n, x) + b3 * Value(n - 1, x)) / b1;
    }
  }

  // Return zeros of the polynomial.
  auto Zeros(int n) {
    assert(n >= 0);
    std::vector<Float> zeros;
    zeros.reserve(n);
    const int maxIter = 30;
    const Float epsilon = std::numeric_limits<Float>::epsilon();
    const Float dth = std::numbers::pi_v<Float> / (2 * n);
    for (int k = 0; k < n; k++) {
      Float r = -std::cos((2 * k + 1) * dth);
      if (k > 0) r = 0.5 * (r + zeros[k - 1]);
      for (int j = 1; j < maxIter; j++) {
        Float fun = Value(n, r);
        Float der = Derivative(n, r);
        Float sum = 0;
        for (int i = 0; i < k; i++)
          sum += static_cast<Float>(1) / (r - zeros[i]);
        Float delr = -fun / (der - sum * fun);
        r += delr;
        if (std::abs(delr) < epsilon) break;
      }
      zeros.push_back(r);
    }
    return zeros;
  }

  // Returns points and weights.
  auto ZerosAndWeights(int n) {
    assert(n >= 0);
    auto zeros = Zeros(n);
    std::vector<Float> weights;
    weights.reserve(n);
    std::transform(zeros.begin(), zeros.end(), std::back_inserter(weights),
                   [&](auto z) { return Derivative(n, z); });
    Float fac =
        std::exp(std::numbers::ln2_v<Float> * (alpha + beta + 1) +
                 std::lgamma(alpha + n + 1) + std::lgamma(beta + n + 1) -
                 std::lgamma(static_cast<Float>(n + 1)) -
                 std::lgamma(alpha + beta + n + 1));
    std::transform(
        zeros.begin(), zeros.end(), weights.begin(), weights.begin(),
        [fac](auto z, auto w) { return fac / (w * w * (1 - z * z)); });
    return std::pair(zeros, weights);
  }

 private:
  Float alpha, beta;

  // Evaluation function by upwards recursion.
  Float Value(int n, Float x) {
    assert(n >= 0);
    Float pm1 = static_cast<Float>(1);
    if (n == 0) return pm1;
    Float p = 0.5 * (alpha - beta + (alpha + beta + 2) * x);
    if (n == 1) return p;
    for (int m = 1; m < n; m++) {
      pm1 = ((A2(m) + A3(m) * x) * p - A4(m) * pm1) / A1(m);
      std::swap(p, pm1);
    }
    return p;
  }
};

template <std::floating_point Float>
class LegendrePolynomial {
 public:
  LegendrePolynomial() : p{JacobiPolynomial<Float>(0, 0)} {}

  // Basic data functions.
  constexpr Float X1() const { return -1; }
  constexpr Float X2() const { return 1; }
  Float Mu() const { return std::numbers::pi_v<Float>; }

  // Matrix coefficients for Goulb and Welsch method.
  Float D(int n) const { return 0; }
  Float E(int n) const { return p.D(n); }

  // Evaluation functions.
  Float operator()(int n, Float x) { return p(n, x); }
  Float Derivative(int n, Float x) { return p.Derivative(n, x); }

  // Zeros and weights
  auto Zeros(int n) { return p.Zeros(n); }
  auto ZerosAndWeights(int n) { return p.ZerosAndWeights(n); }

 private:
  JacobiPolynomial<Float> p;
};

template <std::floating_point Float>
class OrthogonalPolynomial {
 public:
  // Basic data functions.
  virtual Float x1() const = 0;
  virtual Float x2() const = 0;
  virtual Float mu() const = 0;

  // Returns the weight function.
  virtual Float w(Float) const = 0;

  // Recursion functions following Abramowitz & Stegun.
  virtual Float a1(int) const = 0;
  virtual Float a2(int) const = 0;
  virtual Float a3(int) const = 0;
  virtual Float a4(int) const = 0;

  // Evaluation function based on upwards recursion of the
  // three-term formulae. This need NOT be stable is all cases.
  Float operator()(int n, Float x) const {
    if (n == 0) return 1.;
    Float pm1 = 0.;
    Float p = 1.;
    for (int m = 0; m < n; m++) {
      Float pp1 = (a2(n) + a3(n) * x) * p - a4(n) * pm1;
      pp1 /= a1(n);
      pm1 = p;
      p = pp1;
    }
    return p;
  }

  // Quadrature matrix functions following Goulb & Welsch.
  // Default implementation given, but may need modification
  // to avoid divide by zero problems.
  virtual Float d(int n) const { return -a2(n - 1) / a3(n - 1); }
  virtual Float e(int n) const {
    Float num = a4(n) * a1(n - 1);
    Float den = a3(n - 1) * a3(n);
    return std::sqrt(num / den);
  }

  // set the default destructor to default
  virtual ~OrthogonalPolynomial() = default;
};

template <std::floating_point Float>
class Legendre final : public OrthogonalPolynomial<Float> {
 public:
  // Use default constructor.
  Legendre() = default;

  // Basic data functions.
  constexpr Float x1() const override { return Float{-1.0}; }
  constexpr Float x2() const override { return Float{+1.0}; }
  constexpr Float mu() const override { return Float{+2.0}; }

  // Returns the weight function.
  constexpr Float w(Float) const override { return 1.0; };

  // Recursion functions following Abramowitz & Stegun.
  Float a1(int n) const override { return n + 1; }
  constexpr Float a2(int n) const override { return 0; }
  Float a3(int n) const override { return 2 * n + 1; }
  Float a4(int n) const override { return n; }
};

template <std::floating_point Float>
class Jacobi final : public OrthogonalPolynomial<Float> {
 public:
  // Basic data functions.
  constexpr Float x1() const override { return Float{-1.0}; }
  constexpr Float x2() const override { return Float{+1.0}; }
  Float mu() const override {
    using std::exp;
    using std::lgamma;
    using std::log;
    constexpr Float ln2 = std::numbers::ln2_v<Float>;
    return exp(lgamma(alpha + 1) + lgamma(beta + 1) + ln2 * (alpha + beta + 1) -
               lgamma(alpha + beta + 1) - log(alpha + beta + 1));
  }

  // Delete the default constructor.
  Jacobi() = delete;

  // Constructor taking the two parameter.
  Jacobi(Float alpha, Float beta) : alpha{alpha}, beta{beta} {}

  // Returns the weight function.
  Float w(Float x) const override {
    return pow((1.0 - x), alpha) * pow((1.0 + x), beta);
  }

  // Recurence functions following the notations of Abramowitz & Stegun.
  Float a1(int n) const override {
    Float x(n);
    return 2 * (x + 1) * (x + alpha + beta + 1) * (2 * x + alpha + beta);
  }
  Float a2(int n) const override {
    Float x(n);
    return (2 * x + alpha + beta + 1) * (alpha * alpha - beta * beta);
  }
  Float a3(int n) const override {
    Float x(n);
    Float tmp = 2 * x + alpha + beta;
    return tmp * (tmp + 1) * (tmp + 2);
  }
  Float a4(int n) const override {
    Float x(n);
    return 2 * (x + alpha) * (x + beta) * (2 * x + alpha + beta + 2);
  }

  // Quadrature matrix functions following Goulb & Welsch.
  Float d(int n) const override {
    Float x(n);
    Float num = beta * beta - alpha * alpha;
    Float den = (2 * x + alpha + beta - 2) * (2 * x + alpha + beta);
    return num != 0 ? num / den : 0.0;
  }
  Float e(int n) const override {
    using std::sqrt;
    Float x(n);
    Float num = 4 * x * (x + alpha) * (x + beta) * (x + alpha + beta);
    Float den = (2 * x + alpha + beta - 1) * pow((2 * x + alpha + beta), 2) *
                (2 * x + alpha + beta + 1);
    return sqrt(num / den);
  }

 private:
  Float alpha;
  Float beta;
};

}  // namespace GaussQuad

#endif  // GAUSS_QUAD_ORTHOGONAL_POLYNOMIAL_GUARD_H
