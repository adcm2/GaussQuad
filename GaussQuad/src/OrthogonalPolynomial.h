#ifndef GAUSS_QUAD_ORTHOGONAL_POLYNOMIAL_GUARD_H
#define GAUSS_QUAD_ORTHOGONAL_POLYNOMIAL_GUARD_H

#include <cmath>
#include <concepts>
#include <numbers>

namespace GaussQuad {

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
