#ifndef GAUSS_QUAD_QUADRATURE_GUARD_H
#define GAUSS_QUAD_QUADRATURE_GUARD_H

#include <Eigen/Cholesky>
#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include <algorithm>
#include <cassert>
#include <concepts>
#include <execution>
#include <numeric>
#include <vector>

#include "OrthogonalPolynomial.h"

namespace GaussQuad {

// Tags for quadrature types.
struct None {};
struct Radau {};
struct Lobatto {};

// Concept to check for valid quadrature types.
template <typename QuadratureType>
concept ValidQuadratureTypes =
    std::same_as<QuadratureType, None> or std::same_as<QuadratureType, Radau> or
    std::same_as<QuadratureType, Lobatto>;

// Concept for functions that can be integrated using quadrature.
template <typename Float, typename Function, typename FunctionValue>
concept Integrable = requires(Float w, FunctionValue f) {
  requires std::floating_point<Float>;
  requires std::invocable<Function, Float>;
  requires std::same_as<std::invoke_result_t<Function, Float>, FunctionValue>;
  { f* w } -> std::same_as<FunctionValue>;
  { f + f } -> std::same_as<FunctionValue>;
};

template <std::floating_point Float, typename QuadratureType = None>
requires ValidQuadratureTypes<QuadratureType>
class Quadrature {
  // Type aliases for Orthogonal polynomials.
  using Poly = OrthogonalPolynomial<Float>;
  using Leg = Legendre<Float>;

  // Type aliases for Eigen3
  using Matrix = Eigen::Matrix<Float, Eigen::Dynamic, Eigen::Dynamic>;
  using Vector = Eigen::Matrix<Float, Eigen::Dynamic, 1>;

 public:
  // Store the real type.
  using value_type = Float;

  // delete the default constructor
  Quadrature() = delete;

  // Constructor for Gaussian quadrature.
  Quadrature(const int n, const Poly& p = Leg{}) {
    // Check the order is positive.
    assert(n > 0);

    // Store the endpoints for the interval.
    left = p.x1();
    right = p.x2();

    // Build the matrix for the eigenvalue problem.
    Matrix A = build_matrix(n, p);

    // Solve the eigenvalue problem.
    Eigen::SelfAdjointEigenSolver<Matrix> es;
    es.compute(A);

    // Store the points.
    points = std::vector(es.eigenvalues().begin(), es.eigenvalues().end());

    // Store and scale the weights.
    weights = std::vector(es.eigenvectors().row(0).begin(),
                          es.eigenvectors().row(0).end());
    std::transform(weights.cbegin(), weights.cend(), weights.begin(),
                   [&p](Float w) -> Float { return p.mu() * w * w; });
  }

  // Return the number of points.
  int n() const { return points.size(); }

  // Return the end points.
  inline Float x1() const { return left; }
  inline Float x2() const { return right; }

  // Return the ith point or weight.
  inline Float x(int i) const { return points[i]; }
  inline Float w(int i) const { return weights[i]; }

  // Return const reference to the points or weights
  const std::vector<Float>& x() const { return points; }
  const std::vector<Float>& w() const { return weights; }

  //  template <std::invocable<Float> Function>
  template <typename Function,
            typename FunctionValue = std::invoke_result_t<Function, Float> >
  requires Integrable<Float, Function, FunctionValue>
  auto integrate(const Function& f) {
    return std::inner_product(
        points.cbegin(), points.cend(), weights.cbegin(), FunctionValue{},
        std::plus<>(),
        [f](Float x, Float w) -> FunctionValue { return f(x) * w; });
  }

 private:
  // Store the interval.
  Float left;
  Float right;

  // Store the points and weights.
  std::vector<Float> points;
  std::vector<Float> weights;

  // Gauss matrix building function.
  Matrix build_matrix(
      int n, const Poly& p) requires std::same_as<QuadratureType, None> {
    Matrix A{Matrix::Zero(n, n)};
    for (int i = 0; i < n; i++) {
      A.diagonal()(i) = p.d(i + 1);
    }
    for (int i = 0; i < n - 1; i++) {
      Float tmp = p.e(i + 1);
      A.diagonal(+1)(i) = tmp;
      A.diagonal(-1)(i) = tmp;
    }
    return A;
  }

  // Gauss-Radau matrix building function.
  Matrix build_matrix(
      int n, const Poly& p) requires std::same_as<QuadratureType, Radau> {
    int m = n - 1;
    Matrix B{Matrix::Zero(m, m)};
    for (int i = 0; i < m; i++) {
      B.diagonal()(i) = p.d(i + 1) - x1();
    }
    for (int i = 0; i < m - 1; i++) {
      Float tmp = p.e(i + 1);
      B.diagonal(+1)(i) = tmp;
      B.diagonal(-1)(i) = tmp;
    }
    Vector y{Vector::Zero(m)};
    y(m - 1) = pow(p.e(m), 2);
    Vector x = B.llt().solve(y);
    Matrix A{Matrix::Zero(n, n)};
    for (int i = 0; i < m; i++) {
      A.diagonal()(i) = p.d(i + 1);
    }
    A(m, m) = x1() + x(m - 1);
    for (int i = 0; i < m; i++) {
      Float tmp = p.e(i + 1);
      A.diagonal(+1)(i) = tmp;
      A.diagonal(-1)(i) = tmp;
    }
    return A;
  }

  // Gauss-Lobatto matrix building function.
  Matrix build_matrix(
      int n, const Poly& p) requires std::same_as<QuadratureType, Lobatto> {
    int m = n - 1;
    Matrix B{Matrix::Zero(m, m)};
    for (int i = 0; i < m; i++) {
      B.diagonal()(i) = p.d(i + 1) - x1();
    }
    for (int i = 0; i < m - 1; i++) {
      Float tmp = p.e(i + 1);
      B.diagonal(+1)(i) = tmp;
      B.diagonal(-1)(i) = tmp;
    }
    Vector y{Vector::Zero(m)};
    y(m - 1) = 1.0;
    Vector x = B.llt().solve(y);
    Float gamma = x(m - 1);

    for (int i = 0; i < m; i++) {
      B.diagonal()(i) = x2() - p.d(i + 1);
    }
    for (int i = 0; i < m - 1; i++) {
      Float tmp = -p.e(i + 1);
      B.diagonal(+1)(i) = -tmp;
      B.diagonal(-1)(i) = -tmp;
    }
    y(m - 1) = -1.0;
    x = B.llt().solve(y);
    Float mu = x(m - 1);

    Matrix A{Matrix::Zero(n, n)};
    for (int i = 0; i < m; i++) {
      A.diagonal()(i) = p.d(i + 1);
    }
    for (int i = 0; i < m - 1; i++) {
      Float tmp = p.e(i + 1);
      A.diagonal(+1)(i) = tmp;
      A.diagonal(-1)(i) = tmp;
    }
    using std::sqrt;
    Float tmp = sqrt((x2() - x1()) / (gamma - mu));
    A.diagonal(+1)(m - 1) = tmp;
    A.diagonal(-1)(m - 1) = tmp;
    A.diagonal()(m) = x1() + gamma * tmp * tmp;
    return A;
  }
};

}  // namespace GaussQuad

#endif  // GAUSS_QUAD_QUADRATURE_GUARD_H
