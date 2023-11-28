#ifndef GAUSS_QUAD_QUADRATURE_GUARD_H
#define GAUSS_QUAD_QUADRATURE_GUARD_H

#include <algorithm>
#include <cassert>
#include <concepts>
#include <numeric>
#include <ranges>
#include <utility>

#include "OrthogonalPolynomial.h"

namespace GaussQuad {

// Concept for functions that can be integrated using quadrature.
template <typename Real, typename Function, typename FunctionValue>
concept Integrable = requires(Real w, FunctionValue f) {
  requires std::floating_point<Real>;
  requires std::invocable<Function, Real>;
  requires std::same_as<std::invoke_result_t<Function, Real>, FunctionValue>;
  { f* w } -> std::same_as<FunctionValue>;
  { f + f } -> std::same_as<FunctionValue>;
};

template <std::floating_point Real>
class Quadrature1D {
  using Vector = std::vector<Real>;
  using VectorPair = std::pair<Vector, Vector>;

 public:
  // Constructor given pair of vectors for points and weights.
  Quadrature1D(VectorPair pair) : _x{std::get<0>(pair)}, _w{std::get<1>(pair)} {
    assert(_x.size() > 0);
    assert(_x.size() == _w.size());
  }

  // Return the number of points.
  int N() const { return _x.size(); }

  // Return the ith points or weights.
  auto X(int i) const { return _x[i]; }
  auto W(int i) const { return _w[i]; }

  // Return constant references to the points and weights.
  const Vector& Points() const { return _x; }
  const Vector& Weights() const { return _w; }

  // Simple integrator.
  template <typename Function,
            typename FunctionValue = std::invoke_result_t<Function, Real> >
  requires Integrable<Real, Function, FunctionValue>
  auto Integrate(const Function& f) {
    return std::inner_product(
        _x.cbegin(), _x.cend(), _w.cbegin(), FunctionValue{}, std::plus<>(),
        [f](Real x, Real w) -> FunctionValue { return f(x) * w; });
  }

  template <typename Function1, typename Function2>
  void Transform(Function1 f, Function2 df) {
    std::transform(_x.begin(), _x.end(), _x.begin(), f);
    std::transform(_x.begin(), _x.end(), _w.begin(), _w.begin(),
                   [&f, &df](auto x, auto w) { return df(x) * w; });
  }

   private:
    Vector _x;
    Vector _w;
};

// Factory functions for the Quadrature1D type.

template <std::floating_point Real>
auto GaussLegendreQuadrature1D(int n) {
  return Quadrature1D(LegendrePolynomial<Real>{}.GaussQuadrature(n));
}

template <std::floating_point Real>
auto GaussRadauLegendreQuadrature1D(int n) {
  return Quadrature1D(LegendrePolynomial<Real>{}.GaussRadauQuadrature(n));
}

template <std::floating_point Real>
auto GaussLobattoLegendreQuadrature1D(int n) {
  return Quadrature1D(LegendrePolynomial<Real>{}.GaussLobattoQuadrature(n));
}

template <std::floating_point Real>
auto GaussChebyshevQuadrature1D(int n) {
  return Quadrature1D(ChebyshevPolynomial<Real>{}.GaussQuadrature(n));
}

template <std::floating_point Real>
auto GaussRadauChebyshevQuadrature1D(int n) {
  return Quadrature1D(ChebyshevPolynomial<Real>{}.GaussRadauQuadrature(n));
}

template <std::floating_point Real>
auto GaussLobattoChebyshevQuadrature1D(int n) {
  return Quadrature1D(ChebyshevPolynomial<Real>{}.GaussLobattoQuadrature(n));
}

template <std::floating_point Real>
auto GaussJacobiQuadrature1D(int n, Real alpha, Real beta) {
  return Quadrature1D(JacobiPolynomial<Real>{alpha, beta}.GaussQuadrature(n));
}

template <std::floating_point Real>
auto GaussRadauJacobiQuadrature1D(int n, Real alpha, Real beta) {
  return Quadrature1D(
      JacobiPolynomial<Real>{alpha, beta}.GaussRadauQuadrature(n));
}

template <std::floating_point Real>
auto GaussLobattoJacobiQuadrature1D(int n, Real alpha, Real beta) {
  return Quadrature1D(
      JacobiPolynomial<Real>{alpha, beta}.GaussLobattoQuadrature(n));
}

}  // namespace GaussQuad

#endif  // GAUSS_QUAD_QUADRATURE_GUARD_H
