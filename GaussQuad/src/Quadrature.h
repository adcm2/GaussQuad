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
template <typename Float, typename Function, typename FunctionValue>
concept Integrable = requires(Float w, FunctionValue f) {
  requires std::floating_point<Float>;
  requires std::invocable<Function, Float>;
  requires std::same_as<std::invoke_result_t<Function, Float>, FunctionValue>;
  { f* w } -> std::same_as<FunctionValue>;
  { f + f } -> std::same_as<FunctionValue>;
};

template <std::floating_point Float>
class Quadrature1D {
  using Vector = std::vector<Float>;
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
            typename FunctionValue = std::invoke_result_t<Function, Float> >
  requires Integrable<Float, Function, FunctionValue>
  auto Integrate(const Function& f) {
    return std::inner_product(
        _x.cbegin(), _x.cend(), _w.cbegin(), FunctionValue{}, std::plus<>(),
        [f](Float x, Float w) -> FunctionValue { return f(x) * w; });
  }

 private:
  Vector _x;
  Vector _w;
};

// Factory functions for the Quadrature1D type.

template <std::floating_point Float>
auto GaussLegendreQuadrature1D(int n) {
  return Quadrature1D(LegendrePolynomial<Float>{}.GaussQuadrature(n));
}

template <std::floating_point Float>
auto GaussRadauLegendreQuadrature1D(int n) {
  return Quadrature1D(LegendrePolynomial<Float>{}.GaussRadauQuadrature(n));
}

template <std::floating_point Float>
auto GaussLobattoLegendreQuadrature1D(int n) {
  return Quadrature1D(LegendrePolynomial<Float>{}.GaussLobattoQuadrature(n));
}

template <std::floating_point Float>
auto GaussChebyshevQuadrature1D(int n) {
  return Quadrature1D(ChebyshevPolynomial<Float>{}.GaussQuadrature(n));
}

template <std::floating_point Float>
auto GaussRadauChebyshevQuadrature1D(int n) {
  return Quadrature1D(ChebyshevPolynomial<Float>{}.GaussRadauQuadrature(n));
}

template <std::floating_point Float>
auto GaussLobattoChebyshevQuadrature1D(int n) {
  return Quadrature1D(ChebyshevPolynomial<Float>{}.GaussLobattoQuadrature(n));
}

template <std::floating_point Float>
auto GaussJacobiQuadrature1D(int n, Float alpha, Float beta) {
  return Quadrature1D(JacobiPolynomial<Float>{alpha, beta}.GaussQuadrature(n));
}

template <std::floating_point Float>
auto GaussRadauJacobiQuadrature1D(int n, Float alpha, Float beta) {
  return Quadrature1D(
      JacobiPolynomial<Float>{alpha, beta}.GaussRadauQuadrature(n));
}

template <std::floating_point Float>
auto GaussLobattoJacobiQuadrature1D(int n, Float alpha, Float beta) {
  return Quadrature1D(
      JacobiPolynomial<Float>{alpha, beta}.GaussLobattoQuadrature(n));
}

}  // namespace GaussQuad

#endif  // GAUSS_QUAD_QUADRATURE_GUARD_H
