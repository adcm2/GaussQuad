#ifndef GAUSS_QUAD_JACOBI_GUARD_H
#define GAUSS_QUAD_JACOBI_GUARD_H

#include <algorithm>
#include <cassert>
#include <cmath>
#include <concepts>
#include <limits>
#include <utility>
#include <vector>

// Function declarations

template <std::floating_point Float>
Float JacobiPoly(int, Float, Float, Float);

template <std::floating_point Float>
Float JacobiPolyDerivative(int, Float, Float, Float);

template <std::floating_point Float>
std::vector<Float> JacobiPolyZeros(int, Float, Float);

template <std::floating_point Float>
std::pair<std::vector<Float>, std::vector<Float>> JacobiPolyZerosWeights(int,
                                                                         Float,
                                                                         Float);

// Function definitions

template <std::floating_point Float>
Float JacobiPoly(int degree, Float x, Float alpha, Float beta) {
  switch (degree) {
    case 0:
      return static_cast<Float>(1);
    case 1:
      return 0.5 * (alpha - beta + (alpha + beta + 2) * x);
    default:
      Float tmp = 2 * (degree - 1) + alpha + beta;
      Float a1 = 2 * degree * (degree + alpha + beta) * tmp;
      Float a2 = (tmp + 1) * (alpha * alpha - beta * beta);
      Float a3 = tmp * (tmp + 1) * (tmp + 2);
      Float a4 = 2 * (degree + alpha - 1) * (degree + beta - 1) * (tmp + 2);
      return ((a2 + a3 * x) * JacobiPoly(degree - 1, x, alpha, beta) -
              a4 * JacobiPoly(degree - 2, x, alpha, beta)) /
             a1;
  }
}

template <std::floating_point Float>
Float JacobiPolyDerivative(int degree, Float x, Float alpha, Float beta) {
  switch (degree) {
    case 0:
      return 0;
    default:
      Float tmp = 2 * degree + alpha + beta;
      Float b1 = tmp * (1 - x * x);
      Float b2 = degree * (alpha - beta - tmp * x);
      Float b3 = 2 * (degree + alpha) * (degree + beta);
      return (b2 * JacobiPoly(degree, x, alpha, beta) +
              b3 * JacobiPoly(degree - 1, x, alpha, beta)) /
             b1;
  }
}

template <std::floating_point Float>
std::vector<Float> JacobiPolyZeros(int degree, Float alpha, Float beta) {
  assert(degree >= 0);
  std::vector<Float> zeros;
  zeros.reserve(degree);
  const int maxIter = 30;
  const Float epsilon = std::numeric_limits<Float>::epsilon();
  const Float dth = std::numbers::pi_v<Float> / (2 * degree);
  for (int k = 0; k < degree; k++) {
    Float r = -std::cos((2 * k + 1) * dth);
    if (k > 0) r = 0.5 * (r + zeros[k - 1]);
    for (int j = 1; j < maxIter; j++) {
      Float fun = JacobiPoly(degree, r, alpha, beta);
      Float der = JacobiPolyDerivative(degree, r, alpha, beta);
      Float sum = 0;
      for (int i = 0; i < k; i++) sum += static_cast<Float>(1) / (r - zeros[i]);
      Float delr = -fun / (der - sum * fun);
      r += delr;
      if (std::abs(delr) < epsilon) break;
    }
    zeros.push_back(r);
  }
  return zeros;
}

template <std::floating_point Float>
std::pair<std::vector<Float>, std::vector<Float>> JacobiPolyZerosWeights(
    int degree, Float alpha, Float beta) {
  assert(degree >= 0);
  auto zeros = JacobiPolyZeros(degree, alpha, beta);
  std::vector<Float> weights;
  weights.reserve(degree);
  std::transform(
      zeros.begin(), zeros.end(), std::back_inserter(weights),
      [&](auto z) { return JacobiPolyDerivative(degree, z, alpha, beta); });
  Float fac = std::exp(std::numbers::ln2_v<Float> * (alpha + beta + 1) +
                       std::lgamma(alpha + degree + 1) +
                       std::lgamma(beta + degree + 1) -
                       std::lgamma(static_cast<Float>(degree + 1)) -
                       std::lgamma(alpha + beta + degree + 1));
  std::transform(zeros.begin(), zeros.end(), weights.begin(),
                 [fac](auto z, auto w) { return fac / (w * w * (1 - z * z)); });
  return std::pair(zeros, weights);
}

#endif  // GAUSS_QUAD_JACOBI_GUARD_H
