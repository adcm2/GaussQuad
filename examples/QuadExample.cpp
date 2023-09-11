#include <GaussQuad/All>
#include <chrono>
#include <concepts>
#include <fstream>
#include <iostream>

int main() {
  using Float = double;

  using namespace GaussQuad;

  using namespace std::chrono;

  int n = 256;

  auto start = high_resolution_clock::now();
  LegendrePolynomial<Float> p;
  auto quad = p.ZerosAndWeights(n);
  auto stop = high_resolution_clock::now();
  auto duration = duration_cast<microseconds>(stop - start);
  std::cout << duration.count() << std::endl;

  /*

  // Set the output precision
  using std::cout;
  using std::endl;
  cout.setf(std::ios_base::scientific);
  cout.setf(std::ios_base::showpos);
  cout.precision(16);

  // Set the floating point precision
  using Float = double;

  // Set the quadrature type
  using QuadratureType = typename GaussQuad::Radau;

  // Build the quadrature
  int n = 5;
  GaussQuad::Quadrature<Float, QuadratureType> q(n);

  // write out the points and weights
  for (int i = 0; i < n; i++) {
    std::cout << q.x(i) << " " << q.w(i) << std::endl;
  }

  // define a simple function to integrate
  auto fun = [](Float x) { return x * x; };

  // set the exact value for the integral
  Float exact = Float(2.0) / Float(3.0);

  cout << "Numerical value = " << q.integrate(fun)
       << ", exact value = " << exact << endl;

  */
}
