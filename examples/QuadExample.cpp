#include <GaussQuad/All>
#include <chrono>
#include <concepts>
#include <fstream>
#include <iostream>

int main() {
  using Real = double;

  using namespace GaussQuad;

  // Set the output precision
  using std::cout;
  using std::endl;
  cout.setf(std::ios_base::scientific);
  cout.setf(std::ios_base::showpos);
  cout.precision(16);

  // Set the floating point precision
  using Real = double;

  // Build the quadrature
  int n = 5;
  auto q = GaussLobattoLegendreQuadrature1D<Real>(n);

  // write out the points and weights
  for (int i = 0; i < n; i++) {
    std::cout << q.X(i) << " " << q.W(i) << std::endl;
  }

  // define a simple function to integrate
  auto fun = [](Real x) { return x * x; };

  // set the exact value for the integral
  Real exact = Real(2.0) / Real(3.0);

  cout << "Numerical value = " << q.Integrate(fun)
       << ", exact value = " << exact << endl;
}
