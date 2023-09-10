#include <Spectra/SymEigsSolver.h>

#include <Eigen/Core>
// <Spectra/MatOp/DenseSymMatProd.h> is implicitly included
#include <iostream>

using namespace Spectra;

int main() {
  // We are going to calculate the eigenvalues of M
  Eigen::MatrixXd A = Eigen::MatrixXd::Random(10, 10);
  Eigen::MatrixXd M = A + A.transpose();

  // Construct matrix operation object using the wrapper class DenseSymMatProd
  DenseSymMatProd<double> op(M);

  // Construct eigen solver object, requesting the largest three eigenvalues
  SymEigsSolver<DenseSymMatProd<double>> eigs(op, 10, 10);

  // Initialize and compute
  eigs.init();
  int nconv = eigs.compute(SortRule::LargestAlge);

  // Retrieve results
  Eigen::VectorXd evalues;
  if (eigs.info() == CompInfo::Successful) evalues = eigs.eigenvalues();

  std::cout << "Eigenvalues found:\n" << evalues << std::endl;

  return 0;
}
