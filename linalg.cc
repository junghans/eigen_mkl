#include <Eigen/Eigen>

Eigen::VectorXd linalg_constrained_qrsolve(const Eigen::MatrixXd &A,
                                           const Eigen::VectorXd &b,
                                           const Eigen::MatrixXd &constr) {
  const long NoVariables = A.cols();
  const long NoConstrains =
      constr.rows();  // number of constraints is number of rows of constr
  const long deg_of_freedom = NoVariables - NoConstrains;

  Eigen::HouseholderQR<Eigen::MatrixXd> QR(constr.transpose());

  // Calculate A * Q and store the result in A
  auto A_new = A * QR.householderQ();
  // A_new = [A1 A2], so A2 is just a block of A
  // [A1 A2] has N rows. A1 has ysize columns
  // A2 has 2*ngrid-ysize columns
  Eigen::MatrixXd A2 = A_new.rightCols(deg_of_freedom);
  // now perform QR-decomposition of A2 to solve the least-squares problem A2 *
  // z = b A2 has N rows and (2*ngrid-ysize) columns ->
  Eigen::HouseholderQR<Eigen::MatrixXd> QR2(A2);
  Eigen::VectorXd z = QR2.solve(b);

  // Next two steps assemble vector from y (which is zero-vector) and z
  Eigen::VectorXd result = Eigen::VectorXd::Zero(NoVariables);
  result.tail(deg_of_freedom) = z;
  // To get the final answer this vector should be multiplied by matrix Q
  return QR.householderQ() * result;
}
