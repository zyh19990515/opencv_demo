#include<Eigen/Dense>
#include<iostream>
#include<time.h>
#include<vector>
using namespace std;
void MatrixInit(Eigen::MatrixXd& A, Eigen::MatrixXd& B, Eigen::MatrixXd& Q, Eigen::MatrixXd& R);
bool solveRiccatiIterationC(const Eigen::MatrixXd& A, const Eigen::MatrixXd& B, const Eigen::MatrixXd& Q, const Eigen::MatrixXd& R, Eigen::MatrixXd& P, 
	const double dt = 0.001, const double& tolerance = 1.E-5, const uint16_t iter_max = 100000);
bool solveRiccatiIterationD(const Eigen::MatrixXd& Ad, const Eigen::MatrixXd& Bd, const Eigen::MatrixXd& Q, const Eigen::MatrixXd& R, Eigen::MatrixXd& P,
	const double& tolerance = 1.E-5, const uint16_t iter_max = 100000);
void solveK(const Eigen::MatrixXd &R, const Eigen::MatrixXd &B, const Eigen::MatrixXd &P, Eigen::MatrixXd &K);
bool solveRiccatiArimotoPotter(const Eigen::MatrixXd& A, const Eigen::MatrixXd& B, const Eigen::MatrixXd& Q, const Eigen::MatrixXd& R, Eigen::MatrixXd& P);
int main() {
	Eigen::MatrixXd A(2, 2);
	Eigen::MatrixXd B(2, 1);
	Eigen::MatrixXd Q(2, 2);
	Eigen::MatrixXd R(1, 1);

	MatrixInit(A, B, Q, R);

	//Eigen::MatrixXd PC;
	//Eigen::MatrixXd PD;
	Eigen::MatrixXd PP;
	//solveRiccatiIterationC(A, B, Q, R, PC);
	//solveRiccatiIterationD(A, B, Q, R, PD);
	clock_t start = clock();
	solveRiccatiArimotoPotter(A, B, Q, R, PP);
	//cout << "PC:" << endl;
	//cout << PC << endl;
	//cout << "PD:" << endl;
	//cout << PD << endl;
	cout << "PP:" << endl;
	cout << PP << endl;
	Eigen::MatrixXd KC;
	Eigen::MatrixXd KD;
	Eigen::MatrixXd KP;
	//solveK(R, B, PC, KC);
	//solveK(R, B, PD, KD);
	solveK(R, B, PP, KP);
	clock_t end = clock();
	cout << end - start << endl;
	//cout << "KC:" << endl;
	//cout << KC << endl;
	//cout << "KD:" << endl;
	//cout << KD << endl;
	cout << "KP:" << endl;
	cout << KP << endl;
	return 0;
}

void MatrixInit(Eigen::MatrixXd& A, Eigen::MatrixXd& B, Eigen::MatrixXd& Q, Eigen::MatrixXd& R) {
	A(0, 0) = 0;
	A(0, 1) = -3;
	A(1, 0) = -1;
	A(1, 1) = 2;
	
	B(0, 0) = 0;
	B(1, 0) = -1;
	
	Q(0, 0) = 100;
	Q(0, 1) = 0;
	Q(1, 0) = 0;
	Q(1, 1) = 1;

	R(0, 0) = 1.;
	
	cout << "A:" << endl;
	cout << A << endl;
	cout << "B:" << endl;
	cout << B << endl;
	cout << "Q:" << endl;
	cout << Q << endl;
	cout << "R:" << endl;
	cout << R << endl;
}

//迭代法连续模型
bool solveRiccatiIterationC(const Eigen::MatrixXd& A, const Eigen::MatrixXd& B, const Eigen::MatrixXd& Q, const Eigen::MatrixXd& R, Eigen::MatrixXd& P, const double dt, const double& tolerance, const uint16_t iter_max) {
	P = Q;
	Eigen::MatrixXd P_next;

	Eigen::MatrixXd AT = A.transpose();
	Eigen::MatrixXd BT = B.transpose();
	Eigen::MatrixXd Rinv = R.inverse();

	double diff;
	for (uint16_t i = 0; i < iter_max; i++) {
		P_next = P + (P * A + AT * P - P * B * Rinv * BT * P) * dt;
		diff = fabs((P_next - P).maxCoeff());
		P = P_next;
		if (diff < tolerance) {
			cout << "iteration number:" << i << endl;
			return true;
		}
	}
	return false;


}

//迭代法离散模型
bool solveRiccatiIterationD(const Eigen::MatrixXd& Ad, const Eigen::MatrixXd& Bd, const Eigen::MatrixXd& Q, const Eigen::MatrixXd& R, Eigen::MatrixXd& P, const double& tolerance, const uint16_t iter_max) {
	P = Q;
	Eigen::MatrixXd P_next;
	Eigen::MatrixXd AdT = Ad.transpose();
	Eigen::MatrixXd BdT = Bd.transpose();
	Eigen::MatrixXd Rinv = R.inverse();

	double diff;
	for (uint16_t i = 0; i < iter_max; i++) {
		P_next = AdT * P * Ad -
			AdT * P * Bd * (R + BdT * P * Bd).inverse() * BdT * P * Ad + Q;
		diff = fabs((P_next - P).maxCoeff());
		P = P_next;
		if (diff < tolerance) {
			cout << "iteration number:" << i << endl;
			return true;
		}
	}
	return false;
}

//连续模型的Arimoto Potter算法
bool solveRiccatiArimotoPotter(const Eigen::MatrixXd& A, const Eigen::MatrixXd& B, const Eigen::MatrixXd& Q, const Eigen::MatrixXd& R, Eigen::MatrixXd& P) {
	//clock_t start = clock();
	const uint16_t dim_x = A.rows();
	const uint16_t dim_u = B.cols();

	//哈密顿矩阵
	Eigen::MatrixXd Ham = Eigen::MatrixXd::Zero(2 * dim_x, 2 * dim_x);
	Ham << A, -B * R.inverse() * B.transpose(), -Q, -A.transpose();
	
	//计算特征值和特征向量
	Eigen::EigenSolver<Eigen::MatrixXd>Eigs(Ham);

	//检查特征值
	cout << "eigen values:\n" << Eigs.eigenvalues() << endl;
	cout << "eigen vectors:\n" << Eigs.eigenvectors() << endl;

	Eigen::MatrixXcd eigvec = Eigen::MatrixXcd::Zero(2 * dim_x, dim_x);
	int j = 0;
	for (int i = 0; i < 2 * dim_x; ++i) {
		if (Eigs.eigenvalues()[i].real() < 0.) {
			eigvec.col(j) = Eigs.eigenvectors().block(0, i, 2 * dim_x, 1);
			++j;
		}
	}
	cout << "\n" << endl;
	cout << eigvec << endl;
	Eigen::MatrixXcd Vs_1, Vs_2;
	Vs_1 = eigvec.block(0, 0, dim_x, dim_x);
	Vs_2 = eigvec.block(dim_x, 0, dim_x, dim_x);
	P = (Vs_2 * Vs_1.inverse()).real();
	cout << Vs_1 << endl;
	cout << Vs_2 << endl;
	

	return true;

}

void solveK(const Eigen::MatrixXd &R, const Eigen::MatrixXd &B, const Eigen::MatrixXd &P, Eigen::MatrixXd &K) {
	Eigen::MatrixXd BT = B.transpose();
	Eigen::MatrixXd Rinv = R.inverse();
	
	K = Rinv * BT * P;
}
