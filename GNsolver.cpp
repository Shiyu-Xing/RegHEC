#include "GNsolver.h"

void GNcalculation(std::vector<Eigen::Vector3d>& p, std::vector<Eigen::Vector3d>& q, std::vector<int> idx, std::vector<Eigen::Matrix3d>& R, std::vector<Eigen::Vector3d>& t,
	int sum, Eigen::Matrix3d HER, Eigen::Vector3d HEt, Sophus::Vector6d& delta) {

	int idx1, idx2, idx3, idx4, idx5, idx6, idx7, idx8;

	idx1 = idx[0];idx2 = idx[1];idx3 = idx[2];idx4 = idx[3];idx5 = idx[4];idx6 = idx[5];idx7 = idx[6];idx8 = idx[7];


	Sophus::Matrix6d H = Sophus::Matrix6d::Zero();
	Sophus::Vector6d b = Sophus::Vector6d::Zero();

	Eigen::Vector3d residual;
	Eigen::Matrix<double, 3, 6> Jacobian;



	for (int i = 0; i < sum; i++) {

		if (i < idx1) {

			residual = R[0] * HER * p[i] + R[0] * HEt + t[0] - R[1] * HER * q[i] - R[1] * HEt - t[1];
			Eigen::Vector3d temp1 = HER * p[i];
			Eigen::Vector3d temp2 = HER * q[i];
			Eigen::Matrix3d temp1hat;
			Eigen::Matrix3d temp2hat;

			temp1hat << 0, -temp1(2), temp1(1),
				temp1(2), 0, -temp1(0),
				-temp1(1), temp1(0), 0;
			temp2hat << 0, -temp2(2), temp2(1),
				temp2(2), 0, -temp2(0),
				-temp2(1), temp2(0), 0;

			Jacobian << -R[0] * temp1hat + R[1] * temp2hat, R[0] - R[1];

		}
		else if (i < idx2) {

			residual = R[1] * HER * p[i] + R[1] * HEt + t[1] - R[2] * HER * q[i] - R[2] * HEt - t[2];
			Eigen::Vector3d temp1 = HER * p[i];
			Eigen::Vector3d temp2 = HER * q[i];
			Eigen::Matrix3d temp1hat;
			Eigen::Matrix3d temp2hat;

			temp1hat << 0, -temp1(2), temp1(1),
				temp1(2), 0, -temp1(0),
				-temp1(1), temp1(0), 0;
			temp2hat << 0, -temp2(2), temp2(1),
				temp2(2), 0, -temp2(0),
				-temp2(1), temp2(0), 0;

			Jacobian << -R[1] * temp1hat + R[2] * temp2hat, R[1] - R[2];



		}
		else if (i < idx3) {

			residual = R[2] * HER * p[i] + R[2] * HEt + t[2] - R[3] * HER * q[i] - R[3] * HEt - t[3];
			Eigen::Vector3d temp1 = HER * p[i];
			Eigen::Vector3d temp2 = HER * q[i];
			Eigen::Matrix3d temp1hat;
			Eigen::Matrix3d temp2hat;

			temp1hat << 0, -temp1(2), temp1(1),
				temp1(2), 0, -temp1(0),
				-temp1(1), temp1(0), 0;
			temp2hat << 0, -temp2(2), temp2(1),
				temp2(2), 0, -temp2(0),
				-temp2(1), temp2(0), 0;

			Jacobian << -R[2] * temp1hat + R[3] * temp2hat, R[2] - R[3];



		}
		else if (i < idx4) {

			residual = R[3] * HER * p[i] + R[3] * HEt + t[3] - R[4] * HER * q[i] - R[4] * HEt - t[4];
			Eigen::Vector3d temp1 = HER * p[i];
			Eigen::Vector3d temp2 = HER * q[i];
			Eigen::Matrix3d temp1hat;
			Eigen::Matrix3d temp2hat;

			temp1hat << 0, -temp1(2), temp1(1),
				temp1(2), 0, -temp1(0),
				-temp1(1), temp1(0), 0;
			temp2hat << 0, -temp2(2), temp2(1),
				temp2(2), 0, -temp2(0),
				-temp2(1), temp2(0), 0;

			Jacobian << -R[3] * temp1hat + R[4] * temp2hat, R[3] - R[4];



		}
		else if (i < idx5) {

			residual = R[4] * HER * p[i] + R[4] * HEt + t[4] - R[5] * HER * q[i] - R[5] * HEt - t[5];
			Eigen::Vector3d temp1 = HER * p[i];
			Eigen::Vector3d temp2 = HER * q[i];
			Eigen::Matrix3d temp1hat;
			Eigen::Matrix3d temp2hat;

			temp1hat << 0, -temp1(2), temp1(1),
				temp1(2), 0, -temp1(0),
				-temp1(1), temp1(0), 0;
			temp2hat << 0, -temp2(2), temp2(1),
				temp2(2), 0, -temp2(0),
				-temp2(1), temp2(0), 0;

			Jacobian << -R[4] * temp1hat + R[5] * temp2hat, R[4] - R[5];



		}
		else if (i < idx6) {

			residual = R[5] * HER * p[i] + R[5] * HEt + t[5] - R[6] * HER * q[i] - R[6] * HEt - t[6];
			Eigen::Vector3d temp1 = HER * p[i];
			Eigen::Vector3d temp2 = HER * q[i];
			Eigen::Matrix3d temp1hat;
			Eigen::Matrix3d temp2hat;

			temp1hat << 0, -temp1(2), temp1(1),
				temp1(2), 0, -temp1(0),
				-temp1(1), temp1(0), 0;
			temp2hat << 0, -temp2(2), temp2(1),
				temp2(2), 0, -temp2(0),
				-temp2(1), temp2(0), 0;

			Jacobian << -R[5] * temp1hat + R[6] * temp2hat, R[5] - R[6];



		}
		else if (i < idx7) {

			residual = R[6] * HER * p[i] + R[6] * HEt + t[6] - R[7] * HER * q[i] - R[7] * HEt - t[7];
			Eigen::Vector3d temp1 = HER * p[i];
			Eigen::Vector3d temp2 = HER * q[i];
			Eigen::Matrix3d temp1hat;
			Eigen::Matrix3d temp2hat;

			temp1hat << 0, -temp1(2), temp1(1),
				temp1(2), 0, -temp1(0),
				-temp1(1), temp1(0), 0;
			temp2hat << 0, -temp2(2), temp2(1),
				temp2(2), 0, -temp2(0),
				-temp2(1), temp2(0), 0;

			Jacobian << -R[6] * temp1hat + R[7] * temp2hat, R[6] - R[7];



		}
		else if (i < idx8) {

			residual = R[7] * HER * p[i] + R[7] * HEt + t[7] - R[8] * HER * q[i] - R[8] * HEt - t[8];
			Eigen::Vector3d temp1 = HER * p[i];
			Eigen::Vector3d temp2 = HER * q[i];
			Eigen::Matrix3d temp1hat;
			Eigen::Matrix3d temp2hat;

			temp1hat << 0, -temp1(2), temp1(1),
				temp1(2), 0, -temp1(0),
				-temp1(1), temp1(0), 0;
			temp2hat << 0, -temp2(2), temp2(1),
				temp2(2), 0, -temp2(0),
				-temp2(1), temp2(0), 0;

			Jacobian << -R[7] * temp1hat + R[8] * temp2hat, R[7] - R[8];



		}



		H = H + Jacobian.transpose() * Jacobian;
		b = b - Jacobian.transpose() * residual;

	}

	delta = H.inverse() * b;



}