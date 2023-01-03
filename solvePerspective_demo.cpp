#include<iostream>
#include<opencv2\opencv.hpp>
#include<Eigen\Core>
#include<vector>
#include<Eigen\Dense>
#include<time.h>
using namespace std;
Eigen::MatrixXf Vector2Matrix(vector<cv::Point2f>pt1);
Eigen::MatrixXf matrixInit(Eigen::MatrixXf pt1, Eigen::MatrixXf pt2);
Eigen::MatrixXf getperspectiveMatrix(Eigen::MatrixXf A, Eigen::MatrixXf D);
Eigen::MatrixXf matrixNormalize(Eigen::MatrixXf Matrix);
Eigen::MatrixXf ptSort(Eigen::MatrixXf pt);
Eigen::MatrixXf norm2Detect(Eigen::MatrixXf pt_norm, Eigen::MatrixXf perspectiveMatrix);
Eigen::MatrixXf detect2Norm(Eigen::MatrixXf pt_norm, Eigen::MatrixXf perspectiveMatrix);


int main() {
	clock_t start, end;
	Eigen::MatrixXf A(8, 8);
	Eigen::MatrixXf D(8, 1);
	Eigen::MatrixXf pt1(4, 2);
	Eigen::MatrixXf pt2(4, 2);
	vector<cv::Point2f>pt1_vector;
	vector<cv::Point2f>pt2_vector;
	cv::Point2f pt1_1(220, 140);
	cv::Point2f pt1_2(420, 140);
	cv::Point2f pt1_3(220, 340);
	cv::Point2f pt1_4(420, 340);
	cv::Point2f pt2_1(210, 140);
	cv::Point2f pt2_2(424, 148);
	cv::Point2f pt2_3(203, 355);
	cv::Point2f pt2_4(420, 328);
	pt1_vector.push_back(pt1_1);
	pt1_vector.push_back(pt1_2);
	pt1_vector.push_back(pt1_3);
	pt1_vector.push_back(pt1_4);
	pt2_vector.push_back(pt2_1);
	pt2_vector.push_back(pt2_2);
	pt2_vector.push_back(pt2_3);
	pt2_vector.push_back(pt2_4);
	pt1 = Vector2Matrix(pt1_vector);
	pt2 = Vector2Matrix(pt2_vector);
	A = matrixInit(pt1,pt2);

	int cnt = 0;
	for (auto i : pt2_vector) {
		D(cnt) = i.x;
		D(cnt + 4) = i.y;
		cnt++;
	}
	start = clock();
	Eigen::MatrixXf perspectiveMatrix = getperspectiveMatrix(A, D);
	Eigen::MatrixXf pt1_norm = matrixNormalize(pt1);
	Eigen::MatrixXf pt2_norm = matrixNormalize(pt2);
	Eigen::MatrixXf detect = norm2Detect(pt1_norm, perspectiveMatrix);
	Eigen::MatrixXf norm = detect2Norm(pt2_norm, perspectiveMatrix);
	cout << detect << endl;
	cout << norm << endl;
	end = clock();
	cout << "time:" << double(end - start) << endl;

	
	system("pause");
	return 0;
}

Eigen::MatrixXf Vector2Matrix(vector<cv::Point2f>pt1) {
	Eigen::MatrixXf ptMatrix(4, 2);
	int cnt = 0;
	for (auto i : pt1) {
		Eigen::MatrixXf temp(1, 2);
		temp(0, 0) = i.x;
		temp(0, 1) = i.y;
		ptMatrix.block(cnt, 0, 1, 2) = temp;
		cnt++;
	}
	return ptMatrix;
}

Eigen::MatrixXf matrixInit(Eigen::MatrixXf pt1, Eigen::MatrixXf pt2) {
	Eigen::MatrixXf A = Eigen::MatrixXf::Ones(8, 8);
	A.block(0, 3, 4, 3) = Eigen::MatrixXf::Zero(4, 3);
	A.block(4, 0, 4, 3) = Eigen::MatrixXf::Zero(4, 3);
	
	
	A.block(0, 0, 4, 2) = pt1;
	A.block(4, 3, 4, 2) = pt1;

	Eigen::MatrixXf temp_re_1(4, 1);
	Eigen::MatrixXf temp_re_2(4, 1);
	Eigen::MatrixXf temp_re_3(4, 1);
	Eigen::MatrixXf temp_re_4(4, 1);
	temp_re_1 = pt1.block(0, 0, 4, 1).cwiseProduct(-pt2.block(0, 0, 4, 1));
	temp_re_2 = pt1.block(0, 1, 4, 1).cwiseProduct(-pt2.block(0, 0, 4, 1));
	temp_re_3 = pt1.block(0, 0, 4, 1).cwiseProduct(-pt2.block(0, 1, 4, 1));
	temp_re_4 = pt1.block(0, 1, 4, 1).cwiseProduct(-pt2.block(0, 1, 4, 1));
	A.block(0, 6, 4, 1) = temp_re_1;
	A.block(0, 7, 4, 1) = temp_re_2;
	A.block(4, 6, 4, 1) = temp_re_3;
	A.block(4, 7, 4, 1) = temp_re_4;
	return A;
}

Eigen::MatrixXf getperspectiveMatrix(Eigen::MatrixXf A, Eigen::MatrixXf D) {
	
	Eigen::MatrixXf result = A.householderQr().solve(D);
	Eigen::MatrixXf perspectiveMatrix(3, 3);
	perspectiveMatrix.block(0, 0, 3, 1) = result.block(0, 0, 3, 1);

	perspectiveMatrix.block(0, 1, 3, 1) = result.block(3, 0, 3, 1);
	
	perspectiveMatrix.block(0, 2, 2, 1) = result.block(6, 0, 2, 1);
	perspectiveMatrix(2, 2) = 1;
	
	return perspectiveMatrix;
}

Eigen::MatrixXf matrixNormalize(Eigen::MatrixXf Matrix) {
	Eigen::MatrixXf norm = Eigen::MatrixXf::Ones(4, 3);
	norm.block(0, 0, 4, 2) = Matrix;
	return norm;
}

Eigen::MatrixXf norm2Detect(Eigen::MatrixXf pt_norm, Eigen::MatrixXf perspectiveMatrix) {

	Eigen::MatrixXf pt_result(4, 2);//最后返回值
	Eigen::MatrixXf pt_temp(4, 3);
	Eigen::MatrixXf w(4,1);
	pt_temp = pt_norm*perspectiveMatrix;
	w = pt_temp.block(0, 2, 4, 1);
	pt_result = pt_temp.block(0, 0, 4, 2);
	pt_result.block(0, 0, 4, 1) = pt_result.block(0, 0, 4, 1).cwiseQuotient(w);
	pt_result.block(0, 1, 4, 1) = pt_result.block(0, 1, 4, 1).cwiseQuotient(w);

	return pt_result;
}

Eigen::MatrixXf detect2Norm(Eigen::MatrixXf pt_norm, Eigen::MatrixXf perspectiveMatrix) {
	Eigen::MatrixXf pt_result(4, 2);//最后返回值
	Eigen::MatrixXf pt_temp(4, 3);
	Eigen::MatrixXf w(4, 1);
	perspectiveMatrix = perspectiveMatrix.inverse();
	pt_temp = pt_norm*perspectiveMatrix;
	w = pt_temp.block(0, 2, 4, 1);
	pt_result = pt_temp.block(0, 0, 4, 2);
	pt_result.block(0, 0, 4, 1) = pt_result.block(0, 0, 4, 1).cwiseQuotient(w);
	pt_result.block(0, 1, 4, 1) = pt_result.block(0, 1, 4, 1).cwiseQuotient(w);
	return pt_result;
}

Eigen::MatrixXf ptSort(Eigen::MatrixXf pt) {
	
	float one = pt(0, 0) + pt(0, 1);
	float two = pt(1, 0) + pt(1, 1);
	float three = pt(2, 0) + pt(2, 1);
	float four = pt(3, 0) + pt(3, 1);
	float ptSum_array[4] = {one,two,three,four};
	unordered_map<int, float>sort_list = {
		{one,1},{two,2},{three,3},{four,4}
	};
	cout << sort_list[560] << endl;
	sort(ptSum_array, ptSum_array + 4, less<float>());
	for (int i = 0; i < 4; i++) {
		cout << ptSum_array[i] << endl;
	}
	return pt;
}
