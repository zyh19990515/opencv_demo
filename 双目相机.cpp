#include<opencv2\opencv.hpp>
#include<iostream>
using namespace cv;
using namespace std;

int main() {
	int iW = 1024;
	int iH = 1024;
	int cX = iW / 2;
	int cY = iH / 2;
	int Focus = 1000;//世界坐标系原点在左相机光心，相机焦距1000mm

	Mat1f camM = Mat1f::eye(3, 3);
	camM[0][0] = camM[1][1] = Focus;
	camM[0][2] = cX;
	camM[1][2] = cY;
	Mat1f discoef = Mat1f::zeros(5, 1);//以上为相机内参

	Mat1f camCoordR = Mat1f::zeros(3, 1);//右相机在左相机坐标系下的坐标
	camCoordR[0][0] = 200;//右相机在左相机X轴，基线长度为200
	Mat1f rotV = Mat1f::zeros(3, 1);
	rotV[1][0] = 0.1;
	rotV[2][0] = 0.1;
	Mat1f rotM;
	Rodrigues(rotV, rotM);
	//Xr=R*Xl+T
	Mat1f trsV = Mat1f::zeros(3, 1) - rotM*camCoordR;
	Mat1f trsM = Mat1f::zeros(3, 3);
	trsM[0][1] = -trsV[0][0]; trsM[1][0] = trsV[2][0];
	trsM[0][2] = trsV[1][0]; trsM[2][0] = -trsV[1][0];
	trsM[1][2] = -trsV[0][0]; trsM[2][1] = trsV[0][0];
	Mat1f E = trsM*rotM;
	Mat1f F = camM.inv().t()*E*camM.inv();
	cout << trsV << endl;
	cout << trsM << endl;

	Vec3d sphere(100, 200, 900);
	int radius = 100;//球
	vector<Vec3f>objV;
	float radiu2 = radius*radius;
	for (int i = -radius; i < radius; i++) {
		for (int j = -radius; j < radius; j++) {
			float xy = i*i + j*j;
			if (xy < radiu2) {
				float k = -sqrt(radiu2 - xy);
				float x = i + sphere[0];
				float y = j + sphere[1];
				float z = k + sphere[2];
				objV.push_back(Vec3f(x, y, z));
				
			}
		}
	}
	FILE *f1 = fopen("3d.xyz", "w");
	for (auto iter : objV) {
		fprintf(f1, "%f %f %f\n",iter[0],iter[1],iter[2]);
	}
	fclose(f1);
	vector<Vec2f>imgPtVL;
	projectPoints(objV, Mat1f::zeros(3, 1), Mat1f::zeros(3, 1), camM, discoef, imgPtVL);
	Mat1f imgLX = Mat1f::zeros(iH, iW);
	Mat1f imgLY = Mat1f::zeros(iH, iW);
	for (size_t i = 0; i < imgPtVL.size(); i++) {
		Vec2f iter = imgPtVL[i];
		if (iter[0]<0 || iter[0]>iW - 1 || iter[1]<0 || iter[1]>iH - 1) {
			continue;
		}
		imgLX[(int)iter[1]][(int)iter[0]] = imgPtVL[i][0];
		imgLY[(int)iter[1]][(int)iter[0]] = imgPtVL[i][1];
	}
	Mat1f projL = Mat1f::zeros(3, 4);
	projL[0][0] = projL[1][1] = projL[2][2] = 1;
	projL = camM*projL;//左相机投影矩阵3*4=内参*（R|T）
	cout << projL << endl;

	Mat1f projR = Mat1f::zeros(3, 4);
	rotM.copyTo(Mat1f(projR, Rect(0, 0, 3, 3)));
	trsV.copyTo(Mat1f(projR, Rect(3, 0, 1, 3)));
	projR = camM*projR;//右相机投影矩阵

	vector<Vec2f>xLPt, xRPt;
	for (int i = 0; i < iH; i++) {
		for (int j = 0; j < iW; j++) {
			if (imgLX[i][j]>0) {
				xLPt.push_back(Vec2f(j, i));
				Mat1f xxL = Mat1f::ones(3, 1);
				xxL[0][0] = j;
				xxL[1][0] = i;
				Mat1f tmpABC = F*xxL;
				float y0 = (-tmpABC[0][0] * imgLX[i][j] - tmpABC[2][0]) / tmpABC[1][0];
				xRPt.push_back(Vec2f(imgLX[i][j], y0));
			}
		}
	}
	Mat1f ptML(2, xLPt.size()), ptMR(2, xLPt.size());
	for (size_t i = 0; i < xLPt.size(); i++) {
		ptML[0][i] = xLPt[i][0];
		ptML[1][i] = xLPt[i][1];
		ptML[0][i] = xLPt[i][0];
		ptML[1][i] = xLPt[i][1];
	}
	Mat1f pt4M;
	triangulatePoints(projL, projR, ptML, ptMR, pt4M);//交会空间点得出齐次坐标系下的三维坐标阵

	vector<Vec3f>pt4DV(xLPt.size());
	for (int i = 0; i < xLPt.size(); i++) {
		pt4DV[i] = Vec3f(pt4M[0][i] / pt4M[3][i], pt4M[1][i] / pt4M[3][i], pt4M[2][i] / pt4M[3][i]);
	}
	f1 = fopen("D:\\New3d.xyz", "w");
	for (auto iter : pt4DV) {
		fprintf(f1, "%f %f %f\n", iter[0], iter[1], iter[2]);
	}
	fclose(f1);
	system("pause");
	
	

	return 0;
}