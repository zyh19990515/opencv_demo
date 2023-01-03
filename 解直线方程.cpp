#include<iostream>
#include<opencv2/opencv.hpp>
#include<Eigen/Core>
#include<Eigen/Dense>
#include<time.h>
using namespace std;
using namespace cv;
using namespace Eigen;

Mat board_preTreat(Mat img);
Mat ball_preTreat(Mat img);
Mat boardMask(Mat img);
Mat ballMask(Mat img);
vector<Point2f> board_hough(Mat img);
vector<Point2f> ball_hough(Mat img);
vector<Point2f> getPoint(vector<Point2f>kb_h, vector<Point2f>kb_v);
Mat getImg(vector<Point2f> boardPt, vector<Point2f> ballPt, Mat img);

int main() {
	
	vector<Point2f>corners;
	vector<Point2f>ballpts;
	Mat img = imread("E:\\code\\python\\cv2practice\\frame\\122.png");
	Mat img_pre1 = board_preTreat(img);
	corners = board_hough(img_pre1);
	
	Mat img_pre2 = ball_preTreat(img);
	ballpts = ball_hough(img_pre2);
	img = getImg(corners, ballpts, img);
	
	imshow("done", img);
	waitKey(0);
	system("pause");
	return 0;
}

Mat board_preTreat(Mat img) {
	
	img = boardMask(img);
	cvtColor(img, img, COLOR_RGB2GRAY);
	Canny(img, img, 100, 200, 3);
	imshow("3", img);
	return img;
}

Mat ball_preTreat(Mat img) {
	img = ballMask(img);
	Canny(img, img, 100, 200, 3);
	imshow("1", img);
	return img;
}

Mat boardMask(Mat img) {
	
	Mat img_hsv;
	Mat mask;
	Mat img_done;
	cvtColor(img, img_hsv, COLOR_RGB2HSV);
	inRange(img_hsv, Scalar(0, 0, 46), Scalar(180, 30, 220), mask);
	add(img_hsv, img_hsv, img_done, mask);
	cvtColor(img_done, img_done, COLOR_HSV2RGB);

	return img_done;
}
Mat ballMask(Mat img) {
	Mat mask;
	Mat img_hsv;
	Mat img_done;
	cvtColor(img, img_hsv, COLOR_RGB2HSV);
	inRange(img_hsv, Scalar(0, 0, 0), Scalar(180, 255, 100), mask);
	//add(img_hsv, img_hsv, img_done, mask);
	//cvtColor(img_done, img_done, COLOR_HSV2RGB);

	return mask;
}

vector<Point2f> ball_hough(Mat img) {
	int Max_R = 10;
	vector<Vec3f>cirPt;
	vector<Point2f>ballpt;
	HoughCircles(img, cirPt, HOUGH_GRADIENT, 2, 10, 100, 40, 10, 15);
	cout << "pt.size" << endl;
	cout << cirPt.size() << endl;
	for (int i = 0; i < cirPt.size(); i++) {
		Point2f temp;
		temp.x = cirPt[i][0];
		temp.y = cirPt[i][1];
		ballpt.push_back(temp);
	}
	return ballpt;
}
vector<Point2f> board_hough(Mat img) {
	Mat src1(400, 400, CV_16UC3);
	Mat src2(400, 400, CV_16UC3);
	//vector<Vec4f>plines;
	vector<Vec2f>lines;
	vector<Point2f>kb_h;//水平的直线，k<1
	vector<Point2f>kb_v;//竖直的直线，k>1
	//HoughLinesP(img, lines, 1, CV_PI / 180, 90);

	HoughLines(img, lines, 1, CV_PI / 180, 90);
	
	for (int j = 0; j < lines.size(); j++) {
		float rho = lines[j][0], theta = lines[j][1];
		Point pt1, pt2;
		double a = cos(theta), b = sin(theta);
		double x0 = a * rho, y0 = b * rho;
		pt1.x = cvRound(x0 + 1000 * (-b));
		pt1.y = cvRound(y0 + 1000 * (a));
		pt2.x = cvRound(x0 - 1000 * (-b));
		pt2.y = cvRound(y0 - 1000 * (a));
		line(img, pt1, pt2, Scalar(255, 255, 0), 1, LINE_AA);
	}
	for (int j = 0; j < lines.size(); j++) {
		
		Point2f temp;
		if (lines[j][1] != 0) {
			double b = lines[j][0] / sin(lines[j][1]);
			double k = cos(lines[j][1]) / sin(lines[j][1]);
			temp.x = k;
			temp.y = b;
			if (abs(temp.x) < 1) {
				kb_h.push_back(temp);
			}
			else {
				kb_v.push_back(temp);
			}
		}
	}
	
	vector<Point2f>corPt;
	corPt = getPoint(kb_h, kb_v);
	Point2f one;
	Point2f two;
	Point2f three;
	Point2f four;
	for (int i = 0; i < corPt.size(); i++) {
		if (corPt[i].x < 240 && corPt[i].y < 320) {
			one = corPt[i];
		}
		else if (corPt[i].x < 240 && corPt[i].y > 320) {
			two = corPt[i];
		}
		else if (corPt[i].x > 240 && corPt[i].y < 320) {
			three = corPt[i];
		}
		else if (corPt[i].x > 240 && corPt[i].y > 320) {
			four = corPt[i];
		}
	}
	vector<Point2f>temp;
	temp.push_back(one);
	temp.push_back(two);
	temp.push_back(three);
	temp.push_back(four);
	
	return temp;
}

vector<Point2f> getPoint(vector<Point2f>kb_h, vector<Point2f>kb_v) {
	vector<Point2f>ans;
	for (int i = 0; i < kb_h.size(); i++) {
		for (int j = 0; j < kb_h.size(); j++) {
			MatrixXf A(2, 2);
			MatrixXf D(2, 1);
			MatrixXf answer(2, 1);
			A(0, 0) = kb_h[i].x;
			D(0, 0) = kb_h[i].y;
			A(0, 1) = -1;
			A(1, 0) = kb_v[j].x;
			D(1, 0) = kb_v[j].y;
			A(1, 1) = -1;
			
			answer = A.householderQr().solve(D);
			if (abs(answer(0, 0)) < 640 && abs(answer(1, 0)) < 480 && abs(answer(0, 0))>1 && abs(answer(1, 0))>1 ) {
				Point2f temp;
				temp.x = abs(answer(0, 0));
				temp.y = abs(answer(1, 0));
				ans.push_back(temp);
			}
		}
	}
	return ans;
}

Mat getImg(vector<Point2f> boardPt, vector<Point2f> ballPt, Mat img) {
	for (int i = 0; i < boardPt.size(); i++) {
		circle(img, boardPt[i], 3, Scalar(0, 255, 0), 2);
	}
	for (int i = 0; i < ballPt.size(); i++) {
		circle(img, ballPt[i], 3, Scalar(0, 255, 0), 2);
	}
	return img;
}