#include<opencv2\opencv.hpp>
#include<iostream>
#include<time.h>
using namespace std;
using namespace cv;

Mat preTreat(Mat img);
Mat img_hough(Mat img);
Mat imgMask(Mat img);
//Mat La(Mat img);
//Vec2f cornerDetect(Mat img);

int main() {
	
	vector<Point2f>corners;
	Mat img = imread("E:\\code\\python\\cv2practice\\frame\\21.png");
	imshow("first", img);
	img = preTreat(img);
	
	img = img_hough(img);
;
	goodFeaturesToTrack(img, corners, 100, 0.1, 200);
	cvtColor(img, img, COLOR_GRAY2RGB);
	for (int i = 0; i < 10; i++) {

		circle(img, corners[i],5, Scalar(255, 255, 0), 1, 5, 0);
	}
	
	imshow("done", img);
	waitKey(0);
	destroyAllWindows();
	return 0;
}

Mat imgMask(Mat img) {
	Mat img_hsv;
	Mat mask;
	Mat img_done;
	cvtColor(img, img_hsv, COLOR_RGB2HSV);
	//int hsv_low[3] = { 0,0,46 };
	//int hsv_high[3] = { 180,23,220 };
	inRange(img_hsv, Scalar(0, 0, 46), Scalar(180, 30, 220), mask);
	add(img_hsv, img_hsv, img_done, mask);
	cvtColor(img_done, img_done, COLOR_HSV2RGB);
	return img_done;
}

Mat preTreat(Mat img) {
	img = imgMask(img);
	cvtColor(img, img, COLOR_RGB2GRAY);
	//Laplacian(img, img, CV_16S, 3);
	Canny(img, img, 100, 200, 3);
	imshow("pre", img);
	return img;
}

Mat img_hough(Mat img) {
	Mat src1(400, 400, CV_16UC3);
	Mat src2(400, 400, CV_16UC3);
	//vector<Vec4f>plines;
	vector<Vec2f>lines;
	//HoughLinesP(img, lines, 1, CV_PI / 180, 90);
	
	HoughLines(img, lines, 1, CV_PI / 180, 100);
	cout << lines[0] << endl;
	cout << lines.size() << endl;
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
	imshow("hough2", img);
	return img;
}

//Vec2f cornerDetect(Mat img) {
//
//
//
//}

