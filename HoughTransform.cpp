#include<iostream>
#include<opencv2/opencv.hpp>
#include<vector>
using namespace std;

int main() {
	cv::Mat img;
	img = cv::imread("E:\\code\\python\\cv2practice\\yuan.png");
	cv::cvtColor(img, img, cv::COLOR_RGB2GRAY);
	vector<cv::Vec3f>pcircles;
	cv::HoughCircles(img, pcircles, cv::HOUGH_GRADIENT, 1, 10, 100, 36, 0, 10000);
	cout << pcircles.size()  << endl;
	cv::Point2f pt;
	for (auto i : pcircles) {
		cout << i << endl;
		pt.x = int(i[0]);
		pt.y = int(i[1]);
	}
	cout << pt.x << endl;
	cv::circle(img, pt, 5, cv::Scalar(60), 2);
	cv::imshow("1213", img);
	system("pause");
	return 0;
}