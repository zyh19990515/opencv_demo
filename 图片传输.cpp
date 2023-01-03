/*
	read a picture,img[][];
	add head and tail(char h and char t);
	detect char h and transmit to img2;
*/

#include<opencv2\opencv.hpp>
#include<iostream>

using namespace cv;
using namespace std;

int main() {
	Mat img = imread("D:\\1.jpg");
	imshow("1", img);
	
	


	waitKey(0);
	system("pause");
}