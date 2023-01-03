#include<iostream>
#include<opencv2/opencv.hpp>
#include<opencv2/core/core.hpp>
#include<opencv2/features2d/features2d.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<vector>
#include<time.h>
using namespace std;
using namespace cv;

int main() {
	clock_t start, end;
	start = clock();

	Mat img_1 = imread("E:/code/python/cv2practice/picture/ball.jpg");
	Mat img_2 = imread("E:/code/python/cv2practice/picture/ball1.jpg");
	Size dsize(640, 480);
	resize(img_1, img_1, dsize, 0, 0, INTER_AREA);
	resize(img_2, img_2, dsize, 0, 0, INTER_AREA);

	//初始化 
	vector<KeyPoint>keypoints_1, keypoints_2;
	Mat descriptors_1, descriptors_2;
	Ptr<ORB>orb = ORB::create(500, 1.2f, 8, 31, 0, 2, ORB::HARRIS_SCORE, 31, 20);
	 
	orb->detect(img_1, keypoints_1);
	orb->detect(img_2, keypoints_2);

	//2.根据角点位置计算BRIEF描述子
	orb->compute(img_1, keypoints_1, descriptors_1);
	orb->compute(img_2, keypoints_2, descriptors_2);

	Mat outimg1;
	drawKeypoints(img_1, keypoints_1, outimg1, Scalar::all(-1), DrawMatchesFlags::DEFAULT);
	//imshow("ORB特征点", outimg1);

	//3.对两幅图像中的BRIEF描述子进行匹配，使用Hamming距离
	vector<DMatch>matches;
	BFMatcher matcher(NORM_HAMMING);
	matcher.match(descriptors_1, descriptors_2, matches);

	//4.匹配点对筛选
	double min_dist = 10000, max_dist = 0;

	//找出所有匹配之间的最小距离和最大距离，即最相似的和最不相似的两组点之间的距离
	for (int i = 0; i < descriptors_1.rows; i++) {
		double dist = matches[i].distance;
		if (dist < min_dist) {
			min_dist = dist;
		}
		if (dist > max_dist) {
			max_dist = dist;
		}
	}
	printf("--Max dist:%f\n", max_dist);
	printf("--Min dist:%f\n", min_dist);

	//当描述子之间的距离大于两倍的最小距离时，即认为匹配有误
	//但有时候最小距离会非常小，设置一个经验值作为下限
	vector<DMatch>good_matches;
	for (int i = 0; i < descriptors_1.rows; i++) {
		if (matches[i].distance <= max(1 * min_dist, 20.0)) {
			good_matches.push_back(matches[i]);
		}
	}
	end = clock();
	cout << end - start << endl;
	//5.绘制匹配结果
	Mat img_match;
	Mat img_goodmatch;
	drawMatches(img_1, keypoints_1, img_2, keypoints_2, matches, img_match);
	drawMatches(img_1, keypoints_1, img_2, keypoints_2, good_matches, img_goodmatch);
	imshow("所有匹配点对", img_match);
	imshow("优化后匹配点对", img_goodmatch);


	waitKey(0);
	system("pause");
	return 0;
}