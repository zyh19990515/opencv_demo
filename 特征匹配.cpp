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

	//��ʼ�� 
	vector<KeyPoint>keypoints_1, keypoints_2;
	Mat descriptors_1, descriptors_2;
	Ptr<ORB>orb = ORB::create(500, 1.2f, 8, 31, 0, 2, ORB::HARRIS_SCORE, 31, 20);
	 
	orb->detect(img_1, keypoints_1);
	orb->detect(img_2, keypoints_2);

	//2.���ݽǵ�λ�ü���BRIEF������
	orb->compute(img_1, keypoints_1, descriptors_1);
	orb->compute(img_2, keypoints_2, descriptors_2);

	Mat outimg1;
	drawKeypoints(img_1, keypoints_1, outimg1, Scalar::all(-1), DrawMatchesFlags::DEFAULT);
	//imshow("ORB������", outimg1);

	//3.������ͼ���е�BRIEF�����ӽ���ƥ�䣬ʹ��Hamming����
	vector<DMatch>matches;
	BFMatcher matcher(NORM_HAMMING);
	matcher.match(descriptors_1, descriptors_2, matches);

	//4.ƥ����ɸѡ
	double min_dist = 10000, max_dist = 0;

	//�ҳ�����ƥ��֮�����С����������룬�������Ƶĺ�����Ƶ������֮��ľ���
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

	//��������֮��ľ��������������С����ʱ������Ϊƥ������
	//����ʱ����С�����ǳ�С������һ������ֵ��Ϊ����
	vector<DMatch>good_matches;
	for (int i = 0; i < descriptors_1.rows; i++) {
		if (matches[i].distance <= max(1 * min_dist, 20.0)) {
			good_matches.push_back(matches[i]);
		}
	}
	end = clock();
	cout << end - start << endl;
	//5.����ƥ����
	Mat img_match;
	Mat img_goodmatch;
	drawMatches(img_1, keypoints_1, img_2, keypoints_2, matches, img_match);
	drawMatches(img_1, keypoints_1, img_2, keypoints_2, good_matches, img_goodmatch);
	imshow("����ƥ����", img_match);
	imshow("�Ż���ƥ����", img_goodmatch);


	waitKey(0);
	system("pause");
	return 0;
}