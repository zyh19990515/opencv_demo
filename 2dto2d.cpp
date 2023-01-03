#include<iostream>
#include<opencv2/core/core.hpp>
#include<opencv2/features2d/features2d.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/calib3d/calib3d.hpp>

using namespace std;

void find_feature_matches(const cv::Mat& img_1, const cv::Mat& img_2, std::vector<cv::KeyPoint>& keypoints_1, std::vector<cv::KeyPoint>& keypoints_2, std::vector<cv::DMatch>& matches);
void pose_estimation_2d2d(std::vector<cv::KeyPoint> keypoints_1, std::vector<cv::KeyPoint> keypoints_2, std::vector<cv::DMatch>matches, cv::Mat& R, cv::Mat& t);

cv::Point2d pixel2cam(const cv::Point2d& p, const cv::Mat& K);

int main() {
	cv::Mat img_1 = cv::imread("E:/picture/1.png");
	cv::Mat img_2 = cv::imread("E:/picture/2.png");
	vector<cv::KeyPoint>keypoints_1, keypoints_2;
	vector<cv::DMatch>matches;
	find_feature_matches(img_1, img_2, keypoints_1, keypoints_2, matches);
	cout << "一共找到了" << matches.size() << "组匹配点" << endl;

	//估计两张图象间的运动
	cv::Mat R, t;
	pose_estimation_2d2d(keypoints_1, keypoints_2, matches, R, t);

	//验证E
	cv::Mat t_x = (cv::Mat_<double>(3, 3) <<
		0, -t.at<double>(2, 0), t.at<double>(1, 0),
		t.at<double>(2, 0), 0, -t.at<double>(0, 0),
		-t.at<double>(1, 0), t.at<double>(0, 0), 0);

	cout << "t^R=" << endl << t_x * R << endl;
	//验证对极约束
	cv::Mat K = (cv::Mat_<double>(3, 3) << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1);
	for (cv::DMatch m : matches) {
		cv::Point2d pt1 = pixel2cam(keypoints_1[m.queryIdx].pt, K);
		cv::Mat y1 = (cv::Mat_<double>(3, 1) << pt1.x, pt1.y, 1);
		cv::Point2d pt2 = pixel2cam(keypoints_2[m.trainIdx].pt, K);
		cv::Mat y2 = (cv::Mat_<double>(3, 1) << pt2.x, pt2.y, 1);
		cv::Mat d = y2.t() * t_x * R * y1;
		cout << "epipolar constraint =" << d << endl;
	}
	cv::waitKey(0);
	return 0;

}

void find_feature_matches(const cv::Mat& img_1, const cv::Mat& img_2, std::vector<cv::KeyPoint>& keypoints_1, std::vector<cv::KeyPoint>& keypoints_2, std::vector<cv::DMatch>& matches) {
	//初始化
	cv::Mat descriptors_1, descriptors_2;

	cv::Ptr<cv::FeatureDetector> detector = cv::ORB::create();
	cv::Ptr<cv::DescriptorExtractor> descriptor = cv::ORB::create();

	cv::Ptr<cv::DescriptorMatcher> match = cv::DescriptorMatcher::create("BruteForce-Hamming");
	//Oriented FAST 角点位置
	detector->detect(img_1, keypoints_1);
	detector->detect(img_2, keypoints_2);

	//计算描述子
	descriptor->compute(img_1, keypoints_1, descriptors_1);
	descriptor->compute(img_2, keypoints_2, descriptors_2);

	//使用Hamming距离匹配
	vector<cv::DMatch>match_;
	cv::BFMatcher matcher(cv::NORM_HAMMING);
	matcher.match(descriptors_1, descriptors_2, match_);

	//匹配点对筛选
	double min_dist = 10000, max_dist = 0;
	
	for (int i = 0; i < descriptors_1.rows; i++) {
		double dist = match_[i].distance;
		if (dist < min_dist) min_dist = dist;
		if (dist > max_dist) max_dist = dist;
	}
	printf("--Max dist : %f \n", max_dist);
	printf("--Min dist : %f \n", min_dist);
	for (int i = 0; i < descriptors_1.rows; i++) {
		if (match_[i].distance <= max(2 * min_dist, 20.0)) {
			matches.push_back(match_[i]);
		}
	}
}

cv::Point2d pixel2cam(const cv::Point2d& p, const cv::Mat& K) {
	return cv::Point2d((p.x - K.at<double>(0, 2)) / K.at<double>(0, 0), (p.y - K.at<double>(1, 2)) / K.at<double>(1, 1));
}

void pose_estimation_2d2d(std::vector<cv::KeyPoint> keypoints_1, std::vector<cv::KeyPoint> keypoints_2, std::vector<cv::DMatch>matches, cv::Mat& R, cv::Mat& t) {
	//相机内参
	cv::Mat K = (cv::Mat_<double>(3, 3) << 520, 9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1);

	//把匹配点转换为vector
	vector<cv::Point2f> points1;
	vector<cv::Point2f> points2;
	for (int i = 0; i < (int)matches.size(); i++) {
		points1.push_back(keypoints_1[matches[i].queryIdx].pt);
		points2.push_back(keypoints_2[matches[i].trainIdx].pt);
	}

	//计算基础矩阵
	cv::Mat fundamental_matrix;
	fundamental_matrix = cv::findFundamentalMat(points1, points2, cv::FM_8POINT);
	cout << "fundamental_matrix is" << endl << fundamental_matrix << endl;

	//计算本质矩阵
	cv::Point2d principal_point(325.1, 249.7);//相机广信
	double focal_length = 521;
	cv::Mat essential_matrix;
	essential_matrix = cv::findEssentialMat(points1, points2, focal_length, principal_point);
	cout << "essential_matrix is" << endl << essential_matrix << endl;

	//计算单应矩阵
	cv::Mat homography_matrix;
	homography_matrix = cv::findHomography(points1, points2, cv::RANSAC, 3);
	cout << "homography_matrix is" << endl << homography_matrix << endl;

	//从本质矩阵中恢复R,t
	cv::recoverPose(essential_matrix, points1, points2, R, t, focal_length, principal_point);
	cout << "R is" << endl << R << endl;
	cout << "t is" << endl << t << endl;



}