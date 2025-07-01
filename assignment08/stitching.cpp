#include "pch.h"
#include <opencv2/opencv.hpp>
//#include <opencv2/nonfree/nonfree.hpp>
#include <iostream>
#include <vector>
#include <cmath>
#include <random>

#define RATIO_THR 0.4

using namespace std;
using namespace cv;

double euclidDistance(const Mat& vec1, const Mat& vec2);
int nearestNeighbor(const Mat& vec, const vector<KeyPoint>& keypoints, const Mat& descriptors);

void findPairs(vector<KeyPoint>& keypoints1, Mat& descriptors1,
	vector<KeyPoint>& keypoints2, Mat& descriptors2,
	vector<Point2f>& srcPoints, vector<Point2f>& dstPoints, bool crossCheck, bool ratio_threshold);

template <typename T>
Mat cal_affine(vector<int> ptl_x, vector<int> ptl_y, vector<int> ptr_x, vector<int> ptr_y, int number_of_points);

void blend_stitching(const Mat I1, const Mat I2, Mat& I_f, int bound_l, int bound_u, float alpha);
Mat cal_affine_RANSAC(vector<Point2f>& srcPoints, vector<Point2f>& dstPoints, int k, int S, double delta);
void stitching(const Mat& img1, const Mat& img2, vector<Point2f>& srcPoints, vector<Point2f>& dstPoints, String estOpt);

int main() {

	Mat input1 = imread("input1.jpg", IMREAD_COLOR);
	Mat input2 = imread("input2.jpg", IMREAD_COLOR);
	Mat input1_gray, input2_gray;

	if (!input1.data || !input2.data)
	{
		std::cout << "Could not open" << std::endl;
		return -1;
	}

	//resize(input1, input1, Size(input1.cols / 2, input1.rows / 2));
	//resize(input2, input2, Size(input2.cols / 2, input2.rows / 2));

	cvtColor(input1, input1_gray, COLOR_RGB2GRAY);
	cvtColor(input2, input2_gray, COLOR_RGB2GRAY);

	Ptr<SIFT> sift = SIFT::create(
		0,		// nFeatures
		4,		// nOctaveLayers
		0.04,	// contrastThreshold
		10,		// edgeThreshold
		1.6		// sigma
	);

	// Create a image for displaying mathing keypoints
	Size size = input2.size();
	Size sz = Size(size.width + input1_gray.size().width, max(size.height, input1_gray.size().height));
	Mat matchingImage = Mat::zeros(sz, CV_8UC3);

	input1.copyTo(matchingImage(Rect(size.width, 0, input1_gray.size().width, input1_gray.size().height)));
	input2.copyTo(matchingImage(Rect(0, 0, size.width, size.height)));

	// Compute keypoints and descriptor from the source image in advance
	vector<KeyPoint> keypoints1;
	Mat descriptors1;

	sift->detectAndCompute(input1_gray, noArray(), keypoints1, descriptors1);
	printf("input1 : %d keypoints are found.\n", (int)keypoints1.size());

	vector<KeyPoint> keypoints2;
	Mat descriptors2;

	// Detect keypoints

	sift->detectAndCompute(input2_gray, noArray(), keypoints2, descriptors2);
	printf("input2: %zd keypoints are found.\n", keypoints2.size());

	for (int i = 0; i < keypoints1.size(); i++) {
		KeyPoint kp = keypoints1[i];
		kp.pt.x += size.width;
		circle(matchingImage, kp.pt, cvRound(kp.size * 0.25), Scalar(255, 255, 0), 1, 8, 0);
	}

	for (int i = 0; i < keypoints2.size(); i++) {
		KeyPoint kp = keypoints2[i];
		circle(matchingImage, kp.pt, cvRound(kp.size * 0.25), Scalar(255, 255, 0), 1, 8, 0);
	}

	// Find nearest neighbor pairs
	vector<Point2f> srcPoints;
	vector<Point2f> dstPoints;
	bool crossCheck = true;
	bool ratio_threshold = true;
	findPairs(keypoints2, descriptors2, keypoints1, descriptors1, srcPoints, dstPoints, crossCheck, ratio_threshold);
	printf("%zd keypoints are matched.\n", srcPoints.size());

	// Draw line between nearest neighbor pairs
	for (int i = 0; i < (int)srcPoints.size(); ++i) {
		Point2f pt1 = srcPoints[i];
		Point2f pt2 = dstPoints[i];
		Point2f from = pt1;
		Point2f to = Point(size.width + pt2.x, pt2.y);
		line(matchingImage, from, to, Scalar(0, 0, 255));
	}

	// Display mathing image
	namedWindow("Matching");
	imshow("Matching", matchingImage);

	stitching(input1, input2, srcPoints, dstPoints, "case2");
	
	waitKey(0);

	return 0;
}

/**
* Calculate euclid distance
*/
double euclidDistance(const Mat& vec1, const Mat& vec2) {
	double sum = 0.0;
	int dim = vec1.cols;
	for (int i = 0; i < dim; i++) {
		sum += (vec1.at<float>(0, i) - vec2.at<float>(0, i)) * (vec1.at<float>(0, i) - vec2.at<float>(0, i));
	}

	return sqrt(sum);
}

/**
* Find the index of nearest neighbor point from keypoints.
*/
int nearestNeighbor(const Mat& vec, const vector<KeyPoint>& keypoints, const Mat& descriptors) {
	int neighbor = -1;
	double minDist = 1e6;

	for (int i = 0; i < descriptors.rows; i++) {
		Mat v = descriptors.row(i);		// each row of descriptor

		double dist = euclidDistance(vec, v);
		if (dist < minDist) {	
			minDist = dist;		
			neighbor = i;	
		}
	}

	return neighbor;
}

/**
* Find pairs of points with the smallest distace between them
*/
void findPairs(vector<KeyPoint>& keypoints1, Mat& descriptors1,
	vector<KeyPoint>& keypoints2, Mat& descriptors2,
	vector<Point2f>& srcPoints, vector<Point2f>& dstPoints, bool crossCheck, bool ratio_threshold) {
	for (int i = 0; i < descriptors1.rows; i++) {
		KeyPoint pt1 = keypoints1[i];
		Mat desc1 = descriptors1.row(i);

		int nn = nearestNeighbor(desc1, keypoints2, descriptors2);

		// Refine matching points using ratio_based thresholding
		if (ratio_threshold) {

			double closest = euclidDistance(desc1, descriptors2.row(nn));
			double secondClosest = 1e6;
			for (int j = 0; j < descriptors2.rows; j++) {
				if (j == nn) {
					continue;
				}
				double temp = euclidDistance(desc1, descriptors2.row(j));
				if (temp < secondClosest) secondClosest = temp;
			}
			if (closest > RATIO_THR * secondClosest) {
				continue;
			}
		}

		// Refine matching points using cross-checking
		if (crossCheck) {
			Mat desc2 = descriptors2.row(nn);
			int rev = nearestNeighbor(desc2, keypoints1, descriptors1);
			if (rev != i) {
				continue;
			}
		}

		KeyPoint pt2 = keypoints2[nn];
		srcPoints.push_back(pt1.pt);
		dstPoints.push_back(pt2.pt);
	}
} // end of findPiars()

///////////////////////////////// 여기까지 SIFT //////////////////////////////////////

void stitching(const Mat& img1, const Mat& img2, vector<Point2f>& srcPoints, vector<Point2f>& dstPoints, String estOpt) {	// srcPoints - dstPoints 대응점이므로 affine 행렬 M 계산
	// img2를 img1에 붙이는 코드
	// srcPoints = input2 = 오른쪽 이미지 점, dstPoints = input1 = 왼쪽 이미지 점
	Mat I1, I2;
	img1.convertTo(I1, CV_32FC3, 1.0 / 255);
	img2.convertTo(I2, CV_32FC3, 1.0 / 255);

	const float I1_row = I1.rows;
	const float I1_col = I1.cols;
	const float I2_row = I2.rows;
	const float I2_col = I2.cols;

	Mat M12, M21;
	
	if (estOpt == "case1") {	// 단순 Mx=b, outlier 포함한 벡터 전체로 계산하므로 성능 안 좋음
		int src_len = (int)srcPoints.size();	// 배열 크기
		int dst_len = (int)dstPoints.size();

		vector<int> ptl_x(dst_len), ptl_y(dst_len), ptr_x(src_len), ptr_y(src_len);
		for (int i = 0; i < dst_len; i++) {
			ptl_x[i] = (int)dstPoints[i].x;
			ptl_y[i] = (int)dstPoints[i].y;
		}
		for (int i = 0; i < src_len; i++) {
			ptr_x[i] = (int)srcPoints[i].x;
			ptr_y[i] = (int)srcPoints[i].y;
		}

		// M = cal_affine<float>(src_x, src_y, dst_x, dst_y, src_len);	// I1 -> I2 affine 행렬
		M12 = cal_affine<float>(ptl_x, ptl_y, ptr_x, ptr_y, src_len);	// I1 -> I2 affine 행렬
		M21 = cal_affine<float>(ptr_x, ptr_y, ptl_x, ptl_y, dst_len);	// I2 -> I1 affine 행렬

	}
	else {	// Mx=b + RANSAC, outlier 배제하므로 성능 좋음
		double k = 4;		// 샘플 개수
		double S = 20000;	// 반복 횟수
		double delta = 3;	// 임계값
		M12 = cal_affine_RANSAC(dstPoints, srcPoints, k, S, delta);
		M21 = cal_affine_RANSAC(srcPoints, dstPoints, k, S, delta);
	}

	// compute corners (p1, p2, p3, p4)
	// p1: (0,0)
	// p2: (row, 0)
	// p3: (row, col)
	// p4: (0, col)
	Point2f p1(M21.at<float>(0) * 0 + M21.at<float>(1) * 0 + M21.at<float>(2), M21.at<float>(3) * 0 + M21.at<float>(4) * 0 + M21.at<float>(5));				// top left
	Point2f p2(M21.at<float>(0) * 0 + M21.at<float>(1) * I2_row + M21.at<float>(2), M21.at<float>(3) * 0 + M21.at<float>(4) * I2_row + M21.at<float>(5));	// bottom left
	Point2f p3(M21.at<float>(0) * I2_col + M21.at<float>(1) * I2_row + M21.at<float>(2), M21.at<float>(3) * I2_col + M21.at<float>(4) * I2_row + M21.at<float>(5));	// bottom right
	Point2f p4(M21.at<float>(0) * I2_col + M21.at<float>(1) * 0 + M21.at<float>(2), M21.at<float>(3) * I2_col + M21.at<float>(4) * 0 + M21.at<float>(5));	// top right

	// compute boundary for merged image(I_f)
	// bound_u <= 0
	// bound_b >= I1_row-1
	// bound_l <= 0
	// bound_b >= I1_col-1
	int bound_u = (int)round(min(0.0f, min(p1.y, p4.y)));
	int bound_b = (int)round(max(I1_row - 1, max(p2.y, p3.y)));
	int bound_l = (int)round(min(0.0f, min(p1.x, p2.x)));
	int bound_r = (int)round(max(I1_col - 1, max(p3.x, p4.x)));

	// initialize merged image
	Mat I_f(bound_b - bound_u + 1, bound_r - bound_l + 1, CV_32FC3, Scalar(0));

	// inverse warping with bilinear interplolation
	for (int i = bound_u; i <= bound_b; i++) {
		for (int j = bound_l; j <= bound_r; j++) {
			float x = M12.at<float>(0) * j + M12.at<float>(1) * i + M12.at<float>(2) - bound_l;
			float y = M12.at<float>(3) * j + M12.at<float>(4) * i + M12.at<float>(5) - bound_u;

			float y1 = floor(y);
			float y2 = ceil(y);
			float x1 = floor(x);
			float x2 = ceil(x);

			float mu = y - y1;
			float lambda = x - x1;

			if (x1 >= 0 && x2 < I2_col && y1 >= 0 && y2 < I2_row)
				I_f.at<Vec3f>(i - bound_u, j - bound_l) = lambda * (mu * I2.at<Vec3f>(y2, x2) + (1 - mu) * I2.at<Vec3f>(y1, x2)) +
				(1 - lambda) * (mu * I2.at<Vec3f>(y2, x1) + (1 - mu) * I2.at<Vec3f>(y1, x1));
		}
	}

	// image stitching with blend
	blend_stitching(I1, I2, I_f, bound_l, bound_u, 0.5);

	namedWindow("Left Image");
	imshow("Left Image", I1);

	namedWindow("Right Image");
	imshow("Right Image", I2);

	I_f.convertTo(I_f, CV_8UC3, 255.0);
	namedWindow("result");
	imshow("result", I_f);

	// I_f.convertTo(I_f, CV_8UC3, 255.0);
	// imwrite("result.png", I_f);
}

template <typename T>
Mat cal_affine(vector<int> ptl_x, vector<int> ptl_y, vector<int> ptr_x, vector<int> ptr_y, int number_of_points) {

	Mat M(2 * number_of_points, 6, CV_32F, Scalar(0));
	Mat b(2 * number_of_points, 1, CV_32F);

	Mat M_trans, temp, affineM;

	// initialize matrix
	for (int i = 0; i < number_of_points; i++) {
		M.at<T>(2 * i, 0) = ptl_x[i];		M.at<T>(2 * i, 1) = ptl_y[i];		M.at<T>(2 * i, 2) = 1;
		M.at<T>(2 * i + 1, 3) = ptl_x[i];		M.at<T>(2 * i + 1, 4) = ptl_y[i];		M.at<T>(2 * i + 1, 5) = 1;
		b.at<T>(2 * i) = ptr_x[i];		b.at<T>(2 * i + 1) = ptr_y[i];
	}

	// (M^T * M)^(−1) * M^T * b ( * : Matrix multiplication)
	transpose(M, M_trans);
	invert(M_trans * M, temp);
	affineM = temp * M_trans * b;

	return affineM;
}

void blend_stitching(const Mat I1, const Mat I2, Mat& I_f, int bound_l, int bound_u, float alpha) {

	int col = I_f.cols;
	int row = I_f.rows;

	// I2 is already in I_f by inverse warping
	for (int i = 0; i < I1.rows; i++) {
		for (int j = 0; j < I1.cols; j++) {
			bool cond_I2 = I_f.at<Vec3f>(i - bound_u, j - bound_l) != Vec3f(0, 0, 0) ? true : false;	// I2가 있는 픽셀인지

			if (cond_I2)	// I2 있으면 I1, I2 가중치 적용하여 blending
				I_f.at<Vec3f>(i - bound_u, j - bound_l) = alpha * I1.at<Vec3f>(i, j) + (1 - alpha) * I_f.at<Vec3f>(i - bound_u, j - bound_l);
			else	// I1만 있으면 I1 그대로 사용
				I_f.at<Vec3f>(i - bound_u, j - bound_l) = I1.at<Vec3f>(i, j);

		}
	}
}

Mat cal_affine_RANSAC(vector<Point2f>& srcPoints, vector<Point2f>& dstPoints, int k, int S, double delta) {
	/*
	1. 랜덤하게 샘플 데이터 k개 선택
	2. k개의 대응점으로 Affine 행렬 M 계산
	3. M을 이용하여 나머지 대응점에 대한 변환된 점 계산 -> abs(Mp-p')^2<ε^2 인 점을 inlier로 판단
	4. 1~3을 S번 반복
	5. inlier가 가장 많은 Affine 행렬 M을 선택
	6. inlier에 대해 Affine 행렬 M을 다시 계산
	*/

	int srcSize = (int)srcPoints.size();	// srcPoints 크기

	int maxInliers = 0;				// 최대 inlier 개수
	vector<Point2f> bestSrcInliers, bestDstInliers;	// 최적일 때 inlier

	for (int i = 0; i < S; i++) {	// 1~3을 S번 반복

		// 1. 랜덤 대응점 k개 선택
		vector<int> sample_index;	// 샘플 인덱스 벡터
		while (sample_index.size() < k) {	// k개 
			int randIndex = rand() % srcSize;	// 랜덤
			if (find(sample_index.begin(), sample_index.end(), randIndex) == sample_index.end()) {	// 중복 체크
				sample_index.push_back(randIndex);
			}
		}

		vector<Point2f> srcPoints_sample, dstPoints_sample;	// 실제 샘플 데이터 벡터
		for (int j = 0; j < k; j++) {
			srcPoints_sample.push_back(srcPoints[sample_index[j]]);
			dstPoints_sample.push_back(dstPoints[sample_index[j]]);
		}

		// 2. 1로 Affine 행렬 계산
		vector<int> src_x(k), src_y(k), dst_x(k), dst_y(k);	// cal_affine에 전달한 x, y 분리된 벡터
		for (int j = 0; j < k; j++) {
			src_x[j] = (int)srcPoints_sample[j].x;	// srcPoints 샘플
			src_y[j] = (int)srcPoints_sample[j].y;
			dst_x[j] = (int)dstPoints_sample[j].x;	// dstPoints 샘플
			dst_y[j] = (int)dstPoints_sample[j].y;
		}
		Mat Msd = cal_affine<float>(src_x, src_y, dst_x, dst_y, k);	// I1 -> I2 affine 행렬

		// 3. inlier 카운트
		int count = 0;
		vector<Point2f> srcInliers, dstInliers;

		for (int j = 0; j < srcSize; j++) {
			float x = Msd.at<float>(0) * srcPoints[j].x + Msd.at<float>(1) * srcPoints[j].y + Msd.at<float>(2);
			float y = Msd.at<float>(3) * srcPoints[j].x + Msd.at<float>(4) * srcPoints[j].y + Msd.at<float>(5);
			float dist = (x - dstPoints[j].x) * (x - dstPoints[j].x) + (y - dstPoints[j].y) * (y - dstPoints[j].y);	// 얘가 제곱한 거
			
			if (dist < delta * delta) {
				count++;
				srcInliers.push_back(srcPoints[j]);	// inlier에 srcPoints 넣기
				dstInliers.push_back(dstPoints[j]);	// inlier에 dstPoints 넣기
			}
		}
		if (count > maxInliers) {	// 더 최적인 것 나왔을 때 갱신
			maxInliers = count;
			bestSrcInliers = srcInliers;
			bestDstInliers = dstInliers;
		} // 종료되면 bestInliers에 최적 affine 행렬에 대한 inlier들 있음
	} // end of for(0~S)
	
	// 5. bestInliers에 대해 Affine 행렬 M을 다시 계산
	int bestInlierSize = (int)bestSrcInliers.size();

	// bestliniers 원소 개수, 원소 출력
	/*cout << "bestInlierSize: " << bestInlierSize << endl;
	for (int j = 0; j < bestInlierSize; j++) {
		cout << "src: " << bestSrcInliers[j] << ", dst: " << bestDstInliers[j] << endl;
	}*/

	vector<int> best_src_x(bestInlierSize);
	vector<int> best_src_y(bestInlierSize);
	for (int j = 0; j < bestInlierSize; j++) {
		best_src_x[j] = (int)bestSrcInliers[j].x;
		best_src_y[j] = (int)bestSrcInliers[j].y;
	}
	vector<int> best_dst_x(bestInlierSize);
	vector<int> best_dst_y(bestInlierSize);
	for (int j = 0; j < bestInlierSize; j++) {
		best_dst_x[j] = (int)bestDstInliers[j].x;
		best_dst_y[j] = (int)bestDstInliers[j].y;
	}
	Mat best_M = cal_affine<float>(best_src_x, best_src_y, best_dst_x, best_dst_y, bestInlierSize);	// 최종 affine 행렬

	return best_M;
}
