// opencv_test.cpp : Defines the entry point for the console application.
//
#include "pch.h"
//#include "stdafx.h"
#include <opencv2/opencv.hpp>
#include "opencv2/highgui.hpp"
#include <iostream>
#include <string>

using namespace cv;
using namespace std;

void getSegment(const Mat image, const Vec2f& lines, int minLineLength, int maxLineGap, Mat& colorDst);

int main() {

	Mat src = imread("building.jpg", IMREAD_COLOR);
	Mat dst, color_dst;

	// check for validation
	if (!src.data) {
		printf("Could not open\n");
		return -1;
	}

	Canny(src, dst, 50, 200, 3);
	cvtColor(dst, color_dst, COLOR_GRAY2BGR);
	// Mat color_dst_test = color_dst.clone();	// 수동 segmentation 테스트용

	//Standard Hough transform (using 'HoughLines')
#if 1
	vector<Vec2f> lines;
	//Fill this line

	HoughLines(dst, lines, 1, CV_PI / 180, 180);

	for (size_t i = 0; i < lines.size(); i++)
	{
		float rho = lines[i][0];
		float theta = lines[i][1];
		double a = cos(theta), b = sin(theta);
		double x0 = a * rho, y0 = b * rho;
		Point pt1(cvRound(x0 + 1000 * (-b)), cvRound(y0 + 1000 * (a)));
		Point pt2(cvRound(x0 - 1000 * (-b)), cvRound(y0 - 1000 * (a)));
		line(color_dst, pt1, pt2, Scalar(0, 0, 255), 3, 8);
	}

	//if (true) {	// 수동 segmentation 테스트
	//	for (size_t i = 0; i < lines.size(); i++)
	//	{
	//		getSegment(dst, lines[i], 20, 50, color_dst_test);
	//	}
	//	namedWindow("Detected Lines 2", 1);
	//	imshow("Detected Lines 2", color_dst_test);

	//}
	//Probabilistic Hough transform (using 'HoughLinesP')
#else
	vector<Vec4i> lines;
	//Fill this line
	HoughLinesP(dst, lines, 1, CV_PI / 180, 50, 50, 10);

	for (size_t i = 0; i < lines.size(); i++)
	{
		line(color_dst, Point(lines[i][0], lines[i][1]),
			Point(lines[i][2], lines[i][3]), Scalar(0, 0, 255), 3, 8);
	}
#endif
	namedWindow("Source", 1);
	imshow("Source", src);
	namedWindow("Detected Lines", 1);
	imshow("Detected Lines", color_dst);
	waitKey(0);

	return 0;
}

void getSegment(const Mat image, const Vec2f& lines, int minLineLength, int maxLineGap, Mat& colorDst) {

	Mat output = image.clone();	// 결과 이미지

	float rho = lines[0];
	float theta = lines[1];
	double a = cos(theta), b = sin(theta);
	double x0 = a * rho, y0 = b * rho;
	Point pt1(cvRound(x0 + 1000 * (-b)), cvRound(y0 + 1000 * (a)));
	Point pt2(cvRound(x0 - 1000 * (-b)), cvRound(y0 - 1000 * (a)));

	LineIterator it(image, pt1, pt2, 8);	// LineIterator: 직선 내 픽셀 순회
	int gap = 0;	// gap count
	int count = 0;	// line segment count
	Point start;

	for (int i = 0; i < it.count; i++, ++it) {
		Point p = it.pos();	// 현재 픽셀 위치
		if (p.x < 0 || p.x >= image.cols || p.y < 0 || p.y >= image.rows) {
			continue;	// 이미지 범위 밖이면 skip
		}

		bool isEdge = image.at<uchar>(p) == 255;	// 에지 픽셀인지

		if (isEdge) {	// 에지 픽셀이라면
			if (count == 0) {	// line segment 첫 픽셀이라면
				start = p;
			}
			gap++;
			count++;
		}
		// 에지 픽셀이 아닌데
		else if (count > 0) {	// line segment가 존재한다면
			gap++;	// gap count 증가
			if (gap == maxLineGap) {	// gap이 maxLineGap과 같으면 line segment가 아님
				if (count >= minLineLength) {	// line segment가 존재하고 길이가 minLineLength보다 크면 -> line segment 끝내야 함
					Point end = Point(
						it.pos().x - cvRound(gap * (-b)),
						it.pos().y - cvRound(gap * (a))
					);
					line(colorDst, start, end, Scalar(0, 0, 255), 3, 8);
				}
				count = 0;	// line segment count 초기화
				gap = 0;	// gap count 초기화

			}
		}
	}// end of iteration
	// 마지막 segment
	if (count >= minLineLength) {
		Point end = it.pos();
		line(colorDst, start, end, Scalar(0, 0, 255), 3, 8);
	}
}
