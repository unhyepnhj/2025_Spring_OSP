// opencv_test.cpp : Defines the entry point for the console application.
//
//#include "pch.h"
#include <opencv2/opencv.hpp>
#include "opencv2/highgui.hpp"
#include <iostream>
using namespace cv;
using namespace std;

int main() {

	Mat input = imread("lena.jpg", IMREAD_COLOR);
	Mat input_gray, output;

	// check for validation
	if (!input.data) {
		printf("Could not open\n");
		return -1;
	}

	cvtColor(input, input_gray, COLOR_RGB2GRAY);	// convert RGB to Grayscale
	input_gray.convertTo(input_gray, CV_64F, 1.0 / 255);	// 8-bit unsigned char -> 64-bit floating point
	
	//Fill the code using 'Canny' in OpenCV.
	input_gray.convertTo(input_gray, CV_8U, 255.0);	// 64-bit floating point -> 8-bit unsigned char

	double threshold1 = 50.0;   // T_L
	double threshold2 = 150.0;  // T_H
	int    apertureSize = 3;    // Sobel 필터 크기
	bool   L2gradient = false;  // L2 계산할지??

	Canny(input_gray, output, threshold1, threshold2, apertureSize, L2gradient);


	namedWindow("Grayscale", WINDOW_AUTOSIZE);
	imshow("Grayscale", input_gray);

	namedWindow("Canny", WINDOW_AUTOSIZE);
	imshow("Canny", output);

	waitKey(0);

	return 0;
}