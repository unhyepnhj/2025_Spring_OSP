#include <iostream>
#include <opencv2/opencv.hpp>
#include <math.h>       /* exp */
#include<iostream>
#include<tuple> // for tuple
#define IM_TYPE	CV_8UC3
#define L 256		// # of intensity levels

using namespace cv;
using namespace std;

// Image Type
// "G" for GrayScale Image, "C" for Color Image
#if (IM_TYPE == CV_8UC3)
typedef uchar G;
typedef cv::Vec3b C;
#elif (IM_TYPE == CV_16SC3)
typedef short G;
typedef Vec3s C;
#elif (IM_TYPE == CV_32SC3)
typedef int G;
typedef Vec3i C;
#elif (IM_TYPE == CV_32FC3)
typedef float G;
typedef Vec3f C;
#elif (IM_TYPE == CV_64FC3)
typedef double G;
typedef Vec3d C;
#endif

tuple<float, Mat> otsu_gray_seg(const Mat input);

int main() {

	Mat input = imread("lena.jpg", IMREAD_COLOR);
	Mat input_gray;
	Mat output;
	float t;

	cvtColor(input, input_gray, COLOR_RGB2GRAY);



	if (!input.data)
	{
		std::cout << "Could not open" << std::endl;
		return -1;
	}

	namedWindow("Grayscale", WINDOW_AUTOSIZE);
	imshow("Grayscale", input_gray);
	tie(t, output) = otsu_gray_seg(input_gray);

	namedWindow("Otsu", WINDOW_AUTOSIZE);
	imshow("Otsu", output);
	std::cout << t << std::endl;

	//// plot histogram /////////////////////////////////////////////////////////////////
	//Mat hist;
	//int histSize = L;
	//float range[] = { 0, (float)L };
	//const float* histRange = { range };
	//calcHist(&input_gray, 1, 0, Mat(), hist, 1, &histSize, &histRange);

	//int hist_w = 512, hist_h = 400;
	//int bin_w = cvRound((double)hist_w / histSize);
	//Mat histImage(hist_h, hist_w, CV_8UC3, Scalar(255, 255, 255));
	//normalize(hist, hist, 0, histImage.rows, NORM_MINMAX);

	//for (int i = 1; i < histSize; i++) {
	//	line(histImage,
	//		Point(bin_w * (i - 1), hist_h - cvRound(hist.at<float>(i - 1))),
	//		Point(bin_w * i, hist_h - cvRound(hist.at<float>(i))),
	//		Scalar(0, 0, 0), 2, 8, 0);
	//}
	//// T 표시
	//int thresh_x = cvRound(t * bin_w);
	//line(histImage,
	//	Point(thresh_x, 0),
	//	Point(thresh_x, hist_h),
	//	Scalar(0, 0, 255), 2, 8, 0);

	//namedWindow("Histogram", WINDOW_AUTOSIZE);
	//imshow("Histogram", histImage);

	waitKey(0);

	return 0;
}


tuple<float, Mat> otsu_gray_seg(const Mat input) {

	int row = input.rows;
	int col = input.cols;
	Mat output = Mat::zeros(row, col, input.type());
	int n = row*col;
	float T = 0, var = 0, var_max = 0, sum = 0, sumB = 0, q1 = 0, q2 = 0, sigma1 = 0, sigma2 = 0;
	int count = 0;	//추가 변수
	int histogram[L] = { 0 };  // initializing histogram values


	for (int i = 0; i < input.rows; i++) {
		for (int j = 0; j < input.cols; j++) {   // finding histogram of the image
			histogram[input.at<G>(i, j)]++;
		}
	}


	for (int i = 0; i < L; i++) {     //auxiliary value for computing mean value
		//Fill code
		sum += i * histogram[i];
	}
	for (int t = 0; t < L; t++) {  //update q
		//Fill code		
		sumB += t * histogram[t];
		count += histogram[t];	// C1 픽셀 수

		if (count != 0 && count != n) {
			float q1 = (float)count / n;// C1 픽셀 수 / 전체 픽셀 수 -> sum(pdf(i))
			float q2 = 1.0f - q1;		// q1 + q2 = 1
			float m1 = sumB / count;	// m1 = sum(i*pdf(i))/q1 = sum(i*hist(i))*(1/n)/count*(1/n) = sumB/count
			float m2 = (sum - sumB) / (n - count);	// C2 범위에서 m1과 동일하게

			var = q1 * q2 * (m1 - m2) * (m1 - m2);

			if (var > var_max) {
				T = t; //threshold
				var_max = var;
			}
		}
	}


	///*
	//Fill code that makes output image's pixel intensity to 255 if the intensity of the input image is bigger
	// than the threshold value else 0.
	for (int i = 0; i < row; ++i) {
		for (int j = 0; j < col; ++j) {
			if (input.at<G>(i, j) > T) {
				output.at<G>(i, j) = 255;
			}
			else {
				output.at<G>(i, j) = 0;
			}
		}
	}

	return make_tuple(T, output);
}
