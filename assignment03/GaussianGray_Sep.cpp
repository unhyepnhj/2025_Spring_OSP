#include <iostream>
#include <opencv2/opencv.hpp>
#include <math.h>       /* exp */
#define IM_TYPE	CV_8UC3

using namespace cv;

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

Mat gaussianfilter(const Mat input, int n, float sigmaT, float sigmaS, const char* opt);

int main() {

	Mat input = imread("lena.jpg", IMREAD_COLOR);	// IMREAD_COLOR
	Mat input_gray;
	Mat output;


	cvtColor(input, input_gray, COLOR_RGB2GRAY);	// COLOR_RGB2GRAY



	if (!input.data)
	{
		std::cout << "Could not open" << std::endl;
		return -1;
	}

	namedWindow("Grayscale", WINDOW_AUTOSIZE);
	imshow("Grayscale", input_gray);

	// output = gaussianfilter(input_gray, 1, 1, 1, "zero-paddle"); //Boundary process: zero-paddle, mirroring, adjustkernel

	output = gaussianfilter(input_gray, 1, 1, 1, "mirroring"); //Boundary process: zero-paddle, mirroring, adjustkernel
	// output = gaussianfilter(input_gray, 1,1,1, "adjustkernel"); //Boundary process: zero-paddle, mirroring, adjustkernel

	namedWindow("Gaussian Filter", WINDOW_AUTOSIZE);
	imshow("Gaussian Filter", output);
	waitKey(0);

	return 0;
}

Mat gaussianfilter(const Mat input, int n, float sigmaT, float sigmaS, const char* opt) {

	int row = input.rows;
	int col = input.cols;
	int kernel_size = (2 * n + 1);
	int tempa;
	int tempb;
	float denomT, denomS;
	// float kernelvalue;

 // Initialiazing Kernel Matrix 
	Mat kernelT = Mat::zeros(kernel_size, 1, CV_32F);	// N*1
	Mat kernelS = Mat::zeros(1, kernel_size, CV_32F);	// 1*N

	denomT = 0.0;
	for (int i = -n; i <= n; i++) {
		float value1 = exp(-(pow(i, 2) / (2 * pow(sigmaT, 2))));
		kernelT.at<float>(i + n, 0) = value1;	
		denomT += value1;
	}
	denomS = 0.0;
	for (int i = -n; i <= n; i++) {
		float value1 = exp(-(pow(i, 2) / (2 * pow(sigmaS, 2))));
		kernelS.at<float>(0, i + n) = value1;
		denomS += value1;
	}
	//normalization
	for (int i = -n; i <= n; i++) {
		kernelT.at<float>(i + n, 0) /= denomT;
		kernelS.at<float>(0, i + n) /= denomS;
	}
	/*
	for (int a = -n; a <= n; a++) {  // Denominator in m(s,t)
		for (int b = -n; b <= n; b++) {
			kernel.at<float>(a+n, b+n) /= denom;
		}
	}*/

	///////////////////////// N*1 /////////////////////////

	Mat temp = Mat::zeros(row, col, input.type());	// N*1 결과 저장

	for (int i = 0; i < row; i++) {
		for (int j = 0; j < col; j++) {

			if (!strcmp(opt, "zero-paddle")) {
				float sum1 = 0.0;
				for (int b = -n; b <= n; b++) {	// 수평
					if ((j + b <= col - 1) && (j + b >= 0)) {
						sum1 += kernelT.at<float>(b + n, 0) * input.at<G>(i, j + b);
					}
				}
				temp.at<G>(i, j) = (G)sum1;
			}

			else if (!strcmp(opt, "mirroring")) {
				float sum1 = 0.0;
				for (int b = -n; b <= n; b++) {
					if (j + b > col - 1) {
						tempb = j - b;
					}
					else if (j + b < 0) {
						tempb = -(j + b);
					}
					else {
						tempb = j + b;
					}
					sum1 += kernelT.at<float>(b + n, 0) * input.at<G>(i, tempb);
				}
				temp.at<G>(i, j) = (G)sum1;
			}
			else if (!strcmp(opt, "adjustkernel")) {
				float sum1 = 0.0;
				float sum2 = 0.0;
				for (int b = -n; b <= n; b++) {

					if ((j + b <= col - 1) && (j + b >= 0)) {
						sum1 += kernelT.at<float>(b + n, 0) * input.at<G>(i, j + b);
						sum2 += kernelT.at<float>(b + n, 0);
					}
				}
				temp.at<G>(i, j) = (G)(sum1 / sum2);
			}
		}
	}// end of for

	///////////////////////// 1*N /////////////////////////
	Mat output = Mat::zeros(row, col, input.type());


	for (int i = 0; i < row; i++) {
		for (int j = 0; j < col; j++) {

			if (!strcmp(opt, "zero-paddle")) {
				float sum1 = 0.0;
				for (int a = -n; a <= n; a++) {	// 수직
					if ((i + a <= row - 1) && (i + a >= 0)) {
						sum1 += kernelS.at<float>(0, a + n) * temp.at<G>(i + a, j);
					}
				}
				output.at<G>(i, j) = (G)sum1;
			}

			else if (!strcmp(opt, "mirroring")) {
				float sum1 = 0.0;
				for (int a = -n; a <= n; a++) {
					if (i + a > row - 1) {
						tempa = i - a;
					}
					else if (i + a < 0) {
						tempa = -(i + a);
					}
					else {
						tempa = i + a;
					}

					sum1 += kernelS.at<float>(0, a + n) * temp.at<G>(tempa, j);
				}
				output.at<G>(i, j) = (G)sum1;
			}


			else if (!strcmp(opt, "adjustkernel")) {
				float sum1 = 0.0;
				float sum2 = 0.0;
				for (int a = -n; a <= n; a++) {
					if ((i + a <= row - 1) && (i + a >= 0)) {
						sum1 += kernelS.at<float>(0, a + n) * temp.at<G>(i + a, j);
						sum2 += kernelS.at<float>(0, a + n);
					}
				}
				output.at<G>(i, j) = (G)(sum1 / sum2);
			}
		}
	}
	return output;
}