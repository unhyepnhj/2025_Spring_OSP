#include <iostream>
#include <opencv2/opencv.hpp>
#include <math.h>       /* exp */
#define IM_TYPE	CV_8UC3

using namespace cv;

// Image Type
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
Mat unsharpmask(const Mat input, int n, float sigmaT, float sigmaS, float k, const char* opt);

int main() {

	Mat input = imread("lena.jpg", IMREAD_COLOR);
	Mat output;

	if (!input.data)
	{
		std::cout << "Could not open" << std::endl;
		return -1;
	}


	// Unsharp masking
	float k = 0.8f;
	output = unsharpmask(input, 3, 3, 3, k, "mirroring");

	namedWindow("Original", WINDOW_AUTOSIZE);
	imshow("Original", input);

	namedWindow("Unsharp Masking", WINDOW_AUTOSIZE);
	imshow("Unsharp Masking", output);

	waitKey(0);

	return 0;
}

Mat unsharpmask(const Mat input, int n, float sigmaT, float sigmaS, float k, const char* opt) {
	// low-pass filter로 gaussian 사용
	Mat temp = gaussianfilter(input, n, sigmaT, sigmaS, opt);
	namedWindow("Gaussian Filter", WINDOW_AUTOSIZE);
	imshow("Gaussian Filter", temp);


	Mat output = Mat::zeros(input.size(), input.type());

	int row = input.rows;
	int col = input.cols;

	for (int i = 0; i < row; i++) {
		for (int j = 0; j < col; j++) {
			// (I - kL) / (1 - k)
			float o_r = ((float)input.at<C>(i, j)[0] - k * (float)temp.at<C>(i, j)[0]) / (1.0f - k);
			if (o_r < 0) { o_r = 0; }
			if (o_r > 255) { o_r = 255; }
			output.at<C>(i, j)[0] = (G)(o_r);

			float o_g = ((float)input.at<C>(i, j)[1] - k * (float)temp.at<C>(i, j)[1]) / (1.0f - k);
			if (o_g < 0) { o_g = 0; }
			if (o_g > 255) { o_g = 255; }
			output.at<C>(i, j)[1] = (G)(o_g);

			float o_b = ((float)input.at<C>(i, j)[2] - k * (float)temp.at<C>(i, j)[2]) / (1.0f - k);
			if (o_b < 0) { o_b = 0; }
			if (o_b > 255) { o_b = 255; }
			output.at<C>(i, j)[2] = (G)(o_b);
		}
	}

	return output;
}

Mat gaussianfilter(const Mat input, int n, float sigmaT, float sigmaS, const char* opt) {

	Mat kernel;

	int row = input.rows;
	int col = input.cols;
	int kernel_size = (2 * n + 1);
	int tempa;
	int tempb;
	float denom;
	float kernelvalue;

	// Initialiazing Kernel Matrix 
	kernel = Mat::zeros(kernel_size, kernel_size, CV_32F);

	denom = 0.0;
	for (int a = -n; a <= n; a++) {  // Denominator in m(s,t)
		for (int b = -n; b <= n; b++) {
			float value1 = exp(-(pow(a, 2) / (2 * pow(sigmaS, 2))) - (pow(b, 2) / (2 * pow(sigmaT, 2))));
			kernel.at<float>(a + n, b + n) = value1;	// kernelvalue 하지 않고 kernel 값을 꺼내 써야
			denom += value1;
		}
	}

	for (int a = -n; a <= n; a++) {  // Denominator in m(s,t)
		for (int b = -n; b <= n; b++) {
			kernel.at<float>(a + n, b + n) /= denom;
		}
	}

	Mat output = Mat::zeros(row, col, input.type());


	for (int i = 0; i < row; i++) {
		for (int j = 0; j < col; j++) {


			if (!strcmp(opt, "zero-paddle")) {
				// float sum1 = 0.0;
				float sum1_r = 0.0;
				float sum1_g = 0.0;
				float sum1_b = 0.0;

				for (int a = -n; a <= n; a++) {	// 수직
					for (int b = -n; b <= n; b++) {	// 수평

						if ((i + a <= row - 1) && (i + a >= 0) && (j + b <= col - 1) && (j + b >= 0)) {
							// sum1 += kernel.at<float>(a + n, b + n) * input.at<G>(i + a, j + b);	
							sum1_r += kernel.at<float>(a + n, b + n) * (float)(input.at<C>(i + a, j + b)[0]);
							sum1_g += kernel.at<float>(a + n, b + n) * (float)(input.at<C>(i + a, j + b)[1]);
							sum1_b += kernel.at<float>(a + n, b + n) * (float)(input.at<C>(i + a, j + b)[2]);
						}
					}
				}
				output.at<C>(i, j)[0] = (G)sum1_r;
				output.at<C>(i, j)[1] = (G)sum1_g;
				output.at<C>(i, j)[2] = (G)sum1_b;
			}

			else if (!strcmp(opt, "mirroring")) {
				float sum1_r = 0.0;
				float sum1_g = 0.0;
				float sum1_b = 0.0;

				for (int a = -n; a <= n; a++) {
					for (int b = -n; b <= n; b++) {


						if (i + a > row - 1) {
							tempa = i - a;
						}
						else if (i + a < 0) {
							tempa = -(i + a);
						}
						else {
							tempa = i + a;
						}
						// 수평 방향
						if (j + b > col - 1) {
							tempb = j - b;
						}
						else if (j + b < 0) {
							tempb = -(j + b);
						}
						else {
							tempb = j + b;
						}
						sum1_r += kernel.at<float>(a + n, b + n) * (float)(input.at<C>(tempa, tempb)[0]);
						sum1_g += kernel.at<float>(a + n, b + n) * (float)(input.at<C>(tempa, tempb)[1]);
						sum1_b += kernel.at<float>(a + n, b + n) * (float)(input.at<C>(tempa, tempb)[2]);
					}
				}
				output.at<C>(i, j)[0] = (G)sum1_r;
				output.at<C>(i, j)[1] = (G)sum1_g;
				output.at<C>(i, j)[2] = (G)sum1_b;
			}


			else if (!strcmp(opt, "adjustkernel")) {
				// float sum1 = 0.0;
				float sum1_r = 0.0;
				float sum1_g = 0.0;
				float sum1_b = 0.0;
				float sum2 = 0.0;

				for (int a = -n; a <= n; a++) {
					for (int b = -n; b <= n; b++) {

						if ((i + a <= row - 1) && (i + a >= 0) && (j + b <= col - 1) && (j + b >= 0)) {
							sum1_r += kernel.at<float>(a + n, b + n) * (float)(input.at<C>(i + a, j + b)[0]);
							sum1_g += kernel.at<float>(a + n, b + n) * (float)(input.at<C>(i + a, j + b)[1]);
							sum1_b += kernel.at<float>(a + n, b + n) * (float)(input.at<C>(i + a, j + b)[2]);
							sum2 += kernel.at<float>(a + n, b + n);
						}
					}
				}
				output.at<C>(i, j)[0] = (G)(sum1_r / sum2);
				output.at<C>(i, j)[1] = (G)(sum1_g / sum2);
				output.at<C>(i, j)[2] = (G)(sum1_b / sum2);
			}
		}
	}
	return output;
}