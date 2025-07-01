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
	Mat input_gray;
	Mat output;

	if (!input.data)
	{
		std::cout << "Could not open" << std::endl;
		return -1;
	}


	cvtColor(input, input_gray, COLOR_RGB2GRAY); input_gray;

	float k = 0.5f;
	output = unsharpmask(input_gray, 3, 3, 3, k, "mirroring");
	// output = unsharpmask(input_gray, 3, 3, 3, k, "zero-paddle");
	//output = unsharpmask(input_gray, 3, 3, 3, k, "adjustkernel");

	namedWindow("Original", WINDOW_AUTOSIZE);
	imshow("Original", input_gray);

	namedWindow("Unsharp Masking", WINDOW_AUTOSIZE);
	imshow("Unsharp Masking", output);

	waitKey(0);
	return 0;
}

/*
* 1. original
* 2. blur with low-pass filter
* 3. scale with k<1
* 4. subtract
* 5. scale for display
*/
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
			// 원본 이미지 I = input.at<G>(i, j);
			// low-pass 이미지 L = temp.at<G>(i, j);
			float o = ((float)input.at<G>(i, j) - k * (float)temp.at<G>(i, j)) / (1.0f - k);

			if (o < 0) { o = 0; }
			if (o > 255) { o = 255; }

			output.at<G>(i, j) = (G)(o);
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
	// float kernelvalue;

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
				float sum1 = 0.0;
				for (int a = -n; a <= n; a++) {
					for (int b = -n; b <= n; b++) {

						if ((i + a <= row - 1) && (i + a >= 0) && (j + b <= col - 1) && (j + b >= 0)) {
							sum1 += kernel.at<float>(a + n, b + n) * input.at<G>(i + a, j + b);
						}
					}
				}
				output.at<G>(i, j) = (G)sum1;
			}

			else if (!strcmp(opt, "mirroring")) {
				float sum1 = 0.0;
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
						sum1 += kernel.at<float>(a + n, b + n) * input.at<G>(tempa, tempb);
					}
				}
				output.at<G>(i, j) = (G)sum1;
			}


			else if (!strcmp(opt, "adjustkernel")) {
				float sum1 = 0.0;
				float sum2 = 0.0;
				for (int a = -n; a <= n; a++) {
					for (int b = -n; b <= n; b++) {

						if ((i + a <= row - 1) && (i + a >= 0) && (j + b <= col - 1) && (j + b >= 0)) {
							sum1 += kernel.at<float>(a + n, b + n) * input.at<G>(i + a, j + b);
							sum2 += kernel.at<float>(a + n, b + n);
						}
					}
				}
				output.at<G>(i, j) = (G)(sum1 / sum2);
			}
		}
	}
	return output;
}