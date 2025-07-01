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

Mat sobelfilter(const Mat input);

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
	output = sobelfilter(input_gray); //Boundary process: zero-paddle, mirroring, adjustkernel

	namedWindow("Sobel Filter", WINDOW_AUTOSIZE);
	imshow("Sobel Filter", output);


	waitKey(0);

	return 0;
}


Mat sobelfilter(const Mat input) {

	// Mat kernel;

	int row = input.rows;
	int col = input.cols;
	int n = 1; // Sobel Filter Kernel N

	// Initialiazing 2 Kernel Matrix with 3x3 size for Sx and Sy
	int Sx[3][3] = { {-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1} };
	int Sy[3][3] = { {-1, -2, -1}, {0, 0, 0}, {1, 2, 1} };

	//Fill code to initialize Sobel filter kernel matrix for Sx and Sy (Given in the lecture notes)

	Mat output = Mat::zeros(row, col, input.type());

	int tempa;
	int tempb;

	for (int i = 0; i < row; i++) {
		for (int j = 0; j < col; j++) {
			float Xsum = 0.0, Ysum = 0.0;

			for (int a = -n; a <= n; a++) {
				for (int b = -n; b <= n; b++) {
					// Use mirroring boundary process
					// Find output M(x,y) = sqrt( input.at<G>(x, y)*Sx + input.at<G>(x, y)*Sy ) 
					if (i + a > row - 1) {
						tempa = i - a;
					}
					else if (i + a < 0) {
						tempa = -(i + a);
					}
					else {
						tempa = i + a;
					}
					if (j + b > col - 1) {
						tempb = j - b;
					}
					else if (j + b < 0) {
						tempb = -(j + b);
					}
					else {
						tempb = j + b;
					}

					// Xsum += Sx[a][b] * input.at<uchar>(tempa, tempb);
					// Ysum += Sy[a][b] * input.at<uchar>(tempa, tempb);
					Xsum += Sx[a + 1][b + 1] * input.at<G>(tempa, tempb);	// +1 «ÿ¡‡æﬂ «‘
					Ysum += Sy[a + 1][b + 1] * input.at<G>(tempa, tempb);
					// sum1 += kernel.at<float>(a + n, b + n) * input.at<uchar>(tempa, tempb);
				}
			  
			}// end of for(-n~n)
			// M(x, y) = sqrt(input.at<G>(x, y) * Sx + input.at<G>(x, y) * Sy)
			int M = sqrt(Xsum * Xsum + Ysum * Ysum);

			// 0~255
			if (M < 0) { M = 0; }
			if (M > 255) { M = 255; }

			output.at<G>(i, j) = (G)M;
		}
	}
	return output;
}