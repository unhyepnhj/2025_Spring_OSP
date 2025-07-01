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

Mat laplacianfilter(const Mat input);

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
	output = laplacianfilter(input_gray);

	namedWindow("Laplacian Filter", WINDOW_AUTOSIZE);
	imshow("Laplacian Filter", output);


	waitKey(0);

	return 0;
}

Mat laplacianfilter(const Mat input) {
	// Mat kernel;
	int row = input.rows;
	int col = input.cols;
	int n = 1; // Sobel Filter Kernel N

	int L[3][3] = { {0, 1, 0}, {1, -4, 1}, {0, 1, 0} };	
	// int L[3][3] = { {1, 1, 1}, {1, -8, 1}, {1, 1, 1} };	// or 중간만 -8이고 나머지 1인 거
	 
	/*
	0	1	0
	1	-4	1
	0	1	0
	*/

	Mat output = Mat::zeros(row, col, input.type());

	int tempa;
	int tempb;

	for (int i = 0; i < row; i++) {
		for (int j = 0; j < col; j++) {
			float sum1 = 0.0;

			for (int a = -n; a <= n; a++) {
				for (int b = -n; b <= n; b++) {
					// Use mirroring boundary process
					// Find output M(x,y) = sqrt( input.at<G>(x, y)*Sx + input.at<G>(x, y)*Sy ) 

					if (i + a > row - 1) {
						//tempa = i - a;
						tempa = 2 * row - (i + a) - 2;
					}
					else if (i + a < 0) {
						tempa = -(i + a);
					}
					else {
						tempa = i + a;
					}
					if (j + b > col - 1) {
						// tempb = j - b;
						tempa = 2 * col - (j + b) - 2;
					}
					else if (j + b < 0) {
						tempb = -(j + b);
					}
					else {
						tempb = j + b;
					}

					sum1 += L[a + n][b + n] * input.at<G>(tempa, tempb);

				}

			}// end of for(-n~n)
			int M = sum1;
			//int M = abs(sum1);

			// 0~255
			if (M < 0) { M = 0; }
			if (M > 255) { M = 255; }

			output.at<G>(i, j) = (G)(abs(M));
		}
	}//end of for(0~row)
	return output;
}