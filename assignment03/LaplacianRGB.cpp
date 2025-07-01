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
	Mat output;


	if (!input.data)
	{
		std::cout << "Could not open" << std::endl;
		return -1;
	}

	namedWindow("RGB", WINDOW_AUTOSIZE);
	imshow("RGB", input);
	output = laplacianfilter(input);

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
	//int L[3][3] = { {1, 1, 1}, {1, -8, 1}, {1, 1, 1} };	// or 중간만 -8이고 나머지 1인 거

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
			//float sum1 = 0.0;
			float sum1_r = 0.0;
			float sum1_g = 0.0;
			float sum1_b = 0.0;

			for (int a = -n; a <= n; a++) {
				for (int b = -n; b <= n; b++) {
					// Use mirroring boundary process
					// Find output M(x,y) = sqrt( input.at<G>(x, y)*Sx + input.at<G>(x, y)*Sy ) 

					if (i + a > row - 1) {
						// tempa = i - a;
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
					sum1_r += L[a + n][b + n] * (float)(input.at<C>(tempa, tempb)[0]);
					sum1_g += L[a + n][b + n] * (float)(input.at<C>(tempa, tempb)[1]);
					sum1_b += L[a + n][b + n] * (float)(input.at<C>(tempa, tempb)[2]);

				}

			}// end of for(-n~n)


			int M_r = sum1_r;
			if (M_r < 0) { M_r = 0; }
			if (M_r > 255) { M_r = 255; }

			int M_g = sum1_g;
			if (M_g < 0) { M_g = 0; }
			if (M_g > 255) { M_g = 255; }

			int M_b = sum1_b;
			if (M_b < 0) { M_b = 0; }
			if (M_b > 255) { M_b = 255; }

			output.at<C>(i, j)[0] = abs((G)M_r);
			output.at<C>(i, j)[1] = abs((G)M_g);
			output.at<C>(i, j)[2] = abs((G)M_b);
		}
	}//end of for(0~row)
	return output;
}