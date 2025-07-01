#include <iostream>
#include <opencv2/opencv.hpp>

// output: input, output image

#define IM_TYPE	CV_8UC3

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

Mat meanfilter(const Mat input, int n, const char* opt);

int main() {
	
	Mat input = imread("lena.jpg", IMREAD_COLOR);	// IMREAD_COLOR
	Mat input_gray;
	Mat output;

	cvtColor(input, input_gray, COLOR_RGB2GRAY); // Converting image to gray	COLOR_RGB2GRAY
	

	if (!input.data)
	{
		cout << "Could not open" << std::endl;
		return -1;
	}

	namedWindow("Grayscale", WINDOW_AUTOSIZE);
	imshow("Grayscale", input_gray);

	output = meanfilter(input_gray, 3, "mirroring"); //Boundary process: zero-paddle, mirroring, adjustkernel
	
	// output = meanfilter(input_gray, 3, "zero-paddle"); 
	// output = meanfilter(input_gray, 3, "adjustkernel");

	namedWindow("Mean Filter", WINDOW_AUTOSIZE);
	imshow("Mean Filter", output);

	waitKey(0);

	return 0;
}


Mat meanfilter(const Mat input, int n, const char* opt) {

	Mat kernel;
	
	int row = input.rows;
	int col = input.cols;
	int kernel_size = (2 * n + 1);
	int tempa;
	int tempb;
 // Initialiazing Kernel Matrix 
	kernel = Mat::ones(kernel_size, kernel_size, CV_32F) / (float)(kernel_size * kernel_size);
	float kernelvalue=kernel.at<float>(0, 0);  // To simplify, as the filter is uniform. All elements of the kernel value are same.
	
	Mat output = Mat::zeros(row, col, input.type());
	
	
	for (int i = 0; i < row; i++) { //for each pixel in the output
		for (int j = 0; j < col; j++) {
			
			if (!strcmp(opt, "zero-paddle")) {
				float sum1 = 0.0;
				for (int a = -n; a <= n; a++) { // for each kernel window
					for (int b = -n; b <= n; b++) {

						if ((i + a <= row - 1) && (i + a >= 0) && (j + b <= col - 1) && (j + b >= 0)) { //if the pixel is not a border pixel
							sum1 += kernelvalue * (float)(input.at<G>(i + a, j + b));
						}
					}
				}
				output.at<G>(i, j) = (G)sum1;
			}
			
			else if (!strcmp(opt, "mirroring")) {
				float sum1 = 0.0;
				for (int a = -n; a <= n; a++) { // for each kernel window
					for (int b = -n; b <= n; b++) {
						//mirroring for the border pixels
						// 수직 방향
						if (i + a > row - 1) {  // a>0일 때, 이미지 아래 경계 초과
							tempa = i - a;		// (i, j)에서 a만큼 위쪽 픽셀 값
						}
						else if (i + a < 0) {	// a<0일 때, 이미지 위쪽 경계 초과
							tempa = -(i + a);	// (i, j)에서 a만큼 아래쪽 픽셀 값
						}
						else {
							tempa = i + a;	// 범위 내 -> 그대로 사용
						}
						// 수평 방향
						if (j + b > col - 1) {	// b>0일 때, 이미지 우측 경계 초과
							tempb = j - b;		// (i, j)에서 b만큼 왼쪽 픽셀 값
						}
						else if (j + b < 0) {	// b<0일 때, 이미지 좌측 경계 초과
							tempb = -(j + b);	// (i, j)에서 b만큼 오른쪽 픽셀 값
						}
						else {
							tempb = j + b;	// 범위 내 -> 그대로 사용
						}
						sum1 += kernelvalue*(float)(input.at<G>(tempa, tempb));	// sum에 포함
					}
				}
				output.at<G>(i, j) = (G)sum1;	// 결과 이미지에 uniform mean filtering한 (i, j)값 할당
			}
			
			else if (!strcmp(opt, "adjustkernel")) {
				float sum1 = 0.0;
				float sum2 = 0.0;
				for (int a = -n; a <= n; a++) { // for each kernel window
					for (int b = -n; b <= n; b++) {
						if ((i + a <= row - 1) && (i + a >= 0) && (j + b <= col - 1) && (j + b >= 0)) {
							sum1 += kernelvalue*(float)(input.at<G>(i + a, j + b));	
							sum2 += kernelvalue;	// 분모
						}
					}
				}
				output.at<G>(i, j) = (G)(sum1/sum2);
			}
		}
	}
	return output;
}
