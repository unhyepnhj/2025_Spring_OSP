#include <iostream>
#include <opencv2/opencv.hpp>

using namespace cv;


int main() {

	Mat input = imread("lena.jpg", IMREAD_COLOR);
	Mat input_gray, output_gray, output;
	cvtColor(input, input_gray, COLOR_RGB2GRAY);
	cvtColor(input_gray, input_gray, COLOR_GRAY2RGB);

	if (!input.data)
	{
		std::cout << "Could not open" << std::endl;
		return -1;
	}

	imshow("Original Gray", input_gray);

	namedWindow("Original", WINDOW_AUTOSIZE);
	imshow("Original", input);

	pyrMeanShiftFiltering(input_gray, output_gray, 31, 20, 1);	// Grayscale
	imshow("Meanshift Gray", output_gray);
    pyrMeanShiftFiltering(input, output, 31, 20, 3);	// RGB
	imshow("Meanshift", output);

	waitKey();

	return 0;
}