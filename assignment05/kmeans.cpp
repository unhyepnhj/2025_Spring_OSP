#include <iostream>
#include <opencv2/opencv.hpp>

#define IM_TYPE	CV_8UC3
# define L 256 
//#define sigma (float)pow(2, 32);	
#define sigma 512.0f;
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

// Note that this code is for the case when an input data is a color value.
int main() {

	Mat input = imread("lena.jpg", IMREAD_COLOR);
	Mat input_gray, output;
	cvtColor(input, input_gray, COLOR_RGB2GRAY);

	if (!input.data)
	{
		std::cout << "Could not open" << std::endl;
		return -1;
	}

	namedWindow("Original", WINDOW_AUTOSIZE);
	imshow("Original", input);
	namedWindow("Original Gray", WINDOW_AUTOSIZE);
	imshow("Original Gray", input_gray);

	Mat samples(input.rows * input.cols, 3, CV_32F);
	int clusterCount = 10;
	Mat labels;
	int attempts = 5;
	Mat centers;

	int rows = input.rows;
	int cols = input.cols;

	// Grayscale intensity only
	for (int y = 0; y < rows; y++) {
		for (int x = 0; x < cols; x++) {
			samples.at<float>(y + x * rows, 0) = input_gray.at<G>(y, x);
			samples.at<float>(y + x * rows, 1) = 0;
			samples.at<float>(y + x * rows, 2) = 0;
		}
	}
	kmeans(samples, clusterCount, labels, TermCriteria(TermCriteria::COUNT | TermCriteria::EPS, 10000, 0.0001), attempts, KMEANS_PP_CENTERS, centers);
	Mat gray_intensity(input.size(), input_gray.type());
	for (int y = 0; y < rows; y++) {
		for (int x = 0; x < cols; x++) {
			gray_intensity.at<G>(y, x) = (G)(centers.at<float>(labels.at<int>(y + x * rows, 0), 0));
		}
	}
	namedWindow("Grayscale Intensity", WINDOW_AUTOSIZE);
	imshow("Grayscale Intensity", gray_intensity);


	// Grayscale intensity & position
	// intensity: 0~255, position depends on the img size -> pre-processing
	/*
	I <- I/255
	x <- x/w
	y <- y/h
	*/

	for (int y = 0; y < rows; ++y) {
		for (int x = 0; x < cols; ++x) {
			int idx = y + x * rows;
			samples.at<float>(idx, 0) = input_gray.at<G>(y, x) / (float) (L - 1);	// [0,1]
			samples.at<float>(idx, 1) = float(x) / sigma;	
			samples.at<float>(idx, 2) = float(y) / sigma;
		}
	}
	labels = Mat();	// 초기화
	centers = Mat();
	kmeans(samples, clusterCount, labels, TermCriteria(TermCriteria::COUNT | TermCriteria::EPS, 10000, 0.0001), 5, KMEANS_PP_CENTERS, centers);
	Mat gray_intensity_position(input_gray.size(), input_gray.type());
	for (int y = 0; y < rows; ++y) {
		for (int x = 0; x < cols; ++x) {
			gray_intensity_position.at<G>(y, x) = (float)((L - 1) * centers.at<float>(labels.at<int>(y + x * rows, 0), 0));
		}
	}
	namedWindow("Grayscale Intensity & Position", WINDOW_AUTOSIZE);
	imshow("Grayscale Intensity & Position", gray_intensity_position);


	// RGB intensity only
	for (int y = 0; y < input.rows; y++)
		for (int x = 0; x < input.cols; x++)
			for (int z = 0; z < 3; z++)
				samples.at<float>(y + x*input.rows, z) = input.at<Vec3b>(y, x)[z];

	// Clustering is performed for each channel (RGB)
	// Note that the intensity value is not normalized here (0~1). You should normalize both intensity and position when using them simultaneously.
	// int clusterCount = 10;
	labels = Mat();
	//int attempts = 5;
	centers = Mat();	// 초기화
	kmeans(samples, clusterCount, labels, TermCriteria(TermCriteria::COUNT | TermCriteria::EPS, 10000, 0.0001), attempts, KMEANS_PP_CENTERS, centers);

	Mat new_image(input.size(), input.type());
	for (int y = 0; y < input.rows; y++) {
		for (int x = 0; x < input.cols; x++)
		{
			int cluster_idx = labels.at<int>(y + x * input.rows, 0);
			//Fill code that finds for each pixel of each channel of the output image the intensity of the cluster center.
			
			new_image.at<C>(y, x)[0] = (G)(centers.at<float>(cluster_idx, 0));
			new_image.at<C>(y, x)[1] = (G)(centers.at<float>(cluster_idx, 1));
			new_image.at<C>(y, x)[2] = (G)(centers.at<float>(cluster_idx, 2));
		}
	}
	namedWindow("clustered image", WINDOW_AUTOSIZE);
	imshow("clustered image", new_image);

	// RGB intensity & position 5D
	samples = Mat(rows * cols, 5, CV_32F);	// 5D로 재선언
	for (int y = 0; y < rows; y++) {
		for (int x = 0; x < cols; x++) {
			int idx = y + x * rows;
			samples.at<float>(idx, 0) = input.at<C>(y, x)[0] / (float)(L - 1);
			samples.at<float>(idx, 1) = input.at<C>(y, x)[1] / (float)(L - 1);
			samples.at<float>(idx, 2) = input.at<C>(y, x)[2] / (float)(L - 1);
			samples.at<float>(idx, 3) = float(x) / sigma;
			samples.at<float>(idx, 4) = float(y) / sigma;
		}
	}
	labels = Mat();
	centers = Mat();
	kmeans(samples, clusterCount, labels, TermCriteria(TermCriteria::COUNT | TermCriteria::EPS, 10000, 0.0001), attempts, KMEANS_PP_CENTERS, centers);
	Mat RGB_intensity_position_5D(input.size(), input.type());
	for (int y = 0; y < input.rows; y++) {
		for (int x = 0; x < input.cols; x++)
		{
			int cluster_idx = labels.at<int>(y + x * input.rows, 0);
			RGB_intensity_position_5D.at<C>(y, x)[0] = (float)((L - 1) * centers.at<float>(cluster_idx, 0));
			RGB_intensity_position_5D.at<C>(y, x)[1] = (float)((L - 1) * centers.at<float>(cluster_idx, 1));
			RGB_intensity_position_5D.at<C>(y, x)[2] = (float)((L - 1) * centers.at<float>(cluster_idx, 2));
		}
	}
	namedWindow("RGB Intensity & Position 5D", WINDOW_AUTOSIZE);
	imshow("RGB Intensity & Position 5D", RGB_intensity_position_5D);

	waitKey(0);

	return 0;
}

