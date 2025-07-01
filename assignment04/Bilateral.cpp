#include <opencv2/opencv.hpp>
#include <stdio.h>
#include <math.h>       /* exp */
#include <iostream>

#define IM_TYPE	CV_64FC3

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

Mat Add_Gaussian_noise(const Mat input, double mean, double sigma);
Mat bilateralfilter_Gray(const Mat input, int n, float sigmaT, float sigmaS, float sigmaR, const char* opt);
Mat bilateralfilter_RGB(const Mat input, int n, float sigmaT, float sigmaS, float sigmaR, const char* opt);

int main() {

	Mat input = imread("lena.jpg", IMREAD_COLOR);
	Mat input_gray;

	// check for validation
	if (!input.data) {
		printf("Could not open\n");
		return -1;
	}

	cvtColor(input, input_gray, COLOR_RGB2GRAY);	// convert RGB to Grayscale

	// 8-bit unsigned char -> 64-bit floating point
	input.convertTo(input, CV_64FC3, 1.0 / 255);
	input_gray.convertTo(input_gray, CV_64F, 1.0 / 255);

	// Add noise to original image
	Mat noise_Gray = Add_Gaussian_noise(input_gray, 0, 0.1);
	Mat noise_RGB = Add_Gaussian_noise(input, 0, 0.1);

	// Denoise, using bilateral filter
	Mat Denoised_Gray_bilateral = bilateralfilter_Gray(noise_Gray, 5, 3, 3, 0.2, "zero-padding");
	Mat Denoised_RGB_bilateral = bilateralfilter_RGB(noise_RGB, 5, 3, 3, 0.2, "zero-padding");


    namedWindow("Grayscale", WINDOW_AUTOSIZE);
    imshow("Grayscale", input_gray);

    namedWindow("RGB", WINDOW_AUTOSIZE);
    imshow("RGB", input);

    namedWindow("Gaussian Noise (Grayscale)", WINDOW_AUTOSIZE);
    imshow("Gaussian Noise (Grayscale)", noise_Gray);

    namedWindow("Gaussian Noise (RGB)", WINDOW_AUTOSIZE);
    imshow("Gaussian Noise (RGB)", noise_RGB);

	namedWindow("Denoised (Bilateral Gray)", WINDOW_AUTOSIZE);
	imshow("Denoised (Bilateral Gray)", Denoised_Gray_bilateral);

	namedWindow("Denoised (Bilateral RGB)", WINDOW_AUTOSIZE);
	imshow("Denoised (Bilateral RGB)", Denoised_RGB_bilateral);

	waitKey(0);

	return 0;
}

Mat Add_Gaussian_noise(const Mat input, double mean, double sigma) {

	Mat NoiseArr = Mat::zeros(input.rows, input.cols, input.type());
	RNG rng;
	rng.fill(NoiseArr, RNG::NORMAL, mean, sigma);

	add(input, NoiseArr, NoiseArr);

	return NoiseArr;
}

Mat bilateralfilter_Gray(const Mat input, int n, float sigmaT, float sigmaS, float sigmaR, const char* opt) {   // bilateral gaussian filter
    int row = input.rows;
    int col = input.cols;
    int kernel_size = (2 * n + 1);
    int tempa;
    int tempb;
    float denom;

    // Gs = exp(-s^2/2sigmaS^2-t^2/2sitmaT^2)
    Mat Gs = Mat::zeros(kernel_size, kernel_size, CV_32F);
    denom = 0.0;

    for (int a = -n; a <= n; a++) {    // s
        for (int b = -n; b <= n; b++) {    // t
            float value1 = exp(-(pow(a, 2) / (2 * pow(sigmaS, 2))) - (pow(b, 2) / (2 * pow(sigmaT, 2))));
            Gs.at<float>(a + n, b + n) = value1;
            denom += value1;
        }
    }
    for (int a = -n; a <= n; a++) {  // Denominator in m(s,t)
        for (int b = -n; b <= n; b++) {
            Gs.at<float>(a + n, b + n) /= denom;
        }
    }

    Mat output = Mat::zeros(row, col, input.type());

    for (int i = 0; i < row; i++) {
        for (int j = 0; j < col; j++) {

            if (!strcmp(opt, "zero-padding")) {
                float sum1 = 0.0f;
                float W = 0.0f;
                float Gr = 0.0f;
                float GsGr = 0.0f;
                for (int a = -n; a <= n; a++) {
                    for (int b = -n; b <= n; b++) {
                        if ((i + a <= row - 1) && (i + a >= 0) && (j + b <= col - 1) && (j + b >= 0)) {
                            // Gr
                            //p(i, j) - q(i + a, j + b)
                            Gr = exp(-(pow(input.at<G>(i, j) - input.at<G>(i + a, j + b), 2) / (2 * pow(sigmaR, 2))));
                            GsGr = Gs.at<float>(a + n, b + n) * Gr;
                            sum1 += GsGr * input.at<G>(i + a, j + b);      // Gs*Gr*
                        }// 경계 안일 때 I값으로 계산
                        // 경계 밖이면 I(i+a, i+b)=0이므로 계산 안 해도 됨

                        W += GsGr;  // W = sum(GsGr) -> 모든 커널 픽셀 포함시켜야 하므로 경계 안/밖 구분 없이 +=
                    }
                }
                // Op = sum(Gs*Gr*I)/W
                if (W != 0) {
                    output.at<G>(i, j) = (G)(sum1 / W);
                }
                else {
                    output.at<G>(i, j) = (G)sum1;
                }
            }
            else if (!strcmp(opt, "mirroring")) {
                float sum1 = 0.0;
                float W = 0.0;
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

                        if (j + b > col - 1) {
                            tempb = j - b;
                        }
                        else if (j + b < 0) {
                            tempb = -(j + b);
                        }
                        else {
                            tempb = j + b;
                        }

                        float Gr = exp(-(pow(input.at<G>(i, j) - input.at<G>(tempa, tempb), 2) / (2 * pow(sigmaR, 2))));
                        float GsGr = Gs.at<float>(a + n, b + n) * Gr;  // Wp = Gs*Gr
                        sum1 += GsGr * input.at<G>(tempa, tempb);      // Gs*Gr*I
                        W += GsGr;
                    }
                }
                if (W != 0) {
                    output.at<G>(i, j) = (G)(sum1 / W);
                }
                else {
                    output.at<G>(i, j) = (G)sum1;
                }
            }
            else if (!strcmp(opt, "adjustkernel")) {
                float sum1 = 0.0;
                float W = 0.0;
                for (int a = -n; a <= n; a++) {
                    for (int b = -n; b <= n; b++) {
                        if ((i + a >= 0) && (i + a < row) && (j + b >= 0) && (j + b < col)) {
                            float Gr = exp(-(pow(input.at<G>(i, j) - input.at<G>(i + a, j + b), 2) / (2 * pow(sigmaR, 2))));
                            float GsGr = Gs.at<float>(a + n, b + n) * Gr;  // Wp = Gs*Gr
                            sum1 += GsGr * input.at<G>(i + a, j + b);      // Gs*Gr*I
                            W += GsGr;  // 이때는 valid한 픽셀만 고려하므로 경계 안에서만 +=함
                        }
                    }
                }
                if (W != 0) {
                    output.at<G>(i, j) = (G)(sum1 / W);
                }
                else {
                    output.at<G>(i, j) = (G)sum1;
                }
            }
        }
    }
    return output;
}

Mat bilateralfilter_RGB(const Mat input, int n, float sigmaT, float sigmaS, float sigmaR, const char* opt) {	// bilateral gaussian filter
    int row = input.rows;
    int col = input.cols;
    int kernel_size = (2 * n + 1);
    int tempa;
    int tempb;
    float denom;

    // Gs
    Mat Gs = Mat::zeros(kernel_size, kernel_size, CV_32F);
    denom = 0.0;

    for (int a = -n; a <= n; a++) {
        for (int b = -n; b <= n; b++) {
            float value1 = exp(-(pow(a, 2) / (2 * pow(sigmaS, 2))) - (pow(b, 2) / (2 * pow(sigmaT, 2))));
            Gs.at<float>(a + n, b + n) = value1;
            denom += value1;
        }
    }
    for (int a = -n; a <= n; a++) {  // Denominator in m(s,t)
        for (int b = -n; b <= n; b++) {
            Gs.at<float>(a + n, b + n) /= denom;
        }
    }

    Mat output = Mat::zeros(row, col, input.type());

    for (int i = 0; i < row; i++) {
        for (int j = 0; j < col; j++) {
            
            Mat Cp = Mat::zeros(3, 1, CV_32F); // (Rp, Gp, Bp)T = 열벡터
            /*
                                 Rp
            Cp = (Rp, Gp, Bp)T = Gp
                                 Bp
            */
            Cp.at<float>(0, 0) = input.at<C>(i, j)[0];
            Cp.at<float>(1, 0) = input.at<C>(i, j)[1];
            Cp.at<float>(2, 0) = input.at<C>(i, j)[2];

            if (!strcmp(opt, "zero-padding")) {
                //float sum1 = 0.0;
                Mat sum1 = Mat::zeros(3, 1, CV_32F);
                float W = 0.0f;
                Mat Cq = Mat::zeros(3, 1, CV_32F);
                float Gr = 0.0f;
                float GsGr = 0.0f;

                for (int a = -n; a <= n; a++) {
                    for (int b = -n; b <= n; b++) {
                        if ((i + a <= row - 1) && (i + a >= 0) && (j + b <= col - 1) && (j + b >= 0)) {
                            Mat Cq = Mat::zeros(3, 1, CV_32F);
                            Cq.at<float>(0, 0) = input.at<C>(i + a, j + b)[0];
                            Cq.at<float>(1, 0) = input.at<C>(i + a, j + b)[1];
                            Cq.at<float>(2, 0) = input.at<C>(i + a, j + b)[2];

                            Mat d = Cp - Cq;    // 3x1 벡터
                            float powd = d.dot(d);  // d^2
                            Gr = exp(-powd / (2 * pow(sigmaR, 2)));

                            GsGr = Gs.at<float>(a + n, b + n) * Gr;  // Wp = Gs*Gr
                            sum1 += GsGr * Cq;      // Gs*Gr*I   
                        }
                        W += GsGr;
                    }
                }// end of kernel
                // Op = sum(Gs*Gr*I)/W
                Mat Op;

                if (W != 0) {
                    Mat Op = sum1 / W;

                    C o;    // output
                    o[0] = Op.at<float>(0, 0);
                    o[1] = Op.at<float>(1, 0);
                    o[2] = Op.at<float>(2, 0);

                    output.at<C>(i, j) = o;
                }
                else {
                    output.at<C>(i, j) = input.at<C>(i, j); // 그대로?????
                }
            }
            else if (!strcmp(opt, "mirroring")) {
                // float sum1 = 0.0;
                Mat sum1 = Mat::zeros(3, 1, CV_32F);
                float W = 0.0;
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

                        if (j + b > col - 1) {
                            tempb = j - b;
                        }
                        else if (j + b < 0) {
                            tempb = -(j + b);
                        }
                        else {
                            tempb = j + b;
                        }

                        Mat Cq = Mat::zeros(3, 1, CV_32F);
                        Cq.at<float>(0, 0) = input.at<C>(tempa, tempb)[0];
                        Cq.at<float>(1, 0) = input.at<C>(tempa, tempb)[1];
                        Cq.at<float>(2, 0) = input.at<C>(tempa, tempb)[2];

                        Mat d = Cp - Cq;    // 3x1 벡터
                        float powd = d.dot(d);  // d^2
                        float Gr = exp(-powd / (2 * pow(sigmaR, 2)));

                        float GsGr = Gs.at<float>(a + n, b + n) * Gr;  // Wp = Gs*Gr
                        sum1 += GsGr * Cq;      // Gs*Gr*I
                        W += GsGr;
                    }
                }// end of kernel
                Mat Op;

                if (W != 0) {
                    Mat Op = sum1 / W;

                    C o;    // output
                    o[0] = Op.at<float>(0, 0);
                    o[1] = Op.at<float>(1, 0);
                    o[2] = Op.at<float>(2, 0);

                    output.at<C>(i, j) = o;
                }
                else {
                    output.at<C>(i, j) = input.at<C>(i, j); // 그대로?????
                }
            }
            else if (!strcmp(opt, "adjustkernel")) {
                Mat sum1 = Mat::zeros(3, 1, CV_32F);
                float W = 0.0;
                for (int a = -n; a <= n; a++) {
                    for (int b = -n; b <= n; b++) {
                        if ((i + a >= 0) && (i + a < row) && (j + b >= 0) && (j + b < col)) {
                            Mat Cq = Mat::zeros(3, 1, CV_32F);
                            Cq.at<float>(0, 0) = input.at<C>(i + a, j + b)[0];
                            Cq.at<float>(1, 0) = input.at<C>(i + a, j + b)[1];
                            Cq.at<float>(2, 0) = input.at<C>(i + a, j + b)[2];

                            Mat d = Cp - Cq;    // 3x1 벡터
                            float powd = d.dot(d);  // d^2
                            float Gr = exp(-powd / (2 * pow(sigmaR, 2)));

                            float GsGr = Gs.at<float>(a + n, b + n) * Gr;  // Wp = Gs*Gr
                            sum1 += GsGr * Cq;      // Gs*Gr*I
                            W += GsGr;
                        }
                    }
                }
                Mat Op;

                if (W != 0) {
                    Mat Op = sum1 / W;

                    C o;    // output
                    o[0] = Op.at<float>(0, 0);
                    o[1] = Op.at<float>(1, 0);
                    o[2] = Op.at<float>(2, 0);

                    output.at<C>(i, j) = o;
                }
                else {
                    output.at<C>(i, j) = input.at<C>(i, j); // 그대로?????
                }
            }
        }
    }

    return output;
}