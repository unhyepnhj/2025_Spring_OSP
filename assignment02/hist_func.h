#pragma once

#include <opencv2/opencv.hpp>
#include <stdio.h>

#define L 256		// # of intensity levels
#define IM_TYPE	CV_8UC3

using namespace cv;
using namespace std;    // �߰�

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

// generate PDF for single channel image
float* cal_PDF(Mat& input) {

    int count[L] = { 0 };
    float* PDF = (float*)calloc(L, sizeof(float));

    // Count
    for (int i = 0; i < input.rows; i++)
        for (int j = 0; j < input.cols; j++)
            count[input.at<G>(i, j)]++;

    // Compute PDF
    for (int i = 0; i < L; i++)
        PDF[i] = (float)count[i] / (float)(input.rows * input.cols);

    return PDF;
}

// generate PDF for color image
float** cal_PDF_RGB(Mat& input) {

    int count[L][3] = { 0 };
    float** PDF = (float**)malloc(sizeof(float*) * L);

    for (int i = 0; i < L; i++)
        PDF[i] = (float*)calloc(3, sizeof(float));

    ////////////////////////////////////////////////
    //											  //
    // How to access multi channel matrix element //
    //											  //
    // if matrix A is CV_8UC3 type,				  //
    // A(i, j, k) -> A.at<Vec3b>(i, j)[k]		  //
    //											  //
    ////////////////////////////////////////////////

    // Count
    for (int i = 0; i < input.rows; i++) {
        for (int j = 0; j < input.cols; j++) {
            // [0] = B intensity, [1] = G intensity, [2] = R intensity
            for (int k = 0; k < 3; k++) {   // RGB ��ȸ
                count[input.at<Vec3b>(i, j)[k]][k]++; // count[intensity][channel]
            }
        }
    }

    // Compute PDF
    for (int i = 0; i < L; i++) {
        for (int k = 0; k < 3; k++) {
            PDF[i][k] = (float)count[i][k] / (float)(input.rows * input.cols);
        }
    }


    return PDF;
}

// generate CDF for single channel image
float* cal_CDF(Mat& input) {

    int count[L] = { 0 };
    float* CDF = (float*)calloc(L, sizeof(float));

    // Count
    for (int i = 0; i < input.rows; i++)
        for (int j = 0; j < input.cols; j++)
            count[input.at<G>(i, j)]++;

    // Compute CDF
    for (int i = 0; i < L; i++) {
        CDF[i] = (float)count[i] / (float)(input.rows * input.cols);    // PDF[i]�� ����

        if (i != 0) {   // i>=1 ����
            CDF[i] += CDF[i - 1];
        }
    }

    return CDF;
}

// generate CDF for color image
float** cal_CDF_RGB(Mat& input) {
    int count[L][3] = { 0 };
    float** CDF = (float**)malloc(sizeof(float*) * L);

    for (int i = 0; i < L; i++)
        CDF[i] = (float*)calloc(3, sizeof(float));

    // Count
    for (int i = 0; i < input.rows; i++) {
        for (int j = 0; j < input.cols; j++) {
            for (int k = 0; k < 3; k++) {
                count[input.at<Vec3b>(i, j)[k]][k]++;
            }
        }
    }

    // Compute CDF
    for (int k = 0; k < 3; k++) {
        for (int i = 0; i < L; i++) {   // kth channel intensity i
            CDF[i][k] = (float)count[i][k] / (float)(input.rows * input.cols);

            if (i != 0) {   // ����
                CDF[i][k] += CDF[i - 1][k];
            }
        }
    }

    return CDF;
}

////////////////////// Plot the graph ///////////////////////
void drawGraph(Mat& graph, int x, int y, int w, int h, int margin, float maxSize, const string& opt, string title);

void plot(float* function, string opt, string title) {
    // �׷��� ����
    int x = 512;   // x��: 0~255
    int y = 250;   // y��: ������ �ϸ� �ɵ�
    int margin = 50;        // ����
    int w = x + margin * 2; // �ʺ�
    int h = y + margin * 2; // ����

    Mat graph = Mat::zeros(h, w, CV_8UC3);
    float maxSize;

    if (opt == "PDF") {
        maxSize = 0.025f;

        for (int i = 0; i < L; i++) {
            if (function[i] > maxSize) {
                maxSize = function[i];   // �ʰ��ϴ� �� ���� ��� ����
                h += maxSize;
            }
        }
    }
    else {
        maxSize = 1.2f;
    }

    for (int i = 0; i < L; i++) {   // ������׷� ����
        int pointX = margin + cvRound((i / 255.0) * x); // �࿡�� �����ϴ� ����
        int pointY = margin + y - cvRound((function[i] / maxSize) * y);   // �࿡��~

        line(graph, Point(pointX, y + margin), Point(pointX, pointY), Scalar(255, 255, 255));
    }

    drawGraph(graph, x, y, w, h, margin, maxSize, opt, title);

    namedWindow(title, WINDOW_AUTOSIZE);
    imshow(title, graph);
}

/*
    �� �׸���
    x��: PDF, CDF ����(0~255)
    y��:
        PDF: 0.005 ���� 0~0.025
        CDF: 0.2 ���� 0.0~1.2
    */
void drawGraph(Mat& graph, int x, int y, int w, int h, int margin, float maxSize, const string& opt, string title) {    // opt=PDF/CDF/hist, title=�׷����� ǥ�õ� ����
    putText(graph, title, Point(w / 2 - 30, margin / 2), FONT_ITALIC, 0.5, Scalar(255, 255, 255), 1);   // ����

    line(graph, Point(margin, margin + y), Point(margin + x, margin + y), Scalar(255, 255, 255), 1); // x�� ��
    for (int i = 0; i <= 255; i += 15) {   // ��
        int pointX = margin + cvRound((i / 255.0) * x);

        putText(graph, to_string(i), Point(pointX - 10, margin + y + 10), FONT_ITALIC, 0.4, Scalar(255, 255, 255), 1);
    }

    line(graph, Point(margin, margin), Point(margin, margin + y), Scalar(255, 255, 255), 1); // y�� ��

    // y�� ��
    if (opt == "PDF") { // PDF
        vector<float> label = {};
        float iPos = maxSize / 6;
        for (float i = 0.000; i <= maxSize; i += iPos) {
            label.push_back(i);
        }
        //float label[] = { 0.005f, 0.01f, 0.015f, 0.02f, 0.025f };   // ���� �ε���: 0~0.025

        for (int i = 0; i < label.size(); i++) {   // �� ǥ��
            int pointY = margin + y - cvRound((label[i] / maxSize) * y);    // ���� �ߴ� �Ŷ� ����
            putText(graph, to_string(label[i]).substr(0, 5), Point(margin - 40, pointY + 5), FONT_ITALIC, 0.4, Scalar(255, 255, 255), 1);
        }
    }
    else {
        float label[] = { 0.0f, 0.2f, 0.4f, 0.6f, 0.8f, 1.0f, 1.2f };   // 0~1.2

        for (int i = 0; i < 7; i++) {
            int pointY = margin + y - cvRound((label[i] / maxSize) * y);

            putText(graph, to_string(label[i]), Point(margin - 40, pointY + 5), FONT_ITALIC, 0.4, Scalar(255, 255, 255), 1);
        }
    }
}