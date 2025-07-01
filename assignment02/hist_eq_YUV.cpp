#include "hist_func.h"

void hist_eq(Mat &input, Mat &equalized, G *trans_func, float *CDF);

int main() {

	Mat input = imread("input.jpg", IMREAD_COLOR);
	Mat equalized_YUV;

	cvtColor(input, equalized_YUV, COLOR_RGB2YUV);	// RGB -> YUV
	
	// split each channel(Y, U, V)
	Mat channels[3];
	split(equalized_YUV, channels);
	Mat Y = channels[0];						// U = channels[1], V = channels[2]

	// PDF or transfer function txt files
	FILE *f_equalized_PDF_YUV, *f_PDF_RGB;
	FILE *f_trans_func_eq_YUV;

	float **PDF_RGB = cal_PDF_RGB(input);		// PDF of Input image(RGB) : [L][3]
	float *CDF_YUV = cal_CDF(Y);				// CDF of Y channel image

	fopen_s(&f_PDF_RGB, "PDF_YUV.txt", "w+");
	fopen_s(&f_equalized_PDF_YUV, "equalized_PDF_YUV.txt", "w+");
	fopen_s(&f_trans_func_eq_YUV, "trans_func_eq_YUV.txt", "w+");

	G trans_func_eq_YUV[L] = { 0 };			// transfer function

	// histogram equalization on Y channel
	hist_eq(Y, channels[0], trans_func_eq_YUV, CDF_YUV);
	
	// merge Y, U, V channels
	merge(channels, 3, equalized_YUV);
	
	// YUV -> RGB (use "CV_YUV2RGB" flag)
	cvtColor(equalized_YUV, equalized_YUV, COLOR_YUV2RGB);

	// equalized PDF (YUV)
	float* equalized_PDF_Y = cal_PDF(channels[0]);

	for (int i = 0; i < L; i++) {
		// write PDF
		// ...

		// write transfer functions
		// ...
		fprintf(f_PDF_RGB, "%d\t%f\n", i, PDF_RGB[i][0]);	// 원본 이미지의 Y채널에 해당하는 PDF
		fprintf(f_equalized_PDF_YUV, "%d\t%f\n", i, equalized_PDF_Y[i]);
		fprintf(f_trans_func_eq_YUV, "%d\t%d\n", i, trans_func_eq_YUV[i]);
	}

	// memory release
	free(PDF_RGB);
	free(CDF_YUV);
	fclose(f_PDF_RGB);
	fclose(f_equalized_PDF_YUV);
	fclose(f_trans_func_eq_YUV);

	////////////////////// histogram ///////////////////////
	FILE* f_hist_YUV;
	FILE* f_equalized_hist_YUV;

	fopen_s(&f_hist_YUV, "hist_YUV.txt", "w+");
	fopen_s(&f_equalized_hist_YUV, "equalized_hist_YUV.txt", "w+");

	int histSize = L;
	float ran[] = { 0, 256 };
	const float* range = { ran };

	// 원본 RGB 히스토그램
	// !!!!!!!! BGR !!!!!!!!
	vector<Mat> bgr;
	split(input, bgr);	// 채널 분리

	Mat histB, histG, histR;
	calcHist(&bgr[0], 1, 0, Mat(), histB, 1, &histSize, &range);	//b
	calcHist(&bgr[1], 1, 0, Mat(), histG, 1, &histSize, &range);	//g
	calcHist(&bgr[2], 1, 0, Mat(), histR, 1, &histSize, &range);	//r

	// equalized 히스토그램
	vector<Mat> bgr_equalized;
	split(equalized_YUV, bgr_equalized);

	Mat histB_eq, histG_eq, histR_eq;
	calcHist(&bgr_equalized[0], 1, 0, Mat(), histB_eq, 1, &histSize, &range);
	calcHist(&bgr_equalized[1], 1, 0, Mat(), histG_eq, 1, &histSize, &range);
	calcHist(&bgr_equalized[2], 1, 0, Mat(), histR_eq, 1, &histSize, &range);

	// "%d\t%f\t%f\t%f\n", i, PDF_RGB[i][0], PDF_RGB[i][1], PDF_RGB[i][2]
	for (int i = 0; i < histSize; i++) {
		fprintf(f_hist_YUV, "%d\t\t%f\t\t%f\t\t%f\n", i, histR.at<float>(i), histG.at<float>(i), histB.at<float>(i));
		fprintf(f_equalized_hist_YUV, "%d\t\t%f\t\t%f\t\t%f\n", i, histR_eq.at<float>(i), histG_eq.at<float>(i), histB_eq.at<float>(i));
		// R G B 순서 저장
	}

	fclose(f_hist_YUV);
	fclose(f_equalized_hist_YUV);

	////////////////////// Show each image ///////////////////////

	namedWindow("RGB", WINDOW_AUTOSIZE);
	imshow("RGB", input);

	namedWindow("Equalized_YUV", WINDOW_AUTOSIZE);
	imshow("Equalized_YUV", equalized_YUV);

	//////////////////////////////////////////////////////////////

	waitKey(0);

	return 0;
}

// histogram equalization
void hist_eq(Mat &input, Mat &equalized, G *trans_func, float *CDF) {

	// compute transfer function
	for (int i = 0; i < L; i++)
		trans_func[i] = (G)((L - 1) * CDF[i]);

	// perform the transfer function
	for (int i = 0; i < input.rows; i++)
		for (int j = 0; j < input.cols; j++)
			equalized.at<G>(i, j) = trans_func[input.at<G>(i, j)];
}