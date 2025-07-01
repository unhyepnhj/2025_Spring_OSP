#include "hist_func.h"

void hist_eq_Color(Mat &input, Mat &equalized, G(*trans_func)[3], float **CDF);

int main() {

	Mat input = imread("input.jpg", IMREAD_COLOR);
	Mat equalized_RGB = input.clone();

	// PDF or transfer function txt files
	FILE *f_equalized_PDF_RGB, *f_PDF_RGB;
	FILE *f_trans_func_eq_RGB;
	
	fopen_s(&f_PDF_RGB, "PDF_RGB.txt", "w+");
	fopen_s(&f_equalized_PDF_RGB, "equalized_PDF_RGB.txt", "w+");
	fopen_s(&f_trans_func_eq_RGB, "trans_func_eq_RGB.txt", "w+");

	float **PDF_RGB = cal_PDF_RGB(input);	// PDF of Input image(RGB) : [L][3]
	float **CDF_RGB = cal_CDF_RGB(input);	// CDF of Input image(RGB) : [L][3]

	G trans_func_eq_RGB[L][3] = { 0 };		// transfer function

	// histogram equalization on RGB image
	// ...
	hist_eq_Color(input, equalized_RGB, trans_func_eq_RGB, CDF_RGB);

	// equalized PDF (RGB)
	// ...
	float** equalized_PDF_RGB = cal_PDF_RGB(equalized_RGB);	// 추가함
	for (int i = 0; i < L; i++) {
		// write PDF
		// ...

		// write transfer functions
		// ...
		fprintf(f_PDF_RGB, "%d\t%f\t%f\t%f\n", i, PDF_RGB[i][0], PDF_RGB[i][1], PDF_RGB[i][2]);
		fprintf(f_equalized_PDF_RGB, "%d\t%f\t%f\t%f\n", i, equalized_PDF_RGB[i][0], equalized_PDF_RGB[i][1], equalized_PDF_RGB[i][2]);
		fprintf(f_trans_func_eq_RGB, "%d\t%d\t%d\t%d\n", i, trans_func_eq_RGB[i][0], trans_func_eq_RGB[i][1], trans_func_eq_RGB[i][2]);
	}

	// memory release
	free(PDF_RGB);
	free(CDF_RGB);
	fclose(f_PDF_RGB);
	fclose(f_equalized_PDF_RGB);
	fclose(f_trans_func_eq_RGB);

	////////////////////// histogram ///////////////////////
	FILE* f_hist_RGB;
	FILE* f_equalized_hist_RGB;

	fopen_s(&f_hist_RGB, "hist_RGB.txt", "w+");
	fopen_s(&f_equalized_hist_RGB, "equalized_hist_RGB.txt", "w+");

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
	split(equalized_RGB, bgr_equalized);

	Mat histB_eq, histG_eq, histR_eq;
	calcHist(&bgr_equalized[0], 1, 0, Mat(), histB_eq, 1, &histSize, &range);
	calcHist(&bgr_equalized[1], 1, 0, Mat(), histG_eq, 1, &histSize, &range);
	calcHist(&bgr_equalized[2], 1, 0, Mat(), histR_eq, 1, &histSize, &range);

	// "%d\t%f\t%f\t%f\n", i, PDF_RGB[i][0], PDF_RGB[i][1], PDF_RGB[i][2]
	for (int i = 0; i < histSize; i++) {
		fprintf(f_hist_RGB, "%d\t\t%f\t\t%f\t\t%f\n", i, histR.at<float>(i), histG.at<float>(i), histB.at<float>(i));
		fprintf(f_equalized_hist_RGB, "%d\t\t%f\t\t%f\t\t%f\n", i, histR_eq.at<float>(i), histG_eq.at<float>(i), histB_eq.at<float>(i));
		// R G B 순서 저장
	}

	fclose(f_hist_RGB);
	fclose(f_equalized_hist_RGB);

	////////////////////// Show each image ///////////////////////

	namedWindow("RGB", WINDOW_AUTOSIZE);
	imshow("RGB", input);

	namedWindow("Equalized_RGB", WINDOW_AUTOSIZE);
	imshow("Equalized_RGB", equalized_RGB);

	//////////////////////////////////////////////////////////////

	waitKey(0);

	return 0;
}

// histogram equalization on 3 channel image
void hist_eq_Color(Mat &input, Mat &equalized, G(*trans_func)[3], float **CDF) {

	////////////////////////////////////////////////
	//											  //
	// How to access multi channel matrix element //
	//											  //
	// if matrix A is CV_8UC3 type,				  //
	// A(i, j, k) -> A.at<Vec3b>(i, j)[k]		  //
	//											  //
	////////////////////////////////////////////////

	for (int i = 0; i < 3; i++) {	// 채널
		for (int j = 0; j < L; j++) {
			trans_func[j][i] = (G)((L - 1) * CDF[j][i]);	// equalization
		}
	}

	for (int i = 0; i < input.rows; i++) {
		for (int j = 0; j < input.cols; j++) {
			Vec3b temp;	// p를 eq한 값 저장
			for (int k = 0; k < 3; k++) {	// RGB
				temp[k] = trans_func[input.at<Vec3b>(i, j)[k]][k];	// (i, j) 픽셀의 k채널을 equalization 한 값
			}
			equalized.at<Vec3b>(i, j) = temp;	// equalized의 (i, j)의 RGB값 = temp
		}
	}
}