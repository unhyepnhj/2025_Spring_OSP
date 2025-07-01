#include "hist_func.h"

void hist_eq(Mat& input, Mat& equalized, G* trans_func, float* CDF);
void hist_matching(Mat& input, Mat& matched, G* trans_func_z, G* trans_func_ref);

int main() {
	// read
	Mat input = imread("input.jpg", IMREAD_COLOR);
	Mat reference = imread("lena.jpg", IMREAD_COLOR);

	/*
	cvtColor(input, equalized_YUV, COLOR_RGB2YUV);	// RGB -> YUV
	
	// split each channel(Y, U, V)
	Mat channels[3];
	split(equalized_YUV, channels);
	Mat Y = channels[0];	
	*/
	
	Mat equalized_input, equalized_ref;	// YUV
	cvtColor(input, equalized_input, COLOR_RGB2YUV);
	cvtColor(reference, equalized_ref, COLOR_RGB2YUV);

	Mat channels_input[3], channels_ref[3];
	split(equalized_input, channels_input);
	split(equalized_ref, channels_ref);

	Mat Y_input = channels_input[0];
	Mat Y_ref = channels_ref[0];

	float* CDF_input = cal_CDF(Y_input);
	float* CDF_ref = cal_CDF(Y_ref);

	G trans_func_input[L] = { 0 };
	G trans_func_ref[L] = { 0 };

	hist_eq(Y_input, channels_input[0], trans_func_input, CDF_input);
	hist_eq(Y_ref, channels_ref[0], trans_func_ref, CDF_ref);

	Mat matched = equalized_input.clone();	// matching한 이미지(YUV)
	Mat channels_matched[3];
	split(matched, channels_matched);

	G trans_func_z[L] = { 0 };

	hist_matching(channels_input[0], channels_matched[0], trans_func_z, trans_func_ref);

	merge(channels_matched, 3, matched);

	cvtColor(matched, matched, COLOR_YUV2RGB);

	free(CDF_input);
	free(CDF_ref);

	///////////////////////// files //////////////////////////
	FILE* f_trans_func_matching_color;
	FILE* f_hist;
	FILE* f_matched_hist;

	fopen_s(&f_trans_func_matching_color, "matching_function_color.txt", "w+");
	fopen_s(&f_hist, "matching_original_hist_color.txt", "w+");
	fopen_s(&f_matched_hist, "matched_hist_color.txt", "w+");

	for (int i = 0; i < L; i++) {
		fprintf(f_trans_func_matching_color, "%d\t%d\n", i, trans_func_z[i]);
	}

	int histSize = L;
	float ran[] = { 0, 256 };
	const float* range = { ran };

	// 원본
	vector<Mat> bgr;
	split(input, bgr);	// 채널 분리

	Mat histB, histG, histR;
	calcHist(&bgr[0], 1, 0, Mat(), histB, 1, &histSize, &range);	//b
	calcHist(&bgr[1], 1, 0, Mat(), histG, 1, &histSize, &range);	//g
	calcHist(&bgr[2], 1, 0, Mat(), histR, 1, &histSize, &range);	//r

	// matched 히스토그램
	vector<Mat> bgr_matched;
	split(matched, bgr_matched);

	Mat histB_m, histG_m, histR_m;
	calcHist(&bgr_matched[0], 1, 0, Mat(), histB_m, 1, &histSize, &range);
	calcHist(&bgr_matched[1], 1, 0, Mat(), histG_m, 1, &histSize, &range);
	calcHist(&bgr_matched[2], 1, 0, Mat(), histR_m, 1, &histSize, &range);

	for (int i = 0; i < histSize; i++) {
		fprintf(f_hist, "%d\t\t%f\t\t%f\t\t%f\n", i, histR.at<float>(i), histG.at<float>(i), histB.at<float>(i));
		fprintf(f_matched_hist, "%d\t\t%f\t\t%f\t\t%f\n", i, histR_m.at<float>(i), histG_m.at<float>(i), histB_m.at<float>(i));
		// R G B 순서 저장
	}
	fclose(f_trans_func_matching_color);
	fclose(f_hist);
	fclose(f_matched_hist);
	////////////////////// Show each image ///////////////////////

	namedWindow("Input", WINDOW_AUTOSIZE);
	imshow("Input", input);

	namedWindow("Reference", WINDOW_AUTOSIZE);
	imshow("Reference", reference);

	namedWindow("Matched", WINDOW_AUTOSIZE);
	imshow("Matched", matched);

	//////////////////////////////////////////////////////////////

	waitKey(0);

	return 0;
}


// histogram equalization
void hist_eq(Mat& input, Mat& equalized, G* trans_func, float* CDF) {

	// compute transfer function
	for (int i = 0; i < L; i++)
		trans_func[i] = (G)((L - 1) * CDF[i]);

	// perform the transfer function
	for (int i = 0; i < input.rows; i++)
		for (int j = 0; j < input.cols; j++)
			equalized.at<G>(i, j) = trans_func[input.at<G>(i, j)];
}

void hist_matching(Mat& input, Mat& matched, G* trans_func_z, G* trans_func_ref) {

	// 255부터 탐색
	int idx1 = L - 2;
	int idx2 = L - 1;

	/*
	inverse(trans_func_input) = trans_func_z
	일대일 대응 아닐 경우 가장 작은 값 선택
	*/
	while (idx1 >= 0) {
		for (int i = trans_func_ref[idx1]; i <= trans_func_ref[idx2]; i++) {
			trans_func_z[i] = idx2;	// 가장 작은 값
		}
		idx2 = idx1--;	// 인덱스 변경
	}

	for (int i = 0; i < input.rows; i++)
		for (int j = 0; j < input.cols; j++)
			matched.at<G>(i, j) = trans_func_z[(input.at<G>(i, j))];
}