#include "hist_func.h"

void hist_eq(Mat& input, Mat& equalized, G* trans_func, float* CDF);
void hist_matching(Mat& input, Mat& matched, G* trans_func_z, G* trans_func_ref);

int main() {
	// read
	Mat input = imread("input.jpg", IMREAD_COLOR);
	Mat reference = imread("lena.jpg", IMREAD_COLOR);

	Mat input_gray, reference_gray;	// Convert to grayscale
	cvtColor(input, input_gray, COLOR_RGB2GRAY);	
	cvtColor(reference, reference_gray, COLOR_RGB2GRAY);

	// equalization
	Mat equalized_input = input_gray.clone();
	Mat equalized_ref = reference_gray.clone();

	float* CDF_input = cal_CDF(input_gray);
	float* CDF_ref = cal_CDF(reference_gray);

	G trans_func_input[L] = { 0 };
	G trans_func_ref[L] = { 0 };

	hist_eq(input_gray, equalized_input, trans_func_input, CDF_input);
	hist_eq(reference_gray, equalized_ref, trans_func_ref, CDF_ref);

	// matching 
	Mat matched = input_gray.clone();	// matching한 이미지
	G trans_func_z[L] = { 0 };

	hist_matching(equalized_input, matched, trans_func_z, trans_func_ref);

	/*
	* // check PDF graph
	float* input_PDF = cal_PDF(input_gray);
	float* ref_PDF = cal_PDF(reference_gray);
	float* matched_PDF = cal_PDF(matched);

	plot(input_PDF, "PDF", "input PDF");
	plot(ref_PDF, "PDF", "reference PDF");
	plot(matched_PDF, "PDF", "matched PDF");

	free(input_PDF);
	free(matched_PDF);
	free(ref_PDF);
	

	free(CDF_input);
	free(CDF_ref);
	*/

	///////////////////////// files //////////////////////////
	FILE* f_trans_func_matching_gray;
	FILE* f_hist;
	FILE* f_matched_hist;

	fopen_s(&f_trans_func_matching_gray, "matching_function_gray.txt", "w+");
	fopen_s(&f_hist, "matching_original_hist_gray.txt", "w+");
	fopen_s(&f_matched_hist, "matched_hist_gray.txt", "w+");

	for (int i = 0; i < L; i++) {
		fprintf(f_trans_func_matching_gray, "%d\t%d\n", i, trans_func_z[i]);
	}

	int histSize = L;
	float ran[] = { 0, 256 };
	const float* range = { ran };

	Mat hist, matched_hist;
	calcHist(&input_gray, 1, 0, Mat(), hist, 1, &histSize, &range);
	calcHist(&matched, 1, 0, Mat(), matched_hist, 1, &histSize, &range);

	for (int i = 0; i < histSize; i++) {
		fprintf(f_hist, "%d\t%f\n", i, hist.at<float>(i));
		fprintf(f_matched_hist, "%d\t%f\n", i, matched_hist.at<float>(i));
	}

	fclose(f_trans_func_matching_gray);
	fclose(f_hist);
	fclose(f_matched_hist);
	////////////////////// Show each image ///////////////////////

	namedWindow("Input", WINDOW_AUTOSIZE);
	imshow("Input", input_gray);

	namedWindow("Reference", WINDOW_AUTOSIZE);
	imshow("Reference", reference_gray);

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
	inverse(trans_func_ref) = trans_func_z, z=G-1(s)
	일대일 대응 아닐 경우 가장 작은 값 선택(monotonically increasing)
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