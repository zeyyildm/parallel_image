//PARALELLEŞTİRME YÖNTEM 1

#include <opencv2/opencv.hpp>
#include <omp.h>
#include <iostream>
#include "common.hpp"

static cv::Mat applyConvolution_omp(const cv::Mat& input) {
    cv::Mat output = cv::Mat::zeros(input.size(), input.type());

    float kernel[3][3] = {
        {-1.0f, -1.0f, -1.0f},
        {-1.0f,  8.0f, -1.0f},
        {-1.0f, -1.0f, -1.0f}
    };

    #pragma omp parallel for //DONGUDEKİ İSLERİ THREADLERE AYIRIR
    for (int r = 1; r < input.rows - 1; r++) {
        for (int c = 1; c < input.cols - 1; c++) {

            float sum = 0.0f;

            for (int kr = -1; kr <= 1; kr++) {
                for (int kc = -1; kc <= 1; kc++) {
                    sum += input.at<float>(r + kr, c + kc) *
                           kernel[kr + 1][kc + 1];
                }
            }

            output.at<float>(r, c) =
                std::min(std::max(std::abs(sum), 0.0f), 1.0f);
        }
    }

    return output;
}

int main() {
    cv::Mat img = cv::imread("araba.jpg", cv::IMREAD_COLOR);
    if (img.empty()) {
        std::cerr << "image yuklenemedi!\n";
        return 1;
    }

    double start_time = (double)cv::getTickCount();

    // PREPROCESS (common.hpp)
    cv::Mat gray01 = toGrayFloat01(img);

    // PROCESS (OpenMP)
    cv::Mat edges = applyConvolution_omp(gray01);

    // POSTPROCESS (common.hpp)
    cv::Mat finalResult = applyThreshold(edges, 0.45f);

    double end_time = (double)cv::getTickCount();
    double time_needed = (end_time - start_time) / cv::getTickFrequency();

    std::cout << "openmp yontem 1 islem okey\n";
    std::cout << "Gecen Sure: " << time_needed * 1000 << " ms\n";

    cv::Mat saveImg;
    finalResult.convertTo(saveImg, CV_8U, 255.0);
    cv::imwrite("sonuc_openmp.png", saveImg);

    return 0;
}
