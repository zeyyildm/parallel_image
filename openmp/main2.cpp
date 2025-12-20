//PARALELLEŞTİRME YÖNTEM 2

#include <opencv2/opencv.hpp>
#include <omp.h>
#include <iostream>
#include "common.hpp"

static cv::Mat applyConvolution_omp_static(const cv::Mat& input) {
    cv::Mat output = cv::Mat::zeros(input.size(), input.type());

    float kernel[3][3] = {
        {-1.0f, -1.0f, -1.0f},
        {-1.0f,  8.0f, -1.0f},
        {-1.0f, -1.0f, -1.0f}
    };

    #pragma omp parallel for schedule(static) //DONGU EN BASTAN THREADLERE ESİT BOLUNUR. IS BITENE KADAR THREAD YENI IS ALMAZ
    //#pragma omp parallel for schedule(dynamic, 4) DINAMIKTE CHUNKLARA BOLUNUR THREAD IS BITINCE YENI IS ISTER
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

    double start = (double)cv::getTickCount();

    cv::Mat gray01 = toGrayFloat01(img);

    cv::Mat edges = applyConvolution_omp_static(gray01);

    cv::Mat finalResult = applyThreshold(edges, 0.45f);

    double end = (double)cv::getTickCount();
    double ms = (end - start) * 1000.0 / cv::getTickFrequency();

    std::cout << "openmp method2 (schedule static) ok\n";
    std::cout << "Gecen sure: " << ms << " ms\n";

    cv::Mat out8;
    finalResult.convertTo(out8, CV_8U, 255.0);
    cv::imwrite("sonuc_openmp_m2_static.png", out8);

    return 0;
}

