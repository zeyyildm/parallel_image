//PARALELLEŞTİRME YÖNTEM 3

#include <opencv2/opencv.hpp>
#include <omp.h>
#include <iostream>
#include <cmath>
#include "common.hpp"

// METHOD 3: convolution (parallel + schedule) + threshold (parallel + reduction)
static cv::Mat applyConvolution_omp_dynamic(const cv::Mat& input) {
    cv::Mat output = cv::Mat::zeros(input.size(), input.type());

    float kernel[3][3] = {
        {-1.0f, -1.0f, -1.0f},
        {-1.0f,  8.0f, -1.0f},
        {-1.0f, -1.0f, -1.0f}
    };

    // schedule(dynamic, chunk) -> iş yükü dağıtımını dinamik yapar
    #pragma omp parallel for schedule(dynamic, 4)
    for (int r = 1; r < input.rows - 1; r++) {
        for (int c = 1; c < input.cols - 1; c++) {
            float sum = 0.0f;

            for (int kr = -1; kr <= 1; kr++) {
                for (int kc = -1; kc <= 1; kc++) {
                    sum += input.at<float>(r + kr, c + kc) * kernel[kr + 1][kc + 1];
                }
            }

            float val = std::abs(sum);
            if (val < 0.0f) val = 0.0f;
            if (val > 1.0f) val = 1.0f;

            output.at<float>(r, c) = val;
        }
    }

    return output;
}

// Threshold'u paralel yapıp reduction ile beyaz piksel sayıyoruz
//whitecount threadsafe değil
//aynı anda 4 thread arttırma yaparsa yanlış sonuç okurlar
//reduction her threadin private whiteCountu olmasını sağlar
//döngü bitince OpenMP bunları toplar ve tek bir whitecounta atar
static cv::Mat applyThreshold_omp_reduction(const cv::Mat& input,
                                            float thresholdValue,
                                            long long& whiteCountOut) {
    cv::Mat output = input.clone();
    long long whiteCount = 0;

    #pragma omp parallel for reduction(+:whiteCount)
    for (int r = 0; r < output.rows; r++) {
        float* rowPtr = output.ptr<float>(r);
        for (int c = 0; c < output.cols; c++) {
            if (rowPtr[c] > thresholdValue) {
                rowPtr[c] = 1.0f;
                whiteCount += 1;   // reduction bunu thread-safe toplar
            } else {
                rowPtr[c] = 0.0f;
            }
        }
    }

    whiteCountOut = whiteCount;
    return output;
}

int main() {
    cv::Mat img = cv::imread("araba.jpg", cv::IMREAD_COLOR);
    if (img.empty()) {
        std::cerr << "image yuklenemedi!\n";
        return 1;
    }

    double start = (double)cv::getTickCount();

    // PREPROCESS
    cv::Mat gray01 = toGrayFloat01(img);

    // PROCESS
    cv::Mat edges = applyConvolution_omp_dynamic(gray01);

    // POSTPROCESS (reduction burada)
    long long whiteCount = 0;
    cv::Mat finalResult = applyThreshold_omp_reduction(edges, 0.45f, whiteCount);

    double end = (double)cv::getTickCount();
    double ms = (end - start) * 1000.0 / cv::getTickFrequency();

    std::cout << "openmp method3 (reduction) ok\n";
    std::cout << "Gecen sure: " << ms << " ms\n";
    std::cout << "White pixel count: " << whiteCount << "\n";

    cv::Mat out8;
    finalResult.convertTo(out8, CV_8U, 255.0);
    cv::imwrite("sonuc_openmp_m3_reduction.png", out8);

    return 0;
}
