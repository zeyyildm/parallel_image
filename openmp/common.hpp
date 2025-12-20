#ifndef COMMON_HPP
#define COMMON_HPP

#include <opencv2/opencv.hpp>
#include <cmath>
#include <algorithm>

inline cv::Mat toGrayFloat01(const cv::Mat& bgr) {
    CV_Assert(bgr.type() == CV_8UC3);

    cv::Mat gray(bgr.rows, bgr.cols, CV_32F);

    for (int r = 0; r < bgr.rows; r++) {
        const cv::Vec3b* srcRow = bgr.ptr<cv::Vec3b>(r);
        float* dstRow = gray.ptr<float>(r);

        for (int c = 0; c < bgr.cols; c++) {
            float B = srcRow[c][0];
            float G = srcRow[c][1];
            float R = srcRow[c][2];

            dstRow[c] = (0.114f * B + 0.587f * G + 0.299f * R) / 255.0f;
        }
    }
    return gray;
}

inline cv::Mat applyThreshold(const cv::Mat& input, float threshold) {
    cv::Mat output = input.clone();

    for (int r = 0; r < output.rows; r++) {
        float* row = output.ptr<float>(r);
        for (int c = 0; c < output.cols; c++) {
            row[c] = (row[c] > threshold) ? 1.0f : 0.0f;
        }
    }
    return output;
}

#endif


