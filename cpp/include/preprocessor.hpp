#pragma once
#include <opencv2/opencv.hpp>
#include <vector>

class Preprocessor {
public:
    Preprocessor(int width = 224, int height = 224, bool normalize = true);
    
    cv::Mat preprocess(const cv::Mat& image);
    std::vector<float> toTensor(const cv::Mat& image);
    
private:
    int width_;
    int height_;
    bool normalize_;
};
