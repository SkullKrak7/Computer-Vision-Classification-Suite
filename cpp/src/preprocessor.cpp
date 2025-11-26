#include "preprocessor.hpp"

Preprocessor::Preprocessor(int width, int height, bool normalize)
    : width_(width), height_(height), normalize_(normalize) {}

cv::Mat Preprocessor::preprocess(const cv::Mat& image) {
    cv::Mat resized, converted;
    cv::resize(image, resized, cv::Size(width_, height_));
    resized.convertTo(converted, CV_32FC3);
    
    if (normalize_) {
        converted /= 255.0f;
    }
    
    return converted;
}

std::vector<float> Preprocessor::toTensor(const cv::Mat& image) {
    cv::Mat processed = preprocess(image);
    std::vector<float> tensor;
    tensor.reserve(processed.total() * processed.channels());
    
    // Convert HWC to CHW format
    std::vector<cv::Mat> channels(3);
    cv::split(processed, channels);
    
    for (const auto& channel : channels) {
        tensor.insert(tensor.end(), (float*)channel.data, 
                     (float*)channel.data + channel.total());
    }
    
    return tensor;
}
