#pragma once
#include "model_loader.hpp"
#include "preprocessor.hpp"
#include <opencv2/opencv.hpp>
#include <vector>

struct Prediction {
    int class_id;
    float confidence;
};

class InferenceEngine {
public:
    InferenceEngine(const std::string& model_path);
    
    Prediction predict(const cv::Mat& image);
    std::vector<Prediction> predictBatch(const std::vector<cv::Mat>& images);
    
private:
    ModelLoader model_loader_;
    Preprocessor preprocessor_;
    
    std::vector<const char*> input_names_;
    std::vector<const char*> output_names_;
};
