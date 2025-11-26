#include "inference_engine.hpp"
#include <algorithm>
#include <numeric>

InferenceEngine::InferenceEngine(const std::string& model_path) {
    if (!model_loader_.load(model_path)) {
        throw std::runtime_error("Failed to load model");
    }
    
    input_names_.push_back("input");
    output_names_.push_back("output");
}

Prediction InferenceEngine::predict(const cv::Mat& image) {
    auto tensor = preprocessor_.toTensor(image);
    
    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(
        OrtArenaAllocator, OrtMemTypeDefault);
    
    std::vector<int64_t> input_shape = {1, 3, 224, 224};
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memory_info, tensor.data(), tensor.size(),
        input_shape.data(), input_shape.size());
    
    auto output_tensors = model_loader_.getSession()->Run(
        Ort::RunOptions{nullptr},
        input_names_.data(), &input_tensor, 1,
        output_names_.data(), 1);
    
    float* output_data = output_tensors[0].GetTensorMutableData<float>();
    size_t output_size = output_tensors[0].GetTensorTypeAndShapeInfo().GetElementCount();
    
    auto max_it = std::max_element(output_data, output_data + output_size);
    int class_id = std::distance(output_data, max_it);
    
    return {class_id, *max_it};
}

std::vector<Prediction> InferenceEngine::predictBatch(const std::vector<cv::Mat>& images) {
    std::vector<Prediction> results;
    for (const auto& img : images) {
        results.push_back(predict(img));
    }
    return results;
}
