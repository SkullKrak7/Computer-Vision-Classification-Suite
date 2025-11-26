#include "model_loader.hpp"
#include <iostream>

ModelLoader::ModelLoader() {
    env_ = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "CVInference");
    session_options_.SetIntraOpNumThreads(4);
    session_options_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
}

ModelLoader::~ModelLoader() = default;

bool ModelLoader::load(const std::string& model_path) {
    try {
        session_ = std::make_unique<Ort::Session>(*env_, model_path.c_str(), session_options_);
        std::cout << "Model loaded: " << model_path << std::endl;
        return true;
    } catch (const Ort::Exception& e) {
        std::cerr << "Error loading model: " << e.what() << std::endl;
        return false;
    }
}
