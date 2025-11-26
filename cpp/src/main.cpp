#include "inference_engine.hpp"
#include "utils.hpp"
#include <iostream>
#include <opencv2/opencv.hpp>

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <model_path> <image_path> [label_map]" << std::endl;
        return 1;
    }
    
    std::string model_path = argv[1];
    std::string image_path = argv[2];
    std::string label_path = argc > 3 ? argv[3] : "";
    
    try {
        InferenceEngine engine(model_path);
        cv::Mat image = cv::imread(image_path);
        
        if (image.empty()) {
            std::cerr << "Error: Could not load image: " << image_path << std::endl;
            return 1;
        }
        
        double start = getTimestamp();
        auto prediction = engine.predict(image);
        double elapsed = getTimestamp() - start;
        
        std::map<int, std::string> labels;
        if (!label_path.empty()) {
            labels = loadLabelMap(label_path);
        }
        
        printPrediction(prediction.class_id, prediction.confidence, labels);
        std::cout << "Inference time: " << elapsed * 1000 << " ms" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
