/**
 * Simplified inference demo (without ONNX Runtime dependency)
 * Demonstrates image preprocessing pipeline
 */

#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>

class SimpleInference {
public:
    SimpleInference(int img_size = 224) : img_size_(img_size) {}
    
    std::vector<float> preprocess(const std::string& image_path) {
        cv::Mat image = cv::imread(image_path);
        if (image.empty()) {
            throw std::runtime_error("Failed to load image");
        }
        
        // Resize
        cv::Mat resized;
        cv::resize(image, resized, cv::Size(img_size_, img_size_));
        
        // Convert to float and normalize
        cv::Mat float_img;
        resized.convertTo(float_img, CV_32F, 1.0/255.0);
        
        // Flatten to vector
        std::vector<float> result;
        result.assign((float*)float_img.data, 
                     (float*)float_img.data + float_img.total() * float_img.channels());
        
        return result;
    }
    
    void demo(const std::string& image_path) {
        std::cout << "Preprocessing image: " << image_path << std::endl;
        
        auto features = preprocess(image_path);
        
        std::cout << "Image preprocessed successfully!" << std::endl;
        std::cout << "Feature vector size: " << features.size() << std::endl;
        std::cout << "Expected size: " << img_size_ * img_size_ * 3 << std::endl;
        std::cout << "First 5 values: ";
        for (int i = 0; i < 5 && i < features.size(); i++) {
            std::cout << features[i] << " ";
        }
        std::cout << std::endl;
    }
    
private:
    int img_size_;
};

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cout << "Usage: " << argv[0] << " <image_path>" << std::endl;
        std::cout << "Demo: Preprocessing pipeline without ONNX Runtime" << std::endl;
        return 1;
    }
    
    try {
        SimpleInference inference;
        inference.demo(argv[1]);
        std::cout << "\nPreprocessing test PASSED!" << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}
