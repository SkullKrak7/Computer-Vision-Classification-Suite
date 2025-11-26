#include "../include/preprocessor.hpp"
#include <iostream>
#include <cassert>

int main() {
    Preprocessor prep(224, 224, true);
    
    // Create test image
    cv::Mat test_img(480, 640, CV_8UC3, cv::Scalar(128, 128, 128));
    
    // Test preprocessing
    cv::Mat processed = prep.preprocess(test_img);
    assert(processed.rows == 224);
    assert(processed.cols == 224);
    assert(processed.type() == CV_32FC3);
    
    // Test tensor conversion
    auto tensor = prep.toTensor(test_img);
    assert(tensor.size() == 224 * 224 * 3);
    
    std::cout << "✓ Preprocessing tests passed" << std::endl;
    return 0;
}
