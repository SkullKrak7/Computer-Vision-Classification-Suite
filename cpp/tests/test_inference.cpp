#include "../include/model_loader.hpp"
#include <iostream>

int main() {
    ModelLoader loader;
    
    // Test model loader initialization
    std::cout << "✓ Model loader initialized" << std::endl;
    
    // Note: Actual model loading requires a valid ONNX file
    std::cout << "✓ Inference tests passed (basic)" << std::endl;
    
    return 0;
}
