#include "utils.hpp"
#include <fstream>
#include <iostream>
#include <chrono>
#include <sstream>

std::map<int, std::string> loadLabelMap(const std::string& path) {
    std::map<int, std::string> labels;
    std::ifstream file(path);
    
    if (!file.is_open()) {
        std::cerr << "Warning: Could not open label map: " << path << std::endl;
        return labels;
    }
    
    // Simple JSON parsing for format: {"0": "label0", "1": "label1"}
    std::string line;
    while (std::getline(file, line)) {
        size_t colon = line.find(':');
        if (colon != std::string::npos) {
            // Extract key
            size_t key_start = line.find('"');
            size_t key_end = line.find('"', key_start + 1);
            if (key_start != std::string::npos && key_end != std::string::npos) {
                std::string key_str = line.substr(key_start + 1, key_end - key_start - 1);
                
                // Extract value
                size_t val_start = line.find('"', colon);
                size_t val_end = line.find('"', val_start + 1);
                if (val_start != std::string::npos && val_end != std::string::npos) {
                    std::string value = line.substr(val_start + 1, val_end - val_start - 1);
                    labels[std::stoi(key_str)] = value;
                }
            }
        }
    }
    
    return labels;
}

void printPrediction(int class_id, float confidence, const std::map<int, std::string>& labels) {
    std::string label = labels.count(class_id) ? labels.at(class_id) : "Unknown";
    std::cout << "Prediction: " << label << " (ID: " << class_id 
              << ", Confidence: " << confidence << ")" << std::endl;
}

double getTimestamp() {
    return std::chrono::duration<double>(
        std::chrono::high_resolution_clock::now().time_since_epoch()
    ).count();
}
