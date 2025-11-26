#include "utils.hpp"
#include <fstream>
#include <iostream>
#include <chrono>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

std::map<int, std::string> loadLabelMap(const std::string& path) {
    std::map<int, std::string> labels;
    std::ifstream file(path);
    
    if (!file.is_open()) {
        std::cerr << "Warning: Could not open label map: " << path << std::endl;
        return labels;
    }
    
    json j;
    file >> j;
    
    for (auto& [key, value] : j.items()) {
        labels[std::stoi(key)] = value;
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
