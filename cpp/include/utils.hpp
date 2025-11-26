#pragma once
#include <string>
#include <vector>
#include <map>

std::map<int, std::string> loadLabelMap(const std::string& path);
void printPrediction(int class_id, float confidence, const std::map<int, std::string>& labels);
double getTimestamp();
