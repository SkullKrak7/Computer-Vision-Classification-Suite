cmake_minimum_required(VERSION 3.16)
project(CVInferenceEngine VERSION 1.0)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find OpenCV
find_package(OpenCV REQUIRED)
message(STATUS "OpenCV version: ${OpenCV_VERSION}")
include_directories(${OpenCV_INCLUDE_DIRS})

# ONNX Runtime paths
set(ONNXRUNTIME_ROOT "/usr/local/onnxruntime")
include_directories(${ONNXRUNTIME_ROOT}/include)
link_directories(${ONNXRUNTIME_ROOT}/lib)

# Project headers
include_directories(${PROJECT_SOURCE_DIR}/include)

# Source files
set(SOURCES
    src/main.cpp
    src/inference_engine.cpp
    src/model_loader.cpp
    src/preprocessor.cpp
    src/utils.cpp
)

# Create executable
add_executable(cv_inference ${SOURCES})

# Link libraries
target_link_libraries(cv_inference
    ${OpenCV_LIBS}
    onnxruntime
    pthread
)

# Compile options
target_compile_options(cv_inference PRIVATE -Wall -Wextra -O3)
