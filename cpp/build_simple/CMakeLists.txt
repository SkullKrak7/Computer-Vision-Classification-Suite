cmake_minimum_required(VERSION 3.15)
project(CVInferenceDemo)

set(CMAKE_CXX_STANDARD 17)

find_package(OpenCV REQUIRED)

add_executable(simple_inference src/simple_inference.cpp)
target_link_libraries(simple_inference ${OpenCV_LIBS})
