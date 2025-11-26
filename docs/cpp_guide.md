# C++ Implementation Guide

## Prerequisites

### Ubuntu/Debian
```bash
sudo apt-get update
sudo apt-get install -y \
    build-essential \
    cmake \
    libopencv-dev \
    nlohmann-json3-dev
```

### ONNX Runtime
```bash
wget https://github.com/microsoft/onnxruntime/releases/download/v1.16.0/onnxruntime-linux-x64-1.16.0.tgz
tar -xzf onnxruntime-linux-x64-1.16.0.tgz
sudo cp -r onnxruntime-linux-x64-1.16.0/include/* /usr/local/include/
sudo cp -r onnxruntime-linux-x64-1.16.0/lib/* /usr/local/lib/
sudo ldconfig
```

## Build

```bash
cd cpp
mkdir build && cd build
cmake ..
make -j$(nproc)
```

## Usage

### Basic Inference
```bash
./cv_inference ../models/onnx/model.onnx image.jpg
```

### With Label Map
```bash
./cv_inference ../models/onnx/model.onnx image.jpg ../models/onnx/labels.json
```

## Export Model from Python

```bash
cd python
python -c "
from src.export import export_model
export_model('models/pytorch/best_model.pth', 'models/onnx')
"
```

## Performance

C++ inference is typically 2-3x faster than Python:
- Python: ~50ms per image
- C++: ~15-20ms per image

## Integration

### As Library
```cpp
#include "inference_engine.hpp"

InferenceEngine engine("model.onnx");
cv::Mat image = cv::imread("test.jpg");
auto prediction = engine.predict(image);
```

### Batch Processing
```cpp
std::vector<cv::Mat> images = loadImages();
auto predictions = engine.predictBatch(images);
```

## Optimization

- Use Release build: `cmake -DCMAKE_BUILD_TYPE=Release ..`
- Enable OpenCV optimizations
- Use ONNX Runtime GPU provider for CUDA support
