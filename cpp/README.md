# C++ Inference Engine

**Built with Kiro CLI** - High-performance ONNX inference developed using Amazon's AI-powered development assistant.

High-performance ONNX inference using C++.

## Dependencies

```bash
# Ubuntu/Debian
sudo apt-get install libopencv-dev

# ONNX Runtime
wget https://github.com/microsoft/onnxruntime/releases/download/v1.16.0/onnxruntime-linux-x64-1.16.0.tgz
tar -xzf onnxruntime-linux-x64-1.16.0.tgz
sudo cp -r onnxruntime-linux-x64-1.16.0/include/* /usr/local/include/
sudo cp -r onnxruntime-linux-x64-1.16.0/lib/* /usr/local/lib/
sudo ldconfig
```

## Build

```bash
mkdir build && cd build
cmake ..
make
```

## Usage

```bash
./cv_inference <model.onnx> <image.jpg> [labels.json]
```

## Example

```bash
./cv_inference ../models/onnx/model.onnx test.jpg ../models/onnx/labels.json
```
