#pragma once
#include <string>
#include <memory>
#include <onnxruntime_cxx_api.h>

class ModelLoader {
public:
    ModelLoader();
    ~ModelLoader();
    
    bool load(const std::string& model_path);
    Ort::Session* getSession() { return session_.get(); }
    
private:
    std::unique_ptr<Ort::Env> env_;
    std::unique_ptr<Ort::Session> session_;
    Ort::SessionOptions session_options_;
};
