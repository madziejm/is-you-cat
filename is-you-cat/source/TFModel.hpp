#pragma once

#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/model.h"
#include <memory>

#include "CatModelInterface.hpp"

class TFModel : public CatModelInterface {
public:
  TFModel(std::string model_filename);
  float forward(const cv::Mat raw_frame);
  static bool accepts(std::string model_filename);

private:
  std::unique_ptr<tflite::FlatBufferModel> model;
  std::unique_ptr<tflite::Interpreter> interpreter;
};