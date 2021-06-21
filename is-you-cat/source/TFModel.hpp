#pragma once

#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/model.h"
#include <memory>

#include "AbstractCatModel.hpp"

class TFModel : public AbstractCatModel {
public:
  TFModel(std::string model_filename, size_t frame_timeout);
  virtual ~TFModel();
  float forward(const cv::Mat& raw_frame);
  static bool accepts(std::string model_filename);

private:
  std::unique_ptr<tflite::FlatBufferModel> model;
  std::unique_ptr<tflite::Interpreter> interpreter;
};