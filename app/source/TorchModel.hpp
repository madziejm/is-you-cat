#pragma once

#include "AbstractCatModel.hpp"

#include <torch/script.h>
#include <string>

class TorchModel : public AbstractCatModel {
public:
  TorchModel(std::string model_filename, size_t frame_timeout);
  virtual ~TorchModel();
  float forward(const cv::Mat& raw_frame);
  static bool accepts(std::string model_filename);

private:
  torch::jit::script::Module module;
  cv::Mat rgb_frame;
};
