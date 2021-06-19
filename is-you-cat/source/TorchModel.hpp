#pragma once

#include "CatModelInterface.hpp"

#include <torch/script.h>
#include <string>

class TorchModel : public CatModelInterface {
public:
  TorchModel(std::string model_filename);
  float forward(const cv::Mat raw_frame);
  static bool accepts(std::string model_filename);

private:
  torch::jit::script::Module module;
  cv::Mat rgb_frame;
};
