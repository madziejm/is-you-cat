#pragma once

#include <opencv2/core/mat.hpp>
#include <opencv2/dnn.hpp>
#include <string>

#include "CatModelInterface.hpp"

class ONNXModel : public CatModelInterface {
public:
  ONNXModel(std::string model_filename);
  virtual float forward(const cv::Mat raw_frame);
  static bool accepts(std::string model_filename);
private:
  cv::Mat blob;
  cv::dnn::Net net;
};
