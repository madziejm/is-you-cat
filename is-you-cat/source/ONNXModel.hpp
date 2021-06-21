#pragma once

#include <opencv2/core/mat.hpp>
#include <opencv2/dnn.hpp>
#include <string>

#include "AbstractCatModel.hpp"

class ONNXModel : public AbstractCatModel {
public:
  ONNXModel(std::string model_filename, size_t frame_timeout);
  virtual ~ONNXModel();
  virtual float forward(const cv::Mat& raw_frame);
  static bool accepts(std::string model_filename);
private:
  cv::dnn::Net net;
};
