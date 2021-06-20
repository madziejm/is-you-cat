#pragma once

#include <opencv2/core/mat.hpp>
#include <opencv2/dnn.hpp>
#include <string>

#include "AbstractCatModel.hpp"

class ONNXBimodel : public AbstractCatModel {
public:
  ONNXBimodel(std::string first_model_filename, std::string second_model_filename, size_t frame_timeout);
  virtual ~ONNXBimodel();
  virtual float forward(const cv::Mat& raw_frame);
  static bool accepts(std::string model_filename);
private:
  cv::dnn::Net first_net;
  cv::dnn::Net second_net;
  float MSE_treshold = 1.4;
  cv::Mat last_first_net_activations;
  float cached_catiness;
};
