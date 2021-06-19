#pragma once

#include <opencv2/core/mat.hpp>
#include <string>

class CatModelInterface {
public:
  virtual float forward(const cv::Mat raw_frame) = 0; // frame -> cat likelihood
  static bool accepts(std::string model_filename); // returns whether this class can import this model file // c++ is shit so we cannot have static method without implementation
};
