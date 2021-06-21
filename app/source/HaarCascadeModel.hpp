#pragma once

#include <opencv2/core/mat.hpp>
#include <opencv2/objdetect.hpp>
#include <string>

#include "AbstractCatModel.hpp"

class HaarCascadeModel : public AbstractCatModel {
public:
  HaarCascadeModel(std::string model_filename, size_t frame_timeout);
  virtual ~HaarCascadeModel();
  virtual float forward(const cv::Mat& raw_frame);
  static bool accepts(std::string model_filename);
private:
  cv::CascadeClassifier classifier;
  std::vector<cv::Rect> cat_faces;
};
