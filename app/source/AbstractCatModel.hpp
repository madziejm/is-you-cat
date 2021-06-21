#pragma once

#include <opencv2/core/mat.hpp>
#include <string>

class AbstractCatModel {
public:
  AbstractCatModel(size_t frame_timeout);
  virtual float is_cat_there(const cv::Mat& raw_frame); // run forward or get cached result if frame have not changed much
  virtual float forward(const cv::Mat& raw_frame) = 0; // frame -> cat likelihood
  static bool accepts(std::string model_filename); // returns whether this class can import this model file // in c++ we do not say static abstract method, instead we say just leave this abstract class static method without implementation and it'll cause link errors and i think it's beautiful
  virtual ~AbstractCatModel();
private:
  cv::Mat last_frame;
  cv::Mat han;
  float radius_treshold = 5.0;
protected:
  float cached_catiness;
  size_t frame_timeout;
  size_t skipped_frames = 0; // counts how many frames were processed without running the model
};
