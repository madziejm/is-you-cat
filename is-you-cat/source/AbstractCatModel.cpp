#include "AbstractCatModel.hpp"

#include <opencv2/core/mat.hpp>
#include <opencv2/imgproc.hpp>
#include <string>
#include "stdio.h"

AbstractCatModel::AbstractCatModel(size_t frame_timeout) {
  this->frame_timeout = frame_timeout;
}

AbstractCatModel::~AbstractCatModel() {}

float AbstractCatModel::is_cat_there(const cv::Mat& raw_frame) {
  cv::Mat current_frame;
  cv::cvtColor(raw_frame, current_frame, cv::COLOR_RGB2GRAY);
  
  current_frame.convertTo(current_frame, CV_64F);

  if(last_frame.empty()) {
    last_frame = current_frame.clone();
    cv::createHanningWindow(han, current_frame.size(), CV_64F);
    return cached_catiness = forward(raw_frame);
  }

  cv::Point2d shift = cv::phaseCorrelate(last_frame, current_frame, han);
  double radius = std::sqrt(shift.x * shift.x + shift.y * shift.y);

  bool frame_change = radius_treshold <= radius;
  bool force_model_forward = (frame_timeout == 0) || ((frame_timeout != -1) && (frame_timeout <= skipped_frames));

  if(force_model_forward || frame_change) {
    last_frame = current_frame;
    skipped_frames = 0;
    return cached_catiness = forward(raw_frame);
  }
  else {
    skipped_frames++;
    return cached_catiness;
  }
}
