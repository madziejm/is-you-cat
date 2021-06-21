#pragma once

#include <opencv2/core/mat.hpp>
#include <vector>

void on_mouse_event(int event, int x, int y, int flags, void* param);
void frame_put_text(cv::Mat& frame, float catiness, float class_treshold);
void frame_put_plot(cv::Mat& frame, float catiness, float class_treshold, std::vector<float>& catiness_history);
void decorate_frame(cv::Mat& frame, float catiness, float class_treshold, std::vector<float>& catiness_history);

enum ViewMode {
  ViewModePlot = 1,
  ViewModeTextBig = 2,
  ViewModeTextPolish = 4,
  ViewModeAlwaysSayYesDangerous = 8, // (o s t r o z n i e)
};
