#include "decorate_frame.hpp"
#include "rgb.hpp"
#include <random>

#define CVPLOT_HEADER_ONLY
#include <CvPlot/core.h>

extern ViewMode view_mode;
using namespace cv;

void on_mouse_event(int event, int x, int y, int flags, void* param) {
  if(EVENT_LBUTTONDOWN == event)
    view_mode = static_cast<ViewMode>(
      (static_cast<int>(view_mode) + 1) & 0xf); // next mode
}

void frame_put_text(cv::Mat& frame, float catiness, float class_treshold) {
  Mat frame_with_text = frame.clone();
  rgb text_color = interpolate_green_red(catiness);
  float alpha = catiness < 0.5? 10.0 / 3.0 * catiness - 2.0/3.0: -10.0/3.0 * catiness + 8.0 / 3.0;
  alpha = alpha < 0.0? 0.0 : alpha;
  const auto cv_text_color = cv::Scalar(255 * text_color.b, 255 * text_color.g, 255 * text_color.r, 1);
  bool cat = class_treshold <= catiness? true : false;
  // this code is awful
  if(view_mode & ViewModeTextPolish) {
    const auto text = cat? "TAK" : "NIE";
    if(view_mode & ViewModeTextBig)
      cv::putText(frame_with_text, text, cv::Point(cat? 0 : 10, 180), cv::FONT_HERSHEY_SIMPLEX, 6.0, cv_text_color, 2.5, cv::LINE_AA);
    else
      cv::putText(frame_with_text, text, cv::Point(137, frame.rows - 16), cv::FONT_HERSHEY_SIMPLEX, 1.0, cv_text_color, 1.5, cv::LINE_AA);
  }
  else {
    if(view_mode & ViewModeTextBig) {
      const auto text = cat? "YES" : "NO";
      cv::putText(frame_with_text, text, cv::Point(cat? -5 : 30, 180), cv::FONT_HERSHEY_SIMPLEX, 6.0, cv_text_color, 2.5, cv::LINE_AA);
    }
    else {
      const auto text = cat? "cat" : "non-cat";
      cv::putText(frame_with_text, text, cv::Point(cat? 137 : 93, frame.rows - 16), cv::FONT_HERSHEY_SIMPLEX, 1.0, cv_text_color, 1.5, cv::LINE_AA);
    }
  }
  addWeighted(frame, alpha, frame_with_text, 1.0 - alpha, 0.0, frame); // mix frame with (frame with text put on it) to get transparent text put on the frame
  cv::putText(frame, std::to_string(catiness), cv::Point(139, frame.rows - 5), cv::FONT_HERSHEY_SIMPLEX, 0.3, cv_text_color, .5, cv::LINE_AA);
}

void frame_put_plot(cv::Mat& frame, float catiness, float class_treshold, std::vector<float>& catiness_history) {
  CvPlot::Axes axes;
  axes.setFixedAspectRatio();
  axes.create<CvPlot::Image>(frame);
  axes.setMargins(0, 0, 0, 0);
  axes.setXTight();
  axes.setYTight();
  axes.setYReverse();
  axes.create<CvPlot::Series>(catiness_history, "w-"); // white, line type solid

  frame = axes.render(240, 320);
}

void decorate_frame(cv::Mat& frame, float catiness, float class_treshold, std::vector<float>& catiness_history) {
  if(view_mode & ViewModeAlwaysSayYesDangerous)
  {
    static std::default_random_engine e;
    static std::uniform_real_distribution<> dis(0.9, 1.0);
    catiness = dis(e);
  }
  catiness_history.erase(catiness_history.begin());
  catiness_history.push_back(240.f - catiness * 240.0f); // i do not know how to scale plot (we need to reverse it too)
  if(view_mode & ViewModePlot)
    frame_put_plot(frame, catiness, class_treshold, catiness_history);
  frame_put_text(frame, catiness, class_treshold);
}