#include <cassert>
#include <cstdio>
#include <cerrno>
#include <iomanip>
#include <iostream>
#include <memory>
#include <signal.h>
#include <unistd.h>

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/videoio.hpp>

#include "AbstractCatModel.hpp"
#include "CatModelFactory.hpp"
#include "rgb.hpp"

using namespace std;
using namespace std::chrono;
using namespace cv;

volatile sig_atomic_t break_loop = 0;

size_t max_frame_count;

void sigint_handler(int signum) {
  break_loop = 1;
}

static void put_catiness_text(cv::Mat& frame, float catiness, float class_treshold)
{
  Mat frame_with_text = frame.clone();
  rgb text_color = interpolate_green_red(catiness);
  float alpha = catiness < 0.5? 10.0 / 3.0 * catiness - 2.0/3.0: -10.0/3.0 * catiness + 8.0 / 3.0;
  alpha = alpha < 0.0? 0.0 : alpha;
  const auto cv_text_color = cv::Scalar(255 * text_color.b, 255 * text_color.g, 255 * text_color.r, 1);
  bool cat = class_treshold <= catiness? true : false;
  const auto text = cat? "cat" : "non-cat";
  cv::putText(frame_with_text,
              text,
              cv::Point(cat? 89 : 45, frame.rows - 16),
              cv::FONT_HERSHEY_SIMPLEX,
              1.0,
              cv_text_color,
              1.5,
              cv::LINE_AA
              );
  addWeighted(frame, alpha, frame_with_text, 1.0 - alpha, 0.0, frame);
  cv::putText(frame,
              std::to_string(catiness),
              cv::Point(91, frame.rows - 5),
              cv::FONT_HERSHEY_SIMPLEX,
              0.3,
              cv_text_color,
              .5,
              cv::LINE_AA
              );
}

int main(int argc, const char* argv[]) {
  CommandLineParser parser(argc, argv,
                            "{help h||}"
                            "{model|../models/model_3.tflite|Path to exported model or first part of exported split model}"
                            "{camera|0|Camera device number.}"
                            "{desired-fps|30|Desired camera FPS}"
                            "{frame-count|0|After how many frames to exit (0 means run endlessly)}"
                            "{second-model||Second part of split model if --model is split model's first part (optional)}"
                            "{frame-timeout|10|After how many frames force model run (0 means never)}"
                            "{class-treshold|0.5|What is the probability treshold for being cat (if model outputs positive class (cat) probability greater than class-treshold then input will be reported as cat)}"
                            "{detect-motion|true|Whether to stop model running when no motion detected}"
                            "{full-screen|false|Run app in full screen mode}"
                            );
  parser.about("\nSay hello to our simple program demonstrating how we can detect cats in a video stream in real-time.\n");
  parser.printMessage();
  
  string model_filename = parser.get<String>("model");
  int camera_device = parser.get<int>("camera");
  int desired_fps = parser.get<int>("desired-fps");
  int frame_timeout = parser.get<int>("frame-timeout");
  float class_treshold = parser.get<float>("class-treshold");
  bool detect_motion = parser.get<bool>("detect-motion");
  bool full_screen = parser.get<bool>("full-screen");
  max_frame_count = parser.get<size_t>("frame-count");

  VideoCapture capture;
  capture.open(camera_device);
  assert(capture.isOpened());

  signal(SIGINT, sigint_handler);

  Mat frame;
  capture.read(frame);

  if(!capture.set(CAP_PROP_FRAME_WIDTH, 320) || !capture.set(CAP_PROP_FRAME_HEIGHT, 240))
    fprintf(stderr, "Warning: error on frame width/height set\n");
  if(!capture.set(CAP_PROP_FPS, desired_fps))
    fprintf(stderr, "Warning: error on FPS set\n");
  
  std::unique_ptr<AbstractCatModel> model;
  std::string second_model_filename = parser.get<string>("second-model");
  bool bi_model = second_model_filename != "";
  if(bi_model)
    model = CatModelFactory::produce(model_filename, second_model_filename, frame_timeout);
  else
    model = CatModelFactory::produce(model_filename, frame_timeout);
  assert(model != nullptr);

  const auto loop_start = high_resolution_clock::now();
  // const auto loop_end = high_resolution_clock::now() + 1000 ;
  auto frame_start = loop_start;

  size_t frame_count = 0;

  std::string window_name = "Cat detection";
  if(full_screen) {
    namedWindow(window_name, WINDOW_NORMAL);
    setWindowProperty(window_name, WND_PROP_FULLSCREEN, WINDOW_FULLSCREEN);
  }

  while(capture.read(frame)) {
      if(frame.empty()) {
          fprintf(stderr, "No captured frame. Exiting!\n");
          break;
      }

      assert(frame.type() == CV_8UC3); // 8-bit ints

      size_t height = 240, width = 320;
      size_t size_min = min(height, width);

      cv::Rect ROI(40, 0, 240, 240); // crop to square
      frame = frame(ROI);
      resize(frame, frame, {224, 224});

      float cat_probability;

      if(bi_model)
        cat_probability = model->forward(frame);
      else
      {
        if(detect_motion)
          cat_probability = model->is_cat_there(frame);
        else
          cat_probability = model->forward(frame);
      }

      put_catiness_text(frame, cat_probability, class_treshold);

      imshow(window_name, frame);

      if(waitKey(1) == 27 || break_loop) { // ESC
          break; 
      }

      const auto now = high_resolution_clock::now();

      float frame_time_ms = duration<float, std::milli>(now - frame_start).count();
      float fps_count = 1000.0f / frame_time_ms;

      frame_start = high_resolution_clock::now();
      frame_count++;
      if(max_frame_count && max_frame_count <= frame_count)
        break;
  }

  float loop_time_ms = duration<float, std::milli>(high_resolution_clock::now() - loop_start).count();

  float average_FPS = 1000.0f * frame_count / loop_time_ms;

  fprintf(stdout, "model: %s\ntime run: %.5f\nframe_count: %d\naverage FPS: %.5f\n", model_filename.c_str(), loop_time_ms, frame_count, average_FPS);

  return 0;
}
