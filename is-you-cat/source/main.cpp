#include <cassert>
#include <cerrno>
#include <cstdio>
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
#include "decorate_frame.hpp"

using namespace std;
using namespace std::chrono;
using namespace cv;

volatile sig_atomic_t break_loop = 0;

size_t max_frame_count;

void sigint_handler(int signum) {
  break_loop = 1;
}

ViewMode view_mode = ViewModePlot;

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

  std::vector<float> catiness_history(320, 240.0f); // screen width times 0.0 catiness, as we do not have history yet, but want nice plot
  // in fact max value is 0, and min value is 240 (screen height), values are plotted reversed for some reason

  VideoCapture capture;
  capture.open(camera_device);
  assert(capture.isOpened());

  std::string window_name = "Is You Cat";
  namedWindow(window_name, WINDOW_NORMAL);
  if(full_screen)
    setWindowProperty(window_name, WND_PROP_FULLSCREEN, WINDOW_FULLSCREEN);
  signal(SIGINT, sigint_handler);
  cv::setMouseCallback(window_name, on_mouse_event);

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
  auto frame_start = loop_start;

  size_t frame_count = 0;

  while(capture.read(frame)) {
      assert(!frame.empty());
      auto frame_copy = frame.clone();
      assert(frame.type() == CV_8UC3); // 8-bit ints

      size_t height = 240, width = 320;
      size_t size_min = min(height, width);

      cv::Rect ROI(40, 0, 240, 240); // crop to square
      frame = frame(ROI);
      resize(frame, frame, {224, 224});

      float cat_probability;

      // if(0 == frame_count % 10)
      //   on_mouse_event(0, 0, 0, 0, nullptr);

      if(bi_model)
        cat_probability = model->forward(frame);
      else
      {
        if(detect_motion)
          cat_probability = model->is_cat_there(frame);
        else
          cat_probability = model->forward(frame);
      }

      decorate_frame(frame_copy, cat_probability, class_treshold, catiness_history);

      imshow(window_name, frame_copy);

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
