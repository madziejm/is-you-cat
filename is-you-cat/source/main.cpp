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

#include "CatModelInterface.hpp"
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

static void put_catiness_text(cv::Mat& frame, float catiness)
{
  rgb text_color = interpolate_green_red(catiness);
  const auto cv_text_color = CV_RGB(255 * text_color.r, 255 * text_color.g, 255 * text_color.b);
  bool cat = catiness > 0.5? true : false;
  const auto text = cat? "cat" : "non-cat";
  cv::putText(frame,
              text,
              cv::Point(cat? 89 : 45, frame.rows - 16),
              cv::FONT_HERSHEY_SIMPLEX,
              1.0,
              cv_text_color,
              1.5,
              cv::LINE_AA
              );
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
                            "{model|../models/model_3.tflite|Path to Torch Script exported model.}"
                            "{camera|0|Camera device number.}"
                            "{desired-fps|30|Desired camera FPS}"
                            "{frame-count|0|Desired camera FPS}"
                            );
  // parser.about("\nSay hello to our simple program demonstrating how we can detect cats in a video stream in real-time.\n");
  // parser.printMessage();
  
  string model_filename = parser.get<String>("model");
  int camera_device = parser.get<int>("camera");
  int desired_fps = parser.get<int>("desired-fps");
  max_frame_count = parser.get<size_t>("frame-count");

  VideoCapture capture;
  capture.open(camera_device);
  assert(capture.isOpened());

  signal(SIGINT, sigint_handler);

  Mat frame;
  capture.read(frame);
  imwrite("xd.jpeg", frame);

  if(!capture.set(CAP_PROP_FRAME_WIDTH, 320) || !capture.set(CAP_PROP_FRAME_HEIGHT, 240))
    fprintf(stderr, "Warning: error on frame width/height set\n");
  if(!capture.set(CAP_PROP_FPS, desired_fps))
    fprintf(stderr, "Warning: error on FPS\n");
  
  auto model = CatModelFactory::produce(model_filename);
  assert(model != nullptr);

  const auto loop_start = high_resolution_clock::now();
  // const auto loop_end = high_resolution_clock::now() + 1000 ;
  auto frame_start = loop_start;

  size_t frame_count = 0;

  while(capture.read(frame)) {
      if(frame.empty()) {
          fprintf(stderr, "No captured frame. Exiting!\n");
          break;
      }

      assert(frame.type() == CV_8UC3); // 8-bit ints

      std::string window_name = "Cat detection";

      namedWindow(window_name, WINDOW_NORMAL );
      setWindowProperty(window_name, WND_PROP_FULLSCREEN, WINDOW_FULLSCREEN);

      size_t height = 240, width = 320;
      size_t size_min = min(height, width);

      cv::Rect ROI(40, 0, 240, 240); // crop to square
      frame = frame(ROI);
      resize(frame, frame, {224, 224});

      auto cat_probability = model->forward(frame);

      // fprintf(stderr, "cat_probability: %.5f %s\n", cat_probability, cat_probability <= 0.5? "cat" : "non-cat");

      put_catiness_text(frame, cat_probability);

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
