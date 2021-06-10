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

#include <torch/data/transforms/tensor.h>
#include <torch/script.h> // One-stop header whatever it means

using namespace std;
using namespace std::chrono;
using namespace cv;

volatile sig_atomic_t break_loop = 0;

const size_t max_frame_count = 600;

void sigint_handler(int signum) {
  break_loop = 1;
}

int main(int argc, const char* argv[]) {
  // ios_base::sync_with_stdio(false);
  CommandLineParser parser(argc, argv,
                            "{help h||}"
                            "{model|../models/torchscript/nosiemamobilenet.pt|Path to Torch Script exported model.}"
                            "{camera|0|Camera device number.}");
  // parser.about("\nSay hello to our simple program demonstrating how we can use Pytorch model in real-time to detect cats in a video stream.\n");
  // parser.printMessage();

  torch::jit::script::Module module;
  string model_filename = parser.get<String>("model");
  try {
    module = torch::jit::load(model_filename);
  }
  catch(const c10::Error& e) {
    fprintf(stderr, "error loading the model\n");
    return -1;
  }

  int camera_device = parser.get<int>("camera");
  VideoCapture capture;
  capture.open(camera_device);
  if(!capture.isOpened()) {
    fprintf(stderr, "Error opening video capture\n");
    return -1;
  }

  signal(SIGINT, sigint_handler);

  // std::vector<torch::jit::IValue> inputs;
  // inputs.push_back(torch::ones({1, 3, 224, 224})); // test // TODO remove
  Mat frame;
  capture.read(frame);
  imwrite("xd.jpeg", frame);
  fprintf(stderr,
    "capture set%d %d\n",
    capture.set(CAP_PROP_FRAME_WIDTH, 320),
    capture.set(CAP_PROP_FRAME_HEIGHT, 240)
    );

  const auto loop_start = high_resolution_clock::now();
  // const auto loop_end = high_resolution_clock::now() + 1000 ;
  auto frame_start = loop_start;

  size_t frame_count = 0;

  while(capture.read(frame)) {
      if(frame.empty()) {
          fprintf(stderr, "No captured frame. Exiting!\n");
          break;
      }

      // std::string filename = "kot.jpg";
      // frame = imread(filename); // CV_8U C1 (albo jednak C3)
      // std::cerr << filename << "\n";

      // std::cerr << frame.type() << "\n";
      // assert(frame.type() == CV_8UC3 || frame.type() == CV_8UC1); // when converting frame to Libtorch tensor it is expected, that frame contains uint8_t-like values
      assert(frame.type() == CV_8UC3); // when converting frame to Libtorch tensor it is expected, that frame contains uint8_t-like values

      imshow("Cat detection", frame );

      if(waitKey(1) == 27 || break_loop) { // ESC
          break; 
      }

      // // crop for image
      // size_t height = frame.rows, width = frame.cols;
      // size_t size_min = min(height, width);

      // // resize(frame, frame, {224, 224});
      // fprintf(stderr, "%d %d\n", height, width);
      // fprintf(stderr, "%d, %d, %d, %d\n", (width - size_min) / 2, (height - size_min) / 2, size_min, size_min);
      // cv::Rect ROI((width - size_min) / 2, (height - size_min) / 2, size_min, size_min); // crop to square
      // //  before: 224 x 224 from (16, 16)
      // frame = frame(ROI);
      // resize(frame, frame, {224, 224});

      // crop for cam
      size_t height = 240, width = 320;
      size_t size_min = min(height, width);

      cv::Rect ROI(40, 0, 240, 240); // crop to square
      frame = frame(ROI);
      resize(frame, frame, {224, 224});

      // std::cerr << "frame.size()" << frame.size() << "\n";

      Mat rgb_frame;
      // Mat rgb_frame = frame;
      cv::cvtColor(frame, rgb_frame, cv::COLOR_BGR2RGB);
      // std::cerr << rgb_frame.size() << "\n";
      rgb_frame.convertTo(rgb_frame, CV_32FC3, 1.0f / 255.0f);

      at::Tensor frame_tensor = torch::from_blob(rgb_frame.data, {1, rgb_frame.rows, rgb_frame.cols, 3});
      // at::Tensor frame_tensor = torch::from_blob(rgb_frame.data, {1, rgb_frame.rows, rgb_frame.cols, 3}, at::kByte);
      // frame_tensor = frame_tensor.to(at::kFloat);
      frame_tensor = frame_tensor.permute({0, 3, 1, 2});

      frame_tensor = torch::data::transforms::Normalize<>({0.485f, 0.456f, 0.406f}, {0.229f, 0.224f, 0.225f})(frame_tensor);

      // std::cerr << frame_tensor[0][0][0]; // to test whether we are in line with Python
      // fprintf(stderr, "%f\n", frame_tensor[0][0][0][0].item<float>()); // to test whether we are in line with Python

      std::vector<torch::jit::IValue> input;
      input.emplace_back(frame_tensor);
      // input.emplace_back(torch::ones({1, 3, 224, 224}));
      at::Tensor output = module.forward(input).toTensor();

      auto cat_probability = torch::sigmoid(output);

      std::cerr
      // << output << endl
      << (1.0f - cat_probability[0][0].item<float>()) * 100.0f << endl;
      // float positive_logit = output[0][0].item<float>();
      // float negative_logit = output[0][1].item<float>();

      const auto now = high_resolution_clock::now();

      float frame_time_ms = duration<float, std::milli>(now - frame_start).count();
      float fps_count = 1000.0f / frame_time_ms;

      // fprintf(stderr, "positive: %.5f negative: %.5f\n%s\n%.2f ms, %.2f fps\n", positive_logit, negative_logit, (positive_logit < negative_logit)? "non-cat" : "cat", frame_time_ms, fps_count);
      // break;
      frame_start = high_resolution_clock::now();
      frame_count++;
      if(max_frame_count <= frame_count)
        break;
  }

  float loop_time_ms = duration<float, std::milli>(high_resolution_clock::now() - loop_start).count();

  float average_FPS = 1000.0f * frame_count / loop_time_ms;

  fprintf(stdout, "model: %s\ntime run: %.5f\naverage FPS: %.5f\n", model_filename.c_str(), loop_time_ms, average_FPS);

  return 0;
}
