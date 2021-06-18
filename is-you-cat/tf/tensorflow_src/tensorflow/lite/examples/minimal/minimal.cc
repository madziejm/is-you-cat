#include <cassert>
#include <cmath>
#include <cerrno>
#include <cstdio>
#include <iomanip>
#include <iostream>
#include <memory>
#include <opencv2/opencv.hpp>
#include <signal.h>
#include <unistd.h>
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/optional_debug_tools.h"

volatile sig_atomic_t break_loop = 0;

size_t max_frame_count;

void sigint_handler(int signum) {
  break_loop = 1;
}

using namespace std;
using namespace std::chrono;
using namespace cv;

int main(int argc, char* argv[]) {
  CommandLineParser parser(argc, argv,
                            "{help h||}"
                            "{model|../models/model_1.tflite|Path to Torch Script exported model.}"
                            "{camera|0|Camera device number.}"
                            "{desired-fps|30|Desired camera FPS}"
                            "{frame-count|0|Desired camera FPS}"
                            );
  parser.about("\nSay hello to our simple program demonstrating how we can use TF Lite model in real-time to detect cats in a video stream.\n");
  parser.printMessage();

  string model_filename = parser.get<String>("model");
  int camera_device = parser.get<int>("camera");
  int desired_fps = parser.get<int>("desired-fps");
  max_frame_count = parser.get<size_t>("frame-count");
  VideoCapture capture;
  capture.open(camera_device);
  assert(capture.isOpened());

  unique_ptr<tflite::FlatBufferModel> model =
      tflite::FlatBufferModel::BuildFromFile(model_filename.c_str());
  assert(model != nullptr);

  signal(SIGINT, sigint_handler);

  // Build the interpreter with the InterpreterBuilder.
  // Note: all Interpreters should be built with the InterpreterBuilder,
  // which allocates memory for the Interpreter and does various set up
  // tasks so that the Interpreter can read the provided model.
  tflite::ops::builtin::BuiltinOpResolver resolver;
  tflite::InterpreterBuilder builder(*model, resolver);
  unique_ptr<tflite::Interpreter> interpreter;
  builder(&interpreter);
  assert(interpreter != nullptr);
  assert(interpreter->AllocateTensors() == kTfLiteOk);

  if(!capture.set(CAP_PROP_FRAME_WIDTH, 320) || !capture.set(CAP_PROP_FRAME_HEIGHT, 240))
    fprintf(stderr, "Warning: error on frame width/height set\n");
  if!(capture.set(CAP_PROP_FPS, desired_fps))
    fprintf(stderr, "Warning: error on FPS\n");
  
  Mat frame;

  const auto loop_start = high_resolution_clock::now();
  // const auto loop_end = high_resolution_clock::now() + 1000 ;
  auto frame_start = loop_start;

  size_t frame_count = 0;

  float model_time_ms = 0.0f;

  while(capture.read(frame))
  {
    // Mat frame = imread(image_filename);
    assert(frame.type() == CV_8UC3); // when converting frame to Libtorch tensor it is expected, that frame contains uint8_t-like values
    size_t height = frame.rows, width = frame.cols;
    // size_t height = 240, width = 320;
    // size_t size_min = min(height, width);
    
    resize(frame, frame, {320, 240}); // temporarily force 320x240 on my desktop // TODO remove

    // Rect ROI((width - size_min) / 2, (height - size_min) / 2, size_min, size_min); // crop to square
    Rect ROI(40, 0, 240, 240); // crop to square
    frame = frame(ROI);
    resize(frame, frame, {224, 224});

    Mat rgb_frame;
    cvtColor(frame, rgb_frame, COLOR_BGR2RGB);
    // cerr << rgb_frame.size() << "\n";
    // rgb_frame.convertTo(rgb_frame, CV_32FC3, 1.0f / 255.0f); // now its [0.0, 1.0]
    rgb_frame.convertTo(rgb_frame, CV_32FC3, 1.0f / 255.f); // now its [0.0, 255.0] (probably)

    printf("mat sizes: %d %d %d\n", rgb_frame.size[0], rgb_frame.size[1]);

    assert(rgb_frame.channels() == 3);

    // for(size_t row = 0; row < 224; row++)
    //   for(size_t col = 0; col < 224; col++)
    //     {
    //       float* pixelPtr = (float*)rgb_frame.data;
    //       int channels = rgb_frame.channels();
    //       // Vec3b pixel = rgb_frame.at<float>(height, width, 0);
    //       pixelPtr[row * rgb_frame.cols + col * channels + 0] -= 0.485; // r
    //       pixelPtr[row * rgb_frame.cols + col * channels + 1] -= 0.456; // g
    //       pixelPtr[row * rgb_frame.cols + col * channels + 2] -= 0.406; // b
    //       pixelPtr[row * rgb_frame.cols + col * channels + 0] /= 0.229; // r
    //       pixelPtr[row * rgb_frame.cols + col * channels + 1] /= 0.224; // g
    //       pixelPtr[row * rgb_frame.cols + col * channels + 2] /= 0.225; // b
    //     }
    


    // cerr << rgb_frame << endl;
    assert(rgb_frame.isContinuous());
    float* rgb_frame_data_ptr = rgb_frame.ptr<float>(0);

    float* input_tensor_data_ptr  = interpreter->typed_input_tensor<float>(0);
    float* output_tensor = interpreter->typed_output_tensor<float>(0);

    // fprintf(stderr, "%d %d %d\n", frame.at<int>(0), frame.at<int>(1), frame.at<int>(2));
    // fprintf(stderr, "%f %f %f\n", rgb_frame.at<float>(0, 0, 0), rgb_frame.at<float>(0, 0, 1), rgb_frame.at<float>(0, 0, 2));
    // break;

    // imshow("Windoze", rgb_frame);

    // cvtColor(rgb_frame, frame, COLOR_RGB2BGR); // for debug
    // frame.convertTo(frame, CV_8UC3); // for debug
    imshow("Cat detection", frame);

    if(waitKey(1) == 27 || break_loop) { // ESC
        break; 
    }


    // fprintf(stderr, "interpreter inputs size is %ld\n", interpreter->inputs().size()); // 1
    // fprintf(stderr, "interpreter input[0] name is %s\n", interpreter->GetInputName(0));

    // int t_size = interpreter->tensors_size();
    // for (int i = 0; i < t_size; i++) {
    //   if (interpreter->tensor(i)->name)
    //     cerr << i << ": " << interpreter->tensor(i)->name << ", "
    //               << interpreter->tensor(i)->bytes << ", "
    //               << interpreter->tensor(i)->type << ", "
    //               << interpreter->tensor(i)->params.scale << ", "
    //               << interpreter->tensor(i)->params.zero_point << endl;
    //   }

    int input = interpreter->inputs()[0];
    auto type = interpreter->tensor(input)->type;
    // fprintf(stderr, "interpreter input[0] type is %d\n", type);
    assert(kTfLiteFloat32 == type);
    
    TfLiteIntArray* dims = interpreter->tensor(input)->dims;
    int wanted_height = dims->data[3];
    int wanted_width = dims->data[1];
    int wanted_channels = dims->data[2];


    // fprintf(stderr, "%d %d %d\n", dims->data[1], dims->data[2], dims->data[3]);

    // assert(3 == wanted_channels);
    // assert(224 == wanted_width);
    // assert(224 == wanted_height);

    auto input_tensor = interpreter->input_tensor(input);
    // fprintf(stderr, "input_tensor_ptr at %p, byte count: %ld, memcpy count %ld\n", (void*)input_tensor_ptr, input_tensor_ptr->bytes, 3 * 224 * 224 * sizeof(float));

    assert(frame.rows == 224);
    assert(frame.cols == 224);

    assert(input_tensor->bytes == 3 * rgb_frame.rows * rgb_frame.cols * sizeof(float));
    // assert(602112 == 3 * 224 * 224 * sizeof(float));
    // memcpy(frame_data_ptr, frame_data_ptr, 3 * 224 * 224 * sizeof(float));
    auto in = interpreter->typed_tensor<float>(input);
    // for (int i = 0; i <  224 * 224; i++)
    //   rgb_frame_data_ptr[i] = 0.0f;
      // in[i] = frame_data_ptr[i];
    // memcpy(input_tensor_ptr->data.raw, frame_data_ptr, 3 * 224 * 224 * sizeof(float));

    memcpy(input_tensor_data_ptr, rgb_frame_data_ptr, 3 * frame.rows * frame.cols * sizeof(float)); // nie rusz
    // input_tensor->data.f = rgb_frame_data_ptr;



    // fprintf(stderr, "%f %f %f\n", input_tensor_data_ptr[0],  input_tensor_data_ptr[1],  input_tensor_data_ptr[2]); // printf drien debugging is my best friend now

    // Run inference
    assert(interpreter->Invoke() == kTfLiteOk);
    // printf("\n\n=== Post-invoke Interpreter State ===\n");
    // tflite::PrintInterpreterState(interpreter.get());

    assert(1 == interpreter->outputs().size());

    const auto output = interpreter->tensor(interpreter->outputs()[0]);
    size_t output_dims_length = output->dims->size;
    // cerr << "output size: (";
    // for(size_t i = 0; i < output_dims_length; i++) fprintf(stderr, "%d, ", output->dims->data[i]);
    // cerr << ")\n";
    assert(2 == output_dims_length); // output is 2D
    // assert(1 == output->dims->data[0]); assert(2 == output->dims->data[1]); // shape is (1, 2) ie. 1 batch output x 2 items
    assert(1 == output->dims->data[0]); assert(1 == output->dims->data[1]); // shape is (1, 2) ie. 1 batch output x 1 item
    float* output_data_ptr = output->data.f;
    float catiness = output_data_ptr[0]; //, negative_likelihood = output_data_ptr[1];
    float catiness_sigmoid = 1.0f / (1.0f + exp(-catiness)); //, negative_likelihood = output_data_ptr[1];
    // fprintf(stderr, "output: (");
    // for(size_t i = 0; i < output->dims->data[1]; i++) fprintf(stderr, "%f, ", output_data_ptr[i]);
    // fprintf(stderr, ")\n");
    // fprintf(stderr, "output items: %f %f \n", output_data_ptr[0], output_data_ptr[1]);
    fprintf(stderr, "catiness_sigmoid: %.5f %s\n", catiness_sigmoid, catiness_sigmoid <= 0.5? "cat" : "non-cat");
    // fprintf(stderr, "positive: %.5f, negative %.5f, %s\n", positive_likelihood, negative_likelihood, positive_likelihood < negative_likelihood? "non_cat" : "cat");
    // break;

    const auto now = high_resolution_clock::now();

    float frame_time_ms = duration<float, std::milli>(now - frame_start).count();
    float fps_count = 1000.0f / frame_time_ms;

    // fprintf(stderr, "positive: %.5f negative: %.5f\n%s\n%.2f ms, %.2f fps\n", positive_logit, negative_logit, (positive_logit < negative_logit)? "non-cat" : "cat", frame_time_ms, fps_count);
    // break;
    frame_start = high_resolution_clock::now();
    frame_count++;
    if(max_frame_count != 0 && max_frame_count <= frame_count)
      break;
  }

  float loop_time_ms = duration<float, std::milli>(high_resolution_clock::now() - loop_start).count();

  float average_FPS = 1000.0f * frame_count / loop_time_ms;

  fprintf(stdout, "model: %s\ntime run: %.5f\nframe_count: %d\naverage FPS: %.5f\n", model_filename.c_str(), loop_time_ms, frame_count, average_FPS);


  capture.release();

  return 0;
}
