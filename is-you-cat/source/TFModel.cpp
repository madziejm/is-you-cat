#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/optional_debug_tools.h"
#include <opencv2/imgproc.hpp>
#include <filesystem>

#include "TFModel.hpp"

using namespace cv;

TFModel::TFModel(std::string model_filename) {
  model = tflite::FlatBufferModel::BuildFromFile(model_filename.c_str());
  assert(model != nullptr);

  tflite::ops::builtin::BuiltinOpResolver resolver;
  tflite::InterpreterBuilder builder(*model, resolver);
  builder(&interpreter);
  assert(interpreter != nullptr);
  assert(interpreter->AllocateTensors() == kTfLiteOk);
}

float TFModel::forward(const cv::Mat raw_frame) {
  Mat rgb_frame;
  cvtColor(raw_frame, rgb_frame, COLOR_BGR2RGB);
  rgb_frame.convertTo(rgb_frame, CV_32FC3, 1.0f / 255.0f); // [0.0, 1.0]
  // rgb_frame.convertTo(rgb_frame, CV_32FC3); // [0.0, 255.0]

  // printf("mat sizes: %d %d %d\n", rgb_frame.size[0], rgb_frame.size[1]);

  assert(rgb_frame.channels() == 3);
  assert(rgb_frame.isContinuous());
  float* rgb_frame_data_ptr = rgb_frame.ptr<float>(0);

  float* input_tensor_data_ptr  = interpreter->typed_input_tensor<float>(0);
  float* output_tensor = interpreter->typed_output_tensor<float>(0);

  int input = interpreter->inputs()[0];

  auto input_tensor = interpreter->input_tensor(input);
  assert(input_tensor->bytes == 3 * rgb_frame.rows * rgb_frame.cols * sizeof(float));
  auto type = interpreter->tensor(input)->type;
  assert(kTfLiteFloat32 == type);
  auto in = interpreter->typed_tensor<float>(input);

  memcpy(input_tensor_data_ptr, rgb_frame_data_ptr, 3 * raw_frame.rows * raw_frame.cols * sizeof(float));
  
  // fprintf(stderr, "%d %d %d\n", dims->data[1], dims->data[2], dims->data[3]);

  assert(interpreter->Invoke() == kTfLiteOk);
  const auto output = interpreter->tensor(interpreter->outputs()[0]);
  size_t output_dims_length = output->dims->size;
  // cerr << "output size: (";
  // for(size_t i = 0; i < output_dims_length; i++) fprintf(stderr, "%d, ", output->dims->data[i]);
  // cerr << ")\n";
  assert(2 == output_dims_length); // output is 2D
  assert(1 == output->dims->data[0]); assert(1 == output->dims->data[1]); // shape is (1, 2) ie. 1 batch output x 1 item

  float* output_data_ptr = output->data.f;
  float non_catiness = output_data_ptr[0]; // assume model outputs value for negative class
  float non_catiness_sigmoid = 1.0f / (1.0f + exp(-non_catiness));
  return 1.0f - non_catiness_sigmoid;
}

bool TFModel::accepts(std::string model_filename) {
    std::filesystem::path p = model_filename;
    return p.extension() == ".tflite";
}
