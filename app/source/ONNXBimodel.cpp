#include "ONNXBimodel.hpp"

#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/dnn.hpp>
#include <filesystem>
#include <iostream>

using namespace cv;

ONNXBimodel::ONNXBimodel(std::string first_model_filename, std::string second_model_filename, size_t frame_timeout) : AbstractCatModel(frame_timeout) {
  first_net = dnn::readNetFromONNX(first_model_filename);
  second_net = dnn::readNetFromONNX(second_model_filename);
}

ONNXBimodel::~ONNXBimodel() {}

float ONNXBimodel::forward(const cv::Mat& raw_frame)
{
  cv::Mat blob;
  dnn::blobFromImage(raw_frame, blob, 1/255.0, Size(224, 224) , Scalar(0.485, 0.456, 0.406), true, true);
  divide(blob, Scalar(0.229, 0.224, 0.225), blob);

  first_net.setInput(blob);
  Mat activations = first_net.forward();

  if(last_first_net_activations.empty()) {
    last_first_net_activations = activations.clone(); // why is this clone thing needed here aghrh
    second_net.setInput(activations);
    Mat prob = second_net.forward();
    float non_cat_probability = prob.at<float>(0, 0);
    non_cat_probability = 1.0 / (1.0 + exp(-non_cat_probability));
    return cached_catiness = (1.0f - non_cat_probability);
  }

  Mat first_net_activations_diff = activations - last_first_net_activations;
  first_net_activations_diff = first_net_activations_diff.mul(first_net_activations_diff);
  auto mean_scalar = cv::mean(first_net_activations_diff);
  float mean = mean_scalar[0]; // 0-th channel only as activations mat is not image mat, so other channels are zeros

  bool frame_change = MSE_treshold < mean;
  bool force_model_forward = (frame_timeout == 0) || ((frame_timeout != -1) && (frame_timeout <= skipped_frames));

  if(force_model_forward || frame_change) {
    skipped_frames = 0;
    second_net.setInput(activations);
    Mat prob = second_net.forward();
    float non_cat_probability = prob.at<float>(0, 0);
    non_cat_probability = 1.0 / (1.0 + exp(-non_cat_probability));
    return cached_catiness = (1.0f - non_cat_probability);
  }
  else {
    skipped_frames++;
    return cached_catiness;
  }
}

bool ONNXBimodel::accepts(std::string model_filename)
{
  std::filesystem::path p = model_filename;
  return p.extension() == ".onnx";
}
