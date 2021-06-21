#include "TorchModel.hpp"

#include <opencv2/imgproc.hpp>
#include <torch/data/transforms/tensor.h>
#include <torch/script.h>
#include <filesystem>

TorchModel::TorchModel(std::string model_filename, size_t frame_timeout) : AbstractCatModel(frame_timeout) {
  module = torch::jit::load(model_filename);
}

using namespace cv;

float TorchModel::forward(const cv::Mat& raw_frame) {
  cv::cvtColor(raw_frame, rgb_frame, cv::COLOR_BGR2RGB);
  rgb_frame.convertTo(rgb_frame, CV_32FC3, 1.0f / 255.0f);

  at::Tensor frame_tensor = torch::from_blob(rgb_frame.data, {1, rgb_frame.rows, rgb_frame.cols, 3});
  // at::Tensor frame_tensor = torch::from_blob(rgb_frame.data, {1, rgb_frame.rows, rgb_frame.cols, 3}, at::kByte);
  // frame_tensor = frame_tensor.to(at::kFloat);
  frame_tensor = frame_tensor.permute({0, 3, 1, 2});

  frame_tensor = torch::data::transforms::Normalize<>({0.485f, 0.456f, 0.406f}, {0.229f, 0.224f, 0.225f})(frame_tensor);

  std::vector<torch::jit::IValue> input;
  input.emplace_back(frame_tensor);
  at::Tensor output = module.forward(input).toTensor();

  auto non_cat_probability = torch::sigmoid(output).item<float>();

  return 1.0f - non_cat_probability;
}

TorchModel::~TorchModel() {}

bool TorchModel::accepts(std::string model_filename) {
  std::filesystem::path p = model_filename;
  return p.extension() == ".pt";
}
