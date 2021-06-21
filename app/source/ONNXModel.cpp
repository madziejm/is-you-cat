#include "ONNXModel.hpp"


#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/dnn.hpp>
#include <filesystem>

using namespace cv;

ONNXModel::ONNXModel(std::string model_filename, size_t frame_timeout) : AbstractCatModel(frame_timeout) {
  net = dnn::readNetFromONNX(model_filename);
}

ONNXModel::~ONNXModel() {}

float ONNXModel::forward(const cv::Mat& raw_frame) {
  cv::Mat blob;
  dnn::blobFromImage(raw_frame, blob, 1/255.0, Size(224, 224) , Scalar(0.485, 0.456, 0.406), true, true);
  divide(blob, Scalar(0.229, 0.224, 0.225), blob);

  net.setInput(blob);
  Mat prob = net.forward();

  float non_cat_probability = prob.at<float>(0, 0);
  non_cat_probability = 1.0 / (1.0 + exp(-non_cat_probability));

  return 1.0f - non_cat_probability;
}

bool ONNXModel::accepts(std::string model_filename)
{
  std::filesystem::path p = model_filename;
  return p.extension() == ".onnx";
}
