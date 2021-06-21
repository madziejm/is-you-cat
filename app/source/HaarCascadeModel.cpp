#include "HaarCascadeModel.hpp"


#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/dnn.hpp>
#include <filesystem>

#include <cstdio>

using namespace cv;

HaarCascadeModel::HaarCascadeModel(std::string model_filename, size_t frame_timeout)
: AbstractCatModel(frame_timeout), classifier(model_filename) {
}

HaarCascadeModel::~HaarCascadeModel() {}

float HaarCascadeModel::forward(const cv::Mat& raw_frame) {
  cat_faces.clear();
  Mat gray_frame;
  cvtColor(raw_frame, gray_frame, COLOR_BGR2GRAY);
  equalizeHist(gray_frame, gray_frame);
  classifier.detectMultiScale(gray_frame, cat_faces);
  return 0 < cat_faces.size()? 1.0f : 0.0f;
}

bool HaarCascadeModel::accepts(std::string model_filename)
{
  std::filesystem::path p = model_filename;
  return p.extension() == ".xml";
}
