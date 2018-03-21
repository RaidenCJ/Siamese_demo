#ifndef CLASSIFICATION_HPP
#define CLASSIFICATION_HPP
#include "caffe/caffe.hpp"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <algorithm>
#include <iosfwd>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <sstream>

using namespace caffe;
using namespace std;
using std::string;
using namespace cv;

/* Pair (label, confidence) representing a prediction. */
typedef std::pair<string, float> Prediction;

class Classifier {
 public:
  Classifier(const string& model_file,
             const string& trained_file,
             const string& mean_file_1,
             const string& mean_file_2,
             const string& label_file);

  int Classify(const cv::Mat& img1, const cv::Mat& img2, int N = 5);

 private:
  void SetMean(const string& mean_file, cv::Size input_geometry_, int num_channels_, int channel_flag);

  std::vector<float> Predict(const cv::Mat& img, const cv::Mat& img2);

  void WrapInputLayer(std::vector<cv::Mat>* input_channels);

  void Preprocess(const cv::Mat& img,
                  std::vector<cv::Mat>* input_channels, int channel_flag);

 private:
  shared_ptr<Net<float> > net_;
  cv::Size input_geometry_1;
  cv::Size input_geometry_2;
  
  int num_channels_1;
  int num_channels_2;
  cv::Mat mean_1;
  cv::Mat mean_2;
  std::vector<string> labels_;
};
#endif
