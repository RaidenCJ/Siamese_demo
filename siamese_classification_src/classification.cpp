#include "classification.hpp"
#include <caffe/caffe.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/imgproc/types_c.h>
//#include <opencv2/objdetect/objdetect_c.h>

#include <algorithm>
#include <iosfwd>
#include <memory>
#include <string>
#include <utility>
#include <vector>


using namespace caffe;  // NOLINT(build/namespaces)
using std::string;


Classifier::Classifier(const string& model_file,
                       const string& trained_file,
                       const string& mean_file_1,
                       const string& mean_file_2,
                       const string& label_file) {
#ifdef CPU_ONLY
  Caffe::set_mode(Caffe::CPU);
#else
  Caffe::set_mode(Caffe::GPU);
#endif

  /* Load the network. */
  net_.reset(new Net<float>(model_file, TEST));
  net_->CopyTrainedLayersFrom(trained_file);

  CHECK_EQ(net_->num_inputs(), 2) << "Network should have exactly two input.";
  CHECK_EQ(net_->num_outputs(), 1) << "Network should have only one output.";
  
  /* We have 2 images input */
  Blob<float>* input_layer_1 = net_->input_blobs()[0];
  Blob<float>* input_layer_2 = net_->input_blobs()[1];
  
  num_channels_1 = input_layer_1->channels();
  num_channels_2 = input_layer_2->channels();
  
  CHECK(num_channels_1 == 3 || num_channels_1 == 1)
    << "Input layer 1 should have 1 or 3 channels.";
  CHECK(num_channels_2 == 3 || num_channels_2 == 1)
    << "Input layer 2 should have 1 or 3 channels.";
  input_geometry_1 = cv::Size(input_layer_1->width(), input_layer_1->height());
  input_geometry_2 = cv::Size(input_layer_2->width(), input_layer_2->height());

  /* Load the binaryproto mean file. */
  SetMean(mean_file_1,input_geometry_1, num_channels_1, 1);
  SetMean(mean_file_2,input_geometry_2, num_channels_2, 2);

  /* Load labels. */
  std::ifstream labels(label_file.c_str());
  CHECK(labels) << "Unable to open labels file " << label_file;
  string line;
  while (std::getline(labels, line))
    labels_.push_back(string(line));

  Blob<float>* output_layer = net_->output_blobs()[0];
  std::cout<<"output_layer->channels = "<<output_layer->channels()<<std::endl;
  //CHECK_EQ(labels_.size(), output_layer->channels())
    //<< "Number of labels is different from the output layer dimension.";
}

static bool PairCompare(const std::pair<float, int>& lhs,
                        const std::pair<float, int>& rhs) {
  return lhs.first > rhs.first;
}

/* Return the indices of the top N values of vector v. */
static std::vector<int> Argmax(const std::vector<float>& v, int N) {
  std::vector<std::pair<float, int> > pairs;
  for (size_t i = 0; i < v.size(); ++i)
    pairs.push_back(std::make_pair(v[i], i));
  std::partial_sort(pairs.begin(), pairs.begin() + N, pairs.end(), PairCompare);

  std::vector<int> result;
  for (int i = 0; i < N; ++i)
    result.push_back(pairs[i].second);
  return result;
}

/* Return the top N predictions. */
int Classifier::Classify(const cv::Mat& img1, const cv::Mat& img2, int N) {

  std::cout<<"Classify come in 1"<<std::endl;
  std::vector<float> output = Predict(img1, img2);
  std::cout<<"Classify come in 2"<<std::endl;
  if(!output.empty() && 1 == output.size())
  {
	  return static_cast<int>(output[0]);
  }
  else 
	  return -1;
  /*
  N = std::min<int>(labels_.size(), N);
  std::vector<int> maxN = Argmax(output, N);
  std::vector<Prediction> predictions;
  for (int i = 0; i < N; ++i) {
    int idx = maxN[i];
    predictions.push_back(std::make_pair(labels_[idx], output[idx]));
  }

  return predictions;
  */
}

/* Load the mean file in binaryproto format. */
void Classifier::SetMean(const string& mean_file, cv::Size input_geometry_, int num_channels_, int channel_flag) {
  BlobProto blob_proto;
  ReadProtoFromBinaryFileOrDie(mean_file.c_str(), &blob_proto);

  /* Convert from BlobProto to Blob<float> */
  Blob<float> mean_blob;
  mean_blob.FromProto(blob_proto);
  CHECK_EQ(mean_blob.channels(), num_channels_)
    << "Number of channels of mean file doesn't match input layer.";

  /* The format of the mean file is planar 32-bit float BGR or grayscale. */
  std::vector<cv::Mat> channels;
  float* data = mean_blob.mutable_cpu_data();
  for (int i = 0; i < num_channels_; ++i) {
    /* Extract an individual channel. */
    cv::Mat channel(mean_blob.height(), mean_blob.width(), CV_32FC1, data);
    channels.push_back(channel);
    data += mean_blob.height() * mean_blob.width();
  }

  /* Merge the separate channels into a single image. */
  cv::Mat mean;
  cv::merge(channels, mean);

  /* Compute the global mean pixel value and create a mean image
   * filled with this value. */
  cv::Scalar channel_mean = cv::mean(mean);
  if(1 == channel_flag)
     mean_1 = cv::Mat(input_geometry_, mean.type(), channel_mean);
  else if(2 == channel_flag)
     mean_2 = cv::Mat(input_geometry_, mean.type(), channel_mean);
  else
    return;
}

std::vector<float> Classifier::Predict(const cv::Mat& img1, const cv::Mat& img2) {
  std::cout<<"Predict come in 1"<<std::endl;
  Blob<float>* input_layer_1 = net_->input_blobs()[0];
  Blob<float>* input_layer_2 = net_->input_blobs()[1];
  input_layer_1->Reshape(1, num_channels_1,
                       input_geometry_1.height, input_geometry_1.width);
  input_layer_2->Reshape(1, num_channels_2,
                       input_geometry_2.height, input_geometry_2.width);
  /* Forward dimension change to all layers. */
  net_->Reshape();//?
  std::cout<<"Predict come in 2"<<std::endl;
  std::vector<cv::Mat> input_channels_1;
  WrapInputLayer(&input_channels_1);
  std::vector<cv::Mat> input_channels_2;
  WrapInputLayer(&input_channels_2);
  std::cout<<"Predict come in 3"<<std::endl;
  Preprocess(img1, &input_channels_1, 1);
  Preprocess(img2, &input_channels_2, 2);
  
  std::cout<<"Predict come in 4"<<std::endl;
  net_->Forward();
  std::cout<<"Predict come in 5"<<std::endl;

  /* Copy the output layer to a std::vector */
  Blob<float>* output_layer = net_->output_blobs()[0];
  const float* begin = output_layer->cpu_data();
  const float* end = begin + output_layer->channels();
  std::cout<<"Predict come in 6"<<std::endl;
  return std::vector<float>(begin, end);
}

/* Wrap the input layer of the network in separate cv::Mat objects
 * (one per channel). This way we save one memcpy operation and we
 * don't need to rely on cudaMemcpy2D. The last preprocessing
 * operation will write the separate channels directly to the input
 * layer. */
//这个函数要改写
void Classifier::WrapInputLayer(std::vector<cv::Mat>* input_channels) {
  Blob<float>* input_layer = net_->input_blobs()[0];

  int width = input_layer->width();
  int height = input_layer->height();
  float* input_data = input_layer->mutable_cpu_data();
  for (int i = 0; i < input_layer->channels(); ++i) {
    cv::Mat channel(height, width, CV_32FC1, input_data);
    input_channels->push_back(channel);
    input_data += width * height;
  }
}

void Classifier::Preprocess(const cv::Mat& img,
                            std::vector<cv::Mat>* input_channels, int channel_flag) {
  
  cv::Mat mean_;
  cv::Size input_geometry_;
  int num_channels_;
  if(1 == channel_flag)
  {
    mean_ = mean_1;
    input_geometry_ = input_geometry_1;
    num_channels_ = num_channels_1;
  }
  else if(2 == channel_flag)
  {
    mean_ = mean_2;
    input_geometry_ = input_geometry_2;
    num_channels_ = num_channels_2;
  }
  else
    return;
  /* Convert the input image to the input image format of the network. */
  cv::Mat sample;
  
  if (img.channels() == 3 && num_channels_ == 1)
    cv::cvtColor(img, sample, cv::COLOR_BGR2GRAY);
  else if (img.channels() == 4 && num_channels_ == 1)
    cv::cvtColor(img, sample, cv::COLOR_BGRA2GRAY);
  else if (img.channels() == 4 && num_channels_ == 3)
    cv::cvtColor(img, sample, cv::COLOR_BGRA2BGR);
  else if (img.channels() == 1 && num_channels_ == 3)
    cv::cvtColor(img, sample, cv::COLOR_GRAY2BGR);
  else
    sample = img;

  
  
  
  
  cv::Mat sample_resized;
  if (sample.size() != input_geometry_)
    cv::resize(sample, sample_resized, input_geometry_);
  else
    sample_resized = sample;

  cv::Mat sample_float;
  if (num_channels_ == 3)
    sample_resized.convertTo(sample_float, CV_32FC3);
  else
    sample_resized.convertTo(sample_float, CV_32FC1);

  cv::Mat sample_normalized;
  cv::subtract(sample_float, mean_, sample_normalized);

  /* This operation will write the separate BGR planes directly to the
   * input layer of the network because it is wrapped by the cv::Mat
   * objects in input_channels. */
  cv::split(sample_normalized, *input_channels);

  CHECK(reinterpret_cast<float*>(input_channels->at(0).data)
        == net_->input_blobs()[0]->cpu_data())
    << "Input channels are not wrapping the input layer of the network.";
}
