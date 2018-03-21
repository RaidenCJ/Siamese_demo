#include "classification.hpp"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <vector>

using namespace caffe; 
using namespace std;
using namespace cv;

int main(int argc, char** argv) {
  
  std::cout<<"argc ="<<argc<<std::endl;
  if (argc != 8) 
  {
    std::cerr << "Usage: " << argv[0]
              << " deploy.prototxt network.caffemodel"
              << " mean.binaryproto labels.txt img.jpg" << std::endl;
    return 1;
  }

  ::google::InitGoogleLogging(argv[0]);

  string model_file   = argv[1];
  string trained_file = argv[2];
  string mean_file1   = argv[3];
  string mean_file2   = argv[4];
  string label_file   = argv[5];
  Classifier classifier(model_file, trained_file, mean_file1,mean_file2, label_file);

  string file1 = argv[6];
  string file2 = argv[7];

  std::cout << "---------- Siamese Prediction for "
            << file1<<file2 <<" ----------" << std::endl;

  cv::Mat img1 = cv::imread(file1, -1);
  cv::Mat img2 = cv::imread(file2, -1);
  CHECK(!img1.empty()) << "Unable to decode image 1" << file1;
  CHECK(!img2.empty()) << "Unable to decode image 2" << file2;
  int predictions = classifier.Classify(img1, img2);

  /* Print the top N predictions. */
  //for (size_t i = 0; i < predictions.size(); ++i) {
   // Prediction p = predictions[i];
  std::cout <<"Prediction = "<<predictions<< std::endl;
  //}
}
