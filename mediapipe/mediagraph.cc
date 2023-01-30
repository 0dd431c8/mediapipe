#include "mediagraph.h"
#include "mediagraph_impl.h"
#include "opencv2/imgproc.hpp"

namespace mediagraph {

Detector *Detector::Create(const char *graph_config,
                           const uint8_t *detection_model, const size_t d_len,
                           const uint8_t *landmark_model, const size_t l_len,
                           const Output *outputs, uint8_t num_outputs) {
  DetectorImpl *mediagraph = new DetectorImpl();

  absl::Status status =
      mediagraph->Init(graph_config, detection_model, d_len, landmark_model,
                       l_len, outputs, num_outputs);
  if (status.ok()) {
    return mediagraph;
  } else {
    LOG(INFO) << "Error initializing graph " << status.ToString();
    delete mediagraph;
    return nullptr;
  }
}

Detector::~Detector() {
  DetectorImpl *det = static_cast<DetectorImpl *>(this);

  if (det == nullptr)
    return;

  det->Dispose();
}

void flip_mat(std::unique_ptr<cv::Mat> &input, Flip flip) {
  if (flip == Flip::None)
    return;

  cv::Mat flipped;

  cv::flip(*input, flipped, flip);

  input.reset(new cv::Mat(flipped));
}

void color_cvt(std::unique_ptr<cv::Mat> &input, InputType input_type) {
  bool _gpu = false;
#if !MEDIAPIPE_DISABLE_GPU
  _gpu = true;
#endif

  cv::Mat converted;
  if (input_type == InputType::RGBA) {
    if (_gpu)
      return;

    cv::cvtColor(*input, converted, cv::COLOR_RGBA2RGB);
  }

  if (input_type == InputType::RGB) {
    if (!_gpu)
      return;

    cv::cvtColor(*input, converted, cv::COLOR_RGB2RGBA);
  }

  input.reset(new cv::Mat(converted));
}

Landmark *Detector::Process(uint8_t *data, int width, int height,
                            InputType input_type, Flip flip_code,
                            uint8_t *num_features) {
  if (data == nullptr) {
    LOG(INFO) << __FUNCTION__ << " input data is nullptr!";
    return nullptr;
  }

  int mat_type;

  switch (input_type) {
  case InputType::RGB:
    mat_type = CV_8UC3;
    break;
  case InputType::RGBA:
    mat_type = CV_8UC4;
    break;
  default:
    LOG(ERROR) << __FUNCTION__ << " Unsupported input type!";
    return nullptr;
  }

  std::unique_ptr<cv::Mat> input(
      new cv::Mat(cv::Size(width, height), mat_type, data));

  flip_mat(input, flip_code);
  color_cvt(input, input_type);

  return static_cast<DetectorImpl *>(this)->Process(std::move(*input),
                                                    num_features);
}
} // namespace mediagraph
