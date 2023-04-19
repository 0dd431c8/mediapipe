#include "mediagraph.h"
#include "mediagraph_impl.h"
#include "utils.h"

namespace mediagraph {

Detector *Detector::Create(const char *graph_config,
                           const uint8_t *detection_model, const size_t d_len,
                           const uint8_t *landmark_model, const size_t l_len,
                           const uint8_t *hand_model, const size_t h_len,
                           const uint8_t *hand_recrop_model,
                           const size_t hr_len, const Output *outputs,
                           uint8_t num_outputs, PoseCallback callback) {
  DetectorImpl *mediagraph = new DetectorImpl();

  absl::Status status = mediagraph->Init(
      graph_config, detection_model, d_len, landmark_model, l_len, hand_model,
      h_len, hand_recrop_model, hr_len, outputs, num_outputs, callback);
  if (status.ok()) {
    return mediagraph;
  } else {
    LOG(INFO) << "Error initializing graph " << status.ToString();
    delete mediagraph;
    return nullptr;
  }
}

void Detector::Dispose() {
  DetectorImpl *det = static_cast<DetectorImpl *>(this);

  if (det == nullptr)
    return;

  det->Dispose();
}

Landmark *Detector::Process(const uint8_t *data, int width, int height,
                            InputType input_type, Flip flip_code,
                            const void *callback_ctx) {
  if (data == nullptr) {
    LOG(INFO) << __FUNCTION__ << " input data is nullptr!";
    return nullptr;
  }

  int mat_type;

  switch (input_type) {
  case InputType::BGR:
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

  cv::Mat input(cv::Size(width, height), mat_type, (void *)data);

  cv::Mat copied_input = input.clone();

  flip_mat(&copied_input, flip_code);
  color_cvt(&copied_input, input_type);

  return static_cast<DetectorImpl *>(this)->Process(copied_input, callback_ctx);
}
} // namespace mediagraph
