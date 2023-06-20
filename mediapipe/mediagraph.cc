#include "mediagraph.h"
#include "mediagraph_impl.h"
#include "utils.h"

namespace mediagraph {

EGLContext getEglContext() { return eglGetCurrentContext(); }

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

void Detector::Process(const uint8_t *data, int width, int height,
                       InputType input_type, Flip flip_code,
                       const void *callback_ctx, uint texture) {
  // auto input = bytes_to_mat(data, width, height, input_type);
  //
  // if (!input.has_value()) {
  //   return;
  // }
  //
  // flip_mat(&input.value(), flip_code);
  // color_cvt(&input.value(), input_type);

  auto input = new cv::Mat();

  static_cast<DetectorImpl *>(this)->Process(*input, callback_ctx, texture);
}
} // namespace mediagraph
