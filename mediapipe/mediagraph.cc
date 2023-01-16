#include "mediagraph.h"
#include "mediagraph_impl.h"

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

Landmark *Detector::Process(uint8_t *data, int width, int height,
                            uint8_t *num_features) {
  return dynamic_cast<DetectorImpl *>(this)->Process(data, width, height,
                                                     num_features);
}
} // namespace mediagraph
