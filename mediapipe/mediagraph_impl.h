#ifndef MEDIAGRAPH_IMPL
#define MEDIAGRAPH_IMPL

#include <cstdlib>
#include <memory>
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "framework/formats/image_frame.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/output_stream_poller.h"

#include "mediagraph.h"

namespace mediagraph {

class DetectorImpl : public Detector {
public:
  DetectorImpl() {}
  void Dispose();

  absl::Status Init(const char *graph, const uint8_t *detection_model,
                    const size_t d_len, const uint8_t *landmark_model,
                    const size_t l_len, const uint8_t *hand_model,
                    const size_t h_len, const uint8_t *hand_recrop_model,
                    const size_t hr_len, const Output *outputs,
                    uint8_t num_outputs, PoseCallback callback);

  void Process(mediapipe::Packet input, Flip flip_code,
               const void *callback_ctx);

private:
  mediapipe::CalculatorGraph graph_;
  size_t frame_timestamp_ = 0;
  std::vector<Output> outputs_;
  std::vector<std::unique_ptr<mediapipe::OutputStreamPoller>> out_streams_;
  uint8_t num_outputs_;
  PoseCallback callback_;
};
} // namespace mediagraph

#endif
