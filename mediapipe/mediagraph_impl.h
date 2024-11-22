#ifndef MEDIAGRAPH_IMPL
#define MEDIAGRAPH_IMPL

#include <cstdlib>
#include <memory>
#include <string>
#include <vector>

#include "absl/status/status.h"

#include "mediagraph.h"
#include "mediapipe/tasks/cc/vision/pose_landmarker/pose_landmarker.h"
#include "mediapipe/tasks/cc/vision/pose_landmarker/pose_landmarker_result.h"

namespace mediagraph {
using mediapipe::tasks::vision::core::RunningMode;
using mediapipe::tasks::vision::pose_landmarker::PoseLandmarker;
using mediapipe::tasks::vision::pose_landmarker::PoseLandmarkerOptions;
using mediapipe::tasks::vision::pose_landmarker::PoseLandmarkerResult;

class DetectorImpl : public Detector {
public:
  void Dispose();

  absl::Status Init(const uint8_t *pose_landmarker_model, size_t model_len,
                    PoseLandmarkerDelegate delegate, PoseCallback callback);

  size_t Process(mediapipe::Image input, Flip flip_code,
               const void *callback_ctx);

private:
  std::unique_ptr<PoseLandmarker> pose_landmarker_;
  const void *callback_ctx_;
  PoseCallback callback_;
};
} // namespace mediagraph

#endif
