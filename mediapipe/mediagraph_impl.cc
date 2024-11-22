// Pose detector implementation
//
#include <memory>
#include <vector>

#include "mediagraph_impl.h"
#include "mediapipe/mediagraph.h"
#include "utils.h"

namespace mediagraph {
using mediapipe::tasks::core::BaseOptions;

// Cleanup
void DetectorImpl::Dispose() {
  LOG(INFO) << "Shutting down.";
  absl::Status status = pose_landmarker_->Close();
  if (!status.ok()) {
    LOG(INFO) << "Error in CloseInputStream(): " << status.ToString();
  }
}

// Initializer
absl::Status DetectorImpl::Init(const uint8_t *pose_landmarker_model,
                                size_t model_len,
                                PoseLandmarkerDelegate delegate,
                                PoseCallback callback) {
  callback_ = callback;

  std::string model_blob(reinterpret_cast<const char *>(pose_landmarker_model),
                         model_len);

  auto opts = std::make_unique<PoseLandmarkerOptions>();
  opts->num_poses = 1;
  opts->output_segmentation_masks = false;
  opts->base_options.delegate = delegate == PoseLandmarkerDelegate::CPU
                                    ? BaseOptions::Delegate::CPU
                                    : BaseOptions::Delegate::GPU;
  opts->base_options.model_asset_buffer =
      std::make_unique<std::string>(model_blob);

  opts->running_mode = RunningMode::LIVE_STREAM;

  opts->result_callback = [this](absl::StatusOr<PoseLandmarkerResult> result,
                                 mediapipe::Image image, size_t timestamp) {
    if (!result.ok()) {
      return;
    }

    auto res = result.value();

    if (callback_ctx_ == nullptr) {
      LOG(INFO) << "callback_ctx_ is null";
      return;
    }

    if (res.pose_landmarks.size() < 1 || res.pose_world_landmarks.size() < 1) {
      callback_(callback_ctx_, nullptr, 0);
      return;
    }

    auto landmarks = res.pose_landmarks[0].landmarks;
    auto world_landmarks = res.pose_world_landmarks[0].landmarks;
    std::vector<Landmark> landmarks_ = {};

    for (int i = 0; i < landmarks.size(); i++) {
      landmarks_.push_back({landmarks[i].x, landmarks[i].y, landmarks[i].z,
                            landmarks[i].visibility.value_or(0)});
    }

    for (int i = 0; i < world_landmarks.size(); i++) {
      landmarks_.push_back({world_landmarks[i].x, world_landmarks[i].y,
                            world_landmarks[i].z,
                            world_landmarks[i].visibility.value_or(0)});
    }

    callback_(callback_ctx_, landmarks_.data(), timestamp);
  };

  auto status = PoseLandmarker::Create(std::move(opts));
  pose_landmarker_ = std::move(status.value());

  return absl::OkStatus();
}

// Process frame on CPU
size_t DetectorImpl::Process(mediapipe::Image input, Flip flip_code,
                           const void *callback_ctx) {
  callback_ctx_ = callback_ctx;
  auto frame_timestamp = get_timestamp();
  pose_landmarker_->DetectAsync(input, frame_timestamp);

  return frame_timestamp;
}
} // namespace mediagraph
