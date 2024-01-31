#include "mediapipe/gesture_classifier_impl.h"
#include "mediapipe/framework/formats/image.h"
#include "mediapipe/tasks/cc/vision/core/running_mode.h"
#include "mediapipe/tasks/cc/vision/gesture_recognizer/gesture_recognizer.h"
#include "utils.h"
#include <algorithm>
#include <memory>

using absl::StatusOr;
using mediapipe::tasks::vision::core::RunningMode;
using mediapipe::tasks::vision::gesture_recognizer::GestureRecognizerOptions;

namespace mediagraph {
absl::Status GestureClassifierImpl::Init(const uint8_t *model,
                                         const size_t model_len) {
  std::string model_blob(reinterpret_cast<const char *>(model), model_len);

  auto options = std::make_unique<GestureRecognizerOptions>();

  options->base_options.model_asset_buffer =
      std::make_unique<std::string>(std::move(model_blob));

  options->num_hands = 2;
  options->running_mode = RunningMode::LIVE_STREAM;

  options->result_callback =
      [](StatusOr<mediapipe::tasks::vision::gesture_recognizer::
                      GestureRecognizerResult>
             status,
         const mediapipe::Image &image, int64 timestamp) {
        // if (!status.ok()) {
        //   return;
        // }

        // auto res = status.value();

        for (size_t i = 0; i < status->gestures.size(); i++) {
          auto r = status->gestures[i];

          LOG(INFO) << r.ShortDebugString() << std::endl;
        }
      };

  auto status = GestureRecognizer::Create(std::move(options));

  if (status.ok()) {
    gesture_recognizer_ = std::move(status.value());
  }

  return status.status();
}

void GestureClassifierImpl::Recognize(cv::Mat input) {
  std::shared_ptr<mediapipe::ImageFrame> input_frame =
      std::move(mat_to_image_frame(input));

  mediapipe::Image image(input_frame);

  int64 timestamp = get_timestamp();

  gesture_recognizer_->RecognizeAsync(image, timestamp);
}
} // namespace mediagraph
