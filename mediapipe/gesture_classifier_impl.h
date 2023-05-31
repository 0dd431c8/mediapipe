#ifndef GESTURECLASSIFIERIMPL_H
#define GESTURECLASSIFIERIMPL_H

#include "absl/status/status.h"
#include "mediapipe/gesture_classifier.h"
#include "tasks/cc/vision/gesture_recognizer/gesture_recognizer.h"
#include <cstdlib>
#include <mutex>
#include <opencv2/core.hpp>

using mediapipe::tasks::vision::gesture_recognizer::GestureRecognizer;

namespace mediagraph {

class GestureClassifierImpl : public GestureClassifier {
public:
  GestureClassifierImpl(){};
  absl::Status Init(const uint8_t *model, const size_t model_len);
  void Recognize(cv::Mat input);

private:
  std::unique_ptr<GestureRecognizer> gesture_recognizer_;
  std::mutex m_;
};

} // namespace mediagraph

#endif
