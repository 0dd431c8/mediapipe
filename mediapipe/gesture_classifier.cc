#include "gesture_classifier.h"
#include "mediapipe/gesture_classifier_impl.h"
#include "utils.h"

namespace mediagraph {
GestureClassifier *GestureClassifier::Create(const uint8_t *model,
                                             const size_t model_len) {
  GestureClassifierImpl *gesture_classifier_impl = new GestureClassifierImpl();

  auto status = gesture_classifier_impl->Init(model, model_len);

  if (status.ok()) {
    return gesture_classifier_impl;
  } else {
    delete gesture_classifier_impl;
    return nullptr;
  }
}

void GestureClassifier::Recognize(const uint8_t *data, int width, int height,
                                  InputType input_type, Flip flip_code) {
  auto input = bytes_to_mat(data, width, height, input_type);

  if (!input.has_value()) {
    return;
  }

  flip_mat(&input.value(), flip_code);
  color_cvt(&input.value(), input_type);

  static_cast<GestureClassifierImpl *>(this)->Recognize(input.value());
}
} // namespace mediagraph
