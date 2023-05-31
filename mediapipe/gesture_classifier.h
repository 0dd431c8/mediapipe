#ifndef GESTURE_CLASSIFIER_H
#define GESTURE_CLASSIFIER_H

#include "exported.h"
#include "mediagraph.h"
#include <cstdlib>

namespace mediagraph {

class EXPORTED GestureClassifier {
public:
  static GestureClassifier *Create(const uint8_t *model,
                                   const size_t model_len);
  void Recognize(const uint8_t *data, int width, int height,
                 InputType input_type, Flip flip_code);
};

} // namespace mediagraph

#endif
