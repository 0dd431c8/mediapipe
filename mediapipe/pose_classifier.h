#ifndef EXERCISE_DETECTOR
#define EXERCISE_DETECTOR

#include "exported.h"
#include "mediagraph.h"
#include <array>

namespace mediagraph {

enum Feedback {
  FeetWide,
  BodyHeight,
  ArmsHeight,
  KneesHeight,
  Twist,
  Extended,
  __FEEDBACK_OUTPUTS_COUNT
};

const size_t OUTPUT_TENSOR_RANK = __FEEDBACK_OUTPUTS_COUNT + 1;

using Feedbacks = float[__FEEDBACK_OUTPUTS_COUNT];

class EXPORTED PoseClassifier {
public:
  static PoseClassifier *Create(const char *graph, const uint8_t *model,
                                const size_t m_len);
  void Process(const Landmark *landmarks, float *confidence,
               Feedbacks *feedbacks);
  void Dispose();
};

} // namespace mediagraph

#endif
