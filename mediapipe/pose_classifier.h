// Mediapipe graph running arbitrary tflite model

#ifndef EXERCISE_DETECTOR
#define EXERCISE_DETECTOR

#include "exported.h"
#include "mediagraph.h"

namespace mediagraph {

class EXPORTED PoseClassifier {
public:
  static PoseClassifier *Create(const char *graph, const uint8_t *model,
                                const size_t m_len);
  // Run model on landmarks
  void Process(const Landmark *landmarks, float *confidence, float *scores,
               size_t scores_len, float *feedbacks, size_t feedbacks_len);
  void Dispose();
};

} // namespace mediagraph

#endif
