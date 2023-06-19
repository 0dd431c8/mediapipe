#ifndef EXERCISE_DETECTOR
#define EXERCISE_DETECTOR

#include "exported.h"
#include "mediagraph.h"

namespace mediagraph {

class EXPORTED PoseClassifier {
public:
  static PoseClassifier *Create(const char *graph, const uint8_t *model,
                                const size_t m_len);
  void Process(const Landmark *landmarks, float *confidence, float *feedbacks,
               size_t feedbacks_len);
  void Dispose();
};

} // namespace mediagraph

#endif
