#ifndef EXERCISE_DETECTOR
#define EXERCISE_DETECTOR

#include "exported.h"
#include "mediagraph.h"

namespace mediagraph {
class EXPORTED PoseClassifier {
public:
  static PoseClassifier *Create(const char *graph, const uint8_t *model,
                                const size_t m_len);
  float Process(const Landmark *landmarks);
  void Dispose();
};

} // namespace mediagraph

#endif
