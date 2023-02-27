#ifndef EXERCISE_DETECTOR
#define EXERCISE_DETECTOR

#include "exported.h"
#include "mediagraph.h"

namespace mediagraph {
class EXPORTED ExerciseDetector {
public:
  static ExerciseDetector *Create(const char *graph, const uint8_t *model,
                                  const size_t m_len);
  float Process(Landmark *landmarks);
};

} // namespace mediagraph

#endif
