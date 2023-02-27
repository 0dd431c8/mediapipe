#ifndef EXERCISE_DETECTOR
#define EXERCISE_DETECTOR

#include "absl/status/status.h"
#include "absl/synchronization/mutex.h"
#include "mediagraph.h"
#include "mediapipe/framework/calculator_graph.h"

namespace mediagraph {
class ExerciseDetector {
public:
  static ExerciseDetector *Create(const char *graph, const uint8_t *model,
                                  const size_t m_len);
  float Process(std::vector<Landmark> landmarks);
};

} // namespace mediagraph

#endif
