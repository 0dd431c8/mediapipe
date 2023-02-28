#include "exercise_detector.h"
#include "exercise_detector_impl.h"
#include "framework/port/parse_text_proto.h"
#include "mediapipe/framework/deps/status_macros.h"

namespace mediagraph {
ExerciseDetector *ExerciseDetector::Create(const char *graph,
                                           const uint8_t *model,
                                           const size_t m_len) {

  ExerciseDetectorImpl *detector = new ExerciseDetectorImpl();

  auto status = detector->Init(graph, model, m_len);

  if (!status.ok()) {

    return nullptr;
  }

  return detector;
}

float ExerciseDetector::Process(Landmark *landmarks) {
  return static_cast<ExerciseDetectorImpl *>(this)->Process(landmarks);
}

void ExerciseDetector::Dispose() {
  ExerciseDetectorImpl *det = static_cast<ExerciseDetectorImpl *>(this);

  if (det == nullptr)
    return;

  det->Dispose();
}

} // namespace mediagraph
