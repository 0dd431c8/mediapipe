#ifndef EXERCISE_DETECTOR_IMPL
#define EXERCISE_DETECTOR_IMPL

#include "absl/status/status.h"
#include "absl/synchronization/mutex.h"
#include "concurrentqueue.h"
#include "exercise_detector.h"
#include "mediagraph.h"
#include "mediapipe/framework/calculator_graph.h"
#include "tensorflow/lite/interpreter.h"

namespace mediagraph {
class ExerciseDetectorImpl : public ExerciseDetector {
public:
  absl::Status Init(const char *graph, const uint8_t *model,
                    const size_t m_len);
  float Process(Landmark *landmarks);
  void Dispose();

private:
  mediapipe::CalculatorGraph graph_;
  tflite::Interpreter *interpreter_;
  moodycamel::ConcurrentQueue<mediapipe::Packet> out_packets_;
};

} // namespace mediagraph

#endif
