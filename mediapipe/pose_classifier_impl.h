#ifndef EXERCISE_DETECTOR_IMPL
#define EXERCISE_DETECTOR_IMPL

#include "absl/status/status.h"
#include "absl/synchronization/mutex.h"
#include "concurrentqueue.h"
#include "mediagraph.h"
#include "mediapipe/framework/calculator_graph.h"
#include "pose_classifier.h"
#include "tensorflow/lite/interpreter.h"

namespace mediagraph {
class PoseClassifierImpl : public PoseClassifier {
public:
  absl::Status Init(const char *graph, const uint8_t *model,
                    const size_t m_len);
  float Process(const Landmark *landmarks);
  void Dispose();

private:
  mediapipe::CalculatorGraph graph_;
  tflite::Interpreter *interpreter_;
  moodycamel::ConcurrentQueue<mediapipe::Packet> out_packets_;
};

} // namespace mediagraph

#endif
