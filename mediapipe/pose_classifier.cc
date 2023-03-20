#include "pose_classifier.h"
#include "framework/port/parse_text_proto.h"
#include "mediapipe/framework/deps/status_macros.h"
#include "pose_classifier_impl.h"

namespace mediagraph {
PoseClassifier *PoseClassifier::Create(const char *graph, const uint8_t *model,
                                       const size_t m_len) {

  PoseClassifierImpl *detector = new PoseClassifierImpl();

  auto status = detector->Init(graph, model, m_len);

  if (!status.ok()) {

    return nullptr;
  }

  return detector;
}

float PoseClassifier::Process(const Landmark *landmarks) {
  return static_cast<PoseClassifierImpl *>(this)->Process(landmarks);
}

void PoseClassifier::Dispose() {
  PoseClassifierImpl *det = static_cast<PoseClassifierImpl *>(this);

  if (det == nullptr)
    return;

  det->Dispose();
}

} // namespace mediagraph
