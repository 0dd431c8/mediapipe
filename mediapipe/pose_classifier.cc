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

void PoseClassifier::Process(const Landmark *landmarks, float *confidence,
                             float *scores, size_t scores_len, float *feedbacks,
                             size_t feedbacks_len) {
  PoseClassifierImpl *impl = static_cast<PoseClassifierImpl *>(this);
  impl->Process(landmarks, confidence, scores, scores_len, feedbacks,
                feedbacks_len);
}

void PoseClassifier::Dispose() {
  PoseClassifierImpl *impl = static_cast<PoseClassifierImpl *>(this);

  impl->Dispose();
}

} // namespace mediagraph
