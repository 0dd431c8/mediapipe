#include "pose_classifier_impl.h"
#include "framework/port/parse_text_proto.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/deps/status_macros.h"
#include "mediapipe/framework/output_stream_poller.h"
#include "mediapipe/framework/port/status.h"
#include "mediapipe/pose_classifier.h"
#include "opencv2/core.hpp">
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/portable_type_to_tflitetype.h"
#include "utils.h"
#include <algorithm>
#include <iterator>

namespace mediagraph {

constexpr char kInputStream[] = "input_tensors";
constexpr char kOutputStream[] = "output_floats";

absl::Status PoseClassifierImpl::Init(const char *graph, const uint8_t *model,
                                      const size_t m_len) {
  mediapipe::CalculatorGraphConfig config =
      mediapipe::ParseTextProtoOrDie<mediapipe::CalculatorGraphConfig>(graph);

  std::string model_blob(reinterpret_cast<const char *>(model), m_len);

  std::map<std::string, mediapipe::Packet> extra_side_packets;
  extra_side_packets.insert({"model_blob", mediapipe::MakePacket<std::string>(
                                               std::move(model_blob))});

  MP_RETURN_IF_ERROR(graph_.Initialize(config, extra_side_packets));

  interpreter_ = new tflite::Interpreter();

  interpreter_->AddTensors(1);
  interpreter_->SetInputs({0});
  TfLiteQuantization quant;
  quant.type = kTfLiteNoQuantization;

  interpreter_->SetTensorParametersReadWrite(
      0, tflite::typeToTfLiteType<float>(), "", {1, 66, 4}, quant);
  int t = interpreter_->inputs()[0];

  // TfLiteTensor *input_tensor = interpreter_->tensor(t);
  // interpreter_->ResizeInputTensor(t, {1, 66, 4});
  interpreter_->AllocateTensors();

  auto sop = graph_.AddOutputStreamPoller(kOutputStream);

  if (!sop.ok()) {
    return sop.status();
  }

  poller_ =
      std::make_unique<mediapipe::OutputStreamPoller>(std::move(sop.value()));

  return graph_.StartRun({});
}

void PoseClassifierImpl::Process(const Landmark *landmarks, float *confidence,
                                 float *scores, size_t scores_len,
                                 float *feedbacks, size_t feedbacks_len) {
  int t = interpreter_->inputs()[0];
  TfLiteTensor *input_tensor = interpreter_->tensor(t);

  float *input_tensor_buffer = tflite::GetTensorData<float>(input_tensor);

  for (int i = 0; i < 66; i++) {
    input_tensor_buffer[i * 4] = landmarks[i].x;
    input_tensor_buffer[i * 4 + 1] = landmarks[i].y;
    input_tensor_buffer[i * 4 + 2] = landmarks[i].z;
    input_tensor_buffer[i * 4 + 3] = landmarks[i].visibility;
  }

  size_t frame_timestamp_us = get_timestamp();

  auto input_vec = absl::make_unique<std::vector<TfLiteTensor>>();
  input_vec->emplace_back(*input_tensor);

  mediapipe::Status run_status = graph_.AddPacketToInputStream(
      kInputStream, mediapipe::Adopt(input_vec.release())
                        .At(mediapipe::Timestamp(frame_timestamp_us)));

  mediapipe::Packet packet;

  bool found = poller_->Next(&packet);

  if (!found) {
    *confidence = std::nanf("");
    return;
  }

  auto res = packet.Get<std::vector<float>>();
  std::vector<float> result_vec(1 + scores_len + feedbacks_len);

  if (res.size() <= result_vec.size()) {
    std::fill(result_vec.begin(), result_vec.end(), std::nanf(""));
    std::copy(res.begin(), res.end(), result_vec.begin());
  } else {
    std::copy(res.begin(), std::next(res.begin(), result_vec.size()),
              result_vec.begin());
  }

  *confidence = result_vec[0];

  auto first = std::next(result_vec.begin() + 1);
  std::copy(first, std::next(first, scores_len), scores);
  std::copy(std::next(first, scores_len + 1), result_vec.end(), feedbacks);
}

void PoseClassifierImpl::Dispose() {
  LOG(INFO) << "Shutting down.";
  absl::Status status = graph_.CloseInputStream(kInputStream);
  if (status.ok()) {
    absl::Status status1 = graph_.WaitUntilDone();
    if (!status1.ok()) {
      LOG(INFO) << "Error in WaitUntilDone(): " << status1.ToString();
    }
  } else {
    LOG(INFO) << "Error in CloseInputStream(): " << status.ToString();
  }
}

} // namespace mediagraph
