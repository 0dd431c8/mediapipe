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

namespace mediagraph {

constexpr char kInputStream[] = "input_tensors";
constexpr char kOutputStream[] = "output_tensors";

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

  TfLiteTensor *input_tensor = interpreter_->tensor(t);
  interpreter_->ResizeInputTensor(t, {1, 66, 4});
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
                                 Feedbacks *feedbacks) {
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

  if (poller_->QueueSize() < 1) {
    return;
  }

  bool found = poller_->Next(&packet);

  if (!found) {
    return;
  }

  auto res = packet.Get<std::vector<TfLiteTensor>>();
  const TfLiteTensor *result = &res[0];
  const float *result_buffer = tflite::GetTensorData<float>(result);
  auto num_outputs = result->dims->data[0];

  std::array<float, OUTPUT_TENSOR_RANK> result_array;

  for (int i = 0; i < result_array.size(); i++) {
    result_array[i] = std::nan("");
  }

  std::copy(result_buffer, result_buffer + num_outputs, result_array.begin());

  *confidence = result_array[0];
  std::copy(result_array.begin() + 1, result_array.end(), *feedbacks);
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
