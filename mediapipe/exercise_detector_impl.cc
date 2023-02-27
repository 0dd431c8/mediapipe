#include "exercise_detector_impl.h"
#include "framework/port/parse_text_proto.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/deps/status_macros.h"
#include "mediapipe/framework/port/status.h"
#include "opencv2/core.hpp">
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/portable_type_to_tflitetype.h"
#include "utils.h"

namespace mediagraph {

constexpr char kInputStream[] = "input_tensors";
constexpr char kOutputStream[] = "output_tensors";

absl::Status ExerciseDetectorImpl::Init(const char *graph, const uint8_t *model,
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
  auto *affine_quant = static_cast<TfLiteAffineQuantization *>(
      malloc(sizeof(TfLiteAffineQuantization)));
  affine_quant->scale = TfLiteFloatArrayCreate(1);
  affine_quant->zero_point = TfLiteIntArrayCreate(1);
  affine_quant->scale->data[0] = 1.0;
  affine_quant->zero_point->data[0] = 0;
  quant.type = kTfLiteAffineQuantization;
  quant.params = affine_quant;

  interpreter_->SetTensorParametersReadWrite(
      0, tflite::typeToTfLiteType<float>(), "", {4}, quant);
  int t = interpreter_->inputs()[0];

  TfLiteTensor *input_tensor = interpreter_->tensor(t);
  interpreter_->ResizeInputTensor(t, {1, 66, 4});
  interpreter_->AllocateTensors();

  auto out_cb = [&](const mediapipe::Packet &p) {
    absl::MutexLock lock(&out_mutex_);
    out_packets_.push_back(p);
    if (out_packets_.size() > 2) {
      out_packets_.erase(out_packets_.begin(), out_packets_.begin() + 1);
    }
    return absl::OkStatus();
  };

  MP_RETURN_IF_ERROR(graph_.ObserveOutputStream(kOutputStream, out_cb));

  return graph_.StartRun({});
}

float ExerciseDetectorImpl::Process(std::vector<Landmark> landmarks) {
  absl::MutexLock in_lock(&in_mutex_);

  int t = interpreter_->inputs()[0];
  TfLiteTensor *input_tensor = interpreter_->tensor(t);

  float *input_tensor_buffer = tflite::GetTensorData<float>(input_tensor);

  for (int i = 0; i < landmarks.size(); i++) {
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

  absl::MutexLock out_lock(&out_mutex_);
  auto size = out_packets_.size();
  if (size == 0) {
    return 0;
  }
  packet = out_packets_.front();
  auto res = packet.Get<std::vector<TfLiteTensor>>();
  const TfLiteTensor *result = &res[0];
  const float *result_buffer = tflite::GetTensorData<float>(result);

  return result_buffer[0];
}

} // namespace mediagraph
