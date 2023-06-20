#include <memory>
#include <thread>
#include <vector>

#include "mediagraph_impl.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/image.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/framework/formats/image_frame_opencv.h"
#include "mediapipe/framework/formats/landmark.pb.h"
#include "mediapipe/framework/output_stream_poller.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/port/status.h"
#include "utils.h"
#if !MEDIAPIPE_DISABLE_GPU
#include "mediapipe/gpu/gl_base.h"
#include "mediapipe/gpu/gl_calculator_helper.h"
#include "mediapipe/gpu/gl_context.h"
#include "mediapipe/gpu/gl_texture_buffer.h"
#include "mediapipe/gpu/gpu_buffer.h"
#include "mediapipe/gpu/gpu_buffer_format.h"
#include "mediapipe/gpu/gpu_shared_data_internal.h"
#endif // !MEDIAPIPE_DISABLE_GPU

namespace mediagraph {

constexpr char kInputStream[] = "input_video";
constexpr char kFlipHorizontallyStream[] = "flip_horizontal";
constexpr char kFlipVerticallyStream[] = "flip_vertical";

void DetectorImpl::Dispose() {
  LOG(INFO) << "Shutting down.";
  graph_.CloseInputStream(kFlipHorizontallyStream);
  graph_.CloseInputStream(kFlipVerticallyStream);
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

absl::Status
DetectorImpl::Init(const char *graph, const uint8_t *detection_model,
                   const size_t d_len, const uint8_t *landmark_model,
                   const size_t l_len, const uint8_t *hand_model,
                   const size_t h_len, const uint8_t *hand_recrop_model,
                   const size_t hr_len, const Output *outputs,
                   uint8_t num_outputs, PoseCallback callback) {
  num_outputs_ = num_outputs;
  outputs_ = std::vector<Output>(outputs, outputs + num_outputs_);
  mediapipe::CalculatorGraphConfig config =
      mediapipe::ParseTextProtoOrDie<mediapipe::CalculatorGraphConfig>(graph);

  std::string detection_model_blob(
      reinterpret_cast<const char *>(detection_model), d_len);
  std::string landmark_model_blob(
      reinterpret_cast<const char *>(landmark_model), l_len);

  std::map<std::string, mediapipe::Packet> extra_side_packets;
  extra_side_packets.insert(
      {"detection_model_blob",
       mediapipe::MakePacket<std::string>(std::move(detection_model_blob))});
  extra_side_packets.insert(
      {"landmark_model_blob",
       mediapipe::MakePacket<std::string>(std::move(landmark_model_blob))});

  if (hand_model != nullptr && h_len > 0 && hand_recrop_model != nullptr &&
      hr_len > 0) {
    std::string hand_model_blob(reinterpret_cast<const char *>(hand_model),
                                h_len);
    std::string hand_recrop_model_blob(
        reinterpret_cast<const char *>(hand_recrop_model), hr_len);

    extra_side_packets.insert(
        {"hand_model_blob",
         mediapipe::MakePacket<std::string>(std::move(hand_model_blob))});

    extra_side_packets.insert(
        {"hand_recrop_model_blob", mediapipe::MakePacket<std::string>(
                                       std::move(hand_recrop_model_blob))});
  }

  MP_RETURN_IF_ERROR(graph_.Initialize(config, extra_side_packets));

#if !MEDIAPIPE_DISABLE_GPU
  auto ctx = eglGetCurrentContext();
  ASSIGN_OR_RETURN(auto gpu_resources, mediapipe::GpuResources::Create(ctx));
  MP_RETURN_IF_ERROR(graph_.SetGpuResources(std::move(gpu_resources)));
  gpu_helper_.InitializeForTest(graph_.GetGpuResources().get());
#endif

  LOG(INFO) << "Start running the calculator graph.";

  out_streams_ = std::vector<std::unique_ptr<mediapipe::OutputStreamPoller>>();

  for (uint i = 0; i < num_outputs_; ++i) {
    auto sop = graph_.AddOutputStreamPoller(outputs_[i].name);

    std::unique_ptr<mediapipe::OutputStreamPoller> poller =
        std::make_unique<mediapipe::OutputStreamPoller>(std::move(sop.value()));

    out_streams_.push_back(std::move(poller));
  }

  callback_ = callback;

  MP_RETURN_IF_ERROR(graph_.StartRun({}));

  return absl::OkStatus();
}

template <typename T, typename L>
void copyLandmarks(const T &landmarks, std::vector<Landmark> &output,
                   const int start_index) {
  for (int idx = 0; idx < landmarks.landmark_size(); ++idx) {
    const L &landmark = landmarks.landmark(idx);

    output[start_index + idx] = {
        landmark.x(),
        landmark.y(),
        landmark.z(),
        landmark.visibility(),
    };
  }
}

template <typename T, typename L>
std::vector<Landmark> parseLandmarkListPacket(const mediapipe::Packet &packet,
                                              const int num_landmarks,
                                              uint8_t *num_features) {
  auto &landmarks = packet.Get<T>();
  std::vector<Landmark> output(num_landmarks);
  copyLandmarks<T, L>(landmarks, output, 0);
  *num_features = 1;
  return output;
}

std::vector<Landmark> parsePacket(const mediapipe::Packet &packet,
                                  const FeatureType type,
                                  uint8_t *num_features) {
  switch (type) {
  case FeatureType::NORMALIZED_LANDMARKS:
    return parseLandmarkListPacket<mediapipe::NormalizedLandmarkList,
                                   mediapipe::NormalizedLandmark>(packet, 33,
                                                                  num_features);
  case FeatureType::WORLD_LANDMARKS:
    return parseLandmarkListPacket<mediapipe::LandmarkList,
                                   mediapipe::Landmark>(packet, 33,
                                                        num_features);
  case FeatureType::NORMALIZED_HAND_LANDMARKS:
    return parseLandmarkListPacket<mediapipe::NormalizedLandmarkList,
                                   mediapipe::NormalizedLandmark>(packet, 21,
                                                                  num_features);
  case FeatureType::WORLD_HAND_LANDMARKS:
    return parseLandmarkListPacket<mediapipe::LandmarkList,
                                   mediapipe::Landmark>(packet, 21,
                                                        num_features);
  default:
    LOG(INFO) << "NO MATCH\n";
    *num_features = 0;
    return std::vector<Landmark>(0);
  }
}

void DetectorImpl::Process(cv::Mat input, const void *callback_ctx,
                           uint texture) {
  // auto input_frame = mat_to_image_frame(input);

  auto frame_timestamp = mediapipe::Timestamp(get_timestamp());

  graph_.AddPacketToInputStream(
      kFlipVerticallyStream,
      mediapipe::MakePacket<bool>(true).At(frame_timestamp));
  graph_.AddPacketToInputStream(
      kFlipHorizontallyStream,
      mediapipe::MakePacket<bool>(true).At(frame_timestamp));

#if !MEDIAPIPE_DISABLE_GPU
  auto texture_buffer = mediapipe::GlTextureBuffer::Wrap(
      GL_TEXTURE_2D, texture, 640, 480, mediapipe::GpuBufferFormat::kBGRA32,
      [](std::shared_ptr<mediapipe::GlSyncPoint> sync_token) {
        LOG(INFO) << "!!!!";
      });

  std::unique_ptr<mediapipe::GpuBuffer> gpu_buffer =
      std::make_unique<mediapipe::GpuBuffer>(std::move(texture_buffer));

  LOG(INFO) << gpu_buffer->DebugString();

  // auto texture = gpu_helper_.CreateSourceTexture(*input_frame.get());
  // auto gpu_frame = texture.GetFrame<mediapipe::GpuBuffer>();
  // texture.Release();
  // Send GPU image packet into the graph.
  auto run_status = graph_.AddPacketToInputStream(
      kInputStream, mediapipe::Adopt(gpu_buffer.release()).At(frame_timestamp));
#else
  mediapipe::Status run_status = graph_.AddPacketToInputStream(
      kInputStream,
      mediapipe::Adopt(input_frame.release()).At(frame_timestamp));
#endif
  if (!run_status.ok()) {
    LOG(INFO) << "Add Packet error: [" << run_status.message() << "]"
              << std::endl;
    return;
  }

  std::thread([this, callback_ctx]() {
    std::vector<Landmark> landmarks;
    mediapipe::Packet packet;

    std::vector<uint8_t> num_features(num_outputs_);

    for (uint i = 0; i < num_outputs_; ++i) {
      auto &poller = out_streams_[i];

      if (poller->QueueSize() < 1) {
        continue;
      }

      bool found = poller->Next(&packet);
      LOG(INFO) << found;

      if (!found) {
        num_features[i] = 0;
        continue;
      }

      auto result = parsePacket(packet, outputs_[i].type, &num_features[i]);

      if (result.size() > 0) {
        landmarks.insert(landmarks.end(), result.begin(), result.end());
      }
    }

    callback_(callback_ctx, landmarks.data(), num_features.data(),
              num_features.size());
  }).detach();
}
} // namespace mediagraph
