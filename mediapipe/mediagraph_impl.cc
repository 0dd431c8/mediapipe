#include <_types/_uint8_t.h>
#include <vector>

#include "mediagraph_impl.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/framework/formats/image_frame_opencv.h"
#include "mediapipe/framework/formats/landmark.pb.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/port/status.h"

namespace mediagraph {

constexpr char kInputStream[] = "input_video";

void DetectorImpl::Dispose() {
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

absl::Status DetectorImpl::Init(const char *graph,
                                const uint8_t *detection_model,
                                const size_t d_len,
                                const uint8_t *landmark_model,
                                const size_t l_len, const Output *outputs,
                                uint8_t num_outputs) {
  num_outputs_ = num_outputs;
  outputs_ = std::vector<Output>(outputs, outputs + num_outputs_);
  LOG(INFO) << "Parsing graph config " << graph;
  mediapipe::CalculatorGraphConfig config =
      mediapipe::ParseTextProtoOrDie<mediapipe::CalculatorGraphConfig>(graph);

  LOG(INFO) << "Initialize the calculator graph.";

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

  MP_RETURN_IF_ERROR(graph_.Initialize(config, extra_side_packets));

  LOG(INFO) << "Start running the calculator graph.";

  out_packets_ = std::vector<std::deque<mediapipe::Packet>>(num_outputs_);
  out_mutexes_ = std::vector<absl::Mutex>(num_outputs_);

  for (uint i = 0; i < num_outputs_; ++i) {
    auto out_cb = [&, i](const mediapipe::Packet &p) {
      absl::MutexLock lock(&out_mutexes_[i]);
      out_packets_[i].push_back(p);
      if (out_packets_[i].size() > 2) {
        out_packets_[i].erase(out_packets_[i].begin(),
                              out_packets_[i].begin() + 1);
      }
      return absl::OkStatus();
    };

    MP_RETURN_IF_ERROR(graph_.ObserveOutputStream(outputs_[i].name, out_cb));
  }

  MP_RETURN_IF_ERROR(graph_.StartRun({}));

  return absl::OkStatus();
}

template <typename T, typename L>
void copyLandmarks(const T &landmarks, std::vector<Landmark> &output,
                   const int start_index) {
  for (int idx = 0; idx < landmarks.landmark_size(); ++idx) {
    const L &landmark = landmarks.landmark(idx);

    output[start_index + idx] = {
        .x = landmark.x(),
        .y = landmark.y(),
        .z = landmark.z(),
        .visibility = landmark.visibility(),
        .presence = landmark.presence(),
    };
  }
}

template <typename T, typename L>
std::vector<Landmark> parseLandmarkListPacket(const mediapipe::Packet &packet,
                                              uint8_t *num_features) {
  constexpr int num_landmarks = 33;
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
                                   mediapipe::NormalizedLandmark>(packet,
                                                                  num_features);
  case FeatureType::WORLD_LANDMARKS:
    return parseLandmarkListPacket<mediapipe::LandmarkList,
                                   mediapipe::Landmark>(packet, num_features);
  default:
    LOG(INFO) << "NO MATCH\n";
    *num_features = 0;
    return std::vector<Landmark>(0);
  }
}

Landmark *DetectorImpl::Process(uint8_t *data, int width, int height,
                                uint8_t *num_features) {
  if (data == nullptr) {
    LOG(INFO) << __FUNCTION__ << " input data is nullptr!";
    return nullptr;
  }

  int width_step =
      width *
      mediapipe::ImageFrame::ByteDepthForFormat(mediapipe::ImageFormat::SRGB) *
      mediapipe::ImageFrame::NumberOfChannelsForFormat(
          mediapipe::ImageFormat::SRGB);

  cv::Mat input_mat(height, width, CV_8UC3, data);

  auto input_frame = absl::make_unique<mediapipe::ImageFrame>(
      mediapipe::ImageFormat::SRGB, input_mat.cols, input_mat.rows,
      mediapipe::ImageFrame::kDefaultAlignmentBoundary);

  cv::Mat input_frame_mat = mediapipe::formats::MatView(input_frame.get());
  input_mat.copyTo(input_frame_mat);

  // auto input_frame_for_input = absl::make_unique<mediapipe::ImageFrame>(
  //     mediapipe::ImageFormat::SRGB, width, height, width_step, (uint8 *)data,
  //     mediapipe::ImageFrame::PixelDataDeleter::kNone);

  // frame_timestamp_++;
  size_t frame_timestamp_us =
      (double)cv::getTickCount() / (double)cv::getTickFrequency() * 1e6;

  mediapipe::Status run_status = graph_.AddPacketToInputStream(
      kInputStream, mediapipe::Adopt(input_frame.release())
                        .At(mediapipe::Timestamp(frame_timestamp_us)));

  if (!run_status.ok()) {
    LOG(INFO) << "Add Packet error: [" << run_status.message() << "]"
              << std::endl;
    return nullptr;
  }

  std::vector<Landmark> landmarks;
  mediapipe::Packet packet;

  for (uint i = 0; i < num_outputs_; ++i) {
    {
      absl::MutexLock lock(&out_mutexes_[i]);

      auto size = out_packets_[i].size();
      if (size == 0) {
        num_features[i] = 0;
        continue;
      }

      packet = out_packets_[i].front();
    }

    auto result = parsePacket(packet, outputs_[i].type, num_features + i);

    if (result.size() > 0) {
      landmarks.insert(landmarks.end(), result.begin(), result.end());
    }
  }

  return landmarks.data();
}
} // namespace mediagraph
