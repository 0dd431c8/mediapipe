#include <_types/_uint8_t.h>
#include <vector>

#include "mediagraph_impl.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/framework/formats/image_frame_opencv.h"
#include "mediapipe/framework/formats/landmark.pb.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/port/status.h"
#include "opencv2/imgproc.hpp"
#if !MEDIAPIPE_DISABLE_GPU
#include "mediapipe/gpu/gl_calculator_helper.h"
#include "mediapipe/gpu/gpu_buffer.h"
#include "mediapipe/gpu/gpu_shared_data_internal.h"
#endif // !MEDIAPIPE_DISABLE_GPU

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

#if !MEDIAPIPE_DISABLE_GPU
  ASSIGN_OR_RETURN(auto gpu_resources, mediapipe::GpuResources::Create());
  MP_RETURN_IF_ERROR(graph_.SetGpuResources(std::move(gpu_resources)));
  gpu_helper_.InitializeForTest(graph_.GetGpuResources().get());
#endif

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

  cv::Mat raw_frame(cv::Size(width, height), CV_8UC4, data);
  cv::Mat rgb_frame, flip_frame;
#if MEDIAPIPE_DISABLE_GPU
  cv::cvtColor(raw_frame, rgb_frame, cv::COLOR_RGBA2RGB);
  cv::flip(rgb_frame, flip_frame, 1);

  auto input_frame = absl::make_unique<mediapipe::ImageFrame>(
      mediapipe::ImageFormat::SRGB, flip_frame.cols, flip_frame.rows,
      mediapipe::ImageFrame::kDefaultAlignmentBoundary);
#else
  cv::flip(raw_frame, flip_frame, 1);

  auto input_frame = absl::make_unique<mediapipe::ImageFrame>(
      mediapipe::ImageFormat::SRGBA, flip_frame.cols, flip_frame.rows,
      mediapipe::ImageFrame::kDefaultAlignmentBoundary);
#endif

  cv::Mat input_frame_mat = mediapipe::formats::MatView(input_frame.get());
  flip_frame.copyTo(input_frame_mat);

  size_t frame_timestamp_us =
      (double)cv::getTickCount() / (double)cv::getTickFrequency() * 1e6;

#if !MEDIAPIPE_DISABLE_GPU
  mediapipe::Status run_status = gpu_helper_.RunInGlContext(
      [&input_frame, &frame_timestamp_us, this]() -> absl::Status {
        auto texture = gpu_helper_.CreateSourceTexture(*input_frame.get());
        auto gpu_frame = texture.GetFrame<mediapipe::GpuBuffer>();
        glFlush();
        texture.Release();
        // Send GPU image packet into the graph.
        MP_RETURN_IF_ERROR(graph_.AddPacketToInputStream(
            kInputStream, mediapipe::Adopt(gpu_frame.release())
                              .At(mediapipe::Timestamp(frame_timestamp_us))));
        return absl::OkStatus();
      });
#else
  mediapipe::Status run_status = graph_.AddPacketToInputStream(
      kInputStream, mediapipe::Adopt(input_frame.release())
                        .At(mediapipe::Timestamp(frame_timestamp_us)));
#endif
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
