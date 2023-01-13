#include <vector>

#include "mediagraph_impl.h"
// #include "absl/flags/flag.h"
// #include "absl/flags/parse.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/framework/formats/landmark.pb.h"
#include "mediapipe/framework/port/file_helpers.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/port/status.h"

namespace mediagraph {

constexpr char kInputStream[] = "input_video";

void DetectorImpl::Dispose() {
    LOG(INFO) << "Shutting down.";
    absl::Status status = graph_.CloseInputStream(kInputStream);
    if (status.ok()){
    	absl::Status status1 = graph_.WaitUntilDone();
        if (!status1.ok()) {
            LOG(INFO) << "Error in WaitUntilDone(): " << status1.ToString();
        }
    } else {
        LOG(INFO) << "Error in CloseInputStream(): " << status.ToString();
    }
}

absl::Status DetectorImpl::Init(const char* graph, const Output* outputs, uint8_t num_outputs) {
    num_outputs_ = num_outputs;
    outputs_ = std::vector<Output>(outputs, outputs + num_outputs_);
    LOG(INFO) << "Parsing graph config " << graph;
    mediapipe::CalculatorGraphConfig config = mediapipe::ParseTextProtoOrDie<mediapipe::CalculatorGraphConfig>(graph);

    LOG(INFO) << "Initialize the calculator graph.";
    MP_RETURN_IF_ERROR(graph_.Initialize(config));

    LOG(INFO) << "Start running the calculator graph.";

    out_packets_ = std::vector<std::deque<mediapipe::Packet>>(num_outputs_);
    out_mutexes_ = std::vector<absl::Mutex>(num_outputs_);

    for (uint i = 0; i < num_outputs_; ++i) {
        auto out_cb = [&, i](const mediapipe::Packet& p) {
            absl::MutexLock lock(&out_mutexes_[i]);
            out_packets_[i].push_back(p);
            if (out_packets_[i].size() > 2) {
                out_packets_[i].erase(out_packets_[i].begin(), out_packets_[i].begin() + 1);
            }
            return absl::OkStatus();
        };

        MP_RETURN_IF_ERROR(graph_.ObserveOutputStream(outputs_[i].name, out_cb));
    }

    MP_RETURN_IF_ERROR(graph_.StartRun({}));

    return absl::OkStatus();
}

void copyLandmarks(
    const mediapipe::NormalizedLandmarkList& landmarks,
    std::vector<Landmark>& output,
    const int start_index
) {
    for (int idx = 0; idx < landmarks.landmark_size(); ++idx) {
        const mediapipe::NormalizedLandmark& landmark = landmarks.landmark(idx);

        output[start_index + idx] = {
            .x = landmark.x(),
            .y = landmark.y(),
            .z = landmark.z(),
            .visibility = landmark.visibility(),
            .presence = landmark.presence(),
        };
    }
}

void copyWorldLandmarks(
    const mediapipe::LandmarkList& landmarks,
    std::vector<Landmark>& output,
    const int start_index
) {
    for (int idx = 0; idx < landmarks.landmark_size(); ++idx) {
        const mediapipe::Landmark& landmark = landmarks.landmark(idx);

        output[start_index + idx] = {
            .x = landmark.x(),
            .y = landmark.y(),
            .z = landmark.z(),
            .visibility = landmark.visibility(),
            .presence = landmark.presence(),
        };
    }
}

std::vector<Landmark> parseLandmarkListPacket(
    const mediapipe::Packet& packet,
    const int num_landmarks,
    uint8_t* num_features
) {
    auto& landmarks = packet.Get<mediapipe::NormalizedLandmarkList>();
    std::vector<Landmark> output(num_landmarks);
    copyLandmarks(landmarks, output, 0);
    *num_features = 1;
    return output;
}

std::vector<Landmark> parseWorldLandmarkListPacket(
    const mediapipe::Packet& packet,
    const int num_landmarks,
    uint8_t* num_features
) {
    auto& landmarks = packet.Get<mediapipe::LandmarkList>();
    std::vector<Landmark> output(num_landmarks);
    copyWorldLandmarks(landmarks, output, 0);
    *num_features = 1;
    return output;
}

std::vector<Landmark> parseLandmarkListVectorPacket(
    const mediapipe::Packet& packet,
    const int num_landmarks,
    uint8_t* num_features
) {
    auto& landmarks_list = packet.Get<std::vector<mediapipe::NormalizedLandmarkList>>();
    std::vector<Landmark> output(landmarks_list.size() * num_landmarks);

    for (int i = 0; i < landmarks_list.size(); ++i) {
        const auto& landmarks = landmarks_list[i];
        const auto start_index = i * num_landmarks;
        copyLandmarks(landmarks, output, start_index);
    }

    *num_features = landmarks_list.size();

    return output;
}

// std::vector<Landmark> parseWorldLandmarkListVectorPacket(
//     const mediapipe::Packet& packet,
//     const int num_landmarks,
//     uint8_t* num_features
// ) {
//     auto& landmarks_list = packet.Get<std::vector<mediapipe::LandmarkList>>();
//     std::vector<Landmark> output(landmarks_list.size() * num_landmarks);
//
//     for (int i = 0; i < landmarks_list.size(); ++i) {
//         const auto& landmarks = landmarks_list[i];
//         const auto start_index = i * num_landmarks;
//         copyLandmarks(landmarks, output, start_index);
//     }
//
//     *num_features = landmarks_list.size();
//
//     return output;
// }

std::vector<Landmark> parsePosePacket(const mediapipe::Packet& packet, uint8_t* num_features) {
    constexpr int num_landmarks = 33;
    return parseWorldLandmarkListPacket(packet, num_landmarks, num_features);
}

std::vector<Landmark> parsePacket(const mediapipe::Packet& packet, const FeatureType type, uint8_t* num_features) {
    switch (type) {
        case FeatureType::POSE:
            return parsePosePacket(packet, num_features);
        default:
            LOG(INFO) << "NO MATCH\n";
            *num_features = 0;
            return std::vector<Landmark>(0);
    }
}

Landmark* DetectorImpl::Process(uint8_t* data, int width, int height, uint8_t* num_features) {
    if (data == nullptr){
        LOG(INFO) << __FUNCTION__ << " input data is nullptr!";
        return nullptr;
    }

    int width_step = width * mediapipe::ImageFrame::ByteDepthForFormat(mediapipe::ImageFormat::SRGB)
        * mediapipe::ImageFrame::NumberOfChannelsForFormat(mediapipe::ImageFormat::SRGB);

    auto input_frame_for_input = absl::make_unique<mediapipe::ImageFrame>(
        mediapipe::ImageFormat::SRGB, width, height, width_step,
        (uint8*)data, mediapipe::ImageFrame::PixelDataDeleter::kNone
    );

    frame_timestamp_++;

    mediapipe::Status run_status = graph_.AddPacketToInputStream(
        kInputStream,
        mediapipe::Adopt(input_frame_for_input.release()).At(mediapipe::Timestamp(frame_timestamp_))
    );

    if (!run_status.ok()) {
        LOG(INFO) << "Add Packet error: [" << run_status.message() << "]" << std::endl;
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
}
