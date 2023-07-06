#include "mediagraph.h"
#include "mediagraph_impl.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/framework/packet.h"
#include "mediapipe/gpu/gl_texture_buffer.h"
#include "mediapipe/gpu/gpu_buffer.h"
#include <memory>

namespace mediagraph {

Detector *Detector::Create(const char *graph_config,
                           const uint8_t *detection_model, const size_t d_len,
                           const uint8_t *landmark_model, const size_t l_len,
                           const uint8_t *hand_model, const size_t h_len,
                           const uint8_t *hand_recrop_model,
                           const size_t hr_len, const Output *outputs,
                           uint8_t num_outputs, PoseCallback callback) {
  DetectorImpl *mediagraph = new DetectorImpl();

  absl::Status status = mediagraph->Init(
      graph_config, detection_model, d_len, landmark_model, l_len, hand_model,
      h_len, hand_recrop_model, hr_len, outputs, num_outputs, callback);
  if (status.ok()) {
    return mediagraph;
  } else {
    LOG(INFO) << "Error initializing graph " << status.ToString();
    delete mediagraph;
    return nullptr;
  }
}

void Detector::Dispose() {
  DetectorImpl *det = static_cast<DetectorImpl *>(this);

  if (det == nullptr)
    return;

  det->Dispose();
}

void Detector::Process(uint8_t *data, int width, int height,
                       InputType input_type, Flip flip_code,
                       FrameDeleter frame_deleter, const void *callback_ctx) {
  uint8_t channels;
  mediapipe::ImageFormat::Format image_format = mediapipe::ImageFormat::UNKNOWN;
  switch (input_type) {
  case InputType::BGR:
  case InputType::RGB:
    channels = 3;
    image_format = mediapipe::ImageFormat::SRGB;
    break;
  case InputType::RGBA:
    channels = 4;
    image_format = mediapipe::ImageFormat::SRGBA;
  }

  auto width_step = width * channels * sizeof(uint8_t);
  auto input = std::make_unique<mediapipe::ImageFrame>(
      image_format, width, height, width_step, data, frame_deleter);
  auto packet = mediapipe::Adopt(input.release());

  static_cast<DetectorImpl *>(this)->Process(packet, flip_code, callback_ctx);
}

void Detector::ProcessEGL(unsigned int texture, int width, int height,
                          Flip flip_code, TextureDeleter texture_deleter,
                          const void *callback_ctx) {

  std::unique_ptr<mediapipe::GlTextureBuffer> texture_buffer =
      mediapipe::GlTextureBuffer::Wrap(
          GL_TEXTURE_2D, texture, width, height,
          mediapipe::GpuBufferFormat::kBGRA32,
          [texture, texture_deleter](
              std::shared_ptr<mediapipe::GlSyncPoint> sync_token) {
            texture_deleter(texture);
          });

  std::unique_ptr<mediapipe::GpuBuffer> gpu_buffer =
      std::make_unique<mediapipe::GpuBuffer>(std::move(texture_buffer));

  auto packet = mediapipe::Adopt(gpu_buffer.release());
  static_cast<DetectorImpl *>(this)->Process(packet, flip_code, callback_ctx);
}
} // namespace mediagraph
