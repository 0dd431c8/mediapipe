#include "mediagraph.h"
#include "mediagraph_impl.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/framework/packet.h"
#if !MEDIAPIPE_DISABLE_GPU
#include "mediapipe/gpu/gl_texture_buffer.h"
#include "mediapipe/gpu/gpu_buffer.h"
#endif // !MEDIAPIPE_DISABLE_GPU
#include <memory>

namespace mediagraph {

Detector *Detector::Create(const uint8_t *pose_landmarker_model,
                           const size_t model_len,
                           PoseLandmarkerDelegate delegate,
                           PoseCallback callback) {
  DetectorImpl *mediagraph = new DetectorImpl();

  absl::Status status =
      mediagraph->Init(pose_landmarker_model, model_len, delegate, callback);
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

size_t Detector::Process(unsigned int frame_id, const uint8_t *data, int width,
                       int height, InputType input_type, Flip flip_code,
                       FrameDeleter frame_deleter, const void *callback_ctx) {
  uint8_t channels;
  mediapipe::ImageFormat::Format image_format = mediapipe::ImageFormat::UNKNOWN;
  // select image format based on input type
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

  // construct mediapipe input packet
  auto width_step = width * channels * sizeof(uint8_t);
  auto input = std::make_shared<mediapipe::ImageFrame>(
      image_format, width, height, width_step, const_cast<uint8_t *>(data),
      [frame_id, frame_deleter](uint8_t *_data) { frame_deleter(frame_id); });
  mediapipe::Image img(input);

  return static_cast<DetectorImpl *>(this)->Process(img, flip_code, callback_ctx);
}

void Detector::ProcessEGL(unsigned int texture, int width, int height,
                          Flip flip_code, FrameDeleter frame_deleter,
                          const void *callback_ctx) {
  // #if MEDIAPIPE_DISABLE_GPU || !HAS_EGL
  //   return;
  // #else
  //   // wrap input texture id in GlTextureBuffer
  //   std::unique_ptr<mediapipe::GlTextureBuffer> texture_buffer =
  //       mediapipe::GlTextureBuffer::Wrap(
  //           GL_TEXTURE_2D, texture, width, height,
  //           mediapipe::GpuBufferFormat::kBGRA32,
  //           [texture,
  //            frame_deleter](std::shared_ptr<mediapipe::GlSyncPoint>
  //            sync_token) {
  //             frame_deleter(texture);
  //           });
  //
  //   std::unique_ptr<mediapipe::GpuBuffer> gpu_buffer =
  //       std::make_unique<mediapipe::GpuBuffer>(std::move(texture_buffer));
  //
  //   auto packet = mediapipe::Adopt(gpu_buffer.release());
  //   static_cast<DetectorImpl *>(this)->Process(packet, flip_code,
  //   callback_ctx);
  // #endif // MEDIAPIPE_DISABLE_GPU
}
} // namespace mediagraph
