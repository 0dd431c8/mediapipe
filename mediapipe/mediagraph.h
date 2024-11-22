// Pose detector interface that is used by the library consumers

#ifndef MEDIAGRAPH_H
#define MEDIAGRAPH_H

#include "exported.h"
#include <cstdlib>
#include <string>
#include <vector>
#include <cstdint>

namespace mediagraph {
// -calculator_graph_config_file=mediapipe/graphs/pose_tracking/pose_tracking_cpu.pbtxt
struct Landmark {
  float x;
  float y;
  float z;
  float visibility;
};

enum PoseLandmarkerDelegate : unsigned int { CPU, GPU };

enum InputType { RGB, RGBA, BGR };

enum Flip { Horizontal = 1, Vertical = 0, Both = -1, None = -2 };

enum FeatureType {
  NORMALIZED_LANDMARKS,
  WORLD_LANDMARKS,
  NORMALIZED_HAND_LANDMARKS,
  WORLD_HAND_LANDMARKS
};

struct Output {
  FeatureType type;
  char *name;
};

// Pose callback
typedef void (*PoseCallback)(const void *ctx, const Landmark *landmarks, size_t timestamp);

// Called to free the frame
typedef void (*FrameDeleter)(unsigned int frame_id);

// Pose detector
class EXPORTED Detector {
public:
  // Create and initialize using provided graph
  // Returns nullptr if initialization failed
  static Detector *Create(const uint8_t *pose_landmarker_model,
                          const size_t model_len,
                          PoseLandmarkerDelegate delegate,
                          PoseCallback callback);
  void Dispose();

  // Process frame on CPU
  size_t Process(unsigned int frame_id, const uint8_t *data, int width,
               int height, InputType input_type, Flip flip_code,
               FrameDeleter frame_deleter, const void *callback_ctx);

  // Process frame on GPU
  void ProcessEGL(unsigned int texture, int width, int height, Flip flip_code,
                  FrameDeleter texture_deleter, const void *callback_ctx);
};
} // namespace mediagraph

#endif
