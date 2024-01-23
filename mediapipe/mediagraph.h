// Pose detector interface that is used by the library consumers

#ifndef MEDIAGRAPH_H
#define MEDIAGRAPH_H

#include "exported.h"
#include <cstdlib>
#include <string>
#include <vector>

namespace mediagraph {
// -calculator_graph_config_file=mediapipe/graphs/pose_tracking/pose_tracking_cpu.pbtxt
struct Landmark {
  float x;
  float y;
  float z;
  float visibility;
};

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
typedef void (*PoseCallback)(const void *ctx, const Landmark *landmarks,
                             const uint8_t *num_features,
                             uint8_t num_features_len);

// Called to free the frame
typedef void (*FrameDeleter)(unsigned int frame_id);

// Pose detector
class EXPORTED Detector {
public:
  // Create and initialize using provided graph
  // Returns nullptr if initialization failed
  static Detector *Create(const char *graph_config,
                          const uint8_t *detection_model, const size_t d_len,
                          const uint8_t *landmark_model, const size_t l_len,
                          const uint8_t *hand_model, const size_t h_len,
                          const uint8_t *hand_recrop_model, const size_t hr_len,
                          const Output *outputs, uint8_t num_outputs,
                          PoseCallback callback);
  void Dispose();

  // Process frame on CPU
  void Process(unsigned int frame_id, const uint8_t *data, int width,
               int height, InputType input_type, Flip flip_code,
               FrameDeleter frame_deleter, const void *callback_ctx);

  // Process frame on GPU
  void ProcessEGL(unsigned int texture, int width, int height, Flip flip_code,
                  FrameDeleter texture_deleter, const void *callback_ctx);
};
} // namespace mediagraph

#endif
