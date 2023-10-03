// Copyright 2019 The MediaPipe Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// An example of sending OpenCV webcam frames into a MediaPipe graph.

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

typedef void (*PoseCallback)(const void *ctx, const Landmark *landmarks,
                             const uint8_t *num_features,
                             uint8_t num_features_len);

typedef void (*FrameDeleter)(unsigned int frame_id);

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

  void Process(unsigned int frame_id, const uint8_t *data, int width,
               int height, InputType input_type, Flip flip_code,
               FrameDeleter frame_deleter, const void *callback_ctx);

  void ProcessEGL(unsigned int texture, int width, int height, Flip flip_code,
                  FrameDeleter texture_deleter, const void *callback_ctx);
};
} // namespace mediagraph

#endif
