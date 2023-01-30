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

#include <cstdlib>
#include <string>

namespace mediagraph {
// -calculator_graph_config_file=mediapipe/graphs/pose_tracking/pose_tracking_cpu.pbtxt
struct Landmark {
  float x;
  float y;
  float z;
  float visibility;
  float presence;
};

enum InputType { RGB, RGBA, BGR };

enum Flip { Horizontal = 1, Vertical = 0, Both = -1, None = -2 };

enum FeatureType { NORMALIZED_LANDMARKS, WORLD_LANDMARKS };

struct Output {
  FeatureType type;
  char *name;
};

class Detector {
public:
  // Create and initialize using provided graph
  // Returns nullptr if initialization failed
  static Detector *Create(const char *graph_config,
                          const uint8_t *detection_model, const size_t d_len,
                          const uint8_t *landmark_model, const size_t l_len,
                          const Output *outputs, uint8_t num_outputs);
  ~Detector();

  // Processes one frame and returns immediately.
  // If a result is available it is returned.
  // Input data is expected to be ImageFormat::SRGB (24bits)
  // Returns an empty vector if nothing is detected.
  Landmark *Process(uint8_t *data, int width, int height, InputType input_type,
                    Flip flip_code, uint8_t *num_features);
};
} // namespace mediagraph

#endif
