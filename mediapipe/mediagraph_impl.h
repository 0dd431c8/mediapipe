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

#ifndef MEDIAGRAPH_IMPL
#define MEDIAGRAPH_IMPL

#include <cstdlib>
#include <deque>
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/synchronization/mutex.h"
#include <opencv2/core.hpp>
#if !MEDIAPIPE_DISABLE_GPU
#include "gpu/gl_calculator_helper.h"
#endif
#include "mediapipe/framework/calculator_framework.h"

#include "mediagraph.h"

namespace mediagraph {

class DetectorImpl : public Detector {
public:
  DetectorImpl() {}
  void Dispose();

  absl::Status Init(const char *graph, const uint8_t *detection_model,
                    const size_t d_len, const uint8_t *landmark_model,
                    const size_t l_len, const Output *outputs,
                    uint8_t num_outputs);

  Landmark *Process(cv::Mat input, uint8_t *num_features);

private:
  mediapipe::CalculatorGraph graph_;
#if !MEDIAPIPE_DISABLE_GPU
  mediapipe::GlCalculatorHelper gpu_helper_;
#endif
  size_t frame_timestamp_ = 0;
  std::vector<Output> outputs_;
  std::vector<std::deque<mediapipe::Packet>> out_packets_;
  std::vector<absl::Mutex> out_mutexes_;
  uint8_t num_outputs_;
};
} // namespace mediagraph

#endif
