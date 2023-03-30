#include "utils.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/mediagraph.h"
#include "opencv2/imgproc.hpp"

void flip_mat(cv::Mat *input, mediagraph::Flip flip) {

  if (flip == mediagraph::Flip::None)
    return;

  cv::flip(*input, *input, flip);
}

void color_cvt(cv::Mat *input, mediagraph::InputType input_type) {
  bool _gpu = false;
#if !MEDIAPIPE_DISABLE_GPU
  _gpu = true;
#endif

  cv::ColorConversionCodes cvt_code;

  if (input_type == mediagraph::InputType::RGBA) {
    if (_gpu)
      return;

    cvt_code = cv::COLOR_RGBA2RGB;
  }

  if (input_type == mediagraph::InputType::RGB) {
    if (!_gpu)
      return;

    cvt_code = cv::COLOR_RGB2RGBA;
  }

  if (input_type == mediagraph::InputType::BGR) {
    if (!_gpu) {
      cvt_code = cv::COLOR_BGR2RGB;

      return;
    }

    cvt_code = cv::COLOR_BGR2RGBA;
  }

  if (&cvt_code == nullptr) {
    return;
  }

  cv::cvtColor(*input, *input, cvt_code);
}

size_t get_timestamp() {
  return (double)cv::getTickCount() / (double)cv::getTickFrequency() * 1e6;
}
