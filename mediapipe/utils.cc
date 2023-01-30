#include "utils.h"
#include "mediapipe/mediagraph.h"
#include "opencv2/imgproc.hpp"

void flip_mat(std::unique_ptr<cv::Mat> &input, mediagraph::Flip flip) {
  if (flip == mediagraph::Flip::None)
    return;

  cv::Mat flipped;

  cv::flip(*input, flipped, flip);

  input.reset(new cv::Mat(flipped));
}

void color_cvt(std::unique_ptr<cv::Mat> &input,
               mediagraph::InputType input_type) {
  bool _gpu = false;
#if !MEDIAPIPE_DISABLE_GPU
  _gpu = true;
#endif

  cv::Mat converted;
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

  cv::cvtColor(*input, converted, cvt_code);

  input.reset(new cv::Mat(converted));
}
