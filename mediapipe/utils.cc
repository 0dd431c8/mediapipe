#include "utils.h"
#include "mediagraph.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/mediagraph.h"
#include "opencv2/imgproc.hpp"
#include <optional>

using mediagraph::InputType;

void flip_mat(cv::Mat *input, mediagraph::Flip flip) {

  if (flip == mediagraph::Flip::None)
    return;

  cv::flip(*input, *input, flip);
}

void color_cvt(cv::Mat *input, InputType input_type) {
  bool _gpu = false;
#if !MEDIAPIPE_DISABLE_GPU
  _gpu = true;
#endif

  cv::ColorConversionCodes cvt_code;

  if (input_type == InputType::RGBA) {
    if (_gpu)
      return;

    cvt_code = cv::COLOR_RGBA2RGB;
  }

  if (input_type == InputType::RGB) {
    if (!_gpu)
      return;

    cvt_code = cv::COLOR_RGB2RGBA;
  }

  if (input_type == InputType::BGR) {
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

std::optional<cv::Mat> bytes_to_mat(const uint8_t *data, int width, int height,
                                    InputType input_type) {
  if (data == nullptr) {
    return std::nullopt;
  }

  int mat_type;

  switch (input_type) {
  case InputType::BGR:
  case InputType::RGB:
    mat_type = CV_8UC3;
    break;
  case InputType::RGBA:
    mat_type = CV_8UC4;
    break;
  default:
    LOG(ERROR) << __FUNCTION__ << " Unsupported input type!";
    return std::nullopt;
  }

  cv::Mat input(cv::Size(width, height), mat_type, const_cast<uint8_t *>(data));

  return input.clone();
}

std::unique_ptr<mediapipe::ImageFrame> mat_to_image_frame(cv::Mat input) {
#if MEDIAPIPE_DISABLE_GPU
  auto image_format = mediapipe::ImageFormat::SRGB;
  auto alignment = mediapipe::ImageFrame::kDefaultAlignmentBoundary;
#else
  auto image_format = mediapipe::ImageFormat::SRGBA;
  auto alignment = mediapipe::ImageFrame::kGlDefaultAlignmentBoundary;
#endif

  auto input_frame = absl::make_unique<mediapipe::ImageFrame>(
      image_format, input.cols, input.rows, alignment);

  cv::Mat input_frame_mat = mediapipe::formats::MatView(input_frame.get());
  input.copyTo(input_frame_mat);
  input.release();

  return input_frame;
}

size_t get_timestamp() {
  return (double)cv::getTickCount() / (double)cv::getTickFrequency() * 1e6;
}
