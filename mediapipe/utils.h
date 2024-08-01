#include "mediagraph.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/framework/formats/image_frame_opencv.h"
#include "opencv2/imgproc.hpp"
#include <optional>
#include <cstdint>

void flip_mat(cv::Mat *input, mediagraph::Flip flip);
void color_cvt(cv::Mat *input, mediagraph::InputType input_type);

std::optional<cv::Mat> bytes_to_mat(const uint8_t *data, int width, int height,
                                    mediagraph::InputType input_type);

std::unique_ptr<mediapipe::ImageFrame> mat_to_image_frame(cv::Mat input);

size_t get_timestamp();
