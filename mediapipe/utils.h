#include "mediagraph.h"
#include "opencv2/imgproc.hpp"
#include <memory>

void flip_mat(std::unique_ptr<cv::Mat> &input, mediagraph::Flip flip);
void color_cvt(std::unique_ptr<cv::Mat> &input,
               mediagraph::InputType input_type);

size_t get_timestamp();
