#include "mediagraph.h"
#include "opencv2/imgproc.hpp"

void flip_mat(cv::Mat *input, mediagraph::Flip flip);
void color_cvt(cv::Mat *input, mediagraph::InputType input_type);

size_t get_timestamp();
