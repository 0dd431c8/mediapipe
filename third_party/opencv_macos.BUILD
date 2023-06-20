# Description:
#   OpenCV libraries for video/image processing on MacOS

licenses(["notice"])  # BSD license

exports_files(["LICENSE"])

load("@bazel_skylib//lib:paths.bzl", "paths")

# The path to OpenCV is a combination of the path set for "macos_opencv"
# in the WORKSPACE file and the prefix here.
PREFIX = "install"
DEPS = "install/share/OpenCV/3rdparty"

cc_library(
    name = "opencv",
    srcs = 
        glob([
            paths.join(PREFIX, "lib/libopencv_core.a"),
            paths.join(PREFIX, "lib/libopencv_imgproc.a"),
            paths.join(PREFIX, "lib/libopencv_imgcodecs.a"),
            paths.join(DEPS, "lib/libtbb.a"),
            paths.join(DEPS, "lib/libtegra_hal.a"),
            paths.join(DEPS, "lib/libzlib.a")
        ]),
    hdrs = glob([paths.join(PREFIX, "include/opencv2/**/*.h*")]),
    includes = [paths.join(PREFIX, "include/")],
    linkstatic = 1,
    visibility = ["//visibility:public"],
)
