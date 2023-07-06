# Description:
#   OpenCV libraries for video/image processing on Windows

licenses(["notice"])  # BSD license

exports_files(["LICENSE"])

OPENCV_VERSION = "3419"

config_setting(
    name = "opt_build",
    values = {"compilation_mode": "opt"},
)

config_setting(
    name = "dbg_build",
    values = {"compilation_mode": "dbg"},
)

# The following build rule assumes that the executable "opencv-3.4.10-vc14_vc15.exe"
# is downloaded and the files are extracted to local.
# If you install OpenCV separately, please modify the build rule accordingly.
cc_library(
    name = "opencv",
    srcs = [
        "x64/vc17/staticlib/opencv_core" + OPENCV_VERSION + ".lib",
        "x64/vc17/staticlib/opencv_imgproc" + OPENCV_VERSION + ".lib",
        "x64/vc17/staticlib/zlib.lib"
    ],
    hdrs = glob(["include/opencv2/**/*.h*"]),
    includes = ["include/"],
    linkstatic = 1,
    visibility = ["//visibility:public"],
)
