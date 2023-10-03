# Description:
#   OpenCV libraries for video/image processing on Android

licenses(["notice"])  # BSD license

exports_files(["LICENSE"])

OPENCV_LIBRARY_NAME = "libopencv_java3.so"

OPENCVANDROIDSDK_NATIVELIBS_PATH = "sdk/native/staticlibs/"
OPENCVANDROIDSDK_JNI_PATH = "sdk/native/jni/"
THIRD_PARTY = "sdk/native/3rdparty/libs/"

[cc_library(
    name = "libopencv_" + arch,
    srcs = [
        OPENCVANDROIDSDK_NATIVELIBS_PATH + arch + "/" + "libopencv_core.a",
        OPENCVANDROIDSDK_NATIVELIBS_PATH + arch + "/" + "libopencv_imgproc.a",
        THIRD_PARTY + arch + "/libcpufeatures.a",
        THIRD_PARTY + arch + "/libtbb.a",
        THIRD_PARTY + arch + "/libtegra_hal.a",
        THIRD_PARTY + arch + "/libzlib.a"
    ],
    hdrs = glob([
        OPENCVANDROIDSDK_JNI_PATH + "include/**/*.h",
        OPENCVANDROIDSDK_JNI_PATH + "include/**/*.hpp",
    ]),
    includes = [
        OPENCVANDROIDSDK_JNI_PATH + "include",
    ],
    visibility = ["//visibility:public"],
    linkstatic = 1,
    alwayslink = 1,
) for arch in [
    "arm64-v8a",
    "armeabi-v7a",
    "x86",
    "x86_64",
]]

# [alias(
#     name = "libopencv_java3_so_" + arch,
#     actual = OPENCVANDROIDSDK_NATIVELIBS_PATH + arch + "/" + OPENCV_LIBRARY_NAME,
#     visibility = ["//visibility:public"],
# ) for arch in [
#     "arm64-v8a",
#     "armeabi-v7a",
#     "x86",
#     "x86_64",
# ]]
