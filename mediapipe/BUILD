# Copyright 2019 The MediaPipe Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

load("@hedron_compile_commands//:refresh_compile_commands.bzl", "refresh_compile_commands")
load("@build_bazel_rules_apple//apple:ios.bzl", "ios_framework")

licenses(["notice"])  # Apache 2.0

# Note: yes, these need to use "//external:android/crosstool", not
# @androidndk//:default_crosstool.

config_setting(
    name = "android",
    values = {"crosstool_top": "//external:android/crosstool"},
    visibility = ["//visibility:public"],
)

config_setting(
    name = "android_x86",
    values = {
        "crosstool_top": "//external:android/crosstool",
        "cpu": "x86",
    },
    visibility = ["//visibility:public"],
)

config_setting(
    name = "android_x86_64",
    values = {
        "crosstool_top": "//external:android/crosstool",
        "cpu": "x86_64",
    },
    visibility = ["//visibility:public"],
)

config_setting(
    name = "android_armeabi",
    values = {
        "crosstool_top": "//external:android/crosstool",
        "cpu": "armeabi",
    },
    visibility = ["//visibility:public"],
)

config_setting(
    name = "android_arm",
    values = {
        "crosstool_top": "//external:android/crosstool",
        "cpu": "armeabi-v7a",
    },
    visibility = ["//visibility:public"],
)

config_setting(
    name = "android_arm64",
    values = {
        "crosstool_top": "//external:android/crosstool",
        "cpu": "arm64-v8a",
    },
    visibility = ["//visibility:public"],
)

# Note: this cannot just match "apple_platform_type": "macos" because that option
# defaults to "macos" even when building on Linux!
alias(
    name = "macos",
    actual = select({
        ":macos_i386": ":macos_i386",
        ":macos_x86_64": ":macos_x86_64",
        ":macos_arm64": ":macos_arm64",
        "//conditions:default": ":macos_i386",  # Arbitrarily chosen from above.
    }),
    visibility = ["//visibility:public"],
)

# Note: this also matches on crosstool_top so that it does not produce ambiguous
# selectors when used together with "android".
config_setting(
    name = "ios",
    values = {
        "crosstool_top": "@bazel_tools//tools/cpp:toolchain",
        "apple_platform_type": "ios",
    },
    visibility = ["//visibility:public"],
)

alias(
    name = "apple",
    actual = select({
        ":macos": ":macos",
        ":ios": ":ios",
        "//conditions:default": ":ios",  # Arbitrarily chosen from above.
    }),
    visibility = ["//visibility:public"],
)

config_setting(
    name = "macos_i386",
    values = {
        "apple_platform_type": "macos",
        "cpu": "darwin",
    },
    visibility = ["//visibility:public"],
)

config_setting(
    name = "macos_x86_64",
    values = {
        "apple_platform_type": "macos",
        "cpu": "darwin_x86_64",
    },
    visibility = ["//visibility:public"],
)

config_setting(
    name = "macos_arm64",
    values = {
        "apple_platform_type": "macos",
        "cpu": "darwin_arm64",
    },
    visibility = ["//visibility:public"],
)

[
    config_setting(
        name = arch,
        values = {"cpu": arch},
        visibility = ["//visibility:public"],
    )
    for arch in [
        "ios_i386",
        "ios_x86_64",
        "ios_armv7",
        "ios_arm64",
        "ios_arm64e",
    ]
]

config_setting(
    name = "windows",
    values = {"cpu": "x64_windows"},
)

exports_files(
    ["provisioning_profile.mobileprovision"],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "mediapipe_lib",
    alwayslink = 1,
    visibility = ["//visibility:public"],
    deps = [
	    "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework/formats:image_frame",
        "//mediapipe/framework/formats:image_frame_opencv",
        "//mediapipe/framework/port:parse_text_proto",
        "//mediapipe/framework/port:status",
        "@com_google_absl//absl/flags:parse",

        "//mediapipe/calculators/core:constant_side_packet_calculator",
        "//mediapipe/calculators/core:flow_limiter_calculator",
        "//mediapipe/calculators/tflite:tflite_inference_calculator"
	] + select({
            "//mediapipe/gpu:disable_gpu": [
                "//mediapipe/modules/pose_landmark:pose_landmark_cpu"
            ],
            "//conditions:default": [
                "//mediapipe/modules/pose_landmark:pose_landmark_gpu",
                "//mediapipe/gpu:gl_calculator_helper",
                "//mediapipe/gpu:gpu_buffer",
                "//mediapipe/gpu:gpu_shared_data_internal",
            ]
    })
)


cc_library(
    name = "mediagraph_lib",
    srcs = [
	    "mediagraph.cc",
        "mediagraph_impl.cc",
        "exercise_detector.cc",
        "exercise_detector_impl.cc",
        "utils.cc"
    ],
    hdrs = [
        "mediagraph.h",
        "mediagraph_impl.h",
        "exercise_detector.h",
        "exercise_detector_impl.h",
        "utils.h",
        "exported.h",
        "concurrentqueue.h"
    ],
    alwayslink = 1,
    linkopts = [
        "-Wl,-dead_strip"
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":mediapipe_lib"
	],
)

cc_binary(
    name = "libmediagraph.dylib",
    linkshared = 1,
    linkstatic = 1,
    linkopts = [
        "-install_name",
        "@loader_path/libmediagraph.dylib"
    ],
    deps = [
        ":mediagraph_lib",
    ],
)

cc_binary(
    name = "libmediagraph.so",
    linkshared = 1,
    linkstatic = 1,
    deps = [
        ":mediagraph_lib"
    ]
)

cc_binary(
    name = "mediagraph.dll",
    linkshared = 1,
    linkstatic = 1,
    deps = [
        ":mediagraph_lib",
    ],
)

objc_library(
    name = "mediagraph_objc",
    sdk_frameworks = [
        "Accelerate",
        "AVFoundation",
        "CoreVideo",
        "CoreGraphics",
        "CoreMedia",
        "GLKit",
        "OpenGLES",
        "QuartzCore",
    ],
    alwayslink = 1,
    deps = [
        ":mediagraph_lib",
    ],
)

ios_framework(
    name = "mediagraph",
    bundle_id = "com.impaktsdk.ios",
    families = [
        "iphone",
        "ipad",
    ],
    infoplists = ["Info.plist"],
    minimum_os_version = "11.0",
    visibility = ["//visibility:public"],
    bundle_only = True,
    deps = [
        ":mediagraph_objc",
        "@ios_opencv//:OpencvFramework",
    ],
)

refresh_compile_commands(
    name = "refresh_compile_commands",

    # Specify the targets of interest.
    # For example, specify a dict of targets and any flags required to build.
    targets = {
      "//mediapipe:mediagraph_lib": "-c opt",
    },
    # No need to add flags already in .bazelrc. They're automatically picked up.
    # If you don't need flags, a list of targets is also okay, as is a single target string.
    # Wildcard patterns, like //... for everything, *are* allowed here, just like a build.
      # As are additional targets (+) and subtractions (-), like in bazel query https://docs.bazel.build/versions/main/query.html#expressions
    # And if you're working on a header-only library, specify a test or binary target that compiles it.
)

