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
load("@build_bazel_rules_apple//apple:apple.bzl", "apple_universal_binary")

licenses(["notice"])  # Apache 2.0

load("@mediapipe//mediapipe:platforms.bzl", "config_setting_and_platform")

# Generic Android
config_setting(
    name = "android",
    constraint_values = [
        "@platforms//os:android",
    ],
    visibility = ["//visibility:public"],
)

# Android x86 32-bit.
config_setting_and_platform(
    name = "android_x86",
    constraint_values = [
        "@platforms//os:android",
        "@platforms//cpu:x86_32",
    ],
    visibility = ["//visibility:public"],
)

# Android x86 64-bit.
config_setting_and_platform(
    name = "android_x86_64",
    constraint_values = [
        "@platforms//os:android",
        "@platforms//cpu:x86_64",
    ],
    visibility = ["//visibility:public"],
)

# Android ARMv7.
config_setting_and_platform(
    name = "android_arm",
    constraint_values = [
        "@platforms//os:android",
        "@platforms//cpu:armv7",
    ],
    visibility = ["//visibility:public"],
)

# Android ARM64.
config_setting_and_platform(
    name = "android_arm64",
    constraint_values = [
        "@platforms//os:android",
        "@platforms//cpu:arm64",
    ],
    visibility = ["//visibility:public"],
)

# Generic MacOS.
config_setting(
    name = "macos",
    constraint_values = [
        "@platforms//os:macos",
    ],
    visibility = ["//visibility:public"],
)

# MacOS x86 64-bit.
config_setting_and_platform(
    name = "macos_x86_64",
    constraint_values = [
        "@platforms//os:macos",
        "@platforms//cpu:x86_64",
    ],
    visibility = ["//visibility:public"],
)

# MacOS ARM64.
config_setting_and_platform(
    name = "macos_arm64",
    constraint_values = [
        "@platforms//os:macos",
        "@platforms//cpu:arm64",
    ],
    visibility = ["//visibility:public"],
)

# Generic iOS.
config_setting(
    name = "ios",
    constraint_values = [
        "@platforms//os:ios",
    ],
    visibility = ["//visibility:public"],
)

# iOS device ARM32.
config_setting_and_platform(
    name = "ios_armv7",
    constraint_values = [
        "@platforms//os:ios",
        "@platforms//cpu:arm",
    ],
    visibility = ["//visibility:public"],
)

# iOS device ARM64.
config_setting_and_platform(
    name = "ios_arm64",
    constraint_values = [
        "@platforms//os:ios",
        "@platforms//cpu:arm64",
    ],
    visibility = ["//visibility:public"],
)

# iOS device ARM64E.
config_setting_and_platform(
    name = "ios_arm64e",
    constraint_values = [
        "@platforms//os:ios",
        "@platforms//cpu:arm64e",
    ],
    visibility = ["//visibility:public"],
)

# iOS simulator x86 32-bit.
config_setting_and_platform(
    name = "ios_i386",
    constraint_values = [
        "@platforms//os:ios",
        "@platforms//cpu:x86_32",
        "@build_bazel_apple_support//constraints:simulator",
    ],
    visibility = ["//visibility:public"],
)

# iOS simulator x86 64-bit.
config_setting_and_platform(
    name = "ios_x86_64",
    constraint_values = [
        "@platforms//os:ios",
        "@platforms//cpu:x86_64",
        "@build_bazel_apple_support//constraints:simulator",
    ],
    visibility = ["//visibility:public"],
)

# iOS simulator ARM64.
config_setting_and_platform(
    name = "ios_sim_arm64",
    constraint_values = [
        "@platforms//os:ios",
        "@platforms//cpu:arm64",
        "@build_bazel_apple_support//constraints:simulator",
    ],
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

# Windows 64-bit.
config_setting_and_platform(
    name = "windows",
    constraint_values = [
        "@platforms//os:windows",
        "@platforms//cpu:x86_64",
    ],
    visibility = ["//visibility:public"],
)

# Linux 64-bit.
config_setting_and_platform(
    name = "linux",
    constraint_values = [
        "@platforms//os:linux",
        "@platforms//cpu:x86_64",
    ],
    visibility = ["//visibility:public"],
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
        "//mediapipe/tasks/cc/vision/pose_landmarker:pose_landmarker",
        "@com_google_absl//absl/flags:parse",

        "//mediapipe/calculators/core:constant_side_packet_calculator",
        "//mediapipe/calculators/core:flow_limiter_calculator",
        "//mediapipe/calculators/tflite:tflite_inference_calculator",
        "//mediapipe/calculators/tflite:tflite_tensors_to_floats_calculator",
        "//mediapipe/calculators/image:image_transformation_calculator",
        "//mediapipe/calculators/image:color_convert_calculator",
	] + select({
            "//mediapipe/gpu:disable_gpu": [
                "//mediapipe/modules/pose_landmark:pose_landmark_cpu",
                "//mediapipe/modules/holistic_landmark:hand_landmarks_left_and_right_cpu"
            ],
            "//conditions:default": [
                "//mediapipe/modules/pose_landmark:pose_landmark_gpu",
                "//mediapipe/modules/holistic_landmark:hand_landmarks_left_and_right_gpu",
                "//mediapipe/gpu:image_frame_to_gpu_buffer_calculator",
                "//mediapipe/gpu:gl_calculator_helper",
                "//mediapipe/gpu:gpu_buffer",
                "//mediapipe/gpu:gpu_shared_data_internal"
            ]
    })
)


cc_library(
    name = "mediagraph_lib",
    srcs = [
	    "mediagraph.cc",
        "mediagraph_impl.cc",
        "pose_classifier.cc",
        "pose_classifier_impl.cc",
        "utils.cc"
    ],
    hdrs = [
        "mediagraph.h",
        "mediagraph_impl.h",
        "pose_classifier.h",
        "pose_classifier_impl.h",
        "utils.h",
        "exported.h",
        "concurrentqueue.h"
    ],
    alwayslink = 1,
    linkopts = [
        "-Wl,-dead_strip",
        "-flto",
        "-s"
    ],
    copts = [
        "-Ofast",
        "-funroll-loops",
        "-fomit-frame-pointer",
        "-flto=full"
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
        "@loader_path/libmediagraph.dylib",
        "-framework OpenCL"
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
    minimum_os_version = "12.0",
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
      "//mediapipe:libmediagraph.dylib": "-c opt --define MEDIAPIPE_DISABLE_GPU=1",
    },
    # No need to add flags already in .bazelrc. They're automatically picked up.
    # If you don't need flags, a list of targets is also okay, as is a single target string.
    # Wildcard patterns, like //... for everything, *are* allowed here, just like a build.
      # As are additional targets (+) and subtractions (-), like in bazel query https://docs.bazel.build/versions/main/query.html#expressions
    # And if you're working on a header-only library, specify a test or binary target that compiles it.
)

