#include "TensorRTYoloDetector.h"

#include "TensorRTGpuPreprocess.h"

#include <algorithm>
#include <chrono>
#include <exception>
#include <iostream>
#include <string>
#include <vector>

#include "trtyolo.hpp"

namespace {

std::vector<DetectionObject> ConvertDetections(
    const trtyolo::DetectRes& result,
    float conf,
    DetectorFrameStats& stats) {
    std::vector<DetectionObject> detections;
    const int count = std::max(0, result.num);
    detections.reserve(static_cast<std::size_t>(count));
    stats.proposalsBeforeNms = count;

    const std::size_t usable = std::min({
        static_cast<std::size_t>(count),
        result.boxes.size(),
        result.scores.size(),
        result.classes.size(),
    });
    for (std::size_t i = 0; i < usable; ++i) {
        const float score = result.scores[i];
        if (score < conf) {
            continue;
        }

        const trtyolo::Box& box = result.boxes[i];
        DetectionObject detection{};
        detection.bbox.x = box.left;
        detection.bbox.y = box.top;
        detection.bbox.width = box.right - box.left;
        detection.bbox.height = box.bottom - box.top;
        detection.label = result.classes[i];
        detection.prob = score;
        detections.push_back(detection);
    }

    stats.resultsAfterNms = static_cast<int>(detections.size());
    stats.outputRowsBeforeFilter = count;
    stats.outputRowsAfterFilter = stats.resultsAfterNms;
    stats.usedOutputCompaction = true;
    return detections;
}

std::size_t Bgr24PitchBytes(int width) {
    return static_cast<std::size_t>(width) * 3U;
}

void ConfigureInputColor(trtyolo::InferOption& option, TensorRTYoloInputColor inputColor) {
    if (inputColor == TensorRTYoloInputColor::BGR) {
        option.enableSwapRB();
    }
}

const char* InputColorToString(TensorRTYoloInputColor inputColor) {
    return inputColor == TensorRTYoloInputColor::BGR ? "bgr24-swap-rb" : "rgb24-no-swap";
}

} // namespace

struct TensorRTYoloDetector::Impl {
    std::unique_ptr<trtyolo::DetectModel> hostModel;
    std::unique_ptr<trtyolo::DetectModel> cudaModel;
    std::unique_ptr<trtyolo::DetectModel> flippedHostModel;
    std::unique_ptr<trtyolo::DetectModel> flippedCudaModel;
    DetectorFrameStats lastFrameStats;
    std::string enginePath;
    int deviceIndex = 0;
    TensorRTYoloInputColor inputColor = TensorRTYoloInputColor::BGR;
    bool loggedReady = false;
    bool loggedCudaMappedInput = false;
    bool loggedFlipYInput = false;
    bool loggedRuntimeNmsNote = false;
};

TensorRTYoloDetector::TensorRTYoloDetector()
    : m_impl(std::make_unique<Impl>()) {}

TensorRTYoloDetector::~TensorRTYoloDetector() {
    ReleaseResources();
}

bool TensorRTYoloDetector::Initialize(const DetectorSettings& settings) {
    ReleaseResources();
    m_impl = std::make_unique<Impl>();
    m_impl->enginePath = settings.enginePath;
    m_impl->deviceIndex = settings.deviceIndex;
    m_impl->inputColor = settings.tensorRtYoloInputColor;

    if (settings.enginePath.empty()) {
        m_lastError = "TensorRT-YOLO requires a prebuilt engine path. Use trtyolo-export on the YOLO ONNX, then build .engine/.plan with trtexec.";
        return false;
    }

    if (settings.tensorRtRoi.enabled) {
        std::cout << "[TensorRT-YOLO] --trt-roi is ignored by the new backend for now; use a TensorRT-YOLO engine built for the desired input profile."
            << std::endl;
    }
    if (settings.allowTensorRtEngineBuild) {
        std::cout << "[TensorRT-YOLO] --allow-trt-build is ignored. Build the TensorRT-YOLO engine offline with trtyolo-export + trtexec."
            << std::endl;
    }

    try {
        trtyolo::InferOption hostOption;
        hostOption.setDeviceId(settings.deviceIndex);
        ConfigureInputColor(hostOption, settings.tensorRtYoloInputColor);

        m_impl->hostModel = std::make_unique<trtyolo::DetectModel>(settings.enginePath, hostOption);
        m_lastError.clear();
        std::cout << "[TensorRT-YOLO] DetectModel ready engine=" << settings.enginePath
            << " device=" << settings.deviceIndex
            << " input=" << InputColorToString(settings.tensorRtYoloInputColor)
            << std::endl;
        return true;
    }
    catch (const std::exception& ex) {
        m_lastError = std::string("TensorRT-YOLO initialization failed: ") + ex.what();
    }
    catch (...) {
        m_lastError = "TensorRT-YOLO initialization failed with an unknown exception.";
    }

    return false;
}

std::vector<DetectionObject> TensorRTYoloDetector::DetectBGR(
    const unsigned char* imageData,
    int width,
    int height,
    float nms,
    float conf) {
    auto& impl = *m_impl;
    impl.lastFrameStats = DetectorFrameStats{};
    if (imageData == nullptr || width <= 0 || height <= 0 || !impl.hostModel) {
        impl.lastFrameStats.failureStage = "invalid-input";
        return {};
    }
    if (nms > 0.0f && !impl.loggedRuntimeNmsNote) {
        impl.loggedRuntimeNmsNote = true;
        std::cout << "[TensorRT-YOLO] Runtime NMS threshold is engine-defined; snowball applies runtime confidence filtering only."
            << std::endl;
    }

    const auto totalStart = std::chrono::steady_clock::now();
    try {
        trtyolo::Image image(
            const_cast<unsigned char*>(imageData),
            width,
            height,
            3,
            Bgr24PitchBytes(width));
        const auto inferStart = std::chrono::steady_clock::now();
        const trtyolo::DetectRes result = impl.hostModel->predict(image);
        const auto inferEnd = std::chrono::steady_clock::now();

        impl.lastFrameStats.inferMs =
            std::chrono::duration<double, std::milli>(inferEnd - inferStart).count();
        std::vector<DetectionObject> detections = ConvertDetections(result, conf, impl.lastFrameStats);
        impl.lastFrameStats.totalMs =
            std::chrono::duration<double, std::milli>(inferEnd - totalStart).count();
        impl.lastFrameStats.valid = true;

        if (!impl.loggedReady) {
            impl.loggedReady = true;
            std::cout << "[TensorRT-YOLO] frameStats ready path=BGR totalMs="
                << impl.lastFrameStats.totalMs
                << " results=" << impl.lastFrameStats.resultsAfterNms
                << std::endl;
        }
        return detections;
    }
    catch (const std::exception& ex) {
        impl.lastFrameStats.failureStage = "predict";
        m_lastError = std::string("TensorRT-YOLO predict failed: ") + ex.what();
    }
    catch (...) {
        impl.lastFrameStats.failureStage = "predict";
        m_lastError = "TensorRT-YOLO predict failed with an unknown exception.";
    }

    return {};
}

std::vector<DetectionObject> TensorRTYoloDetector::DetectQcapFrameView(
    const QCAPFrameView& frameView,
    float nms,
    float conf) {
    auto& impl = *m_impl;
    impl.lastFrameStats = DetectorFrameStats{};
    if (frameView.width <= 0 || frameView.height <= 0) {
        impl.lastFrameStats.failureStage = "invalid-frameview";
        return {};
    }

    const bool useCudaMappedFrame =
        frameView.colorSpaceType == QCAP_COLORSPACE_TYPE_BGR24 &&
        frameView.sourceDevicePointer != nullptr &&
        frameView.sourceBufferCudaMapped &&
        TensorRTGpuPreprocess::SupportsFrameView(frameView);
    const bool useDirectHostBgrFrame =
        !useCudaMappedFrame &&
        frameView.colorSpaceType == QCAP_COLORSPACE_TYPE_BGR24 &&
        frameView.cpuData != nullptr;

    if (!useCudaMappedFrame && !useDirectHostBgrFrame) {
        impl.lastFrameStats.failureStage = "missing-cpu-frame";
        return {};
    }

    if (nms > 0.0f && !impl.loggedRuntimeNmsNote) {
        impl.loggedRuntimeNmsNote = true;
        std::cout << "[TensorRT-YOLO] Runtime NMS threshold is engine-defined; snowball applies runtime confidence filtering only."
            << std::endl;
    }

    try {
        trtyolo::DetectModel* model = nullptr;
        if (useCudaMappedFrame) {
            std::unique_ptr<trtyolo::DetectModel>& cudaModel =
                frameView.imageUpsideDown ? impl.flippedCudaModel : impl.cudaModel;
            if (!cudaModel) {
                trtyolo::InferOption cudaOption;
                cudaOption.setDeviceId(impl.deviceIndex);
                ConfigureInputColor(cudaOption, impl.inputColor);
                cudaOption.enableCudaMem();
                if (frameView.imageUpsideDown) {
                    cudaOption.enableFlipY();
                }
                cudaModel = std::make_unique<trtyolo::DetectModel>(impl.enginePath, cudaOption);
            }
            model = cudaModel.get();
            if (!impl.loggedCudaMappedInput) {
                impl.loggedCudaMappedInput = true;
                std::cout << "[TensorRT-YOLO] QCAP CUDA-mapped BGR24 input active tensorInput="
                    << InputColorToString(impl.inputColor)
                    << std::endl;
            }
        }
        else {
            if (frameView.imageUpsideDown) {
                if (!impl.flippedHostModel) {
                    trtyolo::InferOption hostOption;
                    hostOption.setDeviceId(impl.deviceIndex);
                    ConfigureInputColor(hostOption, impl.inputColor);
                    hostOption.enableFlipY();
                    impl.flippedHostModel = std::make_unique<trtyolo::DetectModel>(impl.enginePath, hostOption);
                }
                model = impl.flippedHostModel.get();
            }
            else {
                model = impl.hostModel.get();
            }
        }
        if (frameView.imageUpsideDown && !impl.loggedFlipYInput) {
            impl.loggedFlipYInput = true;
            std::cout << "[TensorRT-YOLO] QCAP vertical flip correction active." << std::endl;
        }

        const auto totalStart = std::chrono::steady_clock::now();
        void* imagePointer = nullptr;
        std::size_t imagePitchBytes = Bgr24PitchBytes(frameView.width);
        if (useCudaMappedFrame) {
            imagePointer = const_cast<void*>(frameView.sourceDevicePointer);
        }
        else if (useDirectHostBgrFrame) {
            imagePointer = const_cast<unsigned char*>(frameView.cpuData);
        }
        trtyolo::Image image(
            imagePointer,
            frameView.width,
            frameView.height,
            3,
            imagePitchBytes);
        const auto inferStart = std::chrono::steady_clock::now();
        const trtyolo::DetectRes result = model->predict(image);
        const auto inferEnd = std::chrono::steady_clock::now();

        impl.lastFrameStats.usedGpuPreprocess = useCudaMappedFrame;
        impl.lastFrameStats.inferMs =
            std::chrono::duration<double, std::milli>(inferEnd - inferStart).count();
        std::vector<DetectionObject> detections = ConvertDetections(result, conf, impl.lastFrameStats);
        impl.lastFrameStats.totalMs =
            std::chrono::duration<double, std::milli>(inferEnd - totalStart).count();
        impl.lastFrameStats.valid = true;
        return detections;
    }
    catch (const std::exception& ex) {
        impl.lastFrameStats.failureStage = "predict-frameview";
        m_lastError = std::string("TensorRT-YOLO frame-view predict failed: ") + ex.what();
    }
    catch (...) {
        impl.lastFrameStats.failureStage = "predict-frameview";
        m_lastError = "TensorRT-YOLO frame-view predict failed with an unknown exception.";
    }

    return {};
}

void TensorRTYoloDetector::ReleaseResources() {
    if (m_impl) {
        m_impl->cudaModel.reset();
        m_impl->flippedCudaModel.reset();
        m_impl->flippedHostModel.reset();
        m_impl->hostModel.reset();
        m_impl->lastFrameStats = DetectorFrameStats{};
        m_impl->loggedReady = false;
        m_impl->loggedCudaMappedInput = false;
        m_impl->loggedFlipYInput = false;
        m_impl->loggedRuntimeNmsNote = false;
    }
}

DetectorFrameStats TensorRTYoloDetector::GetLastFrameStats() const {
    return m_impl ? m_impl->lastFrameStats : DetectorFrameStats{};
}
