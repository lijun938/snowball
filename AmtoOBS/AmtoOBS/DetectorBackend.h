#pragma once

#include "DetectionTypes.h"
#include "QCAPCapture.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

enum class InferenceBackendKind {
    TensorRTYolo,
    TensorRTLegacy,
    TensorRTPreferred,
};

enum class TensorRTYoloInputColor {
    BGR,
    RGB,
};

struct TensorRtRoiSettings {
    bool enabled = false;
    int x = 0;
    int y = 0;
    int width = 0;
    int height = 0;
};

struct DetectorSettings {
    InferenceBackendKind backend = InferenceBackendKind::TensorRTYolo;
    std::string modelPath;
    std::string enginePath = "apex-yolov8n-trtyolo.engine";
    int deviceIndex = 0;
    int numThreads = 0;
    bool allowTensorRtEngineBuild = false;
    int tensorRtCandidateTopK = 0;
    TensorRTYoloInputColor tensorRtYoloInputColor = TensorRTYoloInputColor::BGR;
    TensorRtRoiSettings tensorRtRoi;
};

struct DetectorInitReport {
    std::string requestedBackend;
    std::string activeBackend;
    std::string message;
    bool usedFallback = false;
    bool tensorRtRuntimeDetected = false;
    bool tensorRtEngineDetected = false;
    std::string tensorRtRuntimePath;
    std::string resolvedModelPath;
    std::string resolvedEnginePath;
    TensorRtRoiSettings tensorRtRoi;
};

struct DetectorFrameStats {
    bool valid = false;
    bool usedGpuPreprocess = false;
    bool usedOutputCompaction = false;
    bool usedRoi = false;
    double preprocessMs = 0.0;
    double h2dMs = 0.0;
    double inferMs = 0.0;
    double gpuPostprocessMs = 0.0;
    double d2hMs = 0.0;
    double decodeMs = 0.0;
    double totalMs = 0.0;
    std::uint64_t d2hBytes = 0;
    int outputRowsBeforeFilter = 0;
    int outputRowsAfterFilter = 0;
    int proposalsBeforeNms = 0;
    int resultsAfterNms = 0;
    int roiX = 0;
    int roiY = 0;
    int roiWidth = 0;
    int roiHeight = 0;
    std::string failureStage;
};

class IObjectDetector {
public:
    virtual ~IObjectDetector() = default;

    virtual bool Initialize(const DetectorSettings& settings) = 0;
    virtual std::vector<DetectionObject> DetectBGR(
        const unsigned char* imageData,
        int width,
        int height,
        float nms = 0.45f,
        float conf = 0.25f) = 0;
    virtual bool SupportsQcapFrameView() const { return false; }
    virtual bool CanConsumeQcapFrameView(const QCAPFrameView& frameView) const {
        (void)frameView;
        return SupportsQcapFrameView();
    }
    virtual std::vector<DetectionObject> DetectQcapFrameView(
        const QCAPFrameView& frameView,
        float nms = 0.45f,
        float conf = 0.25f) {
        (void)frameView;
        (void)nms;
        (void)conf;
        return {};
    }
    virtual void ReleaseResources() = 0;
    virtual const char* GetBackendName() const = 0;
    virtual const std::string& GetLastError() const = 0;
    virtual DetectorFrameStats GetLastFrameStats() const { return {}; }
};

bool InitializeDetectorWithFallback(
    const DetectorSettings& settings,
    std::unique_ptr<IObjectDetector>& detector,
    DetectorInitReport& report);
