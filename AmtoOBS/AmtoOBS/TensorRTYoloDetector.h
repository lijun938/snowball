#pragma once

#include "DetectorBackend.h"

#include <memory>
#include <string>
#include <vector>

class TensorRTYoloDetector {
public:
    TensorRTYoloDetector();
    ~TensorRTYoloDetector();

    bool Initialize(const DetectorSettings& settings);
    std::vector<DetectionObject> DetectBGR(
        const unsigned char* imageData,
        int width,
        int height,
        float nms = 0.45f,
        float conf = 0.25f);
    std::vector<DetectionObject> DetectQcapFrameView(
        const QCAPFrameView& frameView,
        float nms = 0.45f,
        float conf = 0.25f);
    void ReleaseResources();

    const std::string& GetLastError() const { return m_lastError; }
    DetectorFrameStats GetLastFrameStats() const;

private:
    struct Impl;

    std::unique_ptr<Impl> m_impl;
    std::string m_lastError;
};
