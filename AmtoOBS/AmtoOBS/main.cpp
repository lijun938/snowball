#include <algorithm>
#include <chrono>
#include <cmath>
#include <cctype>
#include <cstdint>
#include <Windows.h>
#include <timeapi.h>
#include <cuda_runtime_api.h>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <memory>
#include <opencv2/opencv.hpp>
#include <sstream>
#include <string>
#include <thread>

#pragma comment(lib, "Winmm.lib")

#include "AppConfig.h"
#include "DetectorBackend.h"
#include "LiveControlPanel.h"
#include "MovementController.h"

namespace {

constexpr float kDefaultDetectionNmsThreshold = 0.45f;
constexpr float kDefaultDetectionConfidenceThreshold = 0.40f;
constexpr const char* kAppDisplayName = "Snowball v1.2.0";

struct RollingMetric {
    double sum = 0.0;
    double min = 0.0;
    double max = 0.0;
    std::uint64_t count = 0;

    void Add(double value) {
        if (count == 0) {
            min = value;
            max = value;
        }
        else {
            min = (std::min)(min, value);
            max = (std::max)(max, value);
        }
        sum += value;
        ++count;
    }

    double Average() const {
        return count == 0 ? 0.0 : sum / static_cast<double>(count);
    }

    void Reset() {
        sum = 0.0;
        min = 0.0;
        max = 0.0;
        count = 0;
    }
};

struct RunWindowStats {
    RollingMetric captureWaitMs;
    RollingMetric detectMs;
    RollingMetric moveSubmitMs;
    RollingMetric detectorPreprocessMs;
    RollingMetric detectorInferMs;
    RollingMetric detectorGpuPostprocessMs;
    RollingMetric detectorD2HMs;
    RollingMetric detectorDecodeMs;
    RollingMetric detectorTotalMs;
    std::uint64_t detectorD2HBytes = 0;
    std::uint64_t detectorCalls = 0;
    std::uint64_t detectorFrames = 0;
    std::uint64_t detectorGpuPreprocessFrames = 0;
    std::uint64_t detectorGpuCompactionFrames = 0;
    std::uint64_t proposalsBeforeNms = 0;
    std::uint64_t resultsAfterNms = 0;
    std::uint64_t outputRowsBeforeFilter = 0;
    std::uint64_t outputRowsAfterFilter = 0;
    std::uint64_t detectorRoiFrames = 0;
    int roiSampleX = 0;
    int roiSampleY = 0;
    int roiSampleWidth = 0;
    int roiSampleHeight = 0;
    bool roiSampleValid = false;

    void Reset() {
        captureWaitMs.Reset();
        detectMs.Reset();
        moveSubmitMs.Reset();
        detectorPreprocessMs.Reset();
        detectorInferMs.Reset();
        detectorGpuPostprocessMs.Reset();
        detectorD2HMs.Reset();
        detectorDecodeMs.Reset();
        detectorTotalMs.Reset();
        detectorD2HBytes = 0;
        detectorCalls = 0;
        detectorFrames = 0;
        detectorGpuPreprocessFrames = 0;
        detectorGpuCompactionFrames = 0;
        proposalsBeforeNms = 0;
        resultsAfterNms = 0;
        outputRowsBeforeFilter = 0;
        outputRowsAfterFilter = 0;
        detectorRoiFrames = 0;
        roiSampleX = 0;
        roiSampleY = 0;
        roiSampleWidth = 0;
        roiSampleHeight = 0;
        roiSampleValid = false;
    }
};

struct LiveControlState {
    MovementSettings movement;
    MovementSettings defaults;
    float detectionConfidenceThreshold = kDefaultDetectionConfidenceThreshold;
    float detectionNmsThreshold = kDefaultDetectionNmsThreshold;
    float defaultDetectionConfidenceThreshold = kDefaultDetectionConfidenceThreshold;
    float defaultDetectionNmsThreshold = kDefaultDetectionNmsThreshold;
    int labelFilter = -1;
    int lastTargetLabel = -1;
    std::vector<int> lastSeenLabels;
};

std::string FormatCudaError(const char* operation, cudaError_t result) {
    std::ostringstream oss;
    oss << operation;
    if (result == cudaSuccess) {
        oss << " succeeded";
    }
    else {
        oss << " failed (" << cudaGetErrorName(result) << ": " << cudaGetErrorString(result) << ")";
    }
    return oss.str();
}

const char* CudaMemoryTypeToString(cudaMemoryType type) {
    switch (type) {
    case cudaMemoryTypeUnregistered:
        return "unregistered";
    case cudaMemoryTypeHost:
        return "host";
    case cudaMemoryTypeDevice:
        return "device";
    case cudaMemoryTypeManaged:
        return "managed";
    default:
        return "unknown";
    }
}

std::string DescribeCudaPointer(const void* pointer) {
    if (pointer == nullptr) {
        return "null";
    }

    cudaPointerAttributes attributes{};
    const cudaError_t result = cudaPointerGetAttributes(&attributes, pointer);
    if (result != cudaSuccess) {
        const char* errorName = cudaGetErrorName(result);
        const char* errorText = cudaGetErrorString(result);
        cudaGetLastError();

        std::ostringstream oss;
        oss << "cudaPointerGetAttributes failed for " << pointer
            << " (" << errorName << ": " << errorText << ")";
        return oss.str();
    }

    std::ostringstream oss;
    oss << "ptr=" << pointer
        << " type=" << CudaMemoryTypeToString(attributes.type)
        << " device=" << attributes.device
        << " devicePtr=" << attributes.devicePointer
        << " hostPtr=" << attributes.hostPointer;
    return oss.str();
}

bool HasLiveControlLabel(const std::vector<int>& labels, int label) {
    return std::find(labels.begin(), labels.end(), label) != labels.end();
}

void AddLiveControlLabel(std::vector<int>& labels, int label) {
    if (label < 0 || HasLiveControlLabel(labels, label)) {
        return;
    }

    labels.push_back(label);
    std::sort(labels.begin(), labels.end());
}

void AccumulateDetectorStats(const DetectorFrameStats& detectorStats, RunWindowStats& runWindowStats) {
    if (!detectorStats.valid) {
        return;
    }

    runWindowStats.detectorPreprocessMs.Add(detectorStats.preprocessMs);
    runWindowStats.detectorInferMs.Add(detectorStats.inferMs);
    runWindowStats.detectorGpuPostprocessMs.Add(detectorStats.gpuPostprocessMs);
    runWindowStats.detectorD2HMs.Add(detectorStats.d2hMs);
    runWindowStats.detectorDecodeMs.Add(detectorStats.decodeMs);
    runWindowStats.detectorTotalMs.Add(detectorStats.totalMs);
    runWindowStats.detectorD2HBytes += detectorStats.d2hBytes;
    runWindowStats.proposalsBeforeNms += static_cast<std::uint64_t>(detectorStats.proposalsBeforeNms);
    runWindowStats.resultsAfterNms += static_cast<std::uint64_t>(detectorStats.resultsAfterNms);
    runWindowStats.outputRowsBeforeFilter += static_cast<std::uint64_t>(detectorStats.outputRowsBeforeFilter);
    runWindowStats.outputRowsAfterFilter += static_cast<std::uint64_t>(detectorStats.outputRowsAfterFilter);
    ++runWindowStats.detectorFrames;
    if (detectorStats.usedGpuPreprocess) {
        ++runWindowStats.detectorGpuPreprocessFrames;
    }
    if (detectorStats.usedOutputCompaction) {
        ++runWindowStats.detectorGpuCompactionFrames;
    }
    if (detectorStats.usedRoi) {
        ++runWindowStats.detectorRoiFrames;
        if (!runWindowStats.roiSampleValid) {
            runWindowStats.roiSampleX = detectorStats.roiX;
            runWindowStats.roiSampleY = detectorStats.roiY;
            runWindowStats.roiSampleWidth = detectorStats.roiWidth;
            runWindowStats.roiSampleHeight = detectorStats.roiHeight;
            runWindowStats.roiSampleValid = true;
        }
    }
}

void PrintFirstDetectorStatsDebug(const DetectorFrameStats& detectorStats, bool& printedDetectorStatsDebug) {
    if (printedDetectorStatsDebug) {
        return;
    }

    printedDetectorStatsDebug = true;
    std::cout << "[Run] detectorStats first valid=" << (detectorStats.valid ? "yes" : "no")
        << " totalMs=" << detectorStats.totalMs
        << " gpuPre=" << (detectorStats.usedGpuPreprocess ? "yes" : "no")
        << " gpuCompact=" << (detectorStats.usedOutputCompaction ? "yes" : "no")
        << " roi=" << (detectorStats.usedRoi
            ? (std::to_string(detectorStats.roiX) + "," +
               std::to_string(detectorStats.roiY) + "," +
               std::to_string(detectorStats.roiWidth) + "," +
               std::to_string(detectorStats.roiHeight))
            : std::string("full-frame"))
        << " fail=" << detectorStats.failureStage
        << std::endl;
}

std::string FormatLiveLabelSet(const std::vector<int>& labels) {
    if (labels.empty()) {
        return "none";
    }

    std::ostringstream oss;
    for (std::size_t i = 0; i < labels.size(); ++i) {
        if (i != 0) {
            oss << ",";
        }
        oss << labels[i];
    }
    return oss.str();
}

void ClampLiveControlState(LiveControlState& controlState) {
    controlState.detectionConfidenceThreshold = std::clamp(controlState.detectionConfidenceThreshold, 0.01f, 0.95f);
    controlState.detectionNmsThreshold = std::clamp(controlState.detectionNmsThreshold, 0.05f, 0.95f);
    controlState.movement.deadzonePixels = std::clamp(controlState.movement.deadzonePixels, 0.0, 120.0);
    controlState.movement.verticalBias = std::clamp(controlState.movement.verticalBias, -0.45, 0.45);
    controlState.movement.centerOffsetX = std::clamp(controlState.movement.centerOffsetX, -320.0, 320.0);
    controlState.movement.centerOffsetY = std::clamp(controlState.movement.centerOffsetY, -240.0, 240.0);
    controlState.movement.titanTwoStickCurve = std::clamp(controlState.movement.titanTwoStickCurve, 0.25, 4.0);
    controlState.movement.titanTwoStickResponseBoost = std::clamp(controlState.movement.titanTwoStickResponseBoost, 0.5, 4.0);
    controlState.movement.titanTwoStickMinPercent = std::clamp(controlState.movement.titanTwoStickMinPercent, 0.0, 100.0);
    controlState.movement.titanTwoStickMaxPercent = std::clamp(controlState.movement.titanTwoStickMaxPercent, 1.0, 100.0);
    controlState.movement.aimTrackConfirmFrames = std::clamp(controlState.movement.aimTrackConfirmFrames, 1, 12);
    controlState.movement.aimTrackLostFrames = std::clamp(controlState.movement.aimTrackLostFrames, 0, 60);
    controlState.movement.aimTrackMatchMaxCost = std::clamp(controlState.movement.aimTrackMatchMaxCost, 0.05, 8.0);
    controlState.movement.aimTargetLockBonus = std::clamp(controlState.movement.aimTargetLockBonus, 0.0, 8.0);
    controlState.movement.aimTargetSwitchMargin = std::clamp(controlState.movement.aimTargetSwitchMargin, 0.0, 8.0);
    controlState.movement.aimOneEuroMinCutoff = std::clamp(controlState.movement.aimOneEuroMinCutoff, 0.05, 20.0);
    controlState.movement.aimOneEuroBeta = std::clamp(controlState.movement.aimOneEuroBeta, 0.0, 2.0);
    controlState.movement.aimOneEuroDerivativeCutoff = std::clamp(controlState.movement.aimOneEuroDerivativeCutoff, 0.05, 20.0);
    controlState.movement.aimPredictionMs = std::clamp(controlState.movement.aimPredictionMs, 0.0, 250.0);
    controlState.movement.aimPredictionMaxBoxFraction = std::clamp(controlState.movement.aimPredictionMaxBoxFraction, 0.0, 1.0);
    controlState.movement.titanTwoStickDerivativeGain = std::clamp(controlState.movement.titanTwoStickDerivativeGain, -2.0, 2.0);
    controlState.movement.titanTwoStickFeedForward = std::clamp(controlState.movement.titanTwoStickFeedForward, -2.0, 2.0);
    controlState.movement.titanTwoStickSlewPercentPerSecond =
        std::clamp(controlState.movement.titanTwoStickSlewPercentPerSecond, 0.0, 5000.0);
    controlState.movement.fovRadius = (std::max)(0.0, controlState.movement.fovRadius);
    if (controlState.movement.titanTwoStickMinPercent > controlState.movement.titanTwoStickMaxPercent) {
        controlState.movement.titanTwoStickMinPercent = controlState.movement.titanTwoStickMaxPercent;
    }
}

std::vector<DetectionObject> FilterDetectionsForLiveControls(
    const std::vector<DetectionObject>& detections,
    const LiveControlState& controlState) {
    if (controlState.labelFilter < 0) {
        return detections;
    }

    std::vector<DetectionObject> filteredDetections;
    filteredDetections.reserve(detections.size());
    for (const auto& detection : detections) {
        if (detection.label == controlState.labelFilter) {
            filteredDetections.push_back(detection);
        }
    }
    return filteredDetections;
}


int ComputeFrameViewTimeoutMs(const QCAPCapture& qcapCapture, const QCAPCaptureConfig& captureConfig) {
    const QCAPFrameInfo frameInfo = qcapCapture.GetFrameInfo();
    const double effectiveFrameRate =
        frameInfo.frameRate > 0.0
        ? frameInfo.frameRate
        : (captureConfig.frameRate > 0.0 ? captureConfig.frameRate : 240.0);
    if (effectiveFrameRate <= 0.0) {
        return captureConfig.requestGpuDirect ? 8 : 30;
    }

    const double expectedFrameMs = 1000.0 / effectiveFrameRate;
    const int timeoutMs = static_cast<int>(std::ceil(expectedFrameMs * (captureConfig.requestGpuDirect ? 2.0 : 4.0)));
    if (captureConfig.requestGpuDirect) {
        return std::clamp(timeoutMs, 2, 20);
    }
    return std::clamp(timeoutMs, 10, 30);
}

std::filesystem::path BuildInferenceScreenshotPath(const AppConfig& appConfig, int frameNumber) {
    const std::filesystem::path rawPath(appConfig.inferenceScreenshotPath);
    if (rawPath.empty()) {
        return {};
    }

    const bool treatAsDirectory =
        !rawPath.has_extension() ||
        (std::filesystem::exists(rawPath) && std::filesystem::is_directory(rawPath));
    if (!treatAsDirectory) {
        return rawPath;
    }

    std::error_code ec;
    std::filesystem::create_directories(rawPath, ec);

    std::ostringstream name;
    name << "inference_" << frameNumber << ".png";
    return rawPath / name.str();
}

std::filesystem::path ResolveSingleOrDirectoryOutputPath(
    const std::filesystem::path& rawPath,
    const std::string& defaultFileName) {
    if (rawPath.empty()) {
        return {};
    }

    const bool treatAsDirectory =
        !rawPath.has_extension() ||
        (std::filesystem::exists(rawPath) && std::filesystem::is_directory(rawPath));
    if (!treatAsDirectory) {
        return rawPath;
    }

    std::error_code ec;
    std::filesystem::create_directories(rawPath, ec);
    return rawPath / defaultFileName;
}

std::filesystem::path BuildInferenceScreenshotMetadataPath(const std::filesystem::path& screenshotPath) {
    if (screenshotPath.empty()) {
        return {};
    }

    std::filesystem::path metadataPath = screenshotPath;
    metadataPath += ".json";
    return metadataPath;
}

std::string PathToUtf8String(const std::filesystem::path& path) {
    return path.u8string();
}

bool ShouldAttemptInferenceScreenshot(
    const AppConfig& appConfig,
    int frameNumber,
    const std::vector<DetectionObject>& detections) {
    if (appConfig.inferenceScreenshotPath.empty()) {
        return false;
    }
    if (appConfig.inferenceScreenshotDetectionsOnly && detections.empty()) {
        return false;
    }
    if (appConfig.inferenceScreenshotEveryNFrames > 0 &&
        (frameNumber % appConfig.inferenceScreenshotEveryNFrames) != 0) {
        return false;
    }
    return true;
}

enum class InferenceScreenshotWriteResult {
    Skipped,
    Succeeded,
    Failed,
};

bool WriteInferenceScreenshotMetadata(
    const std::filesystem::path& screenshotPath,
    int frameNumber,
    const std::vector<DetectionObject>& detections,
    const DetectorFrameStats& detectorStats,
    std::filesystem::path* resolvedMetadataPath = nullptr) {
    const std::filesystem::path metadataPath = BuildInferenceScreenshotMetadataPath(screenshotPath);
    if (metadataPath.empty()) {
        return false;
    }
    if (resolvedMetadataPath != nullptr) {
        *resolvedMetadataPath = metadataPath;
    }

    std::error_code ec;
    if (!metadataPath.parent_path().empty()) {
        std::filesystem::create_directories(metadataPath.parent_path(), ec);
    }

    std::ofstream outputFile(metadataPath, std::ios::binary | std::ios::trunc);
    if (!outputFile) {
        return false;
    }

    outputFile << "{\n";
    outputFile << "  \"frame\": " << frameNumber << ",\n";
    outputFile << "  \"detections\": " << detections.size() << ",\n";
    outputFile << "  \"roiUsed\": " << (detectorStats.usedRoi ? "true" : "false") << ",\n";
    outputFile << "  \"gpuPreprocess\": " << (detectorStats.usedGpuPreprocess ? "true" : "false") << ",\n";
    outputFile << "  \"gpuCompaction\": " << (detectorStats.usedOutputCompaction ? "true" : "false") << ",\n";
    outputFile << "  \"resultsAfterNms\": " << detectorStats.resultsAfterNms << ",\n";
    outputFile << "  \"proposalsBeforeNms\": " << detectorStats.proposalsBeforeNms << ",\n";
    outputFile << "  \"totalMs\": " << detectorStats.totalMs;
    if (detectorStats.usedRoi) {
        outputFile << ",\n";
        outputFile << "  \"roi\": {\n";
        outputFile << "    \"x\": " << detectorStats.roiX << ",\n";
        outputFile << "    \"y\": " << detectorStats.roiY << ",\n";
        outputFile << "    \"width\": " << detectorStats.roiWidth << ",\n";
        outputFile << "    \"height\": " << detectorStats.roiHeight << "\n";
        outputFile << "  }\n";
    }
    else {
        outputFile << "\n";
    }
    outputFile << "}\n";
    outputFile.flush();
    return outputFile.good();
}

InferenceScreenshotWriteResult MaybeWriteInferenceScreenshot(
    const AppConfig& appConfig,
    int frameNumber,
    const std::vector<DetectionObject>& detections,
    const cv::Mat& annotatedFrame,
    std::filesystem::path* resolvedOutputPath = nullptr) {
    if (appConfig.inferenceScreenshotPath.empty() || annotatedFrame.empty()) {
        return InferenceScreenshotWriteResult::Skipped;
    }
    if (appConfig.inferenceScreenshotDetectionsOnly && detections.empty()) {
        return InferenceScreenshotWriteResult::Skipped;
    }
    if (appConfig.inferenceScreenshotEveryNFrames > 0 &&
        (frameNumber % appConfig.inferenceScreenshotEveryNFrames) != 0) {
        return InferenceScreenshotWriteResult::Skipped;
    }

    const std::filesystem::path outputPath = BuildInferenceScreenshotPath(appConfig, frameNumber);
    if (outputPath.empty()) {
        return InferenceScreenshotWriteResult::Skipped;
    }
    if (resolvedOutputPath != nullptr) {
        *resolvedOutputPath = outputPath;
    }

    std::error_code ec;
    if (!outputPath.parent_path().empty()) {
        std::filesystem::create_directories(outputPath.parent_path(), ec);
    }

    std::string extension = outputPath.extension().string();
    if (extension.empty()) {
        extension = ".png";
    }
    std::transform(extension.begin(), extension.end(), extension.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    if (extension != ".png" &&
        extension != ".jpg" &&
        extension != ".jpeg" &&
        extension != ".bmp" &&
        extension != ".webp") {
        extension = ".png";
    }

    std::vector<unsigned char> encoded;
    if (!cv::imencode(extension, annotatedFrame, encoded)) {
        return InferenceScreenshotWriteResult::Failed;
    }

    std::ofstream outputFile(outputPath, std::ios::binary | std::ios::trunc);
    if (!outputFile) {
        return InferenceScreenshotWriteResult::Failed;
    }

    outputFile.write(reinterpret_cast<const char*>(encoded.data()), static_cast<std::streamsize>(encoded.size()));
    outputFile.flush();
    return outputFile.good()
        ? InferenceScreenshotWriteResult::Succeeded
        : InferenceScreenshotWriteResult::Failed;
}

bool WriteImageFile(const std::filesystem::path& outputPath, const cv::Mat& image) {
    if (outputPath.empty() || image.empty()) {
        return false;
    }

    std::error_code ec;
    if (!outputPath.parent_path().empty()) {
        std::filesystem::create_directories(outputPath.parent_path(), ec);
    }

    std::vector<unsigned char> encoded;
    std::string extension = outputPath.extension().string();
    std::transform(extension.begin(), extension.end(), extension.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    if (extension.empty()) {
        extension = ".png";
    }
    if (!cv::imencode(extension, image, encoded)) {
        return false;
    }

    std::ofstream outputFile(outputPath, std::ios::binary | std::ios::trunc);
    if (!outputFile) {
        return false;
    }

    outputFile.write(reinterpret_cast<const char*>(encoded.data()), static_cast<std::streamsize>(encoded.size()));
    outputFile.flush();
    return outputFile.good();
}

bool WriteQcapDebugDump(
    const AppConfig& appConfig,
    const cv::Mat& bgrFrame,
    int frameNumber,
    std::filesystem::path* resolvedBgrPath = nullptr,
    std::filesystem::path* resolvedRgbPath = nullptr) {
    if (appConfig.debugDumpQcapFramePath.empty() || bgrFrame.empty()) {
        return false;
    }

    const std::filesystem::path basePath = ResolveSingleOrDirectoryOutputPath(
        std::filesystem::path(appConfig.debugDumpQcapFramePath),
        "qcap_debug_bgr.png");
    if (basePath.empty()) {
        return false;
    }

    std::filesystem::path bgrPath = basePath;
    if (basePath.has_extension()) {
        const std::filesystem::path parent = basePath.parent_path();
        const std::string stem = basePath.stem().string();
        const std::string ext = basePath.extension().string();
        std::ostringstream name;
        name << stem << "_" << frameNumber << "_bgr" << ext;
        bgrPath = parent / name.str();
    }

    std::filesystem::path rgbPath = bgrPath;
    if (rgbPath.has_extension()) {
        rgbPath.replace_filename(rgbPath.stem().string() + "_rgb_as_bgr" + rgbPath.extension().string());
    }
    else {
        rgbPath += "_rgb_as_bgr.png";
    }

    cv::Mat rgbAsBgr;
    cv::cvtColor(bgrFrame, rgbAsBgr, cv::COLOR_BGR2RGB);

    const bool wroteBgr = WriteImageFile(bgrPath, bgrFrame);
    const bool wroteRgb = WriteImageFile(rgbPath, rgbAsBgr);
    if (resolvedBgrPath != nullptr) {
        *resolvedBgrPath = bgrPath;
    }
    if (resolvedRgbPath != nullptr) {
        *resolvedRgbPath = rgbPath;
    }
    return wroteBgr && wroteRgb;
}

void PrintCudaDeviceInteropReport(int deviceIndex) {
    const cudaError_t flagsResult = cudaSetDeviceFlags(cudaDeviceMapHost);
    std::cout << "[CUDA] " << FormatCudaError("cudaSetDeviceFlags(cudaDeviceMapHost)", flagsResult) << std::endl;
    if (flagsResult != cudaSuccess) {
        cudaGetLastError();
    }

    int deviceCount = 0;
    const cudaError_t countResult = cudaGetDeviceCount(&deviceCount);
    if (countResult != cudaSuccess) {
        std::cout << "[CUDA] " << FormatCudaError("cudaGetDeviceCount", countResult) << std::endl;
        cudaGetLastError();
        return;
    }

    if (deviceIndex < 0 || deviceIndex >= deviceCount) {
        std::cout << "[CUDA] Requested detector device index " << deviceIndex
            << " is outside the detected CUDA device range [0, " << (deviceCount - 1) << "]."
            << std::endl;
        return;
    }

    cudaDeviceProp properties{};
    const cudaError_t propertiesResult = cudaGetDeviceProperties(&properties, deviceIndex);
    if (propertiesResult != cudaSuccess) {
        std::cout << "[CUDA] " << FormatCudaError("cudaGetDeviceProperties", propertiesResult) << std::endl;
        cudaGetLastError();
        return;
    }

    std::cout << "[CUDA] device=" << deviceIndex
        << " name=" << properties.name
        << " canMapHostMemory=" << (properties.canMapHostMemory ? "yes" : "no")
        << " unifiedAddressing=" << (properties.unifiedAddressing ? "yes" : "no")
        << " integrated=" << (properties.integrated ? "yes" : "no")
        << std::endl;
}

void PrintCudaHostMappingProbe(const char* label, const void* pointer, std::size_t size, bool allowRegister = true) {
    std::cout << "[CUDA] " << label << " attr(before): " << DescribeCudaPointer(pointer) << std::endl;
    if (pointer == nullptr || size == 0) {
        std::cout << "[CUDA] " << label << " mapping skipped because the pointer or size is empty." << std::endl;
        return;
    }
    if (!allowRegister) {
        return;
    }

    void* mutablePointer = const_cast<void*>(pointer);
    const unsigned int registerFlags = cudaHostRegisterMapped | cudaHostRegisterPortable;
    const cudaError_t registerResult = cudaHostRegister(mutablePointer, size, registerFlags);
    const bool alreadyRegistered = registerResult == cudaErrorHostMemoryAlreadyRegistered;
    if (registerResult != cudaSuccess && !alreadyRegistered) {
        std::cout << "[CUDA] " << label << " "
            << FormatCudaError("cudaHostRegister", registerResult)
            << " size=" << size << std::endl;
        cudaGetLastError();
        return;
    }

    if (alreadyRegistered) {
        cudaGetLastError();
        std::cout << "[CUDA] " << label << " cudaHostRegister reports already-registered host memory."
            << std::endl;
    }
    else {
        std::cout << "[CUDA] " << label << " cudaHostRegister succeeded for " << size << " bytes."
            << std::endl;
    }

    void* mappedPointer = nullptr;
    const cudaError_t mappedResult = cudaHostGetDevicePointer(&mappedPointer, mutablePointer, 0);
    if (mappedResult != cudaSuccess) {
        std::cout << "[CUDA] " << label << " "
            << FormatCudaError("cudaHostGetDevicePointer", mappedResult)
            << std::endl;
        cudaGetLastError();
    }
    else {
        std::cout << "[CUDA] " << label << " mappedDevicePointer=" << mappedPointer << std::endl;
        std::cout << "[CUDA] " << label << " attr(after): " << DescribeCudaPointer(pointer) << std::endl;
    }

    if (!alreadyRegistered) {
        const cudaError_t unregisterResult = cudaHostUnregister(mutablePointer);
        std::cout << "[CUDA] " << label << " "
            << FormatCudaError("cudaHostUnregister", unregisterResult)
            << std::endl;
        if (unregisterResult != cudaSuccess) {
            cudaGetLastError();
        }
    }
}

bool CanUseQcapFrameViewForMainlineDetection(
    const QCAPFrameView& frameView,
    bool requireGpuDirectDevicePointer) {
    const bool hasFastGpuSource = frameView.sourceDevicePointer != nullptr;
    const bool hasAllowedCpuSource = !requireGpuDirectDevicePointer && frameView.cpuData != nullptr;
    return frameView.colorSpaceType == QCAP_COLORSPACE_TYPE_BGR24 &&
        frameView.width > 0 &&
        frameView.height > 0 &&
        (hasFastGpuSource || hasAllowedCpuSource);
}

bool InitializeQcapCaptureWithSeh(QCAPCapture& qcapCapture, const QCAPCaptureConfig& config, unsigned long& exceptionCode) {
    exceptionCode = 0;
    __try {
        return qcapCapture.Initialize(config);
    }
    __except (EXCEPTION_EXECUTE_HANDLER) {
        exceptionCode = GetExceptionCode();
        return false;
    }
}

void DrawDetections(cv::Mat& image, const std::vector<DetectionObject>& detections) {
    for (const auto& obj : detections) {
        cv::Rect rect(
            static_cast<int>(obj.bbox.x),
            static_cast<int>(obj.bbox.y),
            static_cast<int>(obj.bbox.width),
            static_cast<int>(obj.bbox.height));

        cv::rectangle(image, rect, cv::Scalar(0, 255, 0), 2);

        const std::string label = "label " + std::to_string(obj.label) +
            ": " + std::to_string(static_cast<int>(obj.prob * 100.0f)) + "%";
        cv::putText(
            image,
            label,
            cv::Point(rect.x, rect.y - 5),
            cv::FONT_HERSHEY_SIMPLEX,
            0.5,
            cv::Scalar(0, 255, 0),
            1);
    }
}

std::vector<DetectionObject> ScaleDetectionsForDisplay(
    const std::vector<DetectionObject>& detections,
    double scaleX,
    double scaleY) {
    if (scaleX == 1.0 && scaleY == 1.0) {
        return detections;
    }

    std::vector<DetectionObject> scaled = detections;
    for (auto& obj : scaled) {
        obj.bbox.x = static_cast<float>(obj.bbox.x * scaleX);
        obj.bbox.y = static_cast<float>(obj.bbox.y * scaleY);
        obj.bbox.width = static_cast<float>(obj.bbox.width * scaleX);
        obj.bbox.height = static_cast<float>(obj.bbox.height * scaleY);
    }
    return scaled;
}

MovementDebugState ScaleMovementDebugStateForDisplay(
    MovementDebugState state,
    double scaleX,
    double scaleY) {
    if (scaleX == 1.0 && scaleY == 1.0) {
        return state;
    }

    state.originX = static_cast<int>(std::lround(static_cast<double>(state.originX) * scaleX));
    state.originY = static_cast<int>(std::lround(static_cast<double>(state.originY) * scaleY));
    state.targetX = static_cast<int>(std::lround(static_cast<double>(state.targetX) * scaleX));
    state.targetY = static_cast<int>(std::lround(static_cast<double>(state.targetY) * scaleY));
    return state;
}

void DrawMovementOverlay(cv::Mat& image, const MovementDebugState& movementState, double fovRadiusScaled = 0.0) {
    if (image.empty()) {
        return;
    }

    const cv::Point frameCenter(image.cols / 2, image.rows / 2);
    const cv::Point aimOrigin(
        movementState.originX != 0 ? movementState.originX : frameCenter.x,
        movementState.originY != 0 ? movementState.originY : frameCenter.y);

    if (fovRadiusScaled > 0.0) {
        cv::circle(image, aimOrigin, static_cast<int>(std::lround(fovRadiusScaled)),
            cv::Scalar(0, 255, 255), 1, cv::LINE_AA);
    }

    const cv::Scalar centerColor(0, 255, 100);
    const cv::Scalar originColor(0, 200, 60);
    cv::drawMarker(
        image,
        frameCenter,
        centerColor,
        cv::MARKER_CROSS,
        24,
        2);
    cv::drawMarker(
        image,
        aimOrigin,
        originColor,
        cv::MARKER_DIAMOND,
        22,
        2);

    if (aimOrigin != frameCenter) {
        cv::line(image, frameCenter, aimOrigin, cv::Scalar(0, 180, 60), 1, cv::LINE_AA);
    }

    if (!movementState.hasTarget) {
        return;
    }

    const cv::Point targetPoint(movementState.targetX, movementState.targetY);
    const cv::Scalar targetColor = movementState.active
        ? cv::Scalar(0, 255, 0)
        : cv::Scalar(0, 200, 80);
    cv::drawMarker(
        image,
        targetPoint,
        targetColor,
        cv::MARKER_TILTED_CROSS,
        28,
        2);
    cv::circle(image, targetPoint, 10, targetColor, 2);
    cv::line(image, aimOrigin, targetPoint, targetColor, 1, cv::LINE_AA);

    const std::string label = "aim " +
        std::to_string(movementState.targetX) + "," +
        std::to_string(movementState.targetY) +
        " dx=" + std::to_string(movementState.deltaX) +
        " dy=" + std::to_string(movementState.deltaY);
    cv::putText(
        image,
        label,
        cv::Point(
            (std::min)(image.cols - 320, (std::max)(10, movementState.targetX + 12)),
            (std::max)(20, movementState.targetY - 12)),
        cv::FONT_HERSHEY_SIMPLEX,
        0.55,
        targetColor,
        2);

    std::ostringstream stickStream;
    stickStream << std::fixed << std::setprecision(1)
        << "stick " << movementState.stickXPercent
        << "," << movementState.stickYPercent
        << " score=" << movementState.score;
    cv::putText(
        image,
        stickStream.str(),
        cv::Point(
            (std::min)(image.cols - 360, (std::max)(10, movementState.targetX + 12)),
            (std::max)(44, movementState.targetY + 20)),
        cv::FONT_HERSHEY_SIMPLEX,
        0.55,
        targetColor,
        2);

    std::ostringstream originStream;
    originStream << "origin " << aimOrigin.x << "," << aimOrigin.y
        << " raw " << std::fixed << std::setprecision(1)
        << movementState.rawDeltaX << "," << movementState.rawDeltaY;
    cv::putText(
        image,
        originStream.str(),
        cv::Point(
            (std::min)(image.cols - 380, (std::max)(10, aimOrigin.x + 14)),
            (std::max)(36, aimOrigin.y - 16)),
        cv::FONT_HERSHEY_SIMPLEX,
        0.48,
        originColor,
        2,
        cv::LINE_AA);
}

void DrawDebugHud(
    cv::Mat& image,
    double currentFps,
    const std::vector<DetectionObject>& detections,
    const DetectorInitReport& detectorReport,
    const MovementController& movementController) {
    if (image.empty()) {
        return;
    }

    const MovementSettings& movementSettings = movementController.GetSettings();
    const MovementDebugState& movementState = movementController.GetDebugState();

    const int hudWidth = (std::min)(460, image.cols - 24);
    const int hudHeight = 248;
    const cv::Rect hudRect(12, 12, hudWidth, (std::min)(hudHeight, image.rows - 24));
    const cv::Rect titleRect(hudRect.x, hudRect.y, hudRect.width, 44);

    cv::Mat hudRoi = image(hudRect);
    cv::Mat hudOverlay = hudRoi.clone();
    cv::rectangle(hudOverlay, cv::Rect(0, 0, hudRect.width, hudRect.height), cv::Scalar(8, 16, 8), cv::FILLED);
    cv::rectangle(hudOverlay, titleRect - hudRect.tl(), cv::Scalar(10, 50, 20), cv::FILLED);
    cv::addWeighted(hudOverlay, 0.82, hudRoi, 0.18, 0.0, hudRoi);

    cv::rectangle(image, hudRect, cv::Scalar(80, 230, 0), 2, cv::LINE_AA);
    cv::line(
        image,
        cv::Point(hudRect.x + 12, titleRect.y + titleRect.height),
        cv::Point(hudRect.x + hudRect.width - 12, titleRect.y + titleRect.height),
        cv::Scalar(80, 230, 0),
        1,
        cv::LINE_AA);

    cv::putText(
        image,
        "AI AIM HUD",
        cv::Point(hudRect.x + 16, hudRect.y + 29),
        cv::FONT_HERSHEY_SIMPLEX,
        0.78,
        cv::Scalar(100, 255, 0),
        2,
        cv::LINE_AA);

    auto drawStatusPill = [&](int x, int y, const std::string& label, bool enabled, const cv::Scalar& enabledColor) {
        const cv::Size textSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.48, 1, nullptr);
        const cv::Rect pillRect(x, y, textSize.width + 24, 24);
        cv::rectangle(
            image,
            pillRect,
            enabled ? enabledColor : cv::Scalar(70, 76, 86),
            cv::FILLED,
            cv::LINE_AA);
        cv::rectangle(image, pillRect, cv::Scalar(190, 210, 220), 1, cv::LINE_AA);
        cv::putText(
            image,
            label,
            cv::Point(pillRect.x + 12, pillRect.y + 16),
            cv::FONT_HERSHEY_SIMPLEX,
            0.48,
            cv::Scalar(245, 248, 250),
            1,
            cv::LINE_AA);
    };

    drawStatusPill(hudRect.x + 16, hudRect.y + 54, movementState.transportReady ? "TT2 LINK" : "TT2 WAIT", movementState.transportReady, cv::Scalar(20, 140, 0));
    drawStatusPill(hudRect.x + 126, hudRect.y + 54, movementState.hasTarget ? "TARGET" : "NO TARGET", movementState.hasTarget, cv::Scalar(0, 180, 60));
    drawStatusPill(hudRect.x + 254, hudRect.y + 54, movementState.active ? "PACKET HOT" : "PACKET IDLE", movementState.active, cv::Scalar(0, 200, 80));

    auto putHudText = [&](int x, int y, const std::string& text, double scale, const cv::Scalar& color, int thickness = 1) {
        cv::putText(
            image,
            text,
            cv::Point(x, y),
            cv::FONT_HERSHEY_SIMPLEX,
            scale,
            color,
            thickness,
            cv::LINE_AA);
    };

    std::ostringstream fpsStream;
    fpsStream << std::fixed << std::setprecision(1) << currentFps;
    putHudText(hudRect.x + 16, hudRect.y + 102, "fps " + fpsStream.str(), 0.72, cv::Scalar(100, 255, 0), 2);
    putHudText(hudRect.x + 150, hudRect.y + 102, "obj " + std::to_string(detections.size()), 0.72, cv::Scalar(80, 230, 0), 2);
    putHudText(hudRect.x + 270, hudRect.y + 102, detectorReport.activeBackend, 0.62, cv::Scalar(60, 200, 80), 2);

    putHudText(hudRect.x + 16, hudRect.y + 136, "target  " + std::to_string(movementState.targetX) + "," + std::to_string(movementState.targetY), 0.54, cv::Scalar(180, 220, 180));
    putHudText(hudRect.x + 16, hudRect.y + 162, "delta   " + std::to_string(movementState.deltaX) + "," + std::to_string(movementState.deltaY), 0.54, cv::Scalar(180, 220, 180));

    std::ostringstream stickStream;
    stickStream << std::fixed << std::setprecision(1)
        << "stick   " << movementState.stickXPercent << "," << movementState.stickYPercent;
    putHudText(hudRect.x + 16, hudRect.y + 188, stickStream.str(), 0.54, cv::Scalar(180, 220, 180));

    std::ostringstream scoreStream;
    scoreStream << std::fixed << std::setprecision(2) << movementState.score;
    putHudText(hudRect.x + 16, hudRect.y + 214, "score   " + scoreStream.str(), 0.54, cv::Scalar(180, 220, 180));

    std::ostringstream transportStream;
    transportStream << "dispatch " << movementState.transportDispatchCount
        << " skip " << movementState.transportSkipCount
        << " submitMs " << std::fixed << std::setprecision(2) << movementState.transportLastSubmitMs;
    putHudText(hudRect.x + 16, hudRect.y + 240, transportStream.str(), 0.46, cv::Scalar(120, 160, 120));

    const std::string triggerModeText = movementSettings.backend == MovementBackendKind::TitanTwoGcvGamepad
        ? "trigger TT2 LT/L2 or RT/R2"
        : "trigger external gate";
    putHudText(hudRect.x + 236, hudRect.y + 136, triggerModeText, 0.48, cv::Scalar(100, 255, 0));
    putHudText(hudRect.x + 236, hudRect.y + 148, "physical trigger state not visible on PC yet", 0.38, cv::Scalar(80, 130, 80));

    std::ostringstream tuneStream;
    tuneStream << std::fixed << std::setprecision(2)
        << "bias " << movementSettings.verticalBias
        << " curve " << movementSettings.titanTwoStickCurve
        << " boost " << movementSettings.titanTwoStickResponseBoost;
    putHudText(hudRect.x + 236, hudRect.y + 172, tuneStream.str(), 0.46, cv::Scalar(120, 160, 120));

    std::ostringstream floorStream;
    floorStream << std::fixed << std::setprecision(1)
        << "stick floor " << movementSettings.titanTwoStickMinPercent
        << "% max " << movementSettings.titanTwoStickMaxPercent
        << "% pred " << movementSettings.aimPredictionMs << "ms";
    putHudText(hudRect.x + 236, hudRect.y + 198, floorStream.str(), 0.46, cv::Scalar(120, 160, 120));

    std::ostringstream offsetStream;
    offsetStream << std::fixed << std::setprecision(1)
        << "track " << movementState.targetTrackId
        << " lead " << movementState.leadX
        << "," << movementState.leadY;
    putHudText(hudRect.x + 236, hudRect.y + 212, offsetStream.str(), 0.46, cv::Scalar(120, 160, 120));

    putHudText(
        hudRect.x + 236,
        hudRect.y + 228,
        movementState.transportReady ? "link hot: packet path alive" : "link cold: waiting packet",
        0.46,
        movementState.transportReady ? cv::Scalar(100, 255, 0) : cv::Scalar(80, 120, 80));
}

void DrawTensorRtRoiOverlay(cv::Mat& image, const DetectorFrameStats& detectorStats) {
    if (image.empty() || !detectorStats.usedRoi || detectorStats.roiWidth <= 0 || detectorStats.roiHeight <= 0) {
        return;
    }

    const cv::Rect imageBounds(0, 0, image.cols, image.rows);
    const cv::Rect roiRect(
        detectorStats.roiX,
        detectorStats.roiY,
        detectorStats.roiWidth,
        detectorStats.roiHeight);
    const cv::Rect clippedRoi = roiRect & imageBounds;
    if (clippedRoi.width <= 0 || clippedRoi.height <= 0) {
        return;
    }

    const cv::Scalar roiColor(0, 230, 80);
    cv::rectangle(image, clippedRoi, roiColor, 2);

    const std::string label = "ROI " +
        std::to_string(detectorStats.roiX) + "," +
        std::to_string(detectorStats.roiY) + "," +
        std::to_string(detectorStats.roiWidth) + "," +
        std::to_string(detectorStats.roiHeight);
    const cv::Point labelOrigin(
        clippedRoi.x,
        clippedRoi.y > 12 ? clippedRoi.y - 6 : (std::min)(image.rows - 6, clippedRoi.y + clippedRoi.height + 16));
    cv::putText(
        image,
        label,
        labelOrigin,
        cv::FONT_HERSHEY_SIMPLEX,
        0.55,
        roiColor,
        2);
}

cv::Mat BuildPreviewPlaceholder(
    const AppConfig& appConfig,
    const DetectorInitReport& detectorReport,
    const QCAPCapture* qcapCapture,
    double currentFps) {
    const int width = (std::max)(640, appConfig.capture.width > 0 ? appConfig.capture.width : 1280);
    const int height = (std::max)(360, appConfig.capture.height > 0 ? appConfig.capture.height : 720);

    cv::Mat placeholder(height, width, CV_8UC3, cv::Scalar(8, 12, 8));
    cv::rectangle(
        placeholder,
        cv::Rect(12, 12, (std::max)(0, width - 24), (std::max)(0, height - 24)),
        cv::Scalar(0, 120, 40),
        2);

    std::vector<std::string> lines;
    lines.push_back(qcapCapture ? "Waiting for QCAP preview frame" : "Waiting for OpenCV capture frame");
    lines.push_back(
        "FPS: " + std::to_string(static_cast<int>(currentFps)) +
        "  Det: " + detectorReport.activeBackend);

    if (qcapCapture) {
        const QCAPRuntimeStats runtimeStats = qcapCapture->GetRuntimeStats();
        lines.push_back(
            "callbacks=" + std::to_string(runtimeStats.previewCallbacksObserved) +
            " waitTimeouts=" + std::to_string(runtimeStats.waitTimeouts) +
            " resync=" + std::to_string(runtimeStats.captureResyncAttempts) + "/" +
            std::to_string(runtimeStats.captureResyncSuccesses) + "/" +
            std::to_string(runtimeStats.captureResyncFailures));
        lines.push_back(
            "callbackAgeMs=" + std::to_string(static_cast<int>(runtimeStats.lastCallbackAgeMs)) +
            " activeFps=" + std::to_string(static_cast<int>(qcapCapture->GetFrameInfo().frameRate)));

        std::string lastError = qcapCapture->GetLastError();
        if (!lastError.empty()) {
            if (lastError.size() > 140) {
                lastError.resize(137);
                lastError += "...";
            }
            lines.push_back(lastError);
        }
    }
    int y = 72;
    for (std::size_t i = 0; i < lines.size(); ++i) {
        const double fontScale = i == 0 ? 1.0 : 0.8;
        const int thickness = i == 0 ? 2 : 1;
        cv::putText(
            placeholder,
            lines[i],
            cv::Point(36, y),
            cv::FONT_HERSHEY_SIMPLEX,
            fontScale,
            i == 0 ? cv::Scalar(100, 255, 0) : cv::Scalar(180, 220, 180),
            thickness);
        y += i == 0 ? 42 : 34;
    }

    return placeholder;
}

void PrintCaptureReport(const AppConfig& appConfig, const QCAPCapture& qcapCapture) {
    std::cout << "[Capture] device=" << appConfig.capture.deviceName
        << " index=" << appConfig.capture.deviceIndex
        << " input=" << QcapInputToString(appConfig.capture.videoInput)
        << " size=" << appConfig.capture.width << "x" << appConfig.capture.height
        << " fps(requested)=" << appConfig.capture.frameRate
        << " colorspace=" << QcapColorSpaceToString(appConfig.capture.outputColorSpace)
        << " gpudirect=" << (appConfig.capture.requestGpuDirect ? "requested" : "disabled")
        << std::endl;

    const QCAPGpuDirectStatus gpuDirectStatus = qcapCapture.GetGpuDirectStatus();
    if (gpuDirectStatus.requested) {
        std::cout << "[Capture] GPUDirect active=" << (qcapCapture.IsGpuDirectActive() ? "yes" : "no")
            << " queue=" << gpuDirectStatus.queueSize
            << " bufferSize=" << gpuDirectStatus.bufferSize
            << " cudaMapped=" << (gpuDirectStatus.buffersCudaMapped ? "yes" : "no")
            << " detail=" << gpuDirectStatus.message
            << std::endl;
        std::cout << "[Capture] GPUDirect callbackMatchesBoundBuffer="
            << (gpuDirectStatus.callbackUsesBoundBuffer ? "yes" : "no")
            << " previewFrames=" << gpuDirectStatus.previewFramesObserved
            << " matchedFrames=" << gpuDirectStatus.matchedPreviewFrames
            << " lastBufferIndex=" << gpuDirectStatus.lastBoundBufferIndex
            << std::endl;
    }
    const QCAPFrameInfo frameInfo = qcapCapture.GetFrameInfo();
    std::cout << "[Capture] negotiated size=" << frameInfo.width << "x" << frameInfo.height
        << " fps(active)=" << frameInfo.frameRate
        << " cpuShadow=" << (qcapCapture.HasCpuShadowBuffer()
            ? "ready"
            : (appConfig.capture.requestGpuDirect ? "disabled-on-fast-path" : "on-demand"))
        << std::endl;
}

void PrintDetectorReport(const DetectorInitReport& detectorReport) {
    std::cout << "[Detector] requested=" << detectorReport.requestedBackend
        << " active=" << detectorReport.activeBackend
        << " fallback=" << (detectorReport.usedFallback ? "yes" : "no")
        << std::endl;
    if (!detectorReport.resolvedModelPath.empty()) {
        std::cout << "[Detector] model=" << detectorReport.resolvedModelPath << std::endl;
    }
    if (!detectorReport.resolvedEnginePath.empty()) {
        std::cout << "[Detector] engine=" << detectorReport.resolvedEnginePath
            << " present=" << (detectorReport.tensorRtEngineDetected ? "yes" : "no")
            << std::endl;
    }
    if (detectorReport.tensorRtRoi.enabled) {
        std::cout << "[Detector] trtRoi="
            << detectorReport.tensorRtRoi.x << ","
            << detectorReport.tensorRtRoi.y << ","
            << detectorReport.tensorRtRoi.width << ","
            << detectorReport.tensorRtRoi.height
            << " mode=crop-and-rescale"
            << std::endl;
    }
    std::cout << "[Detector] trtRuntimeBuild="
        << (detectorReport.requestedBackend == "TensorRTYolo"
            ? "offline-required"
            : detectorReport.requestedBackend == "TensorRTLegacy" || detectorReport.requestedBackend == "TensorRTPreferred"
            ? "disabled-by-default"
            : "n/a")
        << std::endl;
    std::cout << "[Detector] " << detectorReport.message << std::endl;
    if (detectorReport.tensorRtRuntimeDetected) {
        std::cout << "[Detector] TensorRT runtime DLL detected at "
            << detectorReport.tensorRtRuntimePath << std::endl;
    }
}

void PrintMovementReport(const MovementController& movementController) {
    const MovementSettings& settings = movementController.GetSettings();
    std::cout << "[Move] backend=" << MovementBackendKindToString(settings.backend)
        << " gain=" << settings.gain
        << " maxStep=" << settings.maxStep
        << " deadzone=" << settings.deadzonePixels
        << " verticalBias=" << settings.verticalBias
        << std::endl;

    if (settings.backend == MovementBackendKind::TitanTwoGcv) {
        std::cout << "[Move] titanTwoGcvPath=" << settings.titanTwoGcvPath
            << " holdMs=" << settings.titanTwoHoldMs
            << std::endl;
    }
    if (settings.backend == MovementBackendKind::TitanTwoGcvGamepad) {
        std::cout << "[Move] titanTwoGcvPath=" << settings.titanTwoGcvPath
            << " holdMs=" << settings.titanTwoHoldMs
            << " stickMaxPercent=" << settings.titanTwoStickMaxPercent
            << " stickCurve=" << settings.titanTwoStickCurve
            << " stickResponseBoost=" << settings.titanTwoStickResponseBoost
            << " stickMinPercent=" << settings.titanTwoStickMinPercent
            << std::endl;
    }
    std::cout << "[Move] aimTracking=" << (settings.aimTrackingEnabled ? "on" : "off")
        << " confirmFrames=" << settings.aimTrackConfirmFrames
        << " lostFrames=" << settings.aimTrackLostFrames
        << " oneEuro=" << (settings.aimOneEuroEnabled ? "on" : "off")
        << " minCutoff=" << settings.aimOneEuroMinCutoff
        << " beta=" << settings.aimOneEuroBeta
        << " predictionMs=" << settings.aimPredictionMs
        << " maxBoxLead=" << settings.aimPredictionMaxBoxFraction
        << " fovRadius=" << settings.fovRadius
        << std::endl;
    std::cout << "[Move] stickPd=" << (settings.titanTwoStickPdEnabled ? "on" : "off")
        << " dGain=" << settings.titanTwoStickDerivativeGain
        << " feedForward=" << settings.titanTwoStickFeedForward
        << " slewPercentPerSecond=" << settings.titanTwoStickSlewPercentPerSecond
        << std::endl;
}

void PrintRunStatsWindow(
    const RunWindowStats& stats,
    const MovementController& movementController,
    const QCAPCapture* qcapCapture) {
    std::ostringstream oss;
    oss << "[Run] captureWait avg=" << stats.captureWaitMs.Average()
        << "ms min=" << stats.captureWaitMs.min
        << " max=" << stats.captureWaitMs.max
        << " detect avg=" << stats.detectMs.Average()
        << "ms move avg=" << stats.moveSubmitMs.Average()
        << "ms\n";
    oss << "[Run] detector calls=" << stats.detectorCalls
        << " validStats=" << stats.detectorFrames
        << "\n";

    if (stats.detectorFrames > 0) {
        const double avgD2HKiB =
            static_cast<double>(stats.detectorD2HBytes) / static_cast<double>(stats.detectorFrames) / 1024.0;
        const double avgRowsBefore =
            static_cast<double>(stats.outputRowsBeforeFilter) / static_cast<double>(stats.detectorFrames);
        const double avgRowsAfter =
            static_cast<double>(stats.outputRowsAfterFilter) / static_cast<double>(stats.detectorFrames);
        const double avgProposals =
            static_cast<double>(stats.proposalsBeforeNms) / static_cast<double>(stats.detectorFrames);
        const double avgResults =
            static_cast<double>(stats.resultsAfterNms) / static_cast<double>(stats.detectorFrames);

        oss << "[Run] det preprocess avg=" << stats.detectorPreprocessMs.Average()
            << "ms infer avg=" << stats.detectorInferMs.Average()
            << "ms gpuPost avg=" << stats.detectorGpuPostprocessMs.Average()
            << "ms d2h avg=" << stats.detectorD2HMs.Average()
            << "ms decode avg=" << stats.detectorDecodeMs.Average()
            << "ms total avg=" << stats.detectorTotalMs.Average()
            << "ms\n";
        oss << "[Run] det d2hKiB/frame=" << avgD2HKiB
            << " rows=" << avgRowsBefore << "->" << avgRowsAfter
            << " proposals=" << avgProposals
            << " results=" << avgResults
            << " gpuPreFrames=" << stats.detectorGpuPreprocessFrames
            << " gpuCompactFrames=" << stats.detectorGpuCompactionFrames
            << "\n";
        if (stats.detectorRoiFrames > 0) {
            oss << "[Run] det roiFrames=" << stats.detectorRoiFrames
                << "/" << stats.detectorFrames;
            if (stats.roiSampleValid) {
                oss << " activeRoiSample="
                    << stats.roiSampleX << ","
                    << stats.roiSampleY << ","
                    << stats.roiSampleWidth << ","
                    << stats.roiSampleHeight;
            }
            oss << "\n";
        }
    }

    const MovementDebugState& movementState = movementController.GetDebugState();
    oss << "[Run] movement backend=" << MovementBackendKindToString(movementController.GetSettings().backend)
        << " target=" << (movementState.hasTarget ? "yes" : "no")
        << " active=" << (movementState.active ? "yes" : "no")
        << " dx=" << movementState.deltaX
        << " dy=" << movementState.deltaY
        << " stickX=" << movementState.stickXPercent
        << " stickY=" << movementState.stickYPercent
        << " track=" << movementState.targetTrackId
        << " leadX=" << movementState.leadX
        << " leadY=" << movementState.leadY
        << " targetVx=" << movementState.targetVelocityX
        << " targetVy=" << movementState.targetVelocityY
        << " seq=" << movementState.sequence
        << "\n";
    oss << "[Run] move transportReady=" << (movementState.transportReady ? "yes" : "no")
        << " dispatch=" << movementState.transportDispatchCount
        << " skip=" << movementState.transportSkipCount
        << " lastSubmitMs=" << movementState.transportLastSubmitMs
        << " userIndex=" << movementState.transportUserIndex
        << "\n";

    if (qcapCapture) {
        const QCAPGpuDirectStatus gpuDirectStatus = qcapCapture->GetGpuDirectStatus();
        if (gpuDirectStatus.requested) {
            oss << "[Run] GPUDirect callbackMatchesBoundBuffer="
                << (gpuDirectStatus.callbackUsesBoundBuffer ? "yes" : "no")
                << " previewFrames=" << gpuDirectStatus.previewFramesObserved
                << " matchedFrames=" << gpuDirectStatus.matchedPreviewFrames
                << " lastBufferIndex=" << gpuDirectStatus.lastBoundBufferIndex
                << "\n";
        }

        const QCAPRuntimeStats runtimeStats = qcapCapture->GetRuntimeStats();
        oss << "[Run] capture waitCalls=" << runtimeStats.waitCalls
            << " waitTimeouts=" << runtimeStats.waitTimeouts
            << " frameViewOk=" << runtimeStats.frameViewSuccesses
            << " frameViewTimeouts=" << runtimeStats.frameViewTimeouts
            << " captureOk=" << runtimeStats.captureSuccesses
            << " captureTimeouts=" << runtimeStats.captureTimeouts
            << " previewCallbacks=" << runtimeStats.previewCallbacksObserved
            << " formatChanges=" << runtimeStats.formatChangesObserved
            << " noSignal=" << runtimeStats.noSignalEventsObserved
            << " signalRemoved=" << runtimeStats.signalRemovedEventsObserved
            << " rebind=" << runtimeStats.gpuDirectRebindAttempts
            << "/" << runtimeStats.gpuDirectRebindSuccesses
            << "/" << runtimeStats.gpuDirectRebindFailures
            << " resync=" << runtimeStats.captureResyncAttempts
            << "/" << runtimeStats.captureResyncSuccesses
            << "/" << runtimeStats.captureResyncFailures
            << " freshAfterRebind=" << runtimeStats.freshFramesAfterGpuDirectRebind
            << "\n";
        oss << "[Run] callback gap avg=" << runtimeStats.avgCallbackGapMs
            << "ms last=" << runtimeStats.lastCallbackGapMs
            << " max=" << runtimeStats.maxCallbackGapMs
            << " age=" << runtimeStats.lastCallbackAgeMs
            << " over2x=" << runtimeStats.callbackGapOver2xExpected
            << " over4x=" << runtimeStats.callbackGapOver4xExpected
            << " resyncs=" << runtimeStats.callbackGapResyncs
            << "\n";
    }

    std::cout << oss.str();
}

} // namespace

int main(int argc, char** argv) {
    ::SetConsoleOutputCP(CP_UTF8);
    ::SetConsoleCP(CP_UTF8);

    ::timeBeginPeriod(1);
    ::SetPriorityClass(::GetCurrentProcess(), HIGH_PRIORITY_CLASS);

    AppConfig appConfig;
    std::string configError;
    if (!ParseAppConfig(argc, argv, appConfig, configError)) {
        std::cerr << configError << std::endl;
        PrintAppUsage();
        return -1;
    }

    if (appConfig.showHelp) {
        PrintAppUsage();
        return 0;
    }

    if (!appConfig.headless) {
        appConfig.capture.autoResyncOnPreviewStall = false;
    }

    PrintCudaDeviceInteropReport(appConfig.detector.deviceIndex);

    const bool useQcap = (appConfig.captureBackend == CaptureBackend::QCAP);
    QCAPCapture qcapCapture;
    cv::VideoCapture cvCapture;
    int cvFrameWidth = 0;
    int cvFrameHeight = 0;

    if (useQcap) {
        unsigned long qcapInitSeh = 0;
        bool qcapReady = false;
        for (int attempt = 0; attempt < 3; ++attempt) {
            if (InitializeQcapCaptureWithSeh(qcapCapture, appConfig.capture, qcapInitSeh)) {
                qcapReady = true;
                break;
            }
            std::string errDetail = qcapCapture.GetLastError();
            if (qcapInitSeh != 0) {
                std::cerr << "QCAP / 710N1 capture initialization raised SEH 0x"
                    << std::hex << std::uppercase << qcapInitSeh << std::dec << std::endl;
            }
            std::cerr << "QCAP / 710N1 capture initialization failed (attempt " << (attempt + 1) << "/3)."
                << " Error: " << errDetail << std::endl;

            if (!appConfig.headless) {
                std::wstring msg = L"采集卡初始化失败 (尝试 " + std::to_wstring(attempt + 1) + L"/3)\n\n";
                if (!errDetail.empty()) {
                    int sz = MultiByteToWideChar(CP_UTF8, 0, errDetail.c_str(), -1, nullptr, 0);
                    std::wstring wErr(sz, 0);
                    MultiByteToWideChar(CP_UTF8, 0, errDetail.c_str(), -1, &wErr[0], sz);
                    msg += L"错误: " + wErr + L"\n\n";
                }
                msg += L"请检查采集卡连接后点击\"重试\"，或点击\"取消\"退出。";
                int res = ::MessageBoxW(nullptr, msg.c_str(), L"Snowball - 采集卡错误", MB_RETRYCANCEL | MB_ICONWARNING);
                if (res != IDRETRY) {
                    return -1;
                }
            } else {
                return -1;
            }
        }
        if (!qcapReady) {
            if (!appConfig.headless) {
                ::MessageBoxW(nullptr, L"采集卡初始化失败，已尝试3次。\n请检查采集卡驱动和连接后重新启动程序。",
                    L"Snowball - 错误", MB_OK | MB_ICONERROR);
            }
            return -1;
        }
        qcapCapture.TryStartupFallbackToActiveInput();
        PrintCaptureReport(appConfig, qcapCapture);
        qcapCapture.PrintInputDiagnostics("startup");
    } else {
        std::cout << "[Capture] OpenCV mode, opening device index " << appConfig.opencvDeviceIndex << std::endl;
        cvCapture.open(appConfig.opencvDeviceIndex, cv::CAP_DSHOW);
        if (!cvCapture.isOpened()) {
            cvCapture.open(appConfig.opencvDeviceIndex, cv::CAP_MSMF);
        }
        if (!cvCapture.isOpened()) {
            std::cerr << "OpenCV capture device " << appConfig.opencvDeviceIndex << " failed to open." << std::endl;
            if (!appConfig.headless) {
                ::MessageBoxW(nullptr,
                    L"OpenCV 采集设备打开失败。\n\n请检查:\n1. 采集卡是否连接\n2. config.ini 中 opencv_index 是否正确\n3. 采集卡驱动是否安装",
                    L"Snowball - 采集卡错误", MB_OK | MB_ICONERROR);
            }
            return -1;
        }
        if (appConfig.capture.width > 0) cvCapture.set(cv::CAP_PROP_FRAME_WIDTH, appConfig.capture.width);
        if (appConfig.capture.height > 0) cvCapture.set(cv::CAP_PROP_FRAME_HEIGHT, appConfig.capture.height);
        cvFrameWidth = static_cast<int>(cvCapture.get(cv::CAP_PROP_FRAME_WIDTH));
        cvFrameHeight = static_cast<int>(cvCapture.get(cv::CAP_PROP_FRAME_HEIGHT));
        std::cout << "[Capture] OpenCV device " << appConfig.opencvDeviceIndex
            << " opened: " << cvFrameWidth << "x" << cvFrameHeight
            << " fps=" << cvCapture.get(cv::CAP_PROP_FPS) << std::endl;
    }

    std::unique_ptr<IObjectDetector> detector;
    DetectorInitReport detectorReport;
    if (!InitializeDetectorWithFallback(appConfig.detector, detector, detectorReport)) {
        std::cerr << "Detector initialization failed. " << detectorReport.message << std::endl;
        if (!appConfig.headless) {
            std::string errMsg = "推理引擎初始化失败:\n" + detectorReport.message
                + "\n\n请检查 config.ini 中的模型路径和 CUDA/TensorRT 环境。";
            int sz = MultiByteToWideChar(CP_UTF8, 0, errMsg.c_str(), -1, nullptr, 0);
            std::wstring wMsg(sz, 0);
            MultiByteToWideChar(CP_UTF8, 0, errMsg.c_str(), -1, &wMsg[0], sz);
            ::MessageBoxW(nullptr, wMsg.c_str(), L"Snowball - 引擎错误", MB_OK | MB_ICONERROR);
        }
        if (useQcap) qcapCapture.Release();
        return -1;
    }
    PrintDetectorReport(detectorReport);

    if (!appConfig.detector.tensorRtRoi.enabled) {
        int srcW, srcH;
        if (useQcap) {
            const QCAPFrameInfo frameInfo = qcapCapture.GetFrameInfo();
            srcW = frameInfo.width > 0 ? frameInfo.width : appConfig.capture.width;
            srcH = frameInfo.height > 0 ? frameInfo.height : appConfig.capture.height;
        } else {
            srcW = cvFrameWidth > 0 ? cvFrameWidth : appConfig.capture.width;
            srcH = cvFrameHeight > 0 ? cvFrameHeight : appConfig.capture.height;
        }
        constexpr int kModelInputSize = 640;
        if (srcW > kModelInputSize && srcH > kModelInputSize) {
            appConfig.detector.tensorRtRoi.enabled = true;
            appConfig.detector.tensorRtRoi.x = (srcW - kModelInputSize) / 2;
            appConfig.detector.tensorRtRoi.y = (srcH - kModelInputSize) / 2;
            appConfig.detector.tensorRtRoi.width = kModelInputSize;
            appConfig.detector.tensorRtRoi.height = kModelInputSize;
            std::cout << "[Detector] Auto-enabled center crop ROI "
                << appConfig.detector.tensorRtRoi.x << ","
                << appConfig.detector.tensorRtRoi.y << ","
                << kModelInputSize << "x" << kModelInputSize
                << " from source " << srcW << "x" << srcH << std::endl;
        }
    }

    MovementController movementController;
    if (!movementController.Initialize(appConfig.movement)) {
        std::cerr << "Movement initialization failed. " << movementController.GetLastError() << std::endl;
        if (!appConfig.headless) {
            std::string errMsg = "运动控制初始化失败:\n" + movementController.GetLastError()
                + "\n\n请检查 TitanTwo 设备连接和驱动。";
            int sz = MultiByteToWideChar(CP_UTF8, 0, errMsg.c_str(), -1, nullptr, 0);
            std::wstring wMsg(sz, 0);
            MultiByteToWideChar(CP_UTF8, 0, errMsg.c_str(), -1, &wMsg[0], sz);
            ::MessageBoxW(nullptr, wMsg.c_str(), L"Snowball - 控制器错误", MB_OK | MB_ICONERROR);
        }
        if (detector) {
            detector->ReleaseResources();
        }
        qcapCapture.Release();
        return -1;
    }
    PrintMovementReport(movementController);

    LiveControlState liveControlState;
    liveControlState.movement = appConfig.movement;
    liveControlState.defaults = appConfig.movement;
    liveControlState.detectionConfidenceThreshold = kDefaultDetectionConfidenceThreshold;
    liveControlState.detectionNmsThreshold = kDefaultDetectionNmsThreshold;
    liveControlState.defaultDetectionConfidenceThreshold = kDefaultDetectionConfidenceThreshold;
    liveControlState.defaultDetectionNmsThreshold = kDefaultDetectionNmsThreshold;
    ClampLiveControlState(liveControlState);

    auto EnumerateEngineFiles = [&]() -> std::vector<std::string> {
        std::vector<std::string> engines;
        std::error_code ec;
        const auto exeDir = std::filesystem::path(argv[0]).parent_path();
        const auto searchDir = exeDir.empty() ? std::filesystem::current_path() : exeDir;
        for (const auto& entry : std::filesystem::directory_iterator(searchDir, ec)) {
            if (!entry.is_regular_file()) continue;
            const std::string ext = entry.path().extension().string();
            if (ext == ".engine" || ext == ".plan") {
                engines.push_back(entry.path().filename().string());
            }
        }
        std::sort(engines.begin(), engines.end());
        return engines;
    };

    bool inferenceRunning = true;

    LiveControlPanel liveControlPanel;
    LiveControlPanelState panelState;
    panelState.movement = liveControlState.movement;
    panelState.defaults = liveControlState.defaults;
    panelState.detectionConfidenceThreshold = liveControlState.detectionConfidenceThreshold;
    panelState.detectionNmsThreshold = liveControlState.detectionNmsThreshold;
    panelState.defaultDetectionConfidenceThreshold = liveControlState.defaultDetectionConfidenceThreshold;
    panelState.defaultDetectionNmsThreshold = liveControlState.defaultDetectionNmsThreshold;
    panelState.labelFilter = liveControlState.labelFilter;
    panelState.backendName = detectorReport.activeBackend;
    panelState.inferenceRunning = inferenceRunning;
    panelState.selectedEnginePath = appConfig.detector.enginePath;
    panelState.availableEngines = EnumerateEngineFiles();
    if (!appConfig.headless && !liveControlPanel.Initialize(panelState)) {
        std::cerr << "Live control panel initialization failed." << std::endl;
    }

    int frameCount = 0;
    int totalFrames = 0;
    int emptyCaptureTicks = 0;
    bool printedCudaInteropProbe = false;
    bool printedDetectorStatsDebug = false;
    bool printedMovementError = false;
    bool debugGpuDirectRebindTriggered = false;
    bool inferenceScreenshotWriteFailed = false;
    bool inferenceScreenshotMetadataWriteFailed = false;
    bool debugDumpQcapFrameWritten = false;
    std::uint64_t inferenceScreenshotsWritten = 0;
    cv::Mat lastGuiDisplayFrame;
    auto fpsStartTime = std::chrono::high_resolution_clock::now();
    auto processStartTime = fpsStartTime;
    auto measuredRunStartTime = fpsStartTime;
    double currentFps = 0.0;
    RunWindowStats runWindowStats;
    const bool useWarmupWindow = appConfig.headless && appConfig.warmupSeconds > 0.0;
    bool warmupCompleted = !useWarmupWindow;
    bool warmupResetDone = false;
    int measuredFrames = 0;
    bool startupActiveInputFallbackAttempted = false;
    constexpr int kGuiPreviewIntervalMs = 33;
    constexpr int kGuiPanelSyncIntervalMs = 250;
    constexpr int kGuiPreviewMaxWidth = 640;
    auto lastGuiPreviewUpdateTime = fpsStartTime - std::chrono::milliseconds(kGuiPreviewIntervalMs);
    auto lastGuiPanelSyncTime = fpsStartTime - std::chrono::milliseconds(kGuiPanelSyncIntervalMs);

    std::cout << (appConfig.headless ? "Headless mode running..." : "Close the ")
        << (!appConfig.headless ? kAppDisplayName : "")
        << (!appConfig.headless ? " window to exit." : "")
        << std::endl;
    if (useWarmupWindow) {
        std::cout << "[Run] warmup enabled seconds=" << appConfig.warmupSeconds << std::endl;
    }

    const bool movementTransportSmokeMode = appConfig.headless && appConfig.movement.movementTestStickEnabled;
    if (movementTransportSmokeMode) {
        const int submitIntervalMs = std::clamp(
            appConfig.movement.titanTwoHoldMs > 0 ? appConfig.movement.titanTwoHoldMs / 2 : 8,
            1,
            16);
        const int fallbackFrameWidth = appConfig.capture.width > 0 ? appConfig.capture.width : 1920;
        const int fallbackFrameHeight = appConfig.capture.height > 0 ? appConfig.capture.height : 1080;
        const std::vector<DetectionObject> noDetections;

        std::cout << "[Run] movement transport smoke mode enabled; fixed stick output will run without waiting for live detections."
            << std::endl;
        std::cout << "[Run] movement transport smoke submitIntervalMs=" << submitIntervalMs
            << " fallbackSize=" << fallbackFrameWidth << "x" << fallbackFrameHeight
            << std::endl;

        while (true) {
            int movementFrameWidth = fallbackFrameWidth;
            int movementFrameHeight = fallbackFrameHeight;
            if (useQcap) {
                const QCAPFrameInfo frameInfo = qcapCapture.GetFrameInfo();
                if (frameInfo.width > 0) {
                    movementFrameWidth = frameInfo.width;
                }
                if (frameInfo.height > 0) {
                    movementFrameHeight = frameInfo.height;
                }
            } else {
                if (cvFrameWidth > 0) movementFrameWidth = cvFrameWidth;
                if (cvFrameHeight > 0) movementFrameHeight = cvFrameHeight;
            }

            const auto moveStart = std::chrono::steady_clock::now();
            if (!movementController.SubmitFromDetections(movementFrameWidth, movementFrameHeight, noDetections) &&
                !printedMovementError) {
                printedMovementError = true;
                std::cout << "[Move] submit failed: " << movementController.GetLastError() << std::endl;
            }
            const auto moveEnd = std::chrono::steady_clock::now();
            runWindowStats.moveSubmitMs.Add(
                std::chrono::duration<double, std::milli>(moveEnd - moveStart).count());

            ++frameCount;
            if (warmupCompleted) {
                ++measuredFrames;
            }

            const auto currentTime = std::chrono::high_resolution_clock::now();
            const auto elapsedMs = std::chrono::duration_cast<std::chrono::milliseconds>(
                currentTime - fpsStartTime).count();

            if (!warmupCompleted) {
                const auto warmupElapsedMs = std::chrono::duration_cast<std::chrono::milliseconds>(
                    currentTime - measuredRunStartTime).count();
                if (warmupElapsedMs >= static_cast<long long>(appConfig.warmupSeconds * 1000.0)) {
                    warmupCompleted = true;
                    warmupResetDone = true;
                    frameCount = 0;
                    totalFrames = 0;
                    measuredFrames = 0;
                    currentFps = 0.0;
                    fpsStartTime = currentTime;
                    measuredRunStartTime = currentTime;
                    runWindowStats.Reset();
                    if (useQcap) qcapCapture.ResetRuntimeTelemetry();
                    std::cout << "[Run] warmup complete, runtime telemetry reset for measured window." << std::endl;
                    std::this_thread::sleep_for(std::chrono::milliseconds(submitIntervalMs));
                    continue;
                }
            }

            if (elapsedMs >= 1000) {
                currentFps = (frameCount * 1000.0) / static_cast<double>(elapsedMs);
                frameCount = 0;
                fpsStartTime = currentTime;
            }

            ++totalFrames;

            if (elapsedMs >= 1000) {
                std::cout << "[Run] movementTicks=" << totalFrames
                    << " rate=" << static_cast<int>(currentFps)
                    << " objects=0"
                    << " backend=" << detectorReport.activeBackend
                    << (warmupResetDone ? " window=measured" : "")
                    << std::endl;
                PrintRunStatsWindow(runWindowStats, movementController, useQcap ? &qcapCapture : nullptr);
                runWindowStats.Reset();
            }

            const auto now = std::chrono::high_resolution_clock::now();
            const auto runElapsedMs = std::chrono::duration_cast<std::chrono::milliseconds>(
                now - measuredRunStartTime).count();
            const auto wallClockElapsedMs = std::chrono::duration_cast<std::chrono::milliseconds>(
                now - processStartTime).count();
            if (appConfig.maxFrames > 0 && measuredFrames >= appConfig.maxFrames) {
                break;
            }
            if (warmupCompleted &&
                appConfig.maxSeconds > 0.0 &&
                runElapsedMs >= static_cast<long long>(appConfig.maxSeconds * 1000.0)) {
                break;
            }
            if (!warmupCompleted &&
                appConfig.maxSeconds > 0.0 &&
                wallClockElapsedMs >= static_cast<long long>((appConfig.warmupSeconds + appConfig.maxSeconds) * 1000.0)) {
                std::cout << "[Run] exiting before measured window because movement transport smoke warmup did not complete in time."
                    << std::endl;
                break;
            }

            std::this_thread::sleep_for(std::chrono::milliseconds(submitIntervalMs));
        }
    }

    while (!movementTransportSmokeMode) {
        if (!appConfig.headless && !liveControlPanel.PumpMessages()) {
            break;
        }
        LiveControlPanelState updatedPanelState;
        if (liveControlPanel.PullState(updatedPanelState)) {
            liveControlState.movement = updatedPanelState.movement;
            liveControlState.defaults = updatedPanelState.defaults;
            liveControlState.detectionConfidenceThreshold = updatedPanelState.detectionConfidenceThreshold;
            liveControlState.detectionNmsThreshold = updatedPanelState.detectionNmsThreshold;
            liveControlState.defaultDetectionConfidenceThreshold = updatedPanelState.defaultDetectionConfidenceThreshold;
            liveControlState.defaultDetectionNmsThreshold = updatedPanelState.defaultDetectionNmsThreshold;
            liveControlState.labelFilter = updatedPanelState.labelFilter;
            ClampLiveControlState(liveControlState);
            movementController.UpdateRuntimeSettings(liveControlState.movement);
            panelState.capturePreviewEnabled = updatedPanelState.capturePreviewEnabled;

            if (updatedPanelState.engineAction == PanelEngineAction::StopInference && inferenceRunning) {
                if (detector) {
                    detector->ReleaseResources();
                    detector.reset();
                }
                inferenceRunning = false;
                panelState.inferenceRunning = false;
                std::cout << "[UI] Inference stopped by user." << std::endl;
            }
            else if (updatedPanelState.engineAction == PanelEngineAction::StartInference) {
                std::string newEngine = updatedPanelState.selectedEnginePath;
                if (!newEngine.empty()) {
                    appConfig.detector.enginePath = newEngine;
                }
                if (detector) {
                    detector->ReleaseResources();
                    detector.reset();
                }
                detectorReport = DetectorInitReport{};
                if (InitializeDetectorWithFallback(appConfig.detector, detector, detectorReport)) {
                    inferenceRunning = true;
                    panelState.inferenceRunning = true;
                    panelState.backendName = detectorReport.activeBackend;
                    panelState.selectedEnginePath = appConfig.detector.enginePath;
                    std::cout << "[UI] Inference started: " << detectorReport.activeBackend
                        << " engine=" << appConfig.detector.enginePath << std::endl;
                } else {
                    inferenceRunning = false;
                    panelState.inferenceRunning = false;
                    std::cerr << "[UI] Inference start failed: " << detectorReport.message << std::endl;
                }
                printedDetectorStatsDebug = false;
            }
            else if (updatedPanelState.engineAction == PanelEngineAction::SaveConfig) {
                appConfig.movement = liveControlState.movement;
                const auto iniPath = std::filesystem::path(argv[0]).parent_path() / "config.ini";
                std::string saveError;
                if (SaveConfigToIniFile(iniPath.string(), appConfig, saveError)) {
                    std::cout << "[UI] Config saved to " << iniPath.string() << std::endl;
                } else {
                    std::cerr << "[UI] Config save failed: " << saveError << std::endl;
                }
            }
            else if (updatedPanelState.engineAction == PanelEngineAction::ExportJson) {
                appConfig.movement = liveControlState.movement;
                const auto jsonPath = std::filesystem::path(argv[0]).parent_path() / "config_export.json";
                std::string exportError;
                if (ExportConfigToJson(jsonPath.string(), appConfig, exportError)) {
                    std::cout << "[UI] Config exported to " << jsonPath.string() << std::endl;
                } else {
                    std::cerr << "[UI] Config export failed: " << exportError << std::endl;
                }
            }
        }

        const auto guiFrameStartTime = std::chrono::high_resolution_clock::now();
        const bool shouldUpdateGuiPreview =
            !appConfig.headless &&
            panelState.capturePreviewEnabled &&
            std::chrono::duration_cast<std::chrono::milliseconds>(
                guiFrameStartTime - lastGuiPreviewUpdateTime).count() >= kGuiPreviewIntervalMs;
        const bool shouldSyncGuiPanel =
            !appConfig.headless &&
            std::chrono::duration_cast<std::chrono::milliseconds>(
                guiFrameStartTime - lastGuiPanelSyncTime).count() >= kGuiPanelSyncIntervalMs;

        cv::Mat frame;
        std::vector<DetectionObject> results;
        std::vector<DetectionObject> movementResults;
        DetectorFrameStats detectorStats;
        bool gotFrame = false;
        bool haveDisplayFrame = false;
        bool displayFrameIsScaledPreview = false;
        bool frameSupportsDisplay = false;
        int sourceFrameWidth = 0;
        int sourceFrameHeight = 0;

        if (useQcap) {
            QCAPFrameView frameView;
            const int frameViewTimeoutMs = ComputeFrameViewTimeoutMs(qcapCapture, appConfig.capture);
            const auto captureWaitStart = std::chrono::steady_clock::now();
            if (qcapCapture.GetLatestFrameView(frameView, frameViewTimeoutMs)) {
                const auto captureWaitEnd = std::chrono::steady_clock::now();
                runWindowStats.captureWaitMs.Add(
                    std::chrono::duration<double, std::milli>(captureWaitEnd - captureWaitStart).count());
                if (!printedCudaInteropProbe && frameView.sourceBufferMatchesGpuDirect) {
                    PrintCudaHostMappingProbe(
                        "GPUDirect sourceBuffer",
                        frameView.sourceBuffer,
                        frameView.bufferSize,
                        false);
                    if (frameView.sourceBufferCudaMapped) {
                        std::cout << "[CUDA] GPUDirect sourceBuffer persistentMappedDevicePointer="
                            << frameView.sourceDevicePointer << std::endl;
                    }
                    std::cout << "[Capture] frameView cpuShadowCopyAvailable="
                        << (frameView.cpuShadowCopyAvailable ? "yes" : "no")
                        << std::endl;
                    printedCudaInteropProbe = true;
                }

                if (inferenceRunning && detector &&
                    CanUseQcapFrameViewForMainlineDetection(frameView, appConfig.capture.requestGpuDirect) &&
                    detector->CanConsumeQcapFrameView(frameView)) {
                    const auto detectStart = std::chrono::steady_clock::now();
                    results = detector->DetectQcapFrameView(
                        frameView,
                        liveControlState.detectionNmsThreshold,
                        liveControlState.detectionConfidenceThreshold);
                    const auto detectEnd = std::chrono::steady_clock::now();
                    runWindowStats.detectMs.Add(
                        std::chrono::duration<double, std::milli>(detectEnd - detectStart).count());
                    detectorStats = detector->GetLastFrameStats();
                    ++runWindowStats.detectorCalls;
                    PrintFirstDetectorStatsDebug(detectorStats, printedDetectorStatsDebug);
                    AccumulateDetectorStats(detectorStats, runWindowStats);

                    gotFrame = true;
                    sourceFrameWidth = frameView.width;
                    sourceFrameHeight = frameView.height;
                    if (shouldUpdateGuiPreview) {
                        if (frameView.cpuData != nullptr && frameView.colorSpaceType == QCAP_COLORSPACE_TYPE_BGR24) {
                            frame = cv::Mat(
                                frameView.height,
                                frameView.width,
                                CV_8UC3,
                                const_cast<BYTE*>(frameView.cpuData)).clone();
                            if (!frame.empty() && frameView.imageUpsideDown) {
                                cv::flip(frame, frame, 0);
                            }
                            haveDisplayFrame = !frame.empty();
                            displayFrameIsScaledPreview = false;
                        }
                        else {
                            haveDisplayFrame =
                                qcapCapture.ConvertFrameToPreviewMat(frame, kGuiPreviewMaxWidth) &&
                                !frame.empty();
                            displayFrameIsScaledPreview =
                                haveDisplayFrame &&
                                sourceFrameWidth > 0 &&
                                sourceFrameHeight > 0 &&
                                (frame.cols != sourceFrameWidth || frame.rows != sourceFrameHeight);
                        }
                        frameSupportsDisplay = haveDisplayFrame;
                    }
                }
            }
        } else {
            cv::Mat cvFrame;
            const auto captureWaitStart = std::chrono::steady_clock::now();
            if (cvCapture.read(cvFrame) && !cvFrame.empty()) {
                const auto captureWaitEnd = std::chrono::steady_clock::now();
                runWindowStats.captureWaitMs.Add(
                    std::chrono::duration<double, std::milli>(captureWaitEnd - captureWaitStart).count());

                sourceFrameWidth = cvFrame.cols;
                sourceFrameHeight = cvFrame.rows;

                if (inferenceRunning && detector) {
                    cv::Mat bgrFrame;
                    if (cvFrame.channels() == 3) {
                        bgrFrame = cvFrame;
                    } else if (cvFrame.channels() == 4) {
                        cv::cvtColor(cvFrame, bgrFrame, cv::COLOR_BGRA2BGR);
                    } else {
                        cv::cvtColor(cvFrame, bgrFrame, cv::COLOR_GRAY2BGR);
                    }
                    if (!bgrFrame.isContinuous()) {
                        bgrFrame = bgrFrame.clone();
                    }

                    const auto detectStart = std::chrono::steady_clock::now();
                    results = detector->DetectBGR(
                        bgrFrame.data,
                        bgrFrame.cols,
                        bgrFrame.rows,
                        liveControlState.detectionNmsThreshold,
                        liveControlState.detectionConfidenceThreshold);
                    const auto detectEnd = std::chrono::steady_clock::now();
                    runWindowStats.detectMs.Add(
                        std::chrono::duration<double, std::milli>(detectEnd - detectStart).count());
                    detectorStats = detector->GetLastFrameStats();
                    ++runWindowStats.detectorCalls;
                    PrintFirstDetectorStatsDebug(detectorStats, printedDetectorStatsDebug);
                    AccumulateDetectorStats(detectorStats, runWindowStats);

                    gotFrame = true;

                    if (shouldUpdateGuiPreview) {
                        frame = cvFrame.clone();
                        haveDisplayFrame = true;
                        displayFrameIsScaledPreview = false;
                        frameSupportsDisplay = true;
                    }
                }
            }
        }

        if (gotFrame) {
            emptyCaptureTicks = 0;

            liveControlState.lastSeenLabels.clear();
            for (const auto& detection : results) {
                AddLiveControlLabel(liveControlState.lastSeenLabels, detection.label);
            }
            movementResults = FilterDetectionsForLiveControls(results, liveControlState);

            int movementFrameWidth = 0;
            int movementFrameHeight = 0;
            if (sourceFrameWidth > 0 && sourceFrameHeight > 0) {
                movementFrameWidth = sourceFrameWidth;
                movementFrameHeight = sourceFrameHeight;
            }
            else if (useQcap) {
                const QCAPFrameInfo frameInfo = qcapCapture.GetFrameInfo();
                movementFrameWidth = frameInfo.width;
                movementFrameHeight = frameInfo.height;
            } else {
                movementFrameWidth = cvFrameWidth;
                movementFrameHeight = cvFrameHeight;
            }

            if (movementFrameWidth > 0 && movementFrameHeight > 0) {
                const auto moveStart = std::chrono::steady_clock::now();
                if (!movementController.SubmitFromDetections(movementFrameWidth, movementFrameHeight, movementResults) &&
                    !printedMovementError) {
                    printedMovementError = true;
                    std::cout << "[Move] submit failed: " << movementController.GetLastError() << std::endl;
                }
                const auto moveEnd = std::chrono::steady_clock::now();
                runWindowStats.moveSubmitMs.Add(
                    std::chrono::duration<double, std::milli>(moveEnd - moveStart).count());
            }

            const MovementDebugState& movementState = movementController.GetDebugState();
            liveControlState.lastTargetLabel = movementState.targetLabel;

            if (shouldSyncGuiPanel) {
                panelState.movement = liveControlState.movement;
                panelState.defaults = liveControlState.defaults;
                panelState.detectionConfidenceThreshold = liveControlState.detectionConfidenceThreshold;
                panelState.detectionNmsThreshold = liveControlState.detectionNmsThreshold;
                panelState.defaultDetectionConfidenceThreshold = liveControlState.defaultDetectionConfidenceThreshold;
                panelState.defaultDetectionNmsThreshold = liveControlState.defaultDetectionNmsThreshold;
                panelState.labelFilter = liveControlState.labelFilter;
                panelState.fps = currentFps;
                panelState.detectionCount = static_cast<int>(results.size());
                panelState.backendName = detectorReport.activeBackend;
                panelState.currentTargetLabel = movementState.targetLabel;
                panelState.hasTarget = movementState.hasTarget;
                panelState.outputActive = movementState.active;
                panelState.transportReady = movementState.transportReady;
                panelState.transportDispatchCount = movementState.transportDispatchCount;
                panelState.transportSkipCount = movementState.transportSkipCount;
                panelState.transportLastSubmitMs = movementState.transportLastSubmitMs;
                panelState.transportUserIndex = movementState.transportUserIndex;
                panelState.originX = movementState.originX;
                panelState.originY = movementState.originY;
                panelState.targetX = movementState.targetX;
                panelState.targetY = movementState.targetY;
                panelState.deltaX = movementState.deltaX;
                panelState.deltaY = movementState.deltaY;
                panelState.rawDeltaX = movementState.rawDeltaX;
                panelState.rawDeltaY = movementState.rawDeltaY;
                panelState.stickXPercent = movementState.stickXPercent;
                panelState.stickYPercent = movementState.stickYPercent;
                panelState.score = movementState.score;
                panelState.observedLabels = liveControlState.lastSeenLabels;
                liveControlPanel.SyncObservedState(panelState);
                lastGuiPanelSyncTime = guiFrameStartTime;
            }

            {
                if (haveDisplayFrame) {
                    double displayScaleX = 1.0;
                    double displayScaleY = 1.0;
                    if (displayFrameIsScaledPreview && sourceFrameWidth > 0 && sourceFrameHeight > 0) {
                        displayScaleX = static_cast<double>(frame.cols) / static_cast<double>(sourceFrameWidth);
                        displayScaleY = static_cast<double>(frame.rows) / static_cast<double>(sourceFrameHeight);
                    }
                    const std::vector<DetectionObject> displayResults =
                        ScaleDetectionsForDisplay(results, displayScaleX, displayScaleY);
                    const MovementDebugState displayMovementState =
                        ScaleMovementDebugStateForDisplay(movementState, displayScaleX, displayScaleY);
                    const double fovRadiusScaled = liveControlState.movement.fovRadius > 0.0
                        ? liveControlState.movement.fovRadius * (std::min)(displayScaleX, displayScaleY)
                        : 0.0;
                    DrawDetections(frame, displayResults);
                    DrawMovementOverlay(frame, displayMovementState, fovRadiusScaled);
                }

                ++frameCount;
                if (warmupCompleted) {
                    ++measuredFrames;
                }
                const int screenshotFrameNumber = warmupCompleted ? measuredFrames : totalFrames;
                if (ShouldAttemptInferenceScreenshot(appConfig, screenshotFrameNumber, results)) {
                    cv::Mat screenshotFrame;
                    if (haveDisplayFrame) {
                        screenshotFrame = frame.clone();
                        DrawTensorRtRoiOverlay(screenshotFrame, detectorStats);
                    }
                    else if (useQcap) {
                        qcapCapture.ConvertFrameToMat(screenshotFrame);
                        if (!screenshotFrame.empty()) {
                            DrawDetections(screenshotFrame, results);
                            DrawMovementOverlay(screenshotFrame, movementState);
                            DrawTensorRtRoiOverlay(screenshotFrame, detectorStats);
                        }
                    }

                    if (!screenshotFrame.empty()) {
                        std::filesystem::path screenshotPath;
                        const InferenceScreenshotWriteResult screenshotWriteResult = MaybeWriteInferenceScreenshot(
                            appConfig,
                            screenshotFrameNumber,
                            results,
                            screenshotFrame,
                            &screenshotPath);
                        if (screenshotWriteResult == InferenceScreenshotWriteResult::Succeeded) {
                            ++inferenceScreenshotsWritten;
                            std::filesystem::path screenshotMetadataPath;
                            if (!WriteInferenceScreenshotMetadata(
                                    screenshotPath,
                                    screenshotFrameNumber,
                                    results,
                                    detectorStats,
                                    &screenshotMetadataPath) &&
                                !inferenceScreenshotMetadataWriteFailed) {
                                inferenceScreenshotMetadataWriteFailed = true;
                                std::cout << "[Run] inference screenshot metadata write attempt failed path="
                                    << PathToUtf8String(screenshotMetadataPath) << std::endl;
                            }
                            if (inferenceScreenshotsWritten == 1 ||
                                appConfig.inferenceScreenshotEveryNFrames > 0) {
                                std::cout << "[Run] inference screenshot written to "
                                    << PathToUtf8String(screenshotPath) << std::endl;
                            }
                        }
                        else if (screenshotWriteResult == InferenceScreenshotWriteResult::Failed &&
                                 !inferenceScreenshotWriteFailed) {
                            inferenceScreenshotWriteFailed = true;
                            std::cout << "[Run] inference screenshot write attempt failed path="
                                << PathToUtf8String(screenshotPath) << std::endl;
                        }
                    }
                }
                if (useQcap && !debugDumpQcapFrameWritten && !appConfig.debugDumpQcapFramePath.empty()) {
                    cv::Mat debugFrame;
                    qcapCapture.ConvertFrameToMat(debugFrame);
                    if (debugFrame.empty() && haveDisplayFrame) {
                        debugFrame = frame.clone();
                    }

                    if (!debugFrame.empty()) {
                        std::filesystem::path bgrPath;
                        std::filesystem::path rgbPath;
                        if (WriteQcapDebugDump(
                                appConfig,
                                debugFrame,
                                screenshotFrameNumber,
                                &bgrPath,
                                &rgbPath)) {
                            std::cout << "[Debug] QCAP frame dump written bgr="
                                << PathToUtf8String(bgrPath)
                                << " rgbAsBgr=" << PathToUtf8String(rgbPath)
                                << std::endl;
                        }
                        else {
                            std::cout << "[Debug] QCAP frame dump write failed path="
                                << appConfig.debugDumpQcapFramePath
                                << std::endl;
                        }
                        debugDumpQcapFrameWritten = true;
                    }
                }
                if (!debugGpuDirectRebindTriggered &&
                    appConfig.debugForceGpuDirectRebindFrame > 0 &&
                    totalFrames >= appConfig.debugForceGpuDirectRebindFrame) {
                    if (useQcap) {
                        std::cout << "[Debug] forcing GPUDirect rebind at frame " << totalFrames << std::endl;
                        qcapCapture.DebugForceGpuDirectRebind();
                    }
                    debugGpuDirectRebindTriggered = true;
                }
                const auto currentTime = std::chrono::high_resolution_clock::now();
                const auto elapsedMs = std::chrono::duration_cast<std::chrono::milliseconds>(
                    currentTime - fpsStartTime).count();

                if (!warmupCompleted) {
                    const auto warmupElapsedMs = std::chrono::duration_cast<std::chrono::milliseconds>(
                        currentTime - measuredRunStartTime).count();
                    if (warmupElapsedMs >= static_cast<long long>(appConfig.warmupSeconds * 1000.0)) {
                        warmupCompleted = true;
                        warmupResetDone = true;
                        frameCount = 0;
                        totalFrames = 0;
                        measuredFrames = 0;
                        debugDumpQcapFrameWritten = false;
                        currentFps = 0.0;
                        fpsStartTime = currentTime;
                        measuredRunStartTime = currentTime;
                        runWindowStats.Reset();
                        printedDetectorStatsDebug = false;
                        if (useQcap) qcapCapture.ResetRuntimeTelemetry();
                        std::cout << "[Run] warmup complete, runtime telemetry reset for measured window." << std::endl;
                        continue;
                    }
                }

                const bool shouldPrintStats = elapsedMs >= 1000;
                if (shouldPrintStats) {
                    currentFps = (frameCount * 1000.0) / static_cast<double>(elapsedMs);
                    frameCount = 0;
                    fpsStartTime = currentTime;
                }

                ++totalFrames;

                if (!appConfig.headless && shouldUpdateGuiPreview) {
                    cv::Mat displayFrame;
                    if (frameSupportsDisplay && !frame.empty()) {
                        displayFrame = frame;
                    }
                    else if (!lastGuiDisplayFrame.empty()) {
                        displayFrame = lastGuiDisplayFrame;
                    }
                    else {
                        displayFrame = BuildPreviewPlaceholder(
                            appConfig,
                            detectorReport,
                            useQcap ? &qcapCapture : nullptr,
                            currentFps);
                    }

                    if (!displayFrame.empty() && displayFrame.cols > kGuiPreviewMaxWidth) {
                        const double previewScale =
                            static_cast<double>(kGuiPreviewMaxWidth) / static_cast<double>(displayFrame.cols);
                        const int previewHeight = (std::max)(
                            1,
                            static_cast<int>(std::round(static_cast<double>(displayFrame.rows) * previewScale)));
                        cv::Mat resizedPreview;
                        cv::resize(
                            displayFrame,
                            resizedPreview,
                            cv::Size(kGuiPreviewMaxWidth, previewHeight),
                            0.0,
                            0.0,
                            cv::INTER_AREA);
                        liveControlPanel.UpdatePreview(resizedPreview);
                    }
                    else {
                        liveControlPanel.UpdatePreview(displayFrame);
                    }
                    if (frameSupportsDisplay && !frame.empty()) {
                        lastGuiDisplayFrame = displayFrame;
                    }
                    lastGuiPreviewUpdateTime = currentTime;
                }
                if (shouldPrintStats) {
                    std::cout << "[Run] frames=" << totalFrames
                        << " fps=" << static_cast<int>(currentFps)
                        << " objects=" << results.size()
                        << " backend=" << detectorReport.activeBackend
                        << (warmupResetDone ? " window=measured" : "")
                        << std::endl;
                    PrintRunStatsWindow(runWindowStats, movementController, useQcap ? &qcapCapture : nullptr);
                    runWindowStats.Reset();
                }
            }
        }
        else if (appConfig.headless) {
            ++emptyCaptureTicks;
            if (emptyCaptureTicks % 30 == 0) {
                std::cout << "[Capture] waiting for frame";
                if (useQcap) {
                    if (!qcapCapture.GetLastError().empty()) {
                        std::cout << " (" << qcapCapture.GetLastError() << ")";
                    }
                    const QCAPRuntimeStats runtimeStats = qcapCapture.GetRuntimeStats();
                    std::cout << " waitTimeouts=" << runtimeStats.waitTimeouts;
                }
                std::cout << std::endl;
            }
        }
        else {
            if (shouldUpdateGuiPreview) {
                if (!lastGuiDisplayFrame.empty()) {
                    liveControlPanel.UpdatePreview(lastGuiDisplayFrame);
                }
                else {
                    cv::Mat placeholder = BuildPreviewPlaceholder(appConfig, detectorReport, useQcap ? &qcapCapture : nullptr, currentFps);
                    liveControlPanel.UpdatePreview(placeholder);
                }
                lastGuiPreviewUpdateTime = std::chrono::high_resolution_clock::now();
            }
        }

        const auto now = std::chrono::high_resolution_clock::now();
        const auto runElapsedMs = std::chrono::duration_cast<std::chrono::milliseconds>(
            now - measuredRunStartTime).count();
        const auto wallClockElapsedMs = std::chrono::duration_cast<std::chrono::milliseconds>(
            now - processStartTime).count();
        if (appConfig.maxFrames > 0 && measuredFrames >= appConfig.maxFrames) {
            break;
        }
        if (warmupCompleted &&
            appConfig.maxSeconds > 0.0 &&
            runElapsedMs >= static_cast<long long>(appConfig.maxSeconds * 1000.0)) {
            break;
        }
        if (!warmupCompleted &&
            appConfig.maxSeconds > 0.0 &&
            wallClockElapsedMs >= static_cast<long long>((appConfig.warmupSeconds + appConfig.maxSeconds) * 1000.0)) {
            if (useQcap && !startupActiveInputFallbackAttempted) {
                    const QCAPRuntimeStats runtimeStats = qcapCapture.GetRuntimeStats();
                    if (runtimeStats.previewCallbacksObserved == 0 &&
                        qcapCapture.TryStartupFallbackToActiveInput()) {
                        startupActiveInputFallbackAttempted = true;
                        frameCount = 0;
                        totalFrames = 0;
                        measuredFrames = 0;
                        emptyCaptureTicks = 0;
                        currentFps = 0.0;
                        processStartTime = now;
                        fpsStartTime = now;
                        measuredRunStartTime = now;
                        runWindowStats.Reset();
                        printedCudaInteropProbe = false;
                        printedDetectorStatsDebug = false;
                        printedMovementError = false;
                        debugGpuDirectRebindTriggered = false;
                        inferenceScreenshotWriteFailed = false;
                        inferenceScreenshotMetadataWriteFailed = false;
                        inferenceScreenshotsWritten = 0;
                        warmupCompleted = !useWarmupWindow;
                        warmupResetDone = false;
                        if (useWarmupWindow) {
                            std::cout << "[Run] startup active-input fallback applied, restarting warmup window." << std::endl;
                        }
                        else {
                            std::cout << "[Run] startup active-input fallback applied, retrying capture loop." << std::endl;
                        }
                        continue;
                    }
                startupActiveInputFallbackAttempted = true;
            }
            std::cout << "[Run] exiting before measured window because no usable frames arrived within "
                << (appConfig.warmupSeconds + appConfig.maxSeconds)
                << " seconds of wall-clock runtime." << std::endl;
            if (useQcap) {
                const QCAPRuntimeStats runtimeStats = qcapCapture.GetRuntimeStats();
                std::cout << "[Run] startup diagnostics previewCallbacks=" << runtimeStats.previewCallbacksObserved
                    << " formatChanges=" << runtimeStats.formatChangesObserved
                    << " noSignal=" << runtimeStats.noSignalEventsObserved
                    << " signalRemoved=" << runtimeStats.signalRemovedEventsObserved
                    << " waitTimeouts=" << runtimeStats.waitTimeouts
                    << " frameViewTimeouts=" << runtimeStats.frameViewTimeouts
                    << " captureTimeouts=" << runtimeStats.captureTimeouts
                    << " callbackGapSamples=" << runtimeStats.callbackGapSamples
                    << " lastCallbackAgeMs=" << runtimeStats.lastCallbackAgeMs
                    << std::endl;
                if (runtimeStats.noSignalEventsObserved > 0) {
                    std::cout << "[Run] startup diagnostics indicate the vendor capture stack reported no signal on the selected input."
                        << std::endl;
                }
                else if (runtimeStats.signalRemovedEventsObserved > 0) {
                    std::cout << "[Run] startup diagnostics indicate the vendor capture stack reported signal removal on the selected input."
                        << std::endl;
                }
                else if (runtimeStats.previewCallbacksObserved == 0) {
                    std::cout << "[Run] startup diagnostics suggest the capture path has not observed any preview callback yet."
                        << std::endl;
                }
                qcapCapture.PrintInputDiagnostics("startup-timeout");
            }
            break;
        }
    }

    if (detector) {
        detector->ReleaseResources();
    }
    liveControlPanel.Shutdown();
    movementController.Release();
    if (useQcap) {
        qcapCapture.Release();
    } else {
        cvCapture.release();
    }

    ::timeEndPeriod(1);
    std::cout << "Program exited." << std::endl;
    return 0;
}
