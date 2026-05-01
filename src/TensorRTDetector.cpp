#include "TensorRTDetector.h"
#include "TensorRTGpuPreprocess.h"
#include "TensorRTGpuPostprocess.h"

#include <cuda_runtime_api.h>
#include <NvInfer.h>
#include <NvInferPlugin.h>
#include <NvOnnxParser.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

namespace {

class TrtLogger final : public nvinfer1::ILogger {
public:
    void log(Severity severity, nvinfer1::AsciiChar const* msg) noexcept override {
        if (severity > Severity::kWARNING || msg == nullptr) {
            return;
        }
        m_messages.emplace_back(msg);
    }

    std::string ConsumeMessages() {
        std::ostringstream oss;
        for (std::size_t i = 0; i < m_messages.size(); ++i) {
            if (i != 0) {
                oss << " | ";
            }
            oss << m_messages[i];
        }
        m_messages.clear();
        return oss.str();
    }

private:
    std::vector<std::string> m_messages;
};

template <typename T>
struct TrtDestroy {
    void operator()(T* ptr) const noexcept {
        delete ptr;
    }
};

using BuilderPtr = std::unique_ptr<nvinfer1::IBuilder, TrtDestroy<nvinfer1::IBuilder>>;
using NetworkPtr = std::unique_ptr<nvinfer1::INetworkDefinition, TrtDestroy<nvinfer1::INetworkDefinition>>;
using ConfigPtr = std::unique_ptr<nvinfer1::IBuilderConfig, TrtDestroy<nvinfer1::IBuilderConfig>>;
using ParserPtr = std::unique_ptr<nvonnxparser::IParser, TrtDestroy<nvonnxparser::IParser>>;
using HostMemoryPtr = std::unique_ptr<nvinfer1::IHostMemory, TrtDestroy<nvinfer1::IHostMemory>>;
using RuntimePtr = std::unique_ptr<nvinfer1::IRuntime, TrtDestroy<nvinfer1::IRuntime>>;
using EnginePtr = std::unique_ptr<nvinfer1::ICudaEngine, TrtDestroy<nvinfer1::ICudaEngine>>;
using ContextPtr = std::unique_ptr<nvinfer1::IExecutionContext, TrtDestroy<nvinfer1::IExecutionContext>>;

std::size_t GetDataTypeSize(nvinfer1::DataType dataType) {
    using nvinfer1::DataType;
    switch (dataType) {
    case DataType::kFLOAT:
        return 4;
    case DataType::kHALF:
        return 2;
    case DataType::kINT32:
        return 4;
    case DataType::kINT8:
        return 1;
    case DataType::kBOOL:
        return 1;
#if NV_TENSORRT_MAJOR >= 10
    case DataType::kUINT8:
        return 1;
    case DataType::kFP8:
        return 1;
    case DataType::kINT64:
        return 8;
    case DataType::kBF16:
        return 2;
#endif
    default:
        return 0;
    }
}

std::size_t GetDimsVolume(const nvinfer1::Dims& dims) {
    if (dims.nbDims < 0) {
        return 0;
    }
    if (dims.nbDims == 0) {
        return 1;
    }

    std::size_t volume = 1;
    for (int i = 0; i < dims.nbDims; ++i) {
        if (dims.d[i] < 0) {
            return 0;
        }
        volume *= static_cast<std::size_t>(dims.d[i]);
    }
    return volume;
}

std::vector<char> ReadBinaryFile(const std::string& path) {
    std::ifstream ifs(path, std::ios::binary);
    if (!ifs) {
        return {};
    }

    ifs.seekg(0, std::ios::end);
    const std::streamoff size = ifs.tellg();
    ifs.seekg(0, std::ios::beg);
    if (size <= 0) {
        return {};
    }

    std::vector<char> data(static_cast<std::size_t>(size));
    ifs.read(data.data(), size);
    if (!ifs) {
        return {};
    }
    return data;
}

bool WriteBinaryFile(const std::string& path, const void* data, std::size_t size) {
    std::ofstream ofs(path, std::ios::binary);
    if (!ofs) {
        return false;
    }
    ofs.write(static_cast<const char*>(data), static_cast<std::streamsize>(size));
    return static_cast<bool>(ofs);
}

struct TrtLetterBoxInfo {
    float scale = 1.0f;
    float padW = 0.0f;
    float padH = 0.0f;
    int resizedWidth = 0;
    int resizedHeight = 0;
    int roiX = 0;
    int roiY = 0;
    int roiWidth = 0;
    int roiHeight = 0;
};

struct RawDetection {
    float x = 0.0f;
    float y = 0.0f;
    float width = 0.0f;
    float height = 0.0f;
    int label = -1;
    float prob = 0.0f;
};

bool LetterBoxPreprocess(
    const unsigned char* imageData,
    int width,
    int height,
    int inputDim,
    std::vector<float>& blob,
    TrtLetterBoxInfo& info,
    const TensorRtRoiSettings* roiSettings = nullptr) {
    if (imageData == nullptr || width <= 0 || height <= 0 || inputDim <= 0) {
        return false;
    }

    int effectiveWidth = width;
    int effectiveHeight = height;
    int roiX = 0;
    int roiY = 0;
    if (roiSettings != nullptr && roiSettings->enabled) {
        roiX = std::clamp(roiSettings->x, 0, width - 1);
        roiY = std::clamp(roiSettings->y, 0, height - 1);
        effectiveWidth = std::clamp(roiSettings->width, 1, width - roiX);
        effectiveHeight = std::clamp(roiSettings->height, 1, height - roiY);
    }

    const float scale = std::min(
        static_cast<float>(inputDim) / static_cast<float>(effectiveWidth),
        static_cast<float>(inputDim) / static_cast<float>(effectiveHeight));

    const int resizedWidth = static_cast<int>(std::round(static_cast<float>(effectiveWidth) * scale));
    const int resizedHeight = static_cast<int>(std::round(static_cast<float>(effectiveHeight) * scale));
    const float padW = (inputDim - resizedWidth) * 0.5f;
    const float padH = (inputDim - resizedHeight) * 0.5f;
    const int padLeft = static_cast<int>(std::round(padW - 0.1f));
    const int padTop = static_cast<int>(std::round(padH - 0.1f));

    info.scale = scale;
    info.padW = static_cast<float>(padLeft);
    info.padH = static_cast<float>(padTop);
    info.resizedWidth = resizedWidth;
    info.resizedHeight = resizedHeight;
    info.roiX = roiX;
    info.roiY = roiY;
    info.roiWidth = effectiveWidth;
    info.roiHeight = effectiveHeight;

    const std::size_t planeSize = static_cast<std::size_t>(inputDim) * static_cast<std::size_t>(inputDim);
    blob.assign(planeSize * 3, 114.0f / 255.0f);

    float* rPlane = blob.data();
    float* gPlane = rPlane + planeSize;
    float* bPlane = gPlane + planeSize;

    const float xRatio = static_cast<float>(effectiveWidth) / static_cast<float>(resizedWidth);
    const float yRatio = static_cast<float>(effectiveHeight) / static_cast<float>(resizedHeight);
    const float norm = 1.0f / 255.0f;

    for (int dstY = 0; dstY < resizedHeight; ++dstY) {
        const float srcY = (static_cast<float>(dstY) + 0.5f) * yRatio - 0.5f;
        const int nearestY = roiY + std::clamp(static_cast<int>(srcY + 0.5f), 0, effectiveHeight - 1);
        const unsigned char* srcRow = imageData + nearestY * width * 3;
        const int dstRowStart = (dstY + padTop) * inputDim + padLeft;

        for (int dstX = 0; dstX < resizedWidth; ++dstX) {
            const float srcX = (static_cast<float>(dstX) + 0.5f) * xRatio - 0.5f;
            const int nearestX = roiX + std::clamp(static_cast<int>(srcX + 0.5f), 0, effectiveWidth - 1);
            const unsigned char* srcPixel = srcRow + nearestX * 3;
            const int dstIndex = dstRowStart + dstX;

            rPlane[dstIndex] = static_cast<float>(srcPixel[2]) * norm;
            gPlane[dstIndex] = static_cast<float>(srcPixel[1]) * norm;
            bPlane[dstIndex] = static_cast<float>(srcPixel[0]) * norm;
        }
    }

    return true;
}

std::string FormatTensorRtRoiRect(int x, int y, int width, int height) {
    std::ostringstream oss;
    oss << x << "," << y << "," << width << "," << height;
    return oss.str();
}

void LogTensorRtRoiActivationOnce(
    bool& logged,
    int frameWidth,
    int frameHeight,
    const TensorRtRoiSettings& requested,
    const TrtLetterBoxInfo& info) {
    if (logged || !requested.enabled || info.roiWidth <= 0 || info.roiHeight <= 0) {
        return;
    }

    logged = true;
    const std::string activeRect = FormatTensorRtRoiRect(
        info.roiX,
        info.roiY,
        info.roiWidth,
        info.roiHeight);
    const std::string requestedRect = FormatTensorRtRoiRect(
        requested.x,
        requested.y,
        requested.width,
        requested.height);

    std::cout << "[TensorRT] ROI request active frame="
        << frameWidth << "x" << frameHeight;
    if (activeRect == requestedRect) {
        std::cout << " roi=" << activeRect;
    }
    else {
        std::cout << " requested=" << requestedRect
            << " active=" << activeRect;
    }
    std::cout << " (cropping source before letterbox)"
        << std::endl;
}

void PopulateTensorRtRoiStats(DetectorFrameStats& stats, const TrtLetterBoxInfo& info) {
    if (info.roiWidth <= 0 || info.roiHeight <= 0) {
        return;
    }

    stats.usedRoi = true;
    stats.roiX = info.roiX;
    stats.roiY = info.roiY;
    stats.roiWidth = info.roiWidth;
    stats.roiHeight = info.roiHeight;
}

float CalculateBoxIoU(
    float ax,
    float ay,
    float aWidth,
    float aHeight,
    float bx,
    float by,
    float bWidth,
    float bHeight) {
    const float left = std::max(ax, bx);
    const float top = std::max(ay, by);
    const float right = std::min(ax + aWidth, bx + bWidth);
    const float bottom = std::min(ay + aHeight, by + bHeight);

    const float overlap = std::max(0.0f, right - left) * std::max(0.0f, bottom - top);
    if (overlap <= 0.0f) {
        return 0.0f;
    }

    const float areaA = std::max(0.0f, aWidth) * std::max(0.0f, aHeight);
    const float areaB = std::max(0.0f, bWidth) * std::max(0.0f, bHeight);
    return overlap / (areaA + areaB - overlap);
}

float CalculateIoU(const RawDetection& a, const RawDetection& b) {
    return CalculateBoxIoU(a.x, a.y, a.width, a.height, b.x, b.y, b.width, b.height);
}

float CalculateIoU(const DetectionObject& a, const DetectionObject& b) {
    return CalculateBoxIoU(
        a.bbox.x,
        a.bbox.y,
        a.bbox.width,
        a.bbox.height,
        b.bbox.x,
        b.bbox.y,
        b.bbox.width,
        b.bbox.height);
}

std::vector<RawDetection> ApplyNms(std::vector<RawDetection> detections, float threshold) {
    std::stable_sort(detections.begin(), detections.end(), [](const RawDetection& lhs, const RawDetection& rhs) {
        return lhs.prob > rhs.prob;
    });

    std::vector<RawDetection> kept;
    std::vector<bool> suppressed(detections.size(), false);

    for (std::size_t i = 0; i < detections.size(); ++i) {
        if (suppressed[i]) {
            continue;
        }

        kept.push_back(detections[i]);
        for (std::size_t j = i + 1; j < detections.size(); ++j) {
            if (suppressed[j] || detections[i].label != detections[j].label) {
                continue;
            }

            if (CalculateIoU(detections[i], detections[j]) >= threshold) {
                suppressed[j] = true;
            }
        }
    }

    return kept;
}

std::vector<DetectionObject> ApplyNms(std::vector<DetectionObject> detections, float threshold) {
    std::stable_sort(detections.begin(), detections.end(), [](const DetectionObject& lhs, const DetectionObject& rhs) {
        return lhs.prob > rhs.prob;
    });

    std::vector<DetectionObject> kept;
    std::vector<bool> suppressed(detections.size(), false);

    for (std::size_t i = 0; i < detections.size(); ++i) {
        if (suppressed[i]) {
            continue;
        }

        kept.push_back(detections[i]);
        for (std::size_t j = i + 1; j < detections.size(); ++j) {
            if (suppressed[j] || detections[i].label != detections[j].label) {
                continue;
            }

            if (CalculateIoU(detections[i], detections[j]) >= threshold) {
                suppressed[j] = true;
            }
        }
    }

    return kept;
}

void ScaleBox(float& x, float& y, float& width, float& height, const TrtLetterBoxInfo& info, int originalWidth, int originalHeight) {
    x = (x - info.padW) / info.scale;
    y = (y - info.padH) / info.scale;
    width /= info.scale;
    height /= info.scale;
    x += static_cast<float>(info.roiX);
    y += static_cast<float>(info.roiY);

    x = std::clamp(x, 0.0f, static_cast<float>(originalWidth - 1));
    y = std::clamp(y, 0.0f, static_cast<float>(originalHeight - 1));
    width = std::clamp(width, 1.0f, static_cast<float>(originalWidth) - x);
    height = std::clamp(height, 1.0f, static_cast<float>(originalHeight) - y);
}

void ScaleBoxes(std::vector<RawDetection>& detections, const TrtLetterBoxInfo& info, int originalWidth, int originalHeight) {
    for (auto& det : detections) {
        ScaleBox(det.x, det.y, det.width, det.height, info, originalWidth, originalHeight);
    }
}

void ScaleBoxes(std::vector<DetectionObject>& detections, const TrtLetterBoxInfo& info, int originalWidth, int originalHeight) {
    for (auto& det : detections) {
        ScaleBox(det.bbox.x, det.bbox.y, det.bbox.width, det.bbox.height, info, originalWidth, originalHeight);
    }
}

void ClampBoxes(std::vector<DetectionObject>& detections, int originalWidth, int originalHeight) {
    for (auto& det : detections) {
        det.bbox.x = std::clamp(det.bbox.x, 0.0f, static_cast<float>(originalWidth - 1));
        det.bbox.y = std::clamp(det.bbox.y, 0.0f, static_cast<float>(originalHeight - 1));
        det.bbox.width = std::clamp(det.bbox.width, 1.0f, static_cast<float>(originalWidth) - det.bbox.x);
        det.bbox.height = std::clamp(det.bbox.height, 1.0f, static_cast<float>(originalHeight) - det.bbox.y);
    }
}

std::vector<DetectionObject> DecodeYoloOutput(
    const float* output,
    const std::vector<int64_t>& outputDims,
    const TrtLetterBoxInfo& info,
    int originalWidth,
    int originalHeight,
    float nms,
    float conf,
    int* proposalsBeforeNms = nullptr,
    bool applyCpuNms = true,
    bool boxesAlreadyScaled = false) {
    std::vector<DetectionObject> results;
    if (output == nullptr || outputDims.size() < 3) {
        if (proposalsBeforeNms != nullptr) {
            *proposalsBeforeNms = 0;
        }
        return results;
    }

    const int numBoxes = static_cast<int>(outputDims[1]);
    const int stride = static_cast<int>(outputDims[2]);
    if (numBoxes <= 0 || stride <= 5) {
        if (proposalsBeforeNms != nullptr) {
            *proposalsBeforeNms = 0;
        }
        return results;
    }

    if (stride == 6) {
        results.reserve(static_cast<std::size_t>(numBoxes));
        for (int boxIdx = 0; boxIdx < numBoxes; ++boxIdx) {
            const int base = boxIdx * stride;
            const float score = output[base + 5];
            if (score <= conf) {
                continue;
            }

            DetectionObject det{};
            det.bbox.x = output[base + 0];
            det.bbox.y = output[base + 1];
            det.bbox.width = output[base + 2];
            det.bbox.height = output[base + 3];
            det.label = static_cast<int>(std::round(output[base + 4]));
            det.prob = score;
            results.push_back(det);
        }
        if (proposalsBeforeNms != nullptr) {
            *proposalsBeforeNms = static_cast<int>(results.size());
        }

        if (applyCpuNms) {
            results = ApplyNms(std::move(results), nms);
        }
        if (boxesAlreadyScaled) {
            ClampBoxes(results, originalWidth, originalHeight);
        }
        else {
            ScaleBoxes(results, info, originalWidth, originalHeight);
        }
        return results;
    }

    std::vector<RawDetection> proposals;
    proposals.reserve(static_cast<std::size_t>(numBoxes));
    {
        for (int boxIdx = 0; boxIdx < numBoxes; ++boxIdx) {
            const int base = boxIdx * stride;
            const float objectness = output[base + 4];
            if (objectness <= conf) {
                continue;
            }

            for (int cls = 5; cls < stride; ++cls) {
                const float score = objectness * output[base + cls];
                if (score <= conf) {
                    continue;
                }

                RawDetection det;
                det.x = output[base + 0] - output[base + 2] * 0.5f;
                det.y = output[base + 1] - output[base + 3] * 0.5f;
                det.width = output[base + 2];
                det.height = output[base + 3];
                det.label = cls - 5;
                det.prob = score;
                proposals.push_back(det);
            }
        }
    }

    if (proposalsBeforeNms != nullptr) {
        *proposalsBeforeNms = static_cast<int>(proposals.size());
    }

    if (applyCpuNms) {
        proposals = ApplyNms(std::move(proposals), nms);
    }
    ScaleBoxes(proposals, info, originalWidth, originalHeight);

    results.reserve(proposals.size());
    for (const auto& det : proposals) {
        DetectionObject out{};
        out.bbox.x = det.x;
        out.bbox.y = det.y;
        out.bbox.width = det.width;
        out.bbox.height = det.height;
        out.label = det.label;
        out.prob = det.prob;
        results.push_back(out);
    }

    return results;
}

float GetCudaElapsedMilliseconds(cudaEvent_t start, cudaEvent_t end) {
    if (start == nullptr || end == nullptr) {
        return 0.0f;
    }

    float elapsedMs = 0.0f;
    const cudaError_t result = cudaEventElapsedTime(&elapsedMs, start, end);
    if (result != cudaSuccess) {
        cudaGetLastError();
        return 0.0f;
    }

    return elapsedMs;
}

std::string GetParentDirectory(const std::string& path) {
    std::error_code ec;
    return std::filesystem::path(path).parent_path().string();
}

std::string FormatCudaError(const char* operation, cudaError_t error) {
    std::ostringstream oss;
    oss << operation << " failed";
    if (error != cudaSuccess) {
        oss << " (" << cudaGetErrorName(error) << ": " << cudaGetErrorString(error) << ")";
    }
    return oss.str();
}

std::string FormatDimsVector(const std::vector<int64_t>& dims) {
    std::ostringstream oss;
    oss << "[";
    for (std::size_t i = 0; i < dims.size(); ++i) {
        if (i != 0) {
            oss << ",";
        }
        oss << dims[i];
    }
    oss << "]";
    return oss.str();
}

std::string FormatTensorLocation(nvinfer1::TensorLocation location) {
    switch (location) {
    case nvinfer1::TensorLocation::kDEVICE:
        return "device";
    case nvinfer1::TensorLocation::kHOST:
        return "host";
    default:
        return "unknown";
    }
}

std::string FormatDataType(nvinfer1::DataType dataType) {
    using nvinfer1::DataType;
    switch (dataType) {
    case DataType::kFLOAT:
        return "float";
    case DataType::kHALF:
        return "half";
    case DataType::kINT32:
        return "int32";
    case DataType::kINT8:
        return "int8";
    case DataType::kBOOL:
        return "bool";
#if NV_TENSORRT_MAJOR >= 10
    case DataType::kUINT8:
        return "uint8";
    case DataType::kFP8:
        return "fp8";
    case DataType::kINT64:
        return "int64";
    case DataType::kBF16:
        return "bf16";
#endif
    default:
        return "unknown";
    }
}

std::string BuildTrtRuntimeFailureMessage(TrtLogger& logger, const char* operation) {
    std::ostringstream oss;
    oss << operation << " failed";

    const std::string logMessages = logger.ConsumeMessages();
    if (!logMessages.empty()) {
        oss << " | " << logMessages;
    }

    const cudaError_t lastCudaError = cudaPeekAtLastError();
    if (lastCudaError != cudaSuccess) {
        oss << " | cudaLastError=" << cudaGetErrorName(lastCudaError)
            << ": " << cudaGetErrorString(lastCudaError);
        cudaGetLastError();
    }

    return oss.str();
}

} // namespace

struct TensorRTDetector::Impl {
    struct OutputTensorBinding {
        std::string name;
        std::vector<int64_t> dims;
        nvinfer1::DataType dataType = nvinfer1::DataType::kFLOAT;
        nvinfer1::TensorLocation location = nvinfer1::TensorLocation::kDEVICE;
        std::size_t elementCount = 0;
        std::size_t byteSize = 0;
        void* buffer = nullptr;
    };

    TrtLogger logger;
    RuntimePtr runtime;
    EnginePtr engine;
    ContextPtr context;
    cudaStream_t stream = nullptr;
    cudaEvent_t eventStart = nullptr;
    cudaEvent_t eventAfterInput = nullptr;
    cudaEvent_t eventAfterInfer = nullptr;
    cudaEvent_t eventAfterGpuPostprocess = nullptr;
    cudaEvent_t eventAfterD2H = nullptr;

    std::string inputTensorName;
    std::string outputTensorName;
    std::vector<int64_t> inputDims;
    std::vector<int64_t> outputDims;
    std::size_t inputElementCount = 0;
    std::size_t outputElementCount = 0;
    int inputWidth = 0;
    int inputHeight = 0;
    TensorRTGpuPreprocess gpuPreprocess;
    TensorRTGpuPostprocess gpuPostprocess;
    bool gpuPreprocessReady = false;
    bool gpuPostprocessReady = false;
    bool loggedGpuPreprocessActive = false;
    bool loggedGpuPreprocessFallback = false;
    bool loggedGpuPostprocessActive = false;
    bool loggedGpuPostprocessFallback = false;
    bool loggedFrameStatsReady = false;
    bool loggedFrameStatsFailure = false;
    bool loggedRoiActivation = false;
    int candidateTopK = 0;
    TensorRtRoiSettings roiSettings;

    void* deviceInput = nullptr;
    void* deviceOutput = nullptr;
    std::vector<float> hostInput;
    std::vector<float> hostOutput;
    std::vector<float> hostCompactedOutput;
    std::vector<OutputTensorBinding> auxiliaryOutputs;
    int hostCompactedCount = 0;
    DetectorFrameStats lastFrameStats;

    void Reset() {
        if (stream != nullptr) {
            const cudaError_t syncResult = cudaStreamSynchronize(stream);
            if (syncResult != cudaSuccess) {
                cudaGetLastError();
            }
        }

        for (auto& output : auxiliaryOutputs) {
            if (output.buffer == nullptr) {
                continue;
            }

            if (output.location == nvinfer1::TensorLocation::kDEVICE) {
                cudaFree(output.buffer);
            }
            else {
                cudaFreeHost(output.buffer);
            }
            output.buffer = nullptr;
        }
        auxiliaryOutputs.clear();
        if (deviceInput != nullptr) {
            cudaFree(deviceInput);
            deviceInput = nullptr;
        }
        if (deviceOutput != nullptr) {
            cudaFree(deviceOutput);
            deviceOutput = nullptr;
        }
        if (eventAfterD2H != nullptr) {
            cudaEventDestroy(eventAfterD2H);
            eventAfterD2H = nullptr;
        }
        if (eventAfterGpuPostprocess != nullptr) {
            cudaEventDestroy(eventAfterGpuPostprocess);
            eventAfterGpuPostprocess = nullptr;
        }
        if (eventAfterInfer != nullptr) {
            cudaEventDestroy(eventAfterInfer);
            eventAfterInfer = nullptr;
        }
        if (eventAfterInput != nullptr) {
            cudaEventDestroy(eventAfterInput);
            eventAfterInput = nullptr;
        }
        if (eventStart != nullptr) {
            cudaEventDestroy(eventStart);
            eventStart = nullptr;
        }
        if (stream != nullptr) {
            cudaStreamDestroy(stream);
            stream = nullptr;
        }
        context.reset();
        engine.reset();
        runtime.reset();
        inputTensorName.clear();
        outputTensorName.clear();
        inputDims.clear();
        outputDims.clear();
        inputElementCount = 0;
        outputElementCount = 0;
        inputWidth = 0;
        inputHeight = 0;
        hostInput.clear();
        hostOutput.clear();
        hostCompactedOutput.clear();
        hostCompactedCount = 0;
        gpuPreprocess.Reset();
        gpuPostprocess.Reset();
        gpuPreprocessReady = false;
        gpuPostprocessReady = false;
        loggedGpuPreprocessActive = false;
        loggedGpuPreprocessFallback = false;
        loggedGpuPostprocessActive = false;
        loggedGpuPostprocessFallback = false;
        loggedFrameStatsReady = false;
        loggedFrameStatsFailure = false;
        loggedRoiActivation = false;
        candidateTopK = 0;
        roiSettings = TensorRtRoiSettings{};
        lastFrameStats = DetectorFrameStats{};
    }
};

TensorRTDetector::TensorRTDetector()
    : m_impl(std::make_unique<Impl>()) {
}

TensorRTDetector::~TensorRTDetector() {
    ReleaseResources();
}

bool TensorRTDetector::Initialize(const DetectorSettings& settings) {
    ReleaseResources();

    auto& impl = *m_impl;
    impl.logger.ConsumeMessages();
    impl.candidateTopK = std::max(0, settings.tensorRtCandidateTopK);
    impl.roiSettings = settings.tensorRtRoi;

    const char* runtimePolicy = settings.allowTensorRtEngineBuild
        ? "development-build-enabled"
        : "offline-engine-only";

    std::cout << "[TensorRT] Initializing with model=" << settings.modelPath
        << " engine=" << settings.enginePath
        << " device=" << settings.deviceIndex
        << " runtimePolicy=" << runtimePolicy
        << std::endl;
    if (impl.roiSettings.enabled) {
        std::cout << "[TensorRT] ROI configured x=" << impl.roiSettings.x
            << " y=" << impl.roiSettings.y
            << " w=" << impl.roiSettings.width
            << " h=" << impl.roiSettings.height
            << " (crop-and-rescale active)"
            << std::endl;
    }

    if (settings.deviceIndex >= 0) {
        const cudaError_t cudaResult = cudaSetDevice(settings.deviceIndex);
        if (cudaResult != cudaSuccess) {
            m_lastError = FormatCudaError("cudaSetDevice", cudaResult);
            return false;
        }
    }

    if (!initLibNvInferPlugins(&impl.logger, "")) {
        m_lastError = "initLibNvInferPlugins failed";
        return false;
    }

    std::vector<char> engineData = ReadBinaryFile(settings.enginePath);
    if (engineData.empty()) {
        if (!settings.allowTensorRtEngineBuild) {
            std::ostringstream oss;
            oss << "TensorRT engine file was not found at '" << settings.enginePath
                << "'. Runtime policy is offline-engine-only, so online ONNX->engine build is disabled."
                << " Offline engine generation is required for the runtime path."
                << " Generate the engine ahead of time and pass --engine=<plan_path>."
                << " Use --allow-trt-build only for development-time engine creation.";
            m_lastError = oss.str();
            return false;
        }

        std::cout << "[TensorRT] No cached engine found. Building from ONNX because"
            << " runtimePolicy=development-build-enabled." << std::endl;
        const auto buildStart = std::chrono::steady_clock::now();

        BuilderPtr builder(nvinfer1::createInferBuilder(impl.logger));
        if (!builder) {
            m_lastError = "createInferBuilder failed";
            return false;
        }

        const auto networkFlags = static_cast<nvinfer1::NetworkDefinitionCreationFlags>(
            1U << static_cast<int>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH));
        NetworkPtr network(builder->createNetworkV2(networkFlags));
        if (!network) {
            m_lastError = "createNetworkV2 failed";
            return false;
        }

        ParserPtr parser(nvonnxparser::createParser(*network, impl.logger));
        if (!parser) {
            m_lastError = "createParser failed";
            return false;
        }

        if (!parser->parseFromFile(settings.modelPath.c_str(), static_cast<int>(nvinfer1::ILogger::Severity::kWARNING))) {
            std::ostringstream oss;
            oss << "TensorRT ONNX parse failed";
            const int nbErrors = parser->getNbErrors();
            for (int i = 0; i < nbErrors; ++i) {
                const auto* error = parser->getError(i);
                if (error == nullptr) {
                    continue;
                }
                oss << " | " << error->desc();
            }
            const std::string logMessages = impl.logger.ConsumeMessages();
            if (!logMessages.empty()) {
                oss << " | " << logMessages;
            }
            m_lastError = oss.str();
            return false;
        }

        const auto parseDone = std::chrono::steady_clock::now();
        std::cout << "[TensorRT] ONNX parse completed in "
            << std::chrono::duration_cast<std::chrono::milliseconds>(parseDone - buildStart).count()
            << " ms" << std::endl;

        ConfigPtr config(builder->createBuilderConfig());
        if (!config) {
            m_lastError = "createBuilderConfig failed";
            return false;
        }

        config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, 1ULL << 30);
        if (builder->platformHasFastFp16()) {
            config->setFlag(nvinfer1::BuilderFlag::kFP16);
        }

        HostMemoryPtr serialized(builder->buildSerializedNetwork(*network, *config));
        if (!serialized) {
            std::ostringstream oss;
            oss << "buildSerializedNetwork failed";
            const std::string logMessages = impl.logger.ConsumeMessages();
            if (!logMessages.empty()) {
                oss << " | " << logMessages;
            }
            m_lastError = oss.str();
            return false;
        }

        const auto buildDone = std::chrono::steady_clock::now();
        std::cout << "[TensorRT] Engine build completed in "
            << std::chrono::duration_cast<std::chrono::milliseconds>(buildDone - parseDone).count()
            << " ms" << std::endl;

        if (!WriteBinaryFile(settings.enginePath, serialized->data(), serialized->size())) {
            m_lastError = "Failed to write TensorRT engine file: " + settings.enginePath;
            return false;
        }

        std::cout << "[TensorRT] Cached engine written to " << settings.enginePath << std::endl;
        engineData.assign(
            static_cast<const char*>(serialized->data()),
            static_cast<const char*>(serialized->data()) + serialized->size());
    }
    else {
        std::cout << "[TensorRT] Loading cached engine from " << settings.enginePath << std::endl;
    }

    impl.runtime.reset(nvinfer1::createInferRuntime(impl.logger));
    if (!impl.runtime) {
        m_lastError = "createInferRuntime failed";
        return false;
    }

    impl.engine.reset(impl.runtime->deserializeCudaEngine(engineData.data(), engineData.size()));
    if (!impl.engine) {
        std::ostringstream oss;
        oss << "deserializeCudaEngine failed";
        const std::string logMessages = impl.logger.ConsumeMessages();
        if (!logMessages.empty()) {
            oss << " | " << logMessages;
        }
        m_lastError = oss.str();
        return false;
    }

    impl.context.reset(impl.engine->createExecutionContext());
    if (!impl.context) {
        m_lastError = "createExecutionContext failed";
        return false;
    }

    std::string preferredOutputTensorName;
    std::vector<int64_t> preferredOutputDims;
    std::string fallbackOutputTensorName;
    std::vector<int64_t> fallbackOutputDims;
    std::vector<Impl::OutputTensorBinding> discoveredOutputs;
    const int numIOTensors = impl.engine->getNbIOTensors();
    for (int i = 0; i < numIOTensors; ++i) {
        const char* tensorName = impl.engine->getIOTensorName(i);
        if (tensorName == nullptr) {
            continue;
        }

        const auto ioMode = impl.engine->getTensorIOMode(tensorName);
        const auto dims = impl.engine->getTensorShape(tensorName);
        std::vector<int64_t> dimsVector;
        dimsVector.reserve(dims.nbDims);
        for (int d = 0; d < dims.nbDims; ++d) {
            dimsVector.push_back(dims.d[d]);
        }

        if (ioMode == nvinfer1::TensorIOMode::kINPUT && impl.inputTensorName.empty()) {
            impl.inputTensorName = tensorName;
            impl.inputDims = std::move(dimsVector);
        }
        else if (ioMode == nvinfer1::TensorIOMode::kOUTPUT) {
            Impl::OutputTensorBinding output{};
            output.name = tensorName;
            output.dims = dimsVector;
            output.dataType = impl.engine->getTensorDataType(tensorName);
            output.location = impl.engine->getTensorLocation(tensorName);
            output.elementCount = GetDimsVolume(dims);
            output.byteSize = output.elementCount * GetDataTypeSize(output.dataType);
            discoveredOutputs.push_back(output);

            const bool looksLikeDecodedDetections =
                output.dims.size() == 3 &&
                output.dims[1] > 0 &&
                output.dims[2] > 5;

            if (std::string(tensorName) == "output0" && looksLikeDecodedDetections) {
                preferredOutputTensorName = tensorName;
                preferredOutputDims = output.dims;
            }
            else if (fallbackOutputTensorName.empty() && looksLikeDecodedDetections) {
                fallbackOutputTensorName = tensorName;
                fallbackOutputDims = output.dims;
            }
        }
    }

    if (!preferredOutputTensorName.empty()) {
        impl.outputTensorName = preferredOutputTensorName;
        impl.outputDims = std::move(preferredOutputDims);
    }
    else if (!fallbackOutputTensorName.empty()) {
        impl.outputTensorName = fallbackOutputTensorName;
        impl.outputDims = std::move(fallbackOutputDims);
    }

    if (impl.inputTensorName.empty() || impl.outputTensorName.empty()) {
        m_lastError = "TensorRT engine input/output tensor discovery failed";
        return false;
    }

    if (impl.inputDims.size() != 4 || impl.inputDims[1] != 3) {
        m_lastError = "Only NCHW 3-channel TensorRT inputs are supported in this stage";
        return false;
    }

    impl.inputHeight = static_cast<int>(impl.inputDims[2]);
    impl.inputWidth = static_cast<int>(impl.inputDims[3]);
    impl.inputElementCount = GetDimsVolume(impl.engine->getTensorShape(impl.inputTensorName.c_str()));
    Impl::OutputTensorBinding* primaryOutput = nullptr;
    for (auto& output : discoveredOutputs) {
        if (output.name == impl.outputTensorName) {
            primaryOutput = &output;
            break;
        }
    }

    if (primaryOutput == nullptr) {
        m_lastError = "Primary TensorRT output tensor metadata discovery failed";
        return false;
    }

    if (primaryOutput->location != nvinfer1::TensorLocation::kDEVICE) {
        std::ostringstream oss;
        oss << "Primary TensorRT output must be device-resident in this stage: "
            << primaryOutput->name << " location=" << FormatTensorLocation(primaryOutput->location);
        m_lastError = oss.str();
        return false;
    }

    if (primaryOutput->dataType != nvinfer1::DataType::kFLOAT) {
        std::ostringstream oss;
        oss << "Primary TensorRT output must be float in this stage: "
            << primaryOutput->name << " type=" << FormatDataType(primaryOutput->dataType);
        m_lastError = oss.str();
        return false;
    }

    impl.outputElementCount = primaryOutput->elementCount;
    if (impl.inputElementCount == 0 || impl.outputElementCount == 0) {
        m_lastError = "TensorRT engine has unsupported dynamic tensor shapes in this stage";
        return false;
    }

    impl.hostInput.resize(impl.inputElementCount);
    impl.hostOutput.resize(impl.outputElementCount);

    const cudaError_t streamResult = cudaStreamCreate(&impl.stream);
    if (streamResult != cudaSuccess) {
        m_lastError = FormatCudaError("cudaStreamCreate", streamResult);
        return false;
    }

    TensorRTGpuPreprocessConfig preprocessConfig{};
    preprocessConfig.inputWidth = impl.inputWidth;
    preprocessConfig.inputHeight = impl.inputHeight;
    preprocessConfig.stream = impl.stream;
    std::string preprocessError;
    impl.gpuPreprocessReady = impl.gpuPreprocess.Initialize(preprocessConfig, preprocessError);
    if (!impl.gpuPreprocessReady && !preprocessError.empty()) {
        std::cout << "[TensorRT] GPU preprocess unavailable: " << preprocessError << std::endl;
    }

    if (impl.outputDims.size() >= 3) {
        TensorRTGpuPostprocessConfig postprocessConfig{};
        postprocessConfig.maxRows = static_cast<int>(impl.outputDims[1]);
        postprocessConfig.stride = static_cast<int>(impl.outputDims[2]);
        postprocessConfig.compactStride = 6;
        postprocessConfig.topK = impl.candidateTopK;
        postprocessConfig.stream = impl.stream;
        std::string postprocessError;
        impl.gpuPostprocessReady = impl.gpuPostprocess.Initialize(postprocessConfig, postprocessError);
        if (!impl.gpuPostprocessReady && !postprocessError.empty()) {
            std::cout << "[TensorRT] GPU postprocess unavailable: " << postprocessError << std::endl;
        }
        else if (impl.gpuPostprocessReady) {
            impl.hostCompactedOutput.resize(
                static_cast<std::size_t>(postprocessConfig.maxRows) * static_cast<std::size_t>(postprocessConfig.compactStride));
        }
    }

    const cudaError_t inputAllocResult = cudaMalloc(&impl.deviceInput, impl.hostInput.size() * sizeof(float));
    if (inputAllocResult != cudaSuccess) {
        m_lastError = FormatCudaError("cudaMalloc(deviceInput)", inputAllocResult);
        return false;
    }

    const cudaError_t outputAllocResult = cudaMalloc(&impl.deviceOutput, impl.hostOutput.size() * sizeof(float));
    if (outputAllocResult != cudaSuccess) {
        m_lastError = FormatCudaError("cudaMalloc(deviceOutput)", outputAllocResult);
        return false;
    }

    if (!impl.context->setInputTensorAddress(impl.inputTensorName.c_str(), impl.deviceInput)) {
        m_lastError = "setInputTensorAddress failed";
        return false;
    }

    if (!impl.context->setOutputTensorAddress(impl.outputTensorName.c_str(), impl.deviceOutput)) {
        m_lastError = "setOutputTensorAddress failed";
        return false;
    }

    for (const auto& output : discoveredOutputs) {
        if (output.name == impl.outputTensorName) {
            continue;
        }

        if (output.byteSize == 0) {
            std::ostringstream oss;
            oss << "Auxiliary TensorRT output has unsupported shape/type: "
                << output.name << " dims=" << FormatDimsVector(output.dims)
                << " type=" << FormatDataType(output.dataType);
            m_lastError = oss.str();
            return false;
        }

        Impl::OutputTensorBinding binding = output;
        cudaError_t allocResult = cudaSuccess;
        if (binding.location == nvinfer1::TensorLocation::kDEVICE) {
            allocResult = cudaMalloc(&binding.buffer, binding.byteSize);
        }
        else {
            allocResult = cudaMallocHost(&binding.buffer, binding.byteSize);
        }

        if (allocResult != cudaSuccess) {
            std::ostringstream oss;
            oss << "Failed to allocate TensorRT auxiliary output buffer for " << binding.name
                << " (" << FormatTensorLocation(binding.location) << ", "
                << binding.byteSize << " bytes): "
                << cudaGetErrorName(allocResult) << ": " << cudaGetErrorString(allocResult);
            m_lastError = oss.str();
            return false;
        }

        if (!impl.context->setOutputTensorAddress(binding.name.c_str(), binding.buffer)) {
            std::ostringstream oss;
            oss << "setOutputTensorAddress failed for auxiliary output " << binding.name;
            m_lastError = oss.str();
            return false;
        }

        impl.auxiliaryOutputs.push_back(std::move(binding));
    }

    const cudaError_t eventStartResult = cudaEventCreate(&impl.eventStart);
    if (eventStartResult != cudaSuccess) {
        m_lastError = FormatCudaError("cudaEventCreate(eventStart)", eventStartResult);
        return false;
    }
    const cudaError_t eventAfterInputResult = cudaEventCreate(&impl.eventAfterInput);
    if (eventAfterInputResult != cudaSuccess) {
        m_lastError = FormatCudaError("cudaEventCreate(eventAfterInput)", eventAfterInputResult);
        return false;
    }
    const cudaError_t eventAfterInferResult = cudaEventCreate(&impl.eventAfterInfer);
    if (eventAfterInferResult != cudaSuccess) {
        m_lastError = FormatCudaError("cudaEventCreate(eventAfterInfer)", eventAfterInferResult);
        return false;
    }
    const cudaError_t eventAfterGpuPostprocessResult = cudaEventCreate(&impl.eventAfterGpuPostprocess);
    if (eventAfterGpuPostprocessResult != cudaSuccess) {
        m_lastError = FormatCudaError("cudaEventCreate(eventAfterGpuPostprocess)", eventAfterGpuPostprocessResult);
        return false;
    }
    const cudaError_t eventAfterD2HResult = cudaEventCreate(&impl.eventAfterD2H);
    if (eventAfterD2HResult != cudaSuccess) {
        m_lastError = FormatCudaError("cudaEventCreate(eventAfterD2H)", eventAfterD2HResult);
        return false;
    }

    std::cout << "[TensorRT] Engine ready"
        << " input=" << impl.inputTensorName << " " << impl.inputWidth << "x" << impl.inputHeight
        << " output=" << impl.outputTensorName
        << " dims=" << FormatDimsVector(impl.outputDims)
        << " totalOutputs=" << discoveredOutputs.size()
        << std::endl;
    for (const auto& output : impl.auxiliaryOutputs) {
        std::cout << "[TensorRT] Bound auxiliary output name=" << output.name
            << " dims=" << FormatDimsVector(output.dims)
            << " type=" << FormatDataType(output.dataType)
            << " location=" << FormatTensorLocation(output.location)
            << " bytes=" << output.byteSize
            << std::endl;
    }

    m_lastError.clear();
    return true;
}

std::vector<DetectionObject> TensorRTDetector::DetectBGR(
    const unsigned char* imageData,
    int width,
    int height,
    float nms,
    float conf) {
    auto& impl = *m_impl;
    impl.lastFrameStats = DetectorFrameStats{};
    if (imageData == nullptr || width <= 0 || height <= 0 || !impl.context || !impl.engine) {
        impl.lastFrameStats.failureStage = "invalid-input";
        return {};
    }

    const auto totalStart = std::chrono::steady_clock::now();
    TrtLetterBoxInfo info;
    const auto preprocessStart = std::chrono::steady_clock::now();
    if (!LetterBoxPreprocess(
            imageData,
            width,
            height,
            impl.inputWidth,
            impl.hostInput,
            info,
            &impl.roiSettings)) {
        impl.lastFrameStats.failureStage = "cpu-preprocess";
        return {};
    }
    if (impl.roiSettings.enabled) {
        LogTensorRtRoiActivationOnce(impl.loggedRoiActivation, width, height, impl.roiSettings, info);
    }
    const auto preprocessEnd = std::chrono::steady_clock::now();
    impl.lastFrameStats.preprocessMs =
        std::chrono::duration<double, std::milli>(preprocessEnd - preprocessStart).count();

    const std::size_t inputBytes = impl.hostInput.size() * sizeof(float);
    const std::size_t outputBytes = impl.hostOutput.size() * sizeof(float);
    impl.lastFrameStats.outputRowsBeforeFilter = impl.outputDims.size() >= 2 ? static_cast<int>(impl.outputDims[1]) : 0;
    impl.lastFrameStats.outputRowsAfterFilter = impl.lastFrameStats.outputRowsBeforeFilter;
    impl.lastFrameStats.d2hBytes = static_cast<std::uint64_t>(outputBytes);

    cudaEventRecord(impl.eventStart, impl.stream);

    if (cudaMemcpyAsync(impl.deviceInput, impl.hostInput.data(), inputBytes, cudaMemcpyHostToDevice, impl.stream) != cudaSuccess) {
        impl.lastFrameStats.failureStage = "h2d";
        return {};
    }
    cudaEventRecord(impl.eventAfterInput, impl.stream);

    if (!impl.context->enqueueV3(impl.stream)) {
        impl.lastFrameStats.failureStage = "enqueue";
        m_lastError = BuildTrtRuntimeFailureMessage(impl.logger, "enqueueV3");
        if (!impl.loggedFrameStatsFailure) {
            impl.loggedFrameStatsFailure = true;
            std::cout << "[TensorRT] " << m_lastError << std::endl;
        }
        return {};
    }
    cudaEventRecord(impl.eventAfterInfer, impl.stream);

    if (cudaMemcpyAsync(impl.hostOutput.data(), impl.deviceOutput, outputBytes, cudaMemcpyDeviceToHost, impl.stream) != cudaSuccess) {
        impl.lastFrameStats.failureStage = "d2h";
        return {};
    }
    cudaEventRecord(impl.eventAfterD2H, impl.stream);

    if (cudaStreamSynchronize(impl.stream) != cudaSuccess) {
        impl.lastFrameStats.failureStage = "stream-sync";
        return {};
    }

    impl.lastFrameStats.h2dMs = GetCudaElapsedMilliseconds(impl.eventStart, impl.eventAfterInput);
    impl.lastFrameStats.inferMs = GetCudaElapsedMilliseconds(impl.eventAfterInput, impl.eventAfterInfer);
    impl.lastFrameStats.d2hMs = GetCudaElapsedMilliseconds(impl.eventAfterInfer, impl.eventAfterD2H);

    int proposalsBeforeNms = 0;
    const auto decodeStart = std::chrono::steady_clock::now();
    std::vector<DetectionObject> results = DecodeYoloOutput(
        impl.hostOutput.data(),
        impl.outputDims,
        info,
        width,
        height,
        nms,
        conf,
        &proposalsBeforeNms);
    const auto decodeEnd = std::chrono::steady_clock::now();

    impl.lastFrameStats.decodeMs =
        std::chrono::duration<double, std::milli>(decodeEnd - decodeStart).count();
    impl.lastFrameStats.proposalsBeforeNms = proposalsBeforeNms;
    impl.lastFrameStats.resultsAfterNms = static_cast<int>(results.size());
    PopulateTensorRtRoiStats(impl.lastFrameStats, info);
    impl.lastFrameStats.totalMs =
        std::chrono::duration<double, std::milli>(decodeEnd - totalStart).count();
    impl.lastFrameStats.valid = true;
    if (!impl.loggedFrameStatsReady) {
        impl.loggedFrameStatsReady = true;
        std::cout << "[TensorRT] frameStats ready path=BGR totalMs=" << impl.lastFrameStats.totalMs
            << " proposals=" << impl.lastFrameStats.proposalsBeforeNms
            << " results=" << impl.lastFrameStats.resultsAfterNms
            << " roi=" << (impl.lastFrameStats.usedRoi
                ? (std::to_string(impl.lastFrameStats.roiX) + "," +
                   std::to_string(impl.lastFrameStats.roiY) + "," +
                   std::to_string(impl.lastFrameStats.roiWidth) + "," +
                   std::to_string(impl.lastFrameStats.roiHeight))
                : std::string("full-frame"))
            << std::endl;
    }
    return results;
}

std::vector<DetectionObject> TensorRTDetector::DetectQcapFrameView(
    const QCAPFrameView& frameView,
    float nms,
    float conf) {
    auto& impl = *m_impl;
    impl.lastFrameStats = DetectorFrameStats{};
    if (frameView.width <= 0 || frameView.height <= 0 || !impl.context || !impl.engine) {
        impl.lastFrameStats.failureStage = "invalid-frameview";
        return {};
    }

    const auto totalStart = std::chrono::steady_clock::now();
    TrtLetterBoxInfo info;
    bool usedGpuPreprocess = false;
    bool usedGpuSuppression = false;
    bool usedGpuDecodeTransform = false;
    std::size_t outputBytes = impl.hostOutput.size() * sizeof(float);
    const float* decodeOutput = impl.hostOutput.data();
    std::vector<int64_t> decodeOutputDims = impl.outputDims;
    impl.hostCompactedCount = 0;
    impl.lastFrameStats.outputRowsBeforeFilter = impl.outputDims.size() >= 2 ? static_cast<int>(impl.outputDims[1]) : 0;
    int compactCountBeforeGpuSuppression = 0;

    const auto preprocessStart = std::chrono::steady_clock::now();
    if (impl.gpuPreprocessReady && TensorRTGpuPreprocess::SupportsFrameView(frameView)) {
        TensorRTGpuPreprocessLetterBoxInfo gpuInfo;
        std::string preprocessError;
        const TensorRTGpuPreprocessRequest preprocessRequest =
            TensorRTGpuPreprocess::BuildRequest(frameView, &impl.roiSettings);
        const bool preprocessOk = impl.gpuPreprocess.PreprocessBgr24ToNchw(
            preprocessRequest,
            static_cast<float*>(impl.deviceInput),
            gpuInfo,
            preprocessError);
        if (preprocessOk) {
            info.scale = gpuInfo.scale;
            info.padW = gpuInfo.padW;
            info.padH = gpuInfo.padH;
            info.resizedWidth = gpuInfo.resizedWidth;
            info.resizedHeight = gpuInfo.resizedHeight;
            info.roiX = gpuInfo.roiX;
            info.roiY = gpuInfo.roiY;
            info.roiWidth = gpuInfo.roiWidth;
            info.roiHeight = gpuInfo.roiHeight;
            usedGpuPreprocess = true;
            if (impl.roiSettings.enabled) {
                LogTensorRtRoiActivationOnce(impl.loggedRoiActivation, frameView.width, frameView.height, impl.roiSettings, info);
            }
            if (!impl.loggedGpuPreprocessActive) {
                impl.loggedGpuPreprocessActive = true;
                std::cout << "[TensorRT] GPU preprocess active: consuming QCAP CUDA-mapped frame memory directly."
                    << std::endl;
            }
        }
        else if (!preprocessError.empty()) {
            if (!impl.loggedGpuPreprocessFallback) {
                impl.loggedGpuPreprocessFallback = true;
                std::cout << "[TensorRT] GPU preprocess fallback: " << preprocessError << std::endl;
            }
        }
    }

    if (!usedGpuPreprocess) {
        if (frameView.cpuData == nullptr) {
            impl.lastFrameStats.failureStage = "cpu-fallback-missing-frame";
            return {};
        }

        if (!LetterBoxPreprocess(
                frameView.cpuData,
                frameView.width,
                frameView.height,
                impl.inputWidth,
                impl.hostInput,
                info,
                &impl.roiSettings)) {
            impl.lastFrameStats.failureStage = "cpu-preprocess";
            return {};
        }
        if (impl.roiSettings.enabled) {
            LogTensorRtRoiActivationOnce(impl.loggedRoiActivation, frameView.width, frameView.height, impl.roiSettings, info);
        }

        const std::size_t inputBytes = impl.hostInput.size() * sizeof(float);
        cudaEventRecord(impl.eventStart, impl.stream);
        if (cudaMemcpyAsync(impl.deviceInput, impl.hostInput.data(), inputBytes, cudaMemcpyHostToDevice, impl.stream) != cudaSuccess) {
            impl.lastFrameStats.failureStage = "h2d";
            return {};
        }
        cudaEventRecord(impl.eventAfterInput, impl.stream);
    }
    else {
        cudaEventRecord(impl.eventStart, impl.stream);
        cudaEventRecord(impl.eventAfterInput, impl.stream);
    }
    const auto preprocessEnd = std::chrono::steady_clock::now();
    impl.lastFrameStats.preprocessMs =
        std::chrono::duration<double, std::milli>(preprocessEnd - preprocessStart).count();
    impl.lastFrameStats.usedGpuPreprocess = usedGpuPreprocess;

    if (!impl.context->enqueueV3(impl.stream)) {
        impl.lastFrameStats.failureStage = "enqueue";
        m_lastError = BuildTrtRuntimeFailureMessage(impl.logger, "enqueueV3");
        if (!impl.loggedFrameStatsFailure) {
            impl.loggedFrameStatsFailure = true;
            std::cout << "[TensorRT] " << m_lastError << std::endl;
        }
        return {};
    }
    cudaEventRecord(impl.eventAfterInfer, impl.stream);

    bool usedOutputCompaction = false;
    if (impl.gpuPostprocessReady) {
        std::string postprocessError;
        const bool compactOk = impl.gpuPostprocess.CompactBestClassDetections(
            static_cast<const float*>(impl.deviceOutput),
            conf,
            postprocessError);
        if (compactOk) {
            int compactCount = 0;
            TensorRTGpuDecodeTransform decodeTransform{};
            decodeTransform.scale = info.scale;
            decodeTransform.padW = info.padW;
            decodeTransform.padH = info.padH;
            decodeTransform.roiX = info.roiX;
            decodeTransform.roiY = info.roiY;
            decodeTransform.originalWidth = frameView.width;
            decodeTransform.originalHeight = frameView.height;
            if (impl.gpuPostprocess.DownloadCompactedRows(
                    impl.hostCompactedOutput.data(),
                    compactCount,
                    impl.hostCompactedOutput.size(),
                    nms,
                    usedGpuSuppression,
                    &compactCountBeforeGpuSuppression,
                    &decodeTransform,
                    postprocessError)) {
                impl.hostCompactedCount = std::clamp(compactCount, 0, impl.lastFrameStats.outputRowsBeforeFilter);
                usedGpuDecodeTransform = impl.hostCompactedCount > 0;
                cudaEventRecord(impl.eventAfterGpuPostprocess, impl.stream);
                if (impl.hostCompactedCount > 0) {
                    outputBytes = static_cast<std::size_t>(impl.hostCompactedCount) *
                        static_cast<std::size_t>(impl.gpuPostprocess.GetConfig().compactStride) * sizeof(float);
                    decodeOutput = impl.hostCompactedOutput.data();
                    decodeOutputDims[1] = impl.hostCompactedCount;
                    decodeOutputDims[2] = impl.gpuPostprocess.GetConfig().compactStride;
                    usedOutputCompaction = true;
                    if (!impl.loggedGpuPostprocessActive) {
                        impl.loggedGpuPostprocessActive = true;
                        std::cout << "[TensorRT] GPU candidate compaction active: downloading best-class candidates with GPU-side suppression when bounded candidate counts allow it."
                            << std::endl;
                    }
                }
                else {
                    decodeOutput = nullptr;
                    decodeOutputDims[1] = 0;
                    decodeOutputDims[2] = impl.gpuPostprocess.GetConfig().compactStride;
                    outputBytes = 0;
                    usedOutputCompaction = true;
                    if (!impl.loggedGpuPostprocessActive) {
                        impl.loggedGpuPostprocessActive = true;
                        std::cout << "[TensorRT] GPU candidate compaction active: downloading best-class candidates with GPU-side suppression when bounded candidate counts allow it."
                            << std::endl;
                    }
                }
            }
        }
        else if (!postprocessError.empty() && !impl.loggedGpuPostprocessFallback) {
            impl.loggedGpuPostprocessFallback = true;
            std::cout << "[TensorRT] GPU postprocess fallback: " << postprocessError << std::endl;
        }
    }

    if (!usedOutputCompaction) {
        cudaEventRecord(impl.eventAfterGpuPostprocess, impl.stream);
    }

    if (!usedOutputCompaction && outputBytes > 0) {
        if (cudaMemcpyAsync(impl.hostOutput.data(), impl.deviceOutput, outputBytes, cudaMemcpyDeviceToHost, impl.stream) != cudaSuccess) {
            impl.lastFrameStats.failureStage = "d2h";
            return {};
        }
    }
    cudaEventRecord(impl.eventAfterD2H, impl.stream);

    if (cudaStreamSynchronize(impl.stream) != cudaSuccess) {
        impl.lastFrameStats.failureStage = "stream-sync";
        return {};
    }

    impl.lastFrameStats.usedOutputCompaction = usedOutputCompaction;
    impl.lastFrameStats.outputRowsAfterFilter = usedOutputCompaction ? impl.hostCompactedCount : impl.lastFrameStats.outputRowsBeforeFilter;
    impl.lastFrameStats.d2hBytes = static_cast<std::uint64_t>(outputBytes);
    impl.lastFrameStats.h2dMs = GetCudaElapsedMilliseconds(impl.eventStart, impl.eventAfterInput);
    impl.lastFrameStats.inferMs = GetCudaElapsedMilliseconds(impl.eventAfterInput, impl.eventAfterInfer);
    impl.lastFrameStats.gpuPostprocessMs = GetCudaElapsedMilliseconds(impl.eventAfterInfer, impl.eventAfterGpuPostprocess);
    impl.lastFrameStats.d2hMs = GetCudaElapsedMilliseconds(impl.eventAfterGpuPostprocess, impl.eventAfterD2H);

    int proposalsBeforeNms = 0;
    const auto decodeStart = std::chrono::steady_clock::now();
    std::vector<DetectionObject> results = DecodeYoloOutput(
        decodeOutput,
        decodeOutputDims,
        info,
        frameView.width,
        frameView.height,
        nms,
        conf,
        &proposalsBeforeNms,
        !usedGpuSuppression,
        usedGpuDecodeTransform);
    const auto decodeEnd = std::chrono::steady_clock::now();

    impl.lastFrameStats.decodeMs =
        std::chrono::duration<double, std::milli>(decodeEnd - decodeStart).count();
    if (usedGpuSuppression) {
        proposalsBeforeNms = compactCountBeforeGpuSuppression;
    }
    impl.lastFrameStats.proposalsBeforeNms = proposalsBeforeNms;
    impl.lastFrameStats.resultsAfterNms = static_cast<int>(results.size());
    PopulateTensorRtRoiStats(impl.lastFrameStats, info);
    impl.lastFrameStats.totalMs =
        std::chrono::duration<double, std::milli>(decodeEnd - totalStart).count();
    impl.lastFrameStats.valid = true;
    if (!impl.loggedFrameStatsReady) {
        impl.loggedFrameStatsReady = true;
        std::cout << "[TensorRT] frameStats ready path=QCAP totalMs=" << impl.lastFrameStats.totalMs
            << " proposals=" << impl.lastFrameStats.proposalsBeforeNms
            << " results=" << impl.lastFrameStats.resultsAfterNms
            << " gpuPre=" << (impl.lastFrameStats.usedGpuPreprocess ? "yes" : "no")
            << " gpuCompact=" << (impl.lastFrameStats.usedOutputCompaction ? "yes" : "no")
            << " roi=" << (impl.lastFrameStats.usedRoi
                ? (std::to_string(impl.lastFrameStats.roiX) + "," +
                   std::to_string(impl.lastFrameStats.roiY) + "," +
                   std::to_string(impl.lastFrameStats.roiWidth) + "," +
                   std::to_string(impl.lastFrameStats.roiHeight))
                : std::string("full-frame"))
            << std::endl;
    }
    return results;
}

void TensorRTDetector::ReleaseResources() {
    if (m_impl) {
        m_impl->Reset();
    }
}

DetectorFrameStats TensorRTDetector::GetLastFrameStats() const {
    if (!m_impl) {
        return {};
    }

    return m_impl->lastFrameStats;
}
