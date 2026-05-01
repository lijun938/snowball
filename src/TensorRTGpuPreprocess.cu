#include "TensorRTGpuPreprocess.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <sstream>

namespace {

std::string FormatCudaError(const char* operation, cudaError_t result) {
    std::ostringstream oss;
    oss << operation << " failed";
    if (result != cudaSuccess) {
        oss << " (" << cudaGetErrorName(result) << ": " << cudaGetErrorString(result) << ")";
    }
    return oss.str();
}

bool ComputeLetterBoxInfo(
    int sourceWidth,
    int sourceHeight,
    int inputWidth,
    int inputHeight,
    TensorRTGpuPreprocessLetterBoxInfo& info,
    std::string& error) {
    if (sourceWidth <= 0 || sourceHeight <= 0 || inputWidth <= 0 || inputHeight <= 0) {
        error = "Invalid source or input dimensions for GPU preprocess.";
        return false;
    }

    const float scale = std::min(
        static_cast<float>(inputWidth) / static_cast<float>(sourceWidth),
        static_cast<float>(inputHeight) / static_cast<float>(sourceHeight));
    const int resizedWidth = static_cast<int>(std::round(static_cast<float>(sourceWidth) * scale));
    const int resizedHeight = static_cast<int>(std::round(static_cast<float>(sourceHeight) * scale));
    const float padW = (static_cast<float>(inputWidth) - static_cast<float>(resizedWidth)) * 0.5f;
    const float padH = (static_cast<float>(inputHeight) - static_cast<float>(resizedHeight)) * 0.5f;

    info.scale = scale;
    info.padW = static_cast<float>(static_cast<int>(std::round(padW - 0.1f)));
    info.padH = static_cast<float>(static_cast<int>(std::round(padH - 0.1f)));
    info.resizedWidth = resizedWidth;
    info.resizedHeight = resizedHeight;
    return true;
}

__global__ void Bgr24LetterBoxToNchwKernel(
    const unsigned char* source,
    std::size_t sourcePitchBytes,
    int sourceWidth,
    int sourceHeight,
    int roiX,
    int roiY,
    float* output,
    int inputWidth,
    int inputHeight,
    float scale,
    float padW,
    float padH,
    int resizedWidth,
    int resizedHeight,
    float padValue,
    bool sourceImageUpsideDown) {
    const int dstX = blockIdx.x * blockDim.x + threadIdx.x;
    const int dstY = blockIdx.y * blockDim.y + threadIdx.y;
    if (dstX >= inputWidth || dstY >= inputHeight) {
        return;
    }

    const std::size_t planeSize = static_cast<std::size_t>(inputWidth) * static_cast<std::size_t>(inputHeight);
    const std::size_t dstIndex = static_cast<std::size_t>(dstY) * static_cast<std::size_t>(inputWidth) + static_cast<std::size_t>(dstX);
    float r = padValue;
    float g = padValue;
    float b = padValue;

    const float localX = static_cast<float>(dstX) - padW;
    const float localY = static_cast<float>(dstY) - padH;
    if (localX >= 0.0f &&
        localY >= 0.0f &&
        localX < static_cast<float>(resizedWidth) &&
        localY < static_cast<float>(resizedHeight)) {
        const float srcX = (localX + 0.5f) / scale - 0.5f;
        const float srcY = (localY + 0.5f) / scale - 0.5f;
        const int nearestX = roiX + max(0, min(static_cast<int>(srcX + 0.5f), sourceWidth - 1));
        int nearestY = roiY + max(0, min(static_cast<int>(srcY + 0.5f), sourceHeight - 1));
        if (sourceImageUpsideDown) {
            nearestY = roiY + (sourceHeight - 1 - (nearestY - roiY));
        }
        const unsigned char* pixel = source + static_cast<std::size_t>(nearestY) * sourcePitchBytes + static_cast<std::size_t>(nearestX) * 3U;
        b = static_cast<float>(pixel[0]) * (1.0f / 255.0f);
        g = static_cast<float>(pixel[1]) * (1.0f / 255.0f);
        r = static_cast<float>(pixel[2]) * (1.0f / 255.0f);
    }

    output[dstIndex] = r;
    output[planeSize + dstIndex] = g;
    output[planeSize * 2 + dstIndex] = b;
}

} // namespace

TensorRTGpuPreprocess::TensorRTGpuPreprocess() = default;

TensorRTGpuPreprocess::~TensorRTGpuPreprocess() = default;

bool TensorRTGpuPreprocess::Initialize(const TensorRTGpuPreprocessConfig& config, std::string& error) {
    error.clear();
    if (config.inputWidth <= 0 || config.inputHeight <= 0) {
        error = "TensorRTGpuPreprocess requires a positive input size.";
        m_initialized = false;
        return false;
    }

    m_config = config;
    m_initialized = true;
    return true;
}

void TensorRTGpuPreprocess::Reset() {
    m_config = TensorRTGpuPreprocessConfig{};
    m_initialized = false;
}

bool TensorRTGpuPreprocess::IsInitialized() const {
    return m_initialized;
}

const TensorRTGpuPreprocessConfig& TensorRTGpuPreprocess::GetConfig() const {
    return m_config;
}

std::size_t TensorRTGpuPreprocess::GetRequiredOutputElementCount(int inputWidth, int inputHeight) {
    return static_cast<std::size_t>(inputWidth) * static_cast<std::size_t>(inputHeight) * 3U;
}

bool TensorRTGpuPreprocess::PreprocessBgr24ToNchw(
    const TensorRTGpuPreprocessRequest& request,
    float* deviceOutput,
    TensorRTGpuPreprocessLetterBoxInfo& info,
    std::string& error) const {
    error.clear();
    info = TensorRTGpuPreprocessLetterBoxInfo{};

    if (!m_initialized) {
        error = "TensorRTGpuPreprocess is not initialized.";
        return false;
    }
    if (request.sourceDevicePointer == nullptr) {
        error = "GPU preprocess requires a CUDA-accessible source pointer. sourceDevicePointer is null.";
        return false;
    }
    if (deviceOutput == nullptr) {
        error = "GPU preprocess requires a valid deviceOutput pointer.";
        return false;
    }
    if (request.sourceWidth <= 0 || request.sourceHeight <= 0) {
        error = "GPU preprocess requires positive source dimensions.";
        return false;
    }

    int effectiveSourceWidth = request.sourceWidth;
    int effectiveSourceHeight = request.sourceHeight;
    int roiX = 0;
    int roiY = 0;
    if (request.useRoi) {
        if (request.roiX < 0 || request.roiY < 0 ||
            request.roiWidth <= 0 || request.roiHeight <= 0 ||
            request.roiX + request.roiWidth > request.sourceWidth ||
            request.roiY + request.roiHeight > request.sourceHeight) {
            error = "GPU preprocess ROI is outside the source frame.";
            return false;
        }
        roiX = request.roiX;
        roiY = request.roiY;
        effectiveSourceWidth = request.roiWidth;
        effectiveSourceHeight = request.roiHeight;
    }
    info.roiX = roiX;
    info.roiY = roiY;
    info.roiWidth = effectiveSourceWidth;
    info.roiHeight = effectiveSourceHeight;

    const std::size_t inferredPitchBytes =
        request.sourcePitchBytes != 0 ? request.sourcePitchBytes : static_cast<std::size_t>(request.sourceWidth) * 3U;
    const std::size_t minimumBufferSize = inferredPitchBytes * static_cast<std::size_t>(request.sourceHeight);
    if (request.sourceBufferSize != 0 && request.sourceBufferSize < minimumBufferSize) {
        error = "GPU preprocess sourceBufferSize is smaller than the required BGR24 frame footprint.";
        return false;
    }

    if (!ComputeLetterBoxInfo(
            effectiveSourceWidth,
            effectiveSourceHeight,
            m_config.inputWidth,
            m_config.inputHeight,
            info,
            error)) {
        return false;
    }

    constexpr int kBlockSizeX = 16;
    constexpr int kBlockSizeY = 16;
    const dim3 block(kBlockSizeX, kBlockSizeY);
    const dim3 grid(
        (m_config.inputWidth + kBlockSizeX - 1) / kBlockSizeX,
        (m_config.inputHeight + kBlockSizeY - 1) / kBlockSizeY);

    Bgr24LetterBoxToNchwKernel<<<grid, block, 0, m_config.stream>>>(
        static_cast<const unsigned char*>(request.sourceDevicePointer),
        inferredPitchBytes,
        effectiveSourceWidth,
        effectiveSourceHeight,
        roiX,
        roiY,
        deviceOutput,
        m_config.inputWidth,
        m_config.inputHeight,
        info.scale,
        info.padW,
        info.padH,
        info.resizedWidth,
        info.resizedHeight,
        m_config.padValue,
        request.sourceImageUpsideDown);

    const cudaError_t launchResult = cudaGetLastError();
    if (launchResult != cudaSuccess) {
        error = FormatCudaError("Bgr24LetterBoxToNchwKernel launch", launchResult);
        return false;
    }

    return true;
}
