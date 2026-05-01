#pragma once

#include <cuda_runtime_api.h>

#include <cstddef>
#include <string>

struct TensorRtRoiSettings;
struct QCAPFrameView;

struct TensorRTGpuPreprocessConfig {
    int inputWidth = 0;
    int inputHeight = 0;
    float padValue = 114.0f / 255.0f;
    cudaStream_t stream = nullptr;
};

struct TensorRTGpuPreprocessLetterBoxInfo {
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

struct TensorRTGpuPreprocessRequest {
    const void* sourceDevicePointer = nullptr;
    const unsigned char* sourceHostPointer = nullptr;
    int sourceWidth = 0;
    int sourceHeight = 0;
    std::size_t sourcePitchBytes = 0;
    std::size_t sourceBufferSize = 0;
    bool sourceImageUpsideDown = false;
    bool useRoi = false;
    int roiX = 0;
    int roiY = 0;
    int roiWidth = 0;
    int roiHeight = 0;
};

class TensorRTGpuPreprocess {
public:
    TensorRTGpuPreprocess();
    ~TensorRTGpuPreprocess();

    bool Initialize(const TensorRTGpuPreprocessConfig& config, std::string& error);
    void Reset();

    bool IsInitialized() const;
    const TensorRTGpuPreprocessConfig& GetConfig() const;

    static std::size_t GetRequiredOutputElementCount(int inputWidth, int inputHeight);
    static bool SupportsFrameView(const QCAPFrameView& frameView);
    static TensorRTGpuPreprocessRequest BuildRequest(
        const QCAPFrameView& frameView,
        const TensorRtRoiSettings* roiSettings = nullptr);

    bool PreprocessBgr24ToNchw(
        const TensorRTGpuPreprocessRequest& request,
        float* deviceOutput,
        TensorRTGpuPreprocessLetterBoxInfo& info,
        std::string& error) const;

private:
    TensorRTGpuPreprocessConfig m_config;
    bool m_initialized = false;
};
