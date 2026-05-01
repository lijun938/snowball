#pragma once

#include <cuda_runtime_api.h>

#include <cstddef>
#include <string>

struct TensorRTGpuPostprocessConfig {
    int maxRows = 0;
    int stride = 0;
    int compactStride = 6;
    int topK = 0;
    cudaStream_t stream = nullptr;
};

struct TensorRTGpuDecodeTransform {
    float scale = 1.0f;
    float padW = 0.0f;
    float padH = 0.0f;
    int roiX = 0;
    int roiY = 0;
    int originalWidth = 0;
    int originalHeight = 0;
};

class TensorRTGpuPostprocess {
public:
    TensorRTGpuPostprocess();
    ~TensorRTGpuPostprocess();

    bool Initialize(const TensorRTGpuPostprocessConfig& config, std::string& error);
    void Reset();

    bool IsInitialized() const;
    const TensorRTGpuPostprocessConfig& GetConfig() const;

    bool CompactRowsByConfidence(const float* deviceOutput, float conf, std::string& error);
    bool CompactBestClassDetections(const float* deviceOutput, float conf, std::string& error);
    bool DownloadCompactedRows(
        float* hostOutput,
        int& compactCount,
        std::size_t hostCapacityFloats,
        float nmsThreshold,
        bool& usedGpuSuppression,
        int* rowsBeforeSuppression,
        const TensorRTGpuDecodeTransform* decodeTransform,
        std::string& error);
    const float* GetCompactedRowsDevicePointer() const;
    const int* GetCompactedCountDevicePointer() const;

private:
    TensorRTGpuPostprocessConfig m_config;
    float* m_deviceCompactedRows = nullptr;
    int* m_deviceCompactedCount = nullptr;
    float* m_deviceCompactedRowsSorted = nullptr;
    float* m_deviceSortKeysIn = nullptr;
    float* m_deviceSortKeysOut = nullptr;
    int* m_deviceSortIndicesIn = nullptr;
    int* m_deviceSortIndicesOut = nullptr;
    void* m_deviceSortTempStorage = nullptr;
    std::size_t m_deviceSortTempStorageBytes = 0;
    int* m_deviceHostVisibleCount = nullptr;
    bool m_initialized = false;
};
