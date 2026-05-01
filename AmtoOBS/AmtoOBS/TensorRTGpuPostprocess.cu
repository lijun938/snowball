#include "TensorRTGpuPostprocess.h"

#include <cuda_runtime.h>
#include <cub/device/device_radix_sort.cuh>

#include <sstream>

namespace {

constexpr int kGpuGreedyNmsMaxCandidates = 512;

std::string FormatCudaError(const char* operation, cudaError_t result) {
    std::ostringstream oss;
    oss << operation << " failed";
    if (result != cudaSuccess) {
        oss << " (" << cudaGetErrorName(result) << ": " << cudaGetErrorString(result) << ")";
    }
    return oss.str();
}

__global__ void CompactRowsByConfidenceKernel(
    const float* inputRows,
    int rowCount,
    int stride,
    float conf,
    float* compactRows,
    int* compactCount) {
    const int rowIndex = blockIdx.x * blockDim.x + threadIdx.x;
    if (rowIndex >= rowCount) {
        return;
    }

    const std::size_t rowOffset = static_cast<std::size_t>(rowIndex) * static_cast<std::size_t>(stride);
    const float objectness = inputRows[rowOffset + 4];
    if (objectness <= conf) {
        return;
    }

    bool keep = false;
    for (int cls = 5; cls < stride; ++cls) {
        if (objectness * inputRows[rowOffset + cls] > conf) {
            keep = true;
            break;
        }
    }

    if (!keep) {
        return;
    }

    const int compactIndex = atomicAdd(compactCount, 1);
    const std::size_t compactOffset = static_cast<std::size_t>(compactIndex) * static_cast<std::size_t>(stride);
    for (int column = 0; column < stride; ++column) {
        compactRows[compactOffset + static_cast<std::size_t>(column)] =
            inputRows[rowOffset + static_cast<std::size_t>(column)];
    }
}

__global__ void CompactBestClassDetectionsKernel(
    const float* inputRows,
    int rowCount,
    int stride,
    float conf,
    float* compactRows,
    int compactStride,
    int* compactCount) {
    const int rowIndex = blockIdx.x * blockDim.x + threadIdx.x;
    if (rowIndex >= rowCount) {
        return;
    }

    const std::size_t rowOffset = static_cast<std::size_t>(rowIndex) * static_cast<std::size_t>(stride);
    const float objectness = inputRows[rowOffset + 4];
    if (objectness <= conf) {
        return;
    }

    int bestClass = -1;
    float bestScore = 0.0f;
    for (int cls = 5; cls < stride; ++cls) {
        const float score = objectness * inputRows[rowOffset + cls];
        if (score > bestScore) {
            bestScore = score;
            bestClass = cls - 5;
        }
    }

    if (bestClass < 0 || bestScore <= conf) {
        return;
    }

    const int compactIndex = atomicAdd(compactCount, 1);
    const std::size_t compactOffset = static_cast<std::size_t>(compactIndex) * static_cast<std::size_t>(compactStride);

    compactRows[compactOffset + 0] = inputRows[rowOffset + 0];
    compactRows[compactOffset + 1] = inputRows[rowOffset + 1];
    compactRows[compactOffset + 2] = inputRows[rowOffset + 2];
    compactRows[compactOffset + 3] = inputRows[rowOffset + 3];
    compactRows[compactOffset + 4] = static_cast<float>(bestClass);
    compactRows[compactOffset + 5] = bestScore;
}

__global__ void FillSortKeysAndIndicesKernel(
    const float* compactRows,
    int count,
    int compactStride,
    float* keys,
    int* indices) {
    const int rowIndex = blockIdx.x * blockDim.x + threadIdx.x;
    if (rowIndex >= count) {
        return;
    }

    keys[rowIndex] = compactRows[static_cast<std::size_t>(rowIndex) * static_cast<std::size_t>(compactStride) + 5];
    indices[rowIndex] = rowIndex;
}

__global__ void GatherSortedRowsKernel(
    const float* inputRows,
    int compactStride,
    const int* sortedIndices,
    int outputCount,
    float* outputRows) {
    const int rowIndex = blockIdx.x * blockDim.x + threadIdx.x;
    if (rowIndex >= outputCount) {
        return;
    }

    const int sourceIndex = sortedIndices[rowIndex];
    const std::size_t srcOffset = static_cast<std::size_t>(sourceIndex) * static_cast<std::size_t>(compactStride);
    const std::size_t dstOffset = static_cast<std::size_t>(rowIndex) * static_cast<std::size_t>(compactStride);
    for (int column = 0; column < compactStride; ++column) {
        outputRows[dstOffset + static_cast<std::size_t>(column)] =
            inputRows[srcOffset + static_cast<std::size_t>(column)];
    }
}

__device__ float ComputeCompactRowIoU(const float* lhs, const float* rhs) {
    const float lhsLeft = lhs[0] - lhs[2] * 0.5f;
    const float lhsTop = lhs[1] - lhs[3] * 0.5f;
    const float lhsRight = lhs[0] + lhs[2] * 0.5f;
    const float lhsBottom = lhs[1] + lhs[3] * 0.5f;

    const float rhsLeft = rhs[0] - rhs[2] * 0.5f;
    const float rhsTop = rhs[1] - rhs[3] * 0.5f;
    const float rhsRight = rhs[0] + rhs[2] * 0.5f;
    const float rhsBottom = rhs[1] + rhs[3] * 0.5f;

    const float overlapLeft = lhsLeft > rhsLeft ? lhsLeft : rhsLeft;
    const float overlapTop = lhsTop > rhsTop ? lhsTop : rhsTop;
    const float overlapRight = lhsRight < rhsRight ? lhsRight : rhsRight;
    const float overlapBottom = lhsBottom < rhsBottom ? lhsBottom : rhsBottom;

    const float overlapWidth = overlapRight - overlapLeft;
    const float overlapHeight = overlapBottom - overlapTop;
    if (overlapWidth <= 0.0f || overlapHeight <= 0.0f) {
        return 0.0f;
    }

    const float overlapArea = overlapWidth * overlapHeight;
    const float lhsArea = lhs[2] * lhs[3];
    const float rhsArea = rhs[2] * rhs[3];
    const float unionArea = lhsArea + rhsArea - overlapArea;
    if (unionArea <= 0.0f) {
        return 0.0f;
    }

    return overlapArea / unionArea;
}

__global__ void GreedyNmsCompactKernel(
    const float* inputRows,
    int inputCount,
    int compactStride,
    float iouThreshold,
    float* outputRows,
    int* outputCount) {
    if (blockIdx.x != 0 || threadIdx.x != 0) {
        return;
    }

    int keptCount = 0;
    for (int inputIndex = 0; inputIndex < inputCount; ++inputIndex) {
        const float* candidateRow =
            inputRows + static_cast<std::size_t>(inputIndex) * static_cast<std::size_t>(compactStride);
        const int candidateLabel = static_cast<int>(candidateRow[4] + 0.5f);

        bool suppressed = false;
        for (int keptIndex = 0; keptIndex < keptCount; ++keptIndex) {
            const float* keptRow =
                outputRows + static_cast<std::size_t>(keptIndex) * static_cast<std::size_t>(compactStride);
            const int keptLabel = static_cast<int>(keptRow[4] + 0.5f);
            if (keptLabel != candidateLabel) {
                continue;
            }

            if (ComputeCompactRowIoU(candidateRow, keptRow) >= iouThreshold) {
                suppressed = true;
                break;
            }
        }

        if (suppressed) {
            continue;
        }

        float* outputRow =
            outputRows + static_cast<std::size_t>(keptCount) * static_cast<std::size_t>(compactStride);
        for (int column = 0; column < compactStride; ++column) {
            outputRow[column] = candidateRow[column];
        }
        ++keptCount;
    }

    *outputCount = keptCount;
}

__global__ void DecodeCompactRowsInPlaceKernel(
    float* rows,
    int count,
    int compactStride,
    float scale,
    float padW,
    float padH,
    int roiX,
    int roiY,
    int originalWidth,
    int originalHeight) {
    const int rowIndex = blockIdx.x * blockDim.x + threadIdx.x;
    if (rowIndex >= count) {
        return;
    }

    float* row = rows + static_cast<std::size_t>(rowIndex) * static_cast<std::size_t>(compactStride);
    const float centerX = row[0];
    const float centerY = row[1];
    const float width = row[2];
    const float height = row[3];

    float x = (centerX - width * 0.5f - padW) / scale;
    float y = (centerY - height * 0.5f - padH) / scale;
    float scaledWidth = width / scale;
    float scaledHeight = height / scale;
    x += static_cast<float>(roiX);
    y += static_cast<float>(roiY);

    const float maxX = static_cast<float>(originalWidth - 1);
    const float maxY = static_cast<float>(originalHeight - 1);
    x = x < 0.0f ? 0.0f : (x > maxX ? maxX : x);
    y = y < 0.0f ? 0.0f : (y > maxY ? maxY : y);

    const float maxWidth = static_cast<float>(originalWidth) - x;
    const float maxHeight = static_cast<float>(originalHeight) - y;
    if (scaledWidth < 1.0f) {
        scaledWidth = 1.0f;
    }
    if (scaledHeight < 1.0f) {
        scaledHeight = 1.0f;
    }
    if (scaledWidth > maxWidth) {
        scaledWidth = maxWidth;
    }
    if (scaledHeight > maxHeight) {
        scaledHeight = maxHeight;
    }

    row[0] = x;
    row[1] = y;
    row[2] = scaledWidth;
    row[3] = scaledHeight;
}

} // namespace

TensorRTGpuPostprocess::TensorRTGpuPostprocess() = default;

TensorRTGpuPostprocess::~TensorRTGpuPostprocess() {
    Reset();
}

bool TensorRTGpuPostprocess::Initialize(const TensorRTGpuPostprocessConfig& config, std::string& error) {
    Reset();
    error.clear();

    if (config.maxRows <= 0 || config.stride <= 5 || config.compactStride < 6) {
        error = "TensorRTGpuPostprocess requires a positive row count and YOLO-like stride.";
        return false;
    }

    m_config = config;

    const std::size_t compactBytes =
        static_cast<std::size_t>(config.maxRows) * static_cast<std::size_t>(config.compactStride) * sizeof(float);
    const cudaError_t rowsResult = cudaMalloc(&m_deviceCompactedRows, compactBytes);
    if (rowsResult != cudaSuccess) {
        error = FormatCudaError("cudaMalloc(deviceCompactedRows)", rowsResult);
        Reset();
        return false;
    }

    const cudaError_t countResult = cudaMalloc(&m_deviceCompactedCount, sizeof(int));
    if (countResult != cudaSuccess) {
        error = FormatCudaError("cudaMalloc(deviceCompactedCount)", countResult);
        Reset();
        return false;
    }

    const cudaError_t hostVisibleCountResult = cudaMallocHost(&m_deviceHostVisibleCount, sizeof(int));
    if (hostVisibleCountResult != cudaSuccess) {
        error = FormatCudaError("cudaMallocHost(hostVisibleCount)", hostVisibleCountResult);
        Reset();
        return false;
    }

    {
        const cudaError_t sortedRowsResult = cudaMalloc(&m_deviceCompactedRowsSorted, compactBytes);
        if (sortedRowsResult != cudaSuccess) {
            error = FormatCudaError("cudaMalloc(deviceCompactedRowsSorted)", sortedRowsResult);
            Reset();
            return false;
        }

        const cudaError_t keysInResult = cudaMalloc(&m_deviceSortKeysIn, static_cast<std::size_t>(config.maxRows) * sizeof(float));
        if (keysInResult != cudaSuccess) {
            error = FormatCudaError("cudaMalloc(deviceSortKeysIn)", keysInResult);
            Reset();
            return false;
        }

        const cudaError_t keysOutResult = cudaMalloc(&m_deviceSortKeysOut, static_cast<std::size_t>(config.maxRows) * sizeof(float));
        if (keysOutResult != cudaSuccess) {
            error = FormatCudaError("cudaMalloc(deviceSortKeysOut)", keysOutResult);
            Reset();
            return false;
        }

        const cudaError_t indicesInResult = cudaMalloc(&m_deviceSortIndicesIn, static_cast<std::size_t>(config.maxRows) * sizeof(int));
        if (indicesInResult != cudaSuccess) {
            error = FormatCudaError("cudaMalloc(deviceSortIndicesIn)", indicesInResult);
            Reset();
            return false;
        }

        const cudaError_t indicesOutResult = cudaMalloc(&m_deviceSortIndicesOut, static_cast<std::size_t>(config.maxRows) * sizeof(int));
        if (indicesOutResult != cudaSuccess) {
            error = FormatCudaError("cudaMalloc(deviceSortIndicesOut)", indicesOutResult);
            Reset();
            return false;
        }

        std::size_t tempStorageBytes = 0;
        const cudaError_t tempQueryResult = cub::DeviceRadixSort::SortPairsDescending(
            nullptr,
            tempStorageBytes,
            m_deviceSortKeysIn,
            m_deviceSortKeysOut,
            m_deviceSortIndicesIn,
            m_deviceSortIndicesOut,
            config.maxRows,
            0,
            sizeof(float) * 8,
            config.stream);
        if (tempQueryResult != cudaSuccess) {
            error = FormatCudaError("DeviceRadixSort::SortPairsDescending(query)", tempQueryResult);
            Reset();
            return false;
        }

        const cudaError_t tempAllocResult = cudaMalloc(&m_deviceSortTempStorage, tempStorageBytes);
        if (tempAllocResult != cudaSuccess) {
            error = FormatCudaError("cudaMalloc(deviceSortTempStorage)", tempAllocResult);
            Reset();
            return false;
        }
        m_deviceSortTempStorageBytes = tempStorageBytes;
    }

    m_initialized = true;
    return true;
}

void TensorRTGpuPostprocess::Reset() {
    if (m_deviceHostVisibleCount != nullptr) {
        cudaFreeHost(m_deviceHostVisibleCount);
        m_deviceHostVisibleCount = nullptr;
    }
    if (m_deviceSortTempStorage != nullptr) {
        cudaFree(m_deviceSortTempStorage);
        m_deviceSortTempStorage = nullptr;
        m_deviceSortTempStorageBytes = 0;
    }
    if (m_deviceSortIndicesOut != nullptr) {
        cudaFree(m_deviceSortIndicesOut);
        m_deviceSortIndicesOut = nullptr;
    }
    if (m_deviceSortIndicesIn != nullptr) {
        cudaFree(m_deviceSortIndicesIn);
        m_deviceSortIndicesIn = nullptr;
    }
    if (m_deviceSortKeysOut != nullptr) {
        cudaFree(m_deviceSortKeysOut);
        m_deviceSortKeysOut = nullptr;
    }
    if (m_deviceSortKeysIn != nullptr) {
        cudaFree(m_deviceSortKeysIn);
        m_deviceSortKeysIn = nullptr;
    }
    if (m_deviceCompactedRowsSorted != nullptr) {
        cudaFree(m_deviceCompactedRowsSorted);
        m_deviceCompactedRowsSorted = nullptr;
    }
    if (m_deviceCompactedRows != nullptr) {
        cudaFree(m_deviceCompactedRows);
        m_deviceCompactedRows = nullptr;
    }
    if (m_deviceCompactedCount != nullptr) {
        cudaFree(m_deviceCompactedCount);
        m_deviceCompactedCount = nullptr;
    }

    m_config = TensorRTGpuPostprocessConfig{};
    m_initialized = false;
}

bool TensorRTGpuPostprocess::IsInitialized() const {
    return m_initialized;
}

const TensorRTGpuPostprocessConfig& TensorRTGpuPostprocess::GetConfig() const {
    return m_config;
}

bool TensorRTGpuPostprocess::CompactRowsByConfidence(const float* deviceOutput, float conf, std::string& error) {
    error.clear();

    if (!m_initialized) {
        error = "TensorRTGpuPostprocess is not initialized.";
        return false;
    }
    if (deviceOutput == nullptr) {
        error = "TensorRTGpuPostprocess requires a valid device output pointer.";
        return false;
    }

    const cudaError_t memsetResult = cudaMemsetAsync(
        m_deviceCompactedCount,
        0,
        sizeof(int),
        m_config.stream);
    if (memsetResult != cudaSuccess) {
        error = FormatCudaError("cudaMemsetAsync(deviceCompactedCount)", memsetResult);
        return false;
    }

    constexpr int kThreadsPerBlock = 256;
    const int blocks = (m_config.maxRows + kThreadsPerBlock - 1) / kThreadsPerBlock;
    CompactRowsByConfidenceKernel<<<blocks, kThreadsPerBlock, 0, m_config.stream>>>(
        deviceOutput,
        m_config.maxRows,
        m_config.stride,
        conf,
        m_deviceCompactedRows,
        m_deviceCompactedCount);

    const cudaError_t launchResult = cudaGetLastError();
    if (launchResult != cudaSuccess) {
        error = FormatCudaError("CompactRowsByConfidenceKernel launch", launchResult);
        return false;
    }

    return true;
}

bool TensorRTGpuPostprocess::CompactBestClassDetections(const float* deviceOutput, float conf, std::string& error) {
    error.clear();

    if (!m_initialized) {
        error = "TensorRTGpuPostprocess is not initialized.";
        return false;
    }
    if (deviceOutput == nullptr) {
        error = "TensorRTGpuPostprocess requires a valid device output pointer.";
        return false;
    }

    const cudaError_t memsetResult = cudaMemsetAsync(
        m_deviceCompactedCount,
        0,
        sizeof(int),
        m_config.stream);
    if (memsetResult != cudaSuccess) {
        error = FormatCudaError("cudaMemsetAsync(deviceCompactedCount)", memsetResult);
        return false;
    }

    constexpr int kThreadsPerBlock = 256;
    const int blocks = (m_config.maxRows + kThreadsPerBlock - 1) / kThreadsPerBlock;
    CompactBestClassDetectionsKernel<<<blocks, kThreadsPerBlock, 0, m_config.stream>>>(
        deviceOutput,
        m_config.maxRows,
        m_config.stride,
        conf,
        m_deviceCompactedRows,
        m_config.compactStride,
        m_deviceCompactedCount);

    const cudaError_t launchResult = cudaGetLastError();
    if (launchResult != cudaSuccess) {
        error = FormatCudaError("CompactBestClassDetectionsKernel launch", launchResult);
        return false;
    }

    return true;
}

bool TensorRTGpuPostprocess::DownloadCompactedRows(
    float* hostOutput,
    int& compactCount,
    std::size_t hostCapacityFloats,
    float nmsThreshold,
    bool& usedGpuSuppression,
    int* rowsBeforeSuppression,
    const TensorRTGpuDecodeTransform* decodeTransform,
    std::string& error) {
    error.clear();
    compactCount = 0;
    usedGpuSuppression = false;
    if (rowsBeforeSuppression != nullptr) {
        *rowsBeforeSuppression = 0;
    }

    if (!m_initialized) {
        error = "TensorRTGpuPostprocess is not initialized.";
        return false;
    }
    if (hostOutput == nullptr && hostCapacityFloats > 0) {
        error = "TensorRTGpuPostprocess requires a host output buffer.";
        return false;
    }
    if (m_deviceHostVisibleCount == nullptr) {
        error = "TensorRTGpuPostprocess host-visible count buffer is unavailable.";
        return false;
    }

    const cudaError_t countCopyResult = cudaMemcpyAsync(
        m_deviceHostVisibleCount,
        m_deviceCompactedCount,
        sizeof(int),
        cudaMemcpyDeviceToHost,
        m_config.stream);
    if (countCopyResult != cudaSuccess) {
        error = FormatCudaError("cudaMemcpyAsync(compactedCount)", countCopyResult);
        return false;
    }

    const cudaError_t syncResult = cudaStreamSynchronize(m_config.stream);
    if (syncResult != cudaSuccess) {
        error = FormatCudaError("cudaStreamSynchronize(compactedCount)", syncResult);
        return false;
    }

    compactCount = *m_deviceHostVisibleCount;
    if (compactCount <= 0) {
        compactCount = 0;
        return true;
    }

    compactCount = compactCount > m_config.maxRows ? m_config.maxRows : compactCount;
    int outputCount = compactCount;
    const float* deviceDownloadRows = m_deviceCompactedRows;
    const bool needsSortedRows =
        compactCount > 1 && (m_config.topK > 0 || nmsThreshold > 0.0f);

    if (needsSortedRows) {
        constexpr int kThreadsPerBlock = 256;
        const int fillBlocks = (compactCount + kThreadsPerBlock - 1) / kThreadsPerBlock;
        FillSortKeysAndIndicesKernel<<<fillBlocks, kThreadsPerBlock, 0, m_config.stream>>>(
            m_deviceCompactedRows,
            compactCount,
            m_config.compactStride,
            m_deviceSortKeysIn,
            m_deviceSortIndicesIn);

        cudaError_t launchResult = cudaGetLastError();
        if (launchResult != cudaSuccess) {
            error = FormatCudaError("FillSortKeysAndIndicesKernel launch", launchResult);
            return false;
        }

        const cudaError_t sortResult = cub::DeviceRadixSort::SortPairsDescending(
            m_deviceSortTempStorage,
            m_deviceSortTempStorageBytes,
            m_deviceSortKeysIn,
            m_deviceSortKeysOut,
            m_deviceSortIndicesIn,
            m_deviceSortIndicesOut,
            compactCount,
            0,
            sizeof(float) * 8,
            m_config.stream);
        if (sortResult != cudaSuccess) {
            error = FormatCudaError("DeviceRadixSort::SortPairsDescending", sortResult);
            return false;
        }

        if (m_config.topK > 0 && outputCount > m_config.topK) {
            outputCount = m_config.topK;
        }
        const int gatherBlocks = (outputCount + kThreadsPerBlock - 1) / kThreadsPerBlock;
        GatherSortedRowsKernel<<<gatherBlocks, kThreadsPerBlock, 0, m_config.stream>>>(
            m_deviceCompactedRows,
            m_config.compactStride,
            m_deviceSortIndicesOut,
            outputCount,
            m_deviceCompactedRowsSorted);

        launchResult = cudaGetLastError();
        if (launchResult != cudaSuccess) {
            error = FormatCudaError("GatherSortedRowsKernel launch", launchResult);
            return false;
        }

        deviceDownloadRows = m_deviceCompactedRowsSorted;
    }
    else if (m_config.topK > 0 && outputCount > m_config.topK) {
        outputCount = m_config.topK;
    }

    if (rowsBeforeSuppression != nullptr) {
        *rowsBeforeSuppression = outputCount;
    }

    if (nmsThreshold > 0.0f && outputCount > 1 && outputCount <= kGpuGreedyNmsMaxCandidates) {
        GreedyNmsCompactKernel<<<1, 1, 0, m_config.stream>>>(
            deviceDownloadRows,
            outputCount,
            m_config.compactStride,
            nmsThreshold,
            const_cast<float*>(deviceDownloadRows),
            m_deviceCompactedCount);

        const cudaError_t launchResult = cudaGetLastError();
        if (launchResult != cudaSuccess) {
            error = FormatCudaError("GreedyNmsCompactKernel launch", launchResult);
            return false;
        }

        const cudaError_t nmsCountCopyResult = cudaMemcpyAsync(
            m_deviceHostVisibleCount,
            m_deviceCompactedCount,
            sizeof(int),
            cudaMemcpyDeviceToHost,
            m_config.stream);
        if (nmsCountCopyResult != cudaSuccess) {
            error = FormatCudaError("cudaMemcpyAsync(gpuNmsCount)", nmsCountCopyResult);
            return false;
        }

        const cudaError_t nmsSyncResult = cudaStreamSynchronize(m_config.stream);
        if (nmsSyncResult != cudaSuccess) {
            error = FormatCudaError("cudaStreamSynchronize(gpuNmsCount)", nmsSyncResult);
            return false;
        }

        outputCount = *m_deviceHostVisibleCount;
        if (outputCount < 0) {
            outputCount = 0;
        }
        if (rowsBeforeSuppression != nullptr && outputCount > *rowsBeforeSuppression) {
            outputCount = *rowsBeforeSuppression;
        }
        usedGpuSuppression = true;
    }

    if (decodeTransform != nullptr &&
        decodeTransform->scale > 0.0f &&
        outputCount > 0) {
        constexpr int kThreadsPerBlock = 256;
        const int decodeBlocks = (outputCount + kThreadsPerBlock - 1) / kThreadsPerBlock;
        DecodeCompactRowsInPlaceKernel<<<decodeBlocks, kThreadsPerBlock, 0, m_config.stream>>>(
            const_cast<float*>(deviceDownloadRows),
            outputCount,
            m_config.compactStride,
            decodeTransform->scale,
            decodeTransform->padW,
            decodeTransform->padH,
            decodeTransform->roiX,
            decodeTransform->roiY,
            decodeTransform->originalWidth,
            decodeTransform->originalHeight);

        const cudaError_t launchResult = cudaGetLastError();
        if (launchResult != cudaSuccess) {
            error = FormatCudaError("DecodeCompactRowsInPlaceKernel launch", launchResult);
            return false;
        }
    }

    const std::size_t requiredFloats =
        static_cast<std::size_t>(outputCount) * static_cast<std::size_t>(m_config.compactStride);
    if (requiredFloats > hostCapacityFloats) {
        error = "TensorRTGpuPostprocess host output capacity is too small.";
        return false;
    }

    const cudaError_t rowsCopyResult = cudaMemcpyAsync(
        hostOutput,
        deviceDownloadRows,
        requiredFloats * sizeof(float),
        cudaMemcpyDeviceToHost,
        m_config.stream);
    if (rowsCopyResult != cudaSuccess) {
        error = FormatCudaError("cudaMemcpyAsync(compactedRows)", rowsCopyResult);
        return false;
    }

    const cudaError_t rowsSyncResult = cudaStreamSynchronize(m_config.stream);
    if (rowsSyncResult != cudaSuccess) {
        error = FormatCudaError("cudaStreamSynchronize(compactedRows)", rowsSyncResult);
        return false;
    }

    compactCount = outputCount;
    return true;
}

const float* TensorRTGpuPostprocess::GetCompactedRowsDevicePointer() const {
    return m_deviceCompactedRows;
}

const int* TensorRTGpuPostprocess::GetCompactedCountDevicePointer() const {
    return m_deviceCompactedCount;
}
