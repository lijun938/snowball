#pragma once

#include <Windows.h>
#include <QCAP.H>

#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

struct QCAPCaptureConfig {
    std::string deviceName = "SC0710 PCI";
    unsigned int deviceIndex = 0;
    ULONG videoInput = QCAP_INPUT_TYPE_HDMI;
    int width = 1920;
    int height = 1080;
    double frameRate = 240.0;
    bool interleaved = false;
    ULONG outputColorSpace = QCAP_COLORSPACE_TYPE_BGR24;
    ULONG downscaleMode = 0;
    ULONG postSkipFrameRate = 0;
    ULONG postAvgFrameRate = 0;
    HWND attachedWindow = nullptr;
    bool thumbDraw = false;
    bool maintainAspectRatio = false;
    bool requestGpuDirect = true;
    bool autoResyncOnPreviewStall = true;
    bool keepCpuShadowCopy = false;
};

struct QCAPFrameInfo {
    ULONG colorSpaceType = QCAP_COLORSPACE_TYPE_BGR24;
    int width = 0;
    int height = 0;
    double frameRate = 0.0;
    double sampleTime = 0.0;
    std::size_t bufferSize = 0;
    std::uint64_t sequence = 0;
};

struct QCAPFrameView {
    const BYTE* cpuData = nullptr;
    const BYTE* sourceBuffer = nullptr;
    const void* sourceDevicePointer = nullptr;
    std::size_t bufferSize = 0;
    ULONG colorSpaceType = QCAP_COLORSPACE_TYPE_BGR24;
    int width = 0;
    int height = 0;
    double frameRate = 0.0;
    double sampleTime = 0.0;
    std::uint64_t sequence = 0;
    bool sourceBufferMatchesGpuDirect = false;
    bool sourceBufferCudaMapped = false;
    bool cpuShadowCopyAvailable = false;
    int gpuDirectBufferIndex = -1;
    bool imageUpsideDown = false;
};

struct QCAPGpuDirectStatus {
    bool requested = false;
    bool queueConfigured = false;
    bool buffersAllocated = false;
    bool buffersBound = false;
    bool buffersCudaRegistered = false;
    bool buffersCudaMapped = false;
    ULONG queueSize = 0;
    std::size_t bufferSize = 0;
    bool callbackUsesBoundBuffer = false;
    std::uint64_t previewFramesObserved = 0;
    std::uint64_t matchedPreviewFrames = 0;
    int lastBoundBufferIndex = -1;
    std::string message;
};

struct QCAPRuntimeStats {
    std::uint64_t waitCalls = 0;
    std::uint64_t waitTimeouts = 0;
    std::uint64_t captureSuccesses = 0;
    std::uint64_t captureTimeouts = 0;
    std::uint64_t frameViewSuccesses = 0;
    std::uint64_t frameViewTimeouts = 0;
    std::uint64_t previewCallbacksObserved = 0;
    std::uint64_t formatChangesObserved = 0;
    std::uint64_t noSignalEventsObserved = 0;
    std::uint64_t signalRemovedEventsObserved = 0;
    std::uint64_t gpuDirectRebindAttempts = 0;
    std::uint64_t gpuDirectRebindSuccesses = 0;
    std::uint64_t gpuDirectRebindFailures = 0;
    std::uint64_t freshFramesAfterGpuDirectRebind = 0;
    std::uint64_t captureResyncAttempts = 0;
    std::uint64_t captureResyncSuccesses = 0;
    std::uint64_t captureResyncFailures = 0;
    std::uint64_t callbackGapSamples = 0;
    double avgCallbackGapMs = 0.0;
    double lastCallbackGapMs = 0.0;
    double maxCallbackGapMs = 0.0;
    double lastCallbackAgeMs = 0.0;
    std::uint64_t callbackGapOver2xExpected = 0;
    std::uint64_t callbackGapOver4xExpected = 0;
    std::uint64_t callbackGapResyncs = 0;
};

struct QCAPEnumeratedDevice {
    unsigned int index = 0;
    std::string name;
    ULONG infoHigh = 0;
    ULONG infoLow = 0;
};

class QCAPCapture {
public:
    QCAPCapture();
    ~QCAPCapture();

    bool Initialize(const QCAPCaptureConfig& config);
    bool Capture(cv::Mat& output, int timeoutMs = 30);
    const unsigned char* CaptureBGR(int timeoutMs = 30);
    bool GetLatestFrameView(QCAPFrameView& frameView, int timeoutMs = 30);
    bool ConvertFrameToMat(cv::Mat& output);
    bool ConvertFrameToPreviewMat(cv::Mat& output, int maxWidth);
    void DebugForceGpuDirectRebind();
    bool TryStartupFallbackToActiveInput();
    void Release();
    static bool IsFrameImageUpsideDown();

    bool IsInitialized() const { return m_initialized; }
    bool IsGpuDirectRequested() const { return m_config.requestGpuDirect; }
    bool IsGpuDirectActive() const { return m_gpuDirectActive; }
    QCAPGpuDirectStatus GetGpuDirectStatus() const;
    QCAPRuntimeStats GetRuntimeStats() const;
    const std::string& GetLastError() const { return m_lastError; }
    QCAPFrameInfo GetFrameInfo() const;
    const std::vector<QCAPEnumeratedDevice>& GetEnumeratedDevices() const { return m_enumeratedDevices; }
    void PrintInputDiagnostics(const char* context = nullptr) const;
    void ResetRuntimeTelemetry();
    bool HasCpuShadowBuffer() const;

private:
    static QRETURN QCAP_EXPORT OnFormatChanged(
        PVOID pDevice,
        ULONG nVideoInput,
        ULONG nAudioInput,
        ULONG nVideoWidth,
        ULONG nVideoHeight,
        BOOL bVideoIsInterleaved,
        double dVideoFrameRate,
        ULONG nAudioChannels,
        ULONG nAudioBitsPerSample,
        ULONG nAudioSampleFrequency,
        PVOID pUserData
    );

    static QRETURN QCAP_EXPORT OnVideoPreview(
        PVOID pDevice,
        double dSampleTime,
        BYTE* pFrameBuffer,
        ULONG nFrameBufferLen,
        PVOID pUserData
    );

    static QRETURN QCAP_EXPORT OnNoSignalDetected(
        PVOID pDevice,
        ULONG nVideoInput,
        ULONG nAudioInput,
        PVOID pUserData
    );

    static QRETURN QCAP_EXPORT OnSignalRemoved(
        PVOID pDevice,
        ULONG nVideoInput,
        ULONG nAudioInput,
        PVOID pUserData
    );

    bool ConfigureDevice();
    bool InitializeRuntimeConfiguration();
    bool EnumerateDevices();
    bool TrySetupGpuDirect();
    void TeardownGpuDirect();
    bool TryMapGpuDirectBuffersToCuda();
    void UnmapGpuDirectBuffersFromCuda();
    bool RebindGpuDirectIfNeeded();
    bool AttemptAutoResyncAfterPreviewStall(double callbackAgeMs);
    bool WaitForFrame(int timeoutMs);
    bool EnsureCpuShadowBufferLocked();
    int FindGpuDirectBufferIndex(const BYTE* frameBuffer) const;
    void MarkGpuDirectRebindPendingLocked(const std::string& reason, std::size_t requiredBufferSize);
    void ResetFrameState();
    void SetLastError(const std::string& message, QRESULT result = QCAP_RS_SUCCESSFUL);
    void HandleNoSignalDetected(ULONG videoInput, ULONG audioInput);
    void HandleSignalRemoved(ULONG videoInput, ULONG audioInput);
    void HandleFormatChanged(ULONG width, ULONG height, BOOL interleaved, double frameRate);
    QRETURN HandleVideoPreview(double sampleTime, BYTE* frameBuffer, ULONG frameBufferLen);

    static std::size_t CalculateFrameBufferSize(ULONG colorSpaceType, int width, int height);

    PVOID m_device;
    QCAPCaptureConfig m_config;
    QCAPFrameInfo m_frameInfo;
    QCAPGpuDirectStatus m_gpuDirectStatus;
    QCAPRuntimeStats m_runtimeStats;
    std::vector<QCAPEnumeratedDevice> m_enumeratedDevices;
    std::vector<unsigned char> m_frameBuffer;
    std::vector<BYTE*> m_gpuDirectBuffers;
    std::vector<void*> m_gpuDirectMappedDevicePointers;

    mutable std::mutex m_frameMutex;
    std::condition_variable m_frameCondition;

    bool m_initialized;
    bool m_hasFrame;
    bool m_hasCpuShadowCopy;
    bool m_gpuDirectActive;
    bool m_shuttingDown;
    bool m_gpuDirectRebindPending;
    bool m_awaitingFreshFrameAfterRebind;
    std::uint64_t m_lastDeliveredSequence;
    const BYTE* m_lastSourceFrameBuffer;
    int m_lastGpuDirectBufferIndex;
    std::uint32_t m_consecutivePreviewWaitTimeouts;
    std::chrono::steady_clock::time_point m_nextAutoResyncAllowedTime;
    bool m_hasLastPreviewArrival;
    std::chrono::steady_clock::time_point m_lastPreviewArrivalTime;
    double m_totalPreviewGapMs;
    std::size_t m_requestedGpuDirectBufferSize;
    std::string m_gpuDirectRebindReason;
    std::string m_lastError;
};
