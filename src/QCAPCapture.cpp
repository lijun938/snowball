#include "QCAPCapture.h"

#include <Windows.h>
#include <cuda_runtime_api.h>
#include <delayimp.h>

#include <array>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <sstream>

namespace {
std::size_t BytesForPackedFrame(int width, int height, int bytesPerPixel) {
    return static_cast<std::size_t>(width) * static_cast<std::size_t>(height) * static_cast<std::size_t>(bytesPerPixel);
}

std::string FormatQcapResult(const char* operation, QRESULT result) {
    std::ostringstream oss;
    oss << operation << " failed (0x" << std::uppercase << std::hex << static_cast<unsigned int>(result) << ")";
    return oss.str();
}

std::string FormatCudaError(const char* operation, cudaError_t result) {
    std::ostringstream oss;
    oss << operation << " failed";
    if (result != cudaSuccess) {
        oss << " (" << cudaGetErrorName(result) << ": " << cudaGetErrorString(result) << ")";
    }
    return oss.str();
}

std::string DecodePackedDeviceName(ULONGLONG encodedName) {
    std::string name;
    for (int shift = 56; shift >= 0; shift -= 8) {
        const unsigned char ch = static_cast<unsigned char>((encodedName >> shift) & 0xFF);
        if (ch == 0) {
            continue;
        }
        name.push_back(static_cast<char>(ch));
    }

    return name;
}

const char* DescribeQcapInput(ULONG inputType) {
    switch (inputType) {
    case QCAP_INPUT_TYPE_HDMI:
        return "HDMI";
    case QCAP_INPUT_TYPE_DVI_D:
        return "DVI";
    case QCAP_INPUT_TYPE_VGA:
        return "VGA";
    case QCAP_INPUT_TYPE_SDI:
        return "SDI";
    case QCAP_INPUT_TYPE_DISPLAY_PORT:
        return "DisplayPort";
    case QCAP_INPUT_TYPE_AUTO:
        return "Auto";
    default:
        return "Unknown";
    }
}

const char* DescribeQcapColorSpace(ULONG colorSpaceType) {
    switch (colorSpaceType) {
    case QCAP_COLORSPACE_TYPE_BGR24:
        return "BGR24";
    case QCAP_COLORSPACE_TYPE_RGB24:
        return "RGB24";
    case QCAP_COLORSPACE_TYPE_YUY2:
        return "YUY2";
    case QCAP_COLORSPACE_TYPE_UYVY:
        return "UYVY";
    case QCAP_COLORSPACE_TYPE_NV12:
        return "NV12";
    case QCAP_COLORSPACE_TYPE_I420:
        return "I420";
    case QCAP_COLORSPACE_TYPE_YV12:
        return "YV12";
    default:
        return "Unknown";
    }
}

std::string g_qcapDelayLoadError;
std::once_flag g_qcapRuntimePathOnce;
std::vector<DLL_DIRECTORY_COOKIE> g_qcapRuntimeDirCookies;
std::vector<std::filesystem::path> g_qcapRuntimeProbeDirs;
std::string g_qcapRuntimePathError;
constexpr bool kQcapBgrPreviewImageUpsideDown = true;
constexpr double kGpuDirectQueueHeadroomMs = 50.0;
constexpr ULONG kMinGpuDirectPreviewQueueSize = 8;
constexpr std::uint32_t kAutoResyncRequiredTimeouts = 3;
constexpr double kAutoResyncMinThresholdMs = 500.0;
constexpr double kAutoResyncExpectedFrameMultiplier = 48.0;

std::filesystem::path GetExecutableDirectory() {
    std::array<wchar_t, MAX_PATH> buffer{};
    const DWORD len = ::GetModuleFileNameW(nullptr, buffer.data(), static_cast<DWORD>(buffer.size()));
    if (len == 0 || len >= buffer.size()) {
        return {};
    }

    std::filesystem::path modulePath(buffer.data());
    return modulePath.parent_path();
}

std::string NarrowPath(const std::filesystem::path& path) {
    return path.u8string();
}

ULONG CalculatePreferredGpuDirectQueueSize(double frameRate) {
    if (!(frameRate > 0.0)) {
        return kMinGpuDirectPreviewQueueSize;
    }

    const double preferredQueue =
        std::ceil((frameRate * kGpuDirectQueueHeadroomMs) / 1000.0);
    const auto preferredQueueSize = static_cast<ULONG>(preferredQueue);
    return (std::max)(kMinGpuDirectPreviewQueueSize, preferredQueueSize);
}

void AppendUniquePath(std::vector<std::filesystem::path>& paths, const std::filesystem::path& candidate) {
    if (candidate.empty()) {
        return;
    }

    std::error_code ec;
    const std::filesystem::path normalized = std::filesystem::weakly_canonical(candidate, ec);
    const std::filesystem::path comparable = ec ? candidate.lexically_normal() : normalized;
    if (!std::filesystem::exists(comparable)) {
        return;
    }

    const bool alreadyPresent = std::find(paths.begin(), paths.end(), comparable) != paths.end();
    if (!alreadyPresent) {
        paths.push_back(comparable);
    }
}

std::filesystem::path GetEnvironmentPath(const wchar_t* variableName) {
    std::wstring value(32767, L'\0');
    const DWORD len = ::GetEnvironmentVariableW(variableName, value.data(), static_cast<DWORD>(value.size()));
    if (len == 0 || len >= value.size()) {
        return {};
    }

    value.resize(len);
    return std::filesystem::path(value);
}

std::vector<std::filesystem::path> CollectQcapRuntimeDirectories() {
    std::vector<std::filesystem::path> paths;
    AppendUniquePath(paths, GetEnvironmentPath(L"GPT_SNOW_AIM_QCAP_CODEC_DIR"));
    AppendUniquePath(paths, GetEnvironmentPath(L"AMTOOBS_QCAP_CODEC_DIR"));
    AppendUniquePath(paths, GetEnvironmentPath(L"MULTIMEDIA_CODEC_DIR"));

    const auto programFilesX86 = GetEnvironmentPath(L"ProgramFiles(x86)");
    if (!programFilesX86.empty()) {
        AppendUniquePath(paths, programFilesX86 / "Multimedia" / "CODECS" / "X64");
    }

    const auto programFiles = GetEnvironmentPath(L"ProgramW6432");
    if (!programFiles.empty()) {
        AppendUniquePath(paths, programFiles / "Multimedia" / "CODECS" / "X64");
    }

    AppendUniquePath(paths, std::filesystem::path(LR"(C:\Program Files (x86)\Multimedia\CODECS\X64)"));
    AppendUniquePath(paths, std::filesystem::path(LR"(C:\Program Files\Multimedia\CODECS\X64)"));
    return paths;
}

void InitializeQcapRuntimeDirectories() {
    std::call_once(g_qcapRuntimePathOnce, []() {
        const DWORD searchFlags = LOAD_LIBRARY_SEARCH_DEFAULT_DIRS | LOAD_LIBRARY_SEARCH_USER_DIRS;
        if (!::SetDefaultDllDirectories(searchFlags)) {
            std::ostringstream oss;
            oss << "SetDefaultDllDirectories failed (Win32=" << ::GetLastError() << ")";
            g_qcapRuntimePathError = oss.str();
            return;
        }

        for (const auto& directory : CollectQcapRuntimeDirectories()) {
            DLL_DIRECTORY_COOKIE cookie = ::AddDllDirectory(directory.c_str());
            if (cookie == nullptr) {
                continue;
            }

            g_qcapRuntimeDirCookies.push_back(cookie);
            AppendUniquePath(g_qcapRuntimeProbeDirs, directory);
        }
    });
}

std::vector<std::filesystem::path> BuildRuntimeProbeDirectories() {
    std::vector<std::filesystem::path> paths;
    AppendUniquePath(paths, GetExecutableDirectory());
    for (const auto& directory : g_qcapRuntimeProbeDirs) {
        AppendUniquePath(paths, directory);
    }
    return paths;
}

std::filesystem::path LocateRuntimeDll(const char* dllName) {
    for (const auto& directory : BuildRuntimeProbeDirectories()) {
        const std::filesystem::path candidate = directory / dllName;
        if (std::filesystem::exists(candidate)) {
            return candidate;
        }
    }

    return {};
}

std::string DescribeRuntimeProbeDirectories() {
    std::ostringstream oss;
    bool first = true;
    for (const auto& directory : BuildRuntimeProbeDirectories()) {
        if (!first) {
            oss << ", ";
        }
        oss << NarrowPath(directory);
        first = false;
    }

    return oss.str();
}

std::string ProbeMiniRuntimeDependencies() {
    static constexpr const char* kRuntimeDlls[] = {
        "QCAP.X64.DLL",
        "AMESDK.X64.DLL",
        "CCUVC.X64.DLL",
        "CCBOX.X64.DLL",
        "EYCOPP.x64.dll"
    };

    if (!g_qcapRuntimePathError.empty()) {
        return g_qcapRuntimePathError;
    }

    std::ostringstream issues;
    bool hasIssue = false;
    const DWORD searchFlags = LOAD_LIBRARY_SEARCH_DEFAULT_DIRS | LOAD_LIBRARY_SEARCH_USER_DIRS;

    for (const char* dllName : kRuntimeDlls) {
        const std::filesystem::path dllPath = LocateRuntimeDll(dllName);
        if (dllPath.empty()) {
            if (hasIssue) {
                issues << "; ";
            }
            issues << dllName << " was not found in [" << DescribeRuntimeProbeDirectories() << "]";
            hasIssue = true;
            continue;
        }

        HMODULE module = ::LoadLibraryExW(dllPath.c_str(), nullptr, searchFlags);
        if (module == nullptr) {
            if (hasIssue) {
                issues << "; ";
            }
            issues << dllName << " exists at " << NarrowPath(dllPath)
                << " but failed to load (Win32=" << ::GetLastError() << ")";
            hasIssue = true;
            continue;
        }

        ::FreeLibrary(module);
    }

    return hasIssue ? issues.str() : std::string{};
}

FARPROC WINAPI QcapDelayLoadFailureHook(unsigned int event, DelayLoadInfo* info) {
    if (event != dliFailLoadLib || info == nullptr || info->szDll == nullptr) {
        return nullptr;
    }

    std::ostringstream oss;
    oss << "Unable to load delayed runtime dependency '" << info->szDll << "'";
    if (GetLastError() != ERROR_SUCCESS) {
        oss << " (Win32=" << GetLastError() << ")";
    }
    g_qcapDelayLoadError = oss.str();
    return nullptr;
}
}

extern "C" {
ExternC const PfnDliHook __pfnDliFailureHook2 = QcapDelayLoadFailureHook;
}

QCAPCapture::QCAPCapture()
    : m_device(nullptr)
    , m_initialized(false)
    , m_hasFrame(false)
    , m_hasCpuShadowCopy(false)
    , m_gpuDirectActive(false)
    , m_shuttingDown(false)
    , m_gpuDirectRebindPending(false)
    , m_awaitingFreshFrameAfterRebind(false)
    , m_lastDeliveredSequence(0)
    , m_lastSourceFrameBuffer(nullptr)
    , m_lastGpuDirectBufferIndex(-1)
    , m_consecutivePreviewWaitTimeouts(0)
    , m_nextAutoResyncAllowedTime(std::chrono::steady_clock::time_point::min())
    , m_hasLastPreviewArrival(false)
    , m_totalPreviewGapMs(0.0)
    , m_requestedGpuDirectBufferSize(0) {
}

QCAPCapture::~QCAPCapture() {
    Release();
}

bool QCAPCapture::Initialize(const QCAPCaptureConfig& config) {
    Release();

    m_config = config;
    m_gpuDirectStatus = QCAPGpuDirectStatus{};
    m_gpuDirectStatus.requested = m_config.requestGpuDirect;
    m_runtimeStats = QCAPRuntimeStats{};
    m_shuttingDown = false;
    m_gpuDirectRebindPending = false;
    m_awaitingFreshFrameAfterRebind = false;
    m_consecutivePreviewWaitTimeouts = 0;
    m_nextAutoResyncAllowedTime = std::chrono::steady_clock::time_point::min();
    m_requestedGpuDirectBufferSize = 0;
    m_gpuDirectRebindReason.clear();
    g_qcapDelayLoadError.clear();
    InitializeQcapRuntimeDirectories();
    if (!InitializeRuntimeConfiguration()) {
        return false;
    }

    if (!EnumerateDevices()) {
        return false;
    }

    const std::string runtimeProbeIssues = ProbeMiniRuntimeDependencies();
    if (!runtimeProbeIssues.empty()) {
        SetLastError("QCAP runtime preflight failed: " + runtimeProbeIssues);
        return false;
    }

    std::cout << "[QCAP] Create request device='" << m_config.deviceName
        << "' index=" << m_config.deviceIndex
        << " hwnd=" << m_config.attachedWindow
        << " thumbDraw=" << (m_config.thumbDraw ? "yes" : "no")
        << " keepAspect=" << (m_config.maintainAspectRatio ? "yes" : "no")
        << std::endl;

    HMODULE qcapProbe = ::LoadLibraryExW(L"QCAP.X64.DLL", nullptr, LOAD_LIBRARY_SEARCH_DEFAULT_DIRS | LOAD_LIBRARY_SEARCH_USER_DIRS);
    if (qcapProbe == nullptr) {
        const DWORD win32Error = ::GetLastError();
        std::ostringstream oss;
        oss << "QCAP runtime preflight failed while loading QCAP.X64.DLL";
        if (!g_qcapDelayLoadError.empty()) {
            oss << ": " << g_qcapDelayLoadError;
        }
        else if (win32Error != ERROR_SUCCESS) {
            oss << " (Win32=" << win32Error << ")";
        }
        SetLastError(oss.str());
        return false;
    }
    ::FreeLibrary(qcapProbe);

    QRESULT result = QCAP_CREATE(
        const_cast<char*>(m_config.deviceName.c_str()),
        m_config.deviceIndex,
        m_config.attachedWindow,
        &m_device,
        m_config.thumbDraw ? TRUE : FALSE,
        m_config.maintainAspectRatio ? TRUE : FALSE);
    if (result != QCAP_RT_OK || m_device == nullptr) {
        if (!g_qcapDelayLoadError.empty()) {
            SetLastError(g_qcapDelayLoadError);
        }
        else {
            SetLastError("QCAP_CREATE failed", result);
        }
        Release();
        return false;
    }

    if (!ConfigureDevice()) {
        Release();
        return false;
    }

    result = QCAP_REGISTER_FORMAT_CHANGED_CALLBACK(m_device, &QCAPCapture::OnFormatChanged, this);
    if (result != QCAP_RT_OK) {
        SetLastError("QCAP_REGISTER_FORMAT_CHANGED_CALLBACK failed", result);
        Release();
        return false;
    }

    result = QCAP_REGISTER_VIDEO_PREVIEW_CALLBACK(m_device, &QCAPCapture::OnVideoPreview, this);
    if (result != QCAP_RT_OK) {
        SetLastError("QCAP_REGISTER_VIDEO_PREVIEW_CALLBACK failed", result);
        Release();
        return false;
    }

    result = QCAP_REGISTER_NO_SIGNAL_DETECTED_CALLBACK(m_device, &QCAPCapture::OnNoSignalDetected, this);
    if (result != QCAP_RT_OK) {
        SetLastError("QCAP_REGISTER_NO_SIGNAL_DETECTED_CALLBACK failed", result);
        Release();
        return false;
    }

    result = QCAP_REGISTER_SIGNAL_REMOVED_CALLBACK(m_device, &QCAPCapture::OnSignalRemoved, this);
    if (result != QCAP_RT_OK) {
        SetLastError("QCAP_REGISTER_SIGNAL_REMOVED_CALLBACK failed", result);
        Release();
        return false;
    }

    m_initialized = true;

    result = QCAP_RUN(m_device);
    if (result != QCAP_RT_OK) {
        SetLastError("QCAP_RUN failed", result);
        Release();
        return false;
    }

    m_frameInfo.colorSpaceType = m_config.outputColorSpace;
    m_frameInfo.width = m_config.width;
    m_frameInfo.height = m_config.height;
    m_frameInfo.frameRate = m_config.frameRate;

    if (m_config.requestGpuDirect) {
        TrySetupGpuDirect();
    }

    std::cout << "[QCAP] Initialized device '" << m_config.deviceName
        << "' index " << m_config.deviceIndex
        << " input(requested)=" << DescribeQcapInput(m_config.videoInput)
        << "(" << m_config.videoInput << ")"
        << " outputColorSpace=" << m_frameInfo.colorSpaceType
        << " size=" << m_frameInfo.width << "x" << m_frameInfo.height
        << " fps(requested)=" << m_config.frameRate
        << " fps(active)=" << m_frameInfo.frameRate
        << std::endl;
    if (m_gpuDirectStatus.requested) {
        std::cout << "[QCAP] GPUDirect queue=" << m_gpuDirectStatus.queueSize
            << " allocated=" << (m_gpuDirectStatus.buffersAllocated ? "yes" : "no")
            << " bound=" << (m_gpuDirectStatus.buffersBound ? "yes" : "no")
            << " status=" << (m_gpuDirectStatus.message.empty() ? "n/a" : m_gpuDirectStatus.message)
            << std::endl;
    }

    return true;
}

bool QCAPCapture::InitializeRuntimeConfiguration() {
    QRESULT result = QCAP_SET_SYSTEM_CONFIGURATION(
        TRUE,
        TRUE,
        TRUE,
        TRUE,
        TRUE,
        3000,
        FALSE,
        nullptr,
        FALSE,
        TRUE,
        FALSE,
        TRUE,
        FALSE,
        const_cast<char*>("C:\\"));
    if (result != QCAP_RT_OK) {
        SetLastError("QCAP_SET_SYSTEM_CONFIGURATION failed", result);
        return false;
    }

    return true;
}

bool QCAPCapture::EnumerateDevices() {
    m_enumeratedDevices.clear();

    ULONGLONG* videoDeviceList = nullptr;
    ULONGLONG* videoEncoderDeviceList = nullptr;
    ULONGLONG* audioDeviceList = nullptr;
    ULONGLONG* audioEncoderDeviceList = nullptr;
    ULONG videoDeviceCount = 0;
    ULONG videoEncoderDeviceCount = 0;
    ULONG audioDeviceCount = 0;
    ULONG audioEncoderDeviceCount = 0;

    const QRESULT result = QCAP_DEVICE_ENUMERATION(
        &videoDeviceList,
        &videoDeviceCount,
        &videoEncoderDeviceList,
        &videoEncoderDeviceCount,
        &audioDeviceList,
        &audioDeviceCount,
        &audioEncoderDeviceList,
        &audioEncoderDeviceCount,
        QCAP_ENUM_TYPE_DEVICE_NAME);
    if (result != QCAP_RT_OK) {
        SetLastError("QCAP_DEVICE_ENUMERATION failed", result);
        return false;
    }

    for (ULONG i = 0; i < videoDeviceCount; ++i) {
        QCAPEnumeratedDevice device;
        device.index = static_cast<unsigned int>(i);
        device.name = DecodePackedDeviceName(videoDeviceList[i]);
        QCAP_GET_DEVICE_ENUMERATION_ITEM_INFO(static_cast<UINT>(i), videoDeviceList, &device.infoHigh, &device.infoLow);
        m_enumeratedDevices.push_back(device);
    }

    if (!m_enumeratedDevices.empty()) {
        std::cout << "[QCAP] Enumerated video devices:" << std::endl;
        for (const auto& device : m_enumeratedDevices) {
            std::cout << "  - [" << device.index << "] " << device.name
                << " infoH=0x" << std::hex << std::uppercase << device.infoHigh
                << " infoL=0x" << device.infoLow << std::dec
                << std::nouppercase << std::endl;
        }
    }
    else {
        std::cout << "[QCAP] Enumerated video devices: none" << std::endl;
    }

    return true;
}

bool QCAPCapture::ConfigureDevice() {
    QRESULT result = QCAP_SET_VIDEO_INPUT(m_device, m_config.videoInput);
    if (result != QCAP_RT_OK) {
        SetLastError("QCAP_SET_VIDEO_INPUT failed", result);
        return false;
    }

    result = QCAP_SET_VIDEO_DEFAULT_OUTPUT_FORMAT(
        m_device,
        m_config.outputColorSpace,
        static_cast<ULONG>(m_config.width),
        static_cast<ULONG>(m_config.height),
        m_config.interleaved ? TRUE : FALSE,
        m_config.frameRate);
    if (result != QCAP_RT_OK) {
        SetLastError("QCAP_SET_VIDEO_DEFAULT_OUTPUT_FORMAT failed", result);
        return false;
    }

    result = QCAP_SET_VIDEO_PREVIEW_PROPERTY_EX(
        m_device,
        m_config.downscaleMode,
        m_config.postSkipFrameRate,
        m_config.postAvgFrameRate);
    if (result != QCAP_RT_OK) {
        SetLastError("QCAP_SET_VIDEO_PREVIEW_PROPERTY_EX failed", result);
        return false;
    }

    return true;
}

bool QCAPCapture::TryStartupFallbackToActiveInput() {
    if (m_device == nullptr) {
        return false;
    }
    if (m_shuttingDown || !m_initialized) {
        return false;
    }

    ULONG activeInput = 0;
    const QRESULT inputResult = QCAP_GET_VIDEO_INPUT(m_device, &activeInput);
    if (inputResult != QCAP_RT_OK) {
        return false;
    }

    ULONG colorSpaceType = 0;
    ULONG width = 0;
    ULONG height = 0;
    BOOL interleaved = FALSE;
    double frameRate = 0.0;
    const QRESULT formatResult = QCAP_GET_VIDEO_CURRENT_INPUT_FORMAT(
        m_device,
        &colorSpaceType,
        &width,
        &height,
        &interleaved,
        &frameRate);
    if (formatResult != QCAP_RT_OK) {
        return false;
    }

    if (activeInput == m_config.videoInput || width == 0 || height == 0 || frameRate <= 0.0) {
        return false;
    }

    const ULONG requestedInput = m_config.videoInput;
    std::cout << "[QCAP] startup fallback reconfiguring input from "
        << DescribeQcapInput(requestedInput) << "(" << requestedInput << ")"
        << " to vendor-active input "
        << DescribeQcapInput(activeInput) << "(" << activeInput << ")"
        << " because it already reports a valid format "
        << width << "x" << height << " @" << frameRate
        << std::endl;

    const QRESULT stopResult = QCAP_STOP(m_device);
    if (stopResult != QCAP_RT_OK) {
        SetLastError("QCAP_STOP failed during startup active-input fallback", stopResult);
        return false;
    }

    {
        std::lock_guard<std::mutex> lock(m_frameMutex);
        ResetFrameState();
        m_lastDeliveredSequence = 0;
        m_runtimeStats = QCAPRuntimeStats{};
        m_gpuDirectStatus.callbackUsesBoundBuffer = false;
        m_gpuDirectStatus.previewFramesObserved = 0;
        m_gpuDirectStatus.matchedPreviewFrames = 0;
        m_gpuDirectStatus.lastBoundBufferIndex = -1;
        m_consecutivePreviewWaitTimeouts = 0;
        m_hasLastPreviewArrival = false;
        m_totalPreviewGapMs = 0.0;
    }

    TeardownGpuDirect();

    m_config.videoInput = activeInput;
    if (!ConfigureDevice()) {
        return false;
    }

    m_frameInfo.colorSpaceType = m_config.outputColorSpace;
    m_frameInfo.width = m_config.width;
    m_frameInfo.height = m_config.height;
    m_frameInfo.frameRate = m_config.frameRate;

    const QRESULT runResult = QCAP_RUN(m_device);
    if (runResult != QCAP_RT_OK) {
        SetLastError("QCAP_RUN failed during startup active-input fallback", runResult);
        return false;
    }

    if (m_config.requestGpuDirect) {
        TrySetupGpuDirect();
    }

    ResetRuntimeTelemetry();

    m_lastError.clear();
    std::cout << "[QCAP] startup fallback input reconfiguration completed." << std::endl;
    return true;
}

bool QCAPCapture::TrySetupGpuDirect() {
    m_gpuDirectActive = false;
    m_gpuDirectStatus.queueConfigured = false;
    m_gpuDirectStatus.buffersAllocated = false;
    m_gpuDirectStatus.buffersBound = false;
    m_gpuDirectStatus.buffersCudaRegistered = false;
    m_gpuDirectStatus.buffersCudaMapped = false;
    m_gpuDirectStatus.queueSize = 0;
    m_gpuDirectStatus.bufferSize = 0;
    m_gpuDirectStatus.callbackUsesBoundBuffer = false;
    m_gpuDirectStatus.previewFramesObserved = 0;
    m_gpuDirectStatus.matchedPreviewFrames = 0;
    m_gpuDirectStatus.lastBoundBufferIndex = -1;
    m_gpuDirectStatus.message.clear();

    const std::size_t bufferSize = m_requestedGpuDirectBufferSize != 0
        ? m_requestedGpuDirectBufferSize
        : CalculateFrameBufferSize(
        m_frameInfo.colorSpaceType,
        m_frameInfo.width,
        m_frameInfo.height);
    if (bufferSize == 0) {
        m_gpuDirectStatus.message = "GPUDirect skipped because the output color space does not have a known buffer size yet.";
        return false;
    }

    ULONG queueSize = 0;
    QRESULT result = QCAP_GET_VIDEO_GPUDIRECT_PREVIEW_QUEUE_SIZE(m_device, &queueSize);
    const ULONG preferredQueueSize = CalculatePreferredGpuDirectQueueSize(
        m_frameInfo.frameRate > 0.0 ? m_frameInfo.frameRate : m_config.frameRate);
    if (result != QCAP_RT_OK || queueSize < preferredQueueSize) {
        queueSize = preferredQueueSize;
        result = QCAP_SET_VIDEO_GPUDIRECT_PREVIEW_QUEUE_SIZE(m_device, queueSize);
        if (result != QCAP_RT_OK) {
            m_gpuDirectStatus.message = FormatQcapResult("QCAP_SET_VIDEO_GPUDIRECT_PREVIEW_QUEUE_SIZE", result);
            return false;
        }
    }

    m_gpuDirectStatus.queueConfigured = true;
    m_gpuDirectStatus.queueSize = queueSize;
    m_gpuDirectStatus.bufferSize = bufferSize;
    m_requestedGpuDirectBufferSize = bufferSize;
    m_gpuDirectRebindPending = false;
    m_gpuDirectRebindReason.clear();

    m_gpuDirectBuffers.assign(queueSize, nullptr);

    for (ULONG i = 0; i < queueSize; ++i) {
        BYTE* frameBuffer = nullptr;
        result = QCAP_ALLOC_VIDEO_GPUDIRECT_PREVIEW_BUFFER(
            m_device,
            &frameBuffer,
            static_cast<ULONG>(bufferSize));
        if (result != QCAP_RT_OK || frameBuffer == nullptr) {
            m_gpuDirectStatus.message = FormatQcapResult("QCAP_ALLOC_VIDEO_GPUDIRECT_PREVIEW_BUFFER", result);
            TeardownGpuDirect();
            return false;
        }

        m_gpuDirectBuffers[i] = frameBuffer;
    }

    m_gpuDirectStatus.buffersAllocated = true;

    for (ULONG i = 0; i < queueSize; ++i) {
        result = QCAP_BIND_VIDEO_GPUDIRECT_PREVIEW_BUFFER(
            m_device,
            static_cast<UINT>(i),
            m_gpuDirectBuffers[i],
            static_cast<ULONG>(bufferSize));
        if (result != QCAP_RT_OK) {
            m_gpuDirectStatus.message = FormatQcapResult("QCAP_BIND_VIDEO_GPUDIRECT_PREVIEW_BUFFER", result);
            TeardownGpuDirect();
            return false;
        }
    }

    m_gpuDirectStatus.buffersBound = true;
    if (TryMapGpuDirectBuffersToCuda()) {
        m_gpuDirectStatus.message = "QCAP GPUDirect buffers are allocated, bound, and CUDA-mapped.";
    }
    else if (m_gpuDirectStatus.message.empty()) {
        m_gpuDirectStatus.message = "QCAP GPUDirect buffers are allocated and bound, but CUDA mapping is not active yet.";
    }
    m_gpuDirectActive = true;
    return true;
}

void QCAPCapture::TeardownGpuDirect() {
    UnmapGpuDirectBuffersFromCuda();

    const std::size_t bufferSize = m_gpuDirectStatus.bufferSize;
    for (std::size_t i = 0; i < m_gpuDirectBuffers.size(); ++i) {
        BYTE* frameBuffer = m_gpuDirectBuffers[i];
        if (frameBuffer == nullptr || m_device == nullptr || bufferSize == 0) {
            continue;
        }

        QCAP_UNBIND_VIDEO_GPUDIRECT_PREVIEW_BUFFER(
            m_device,
            static_cast<UINT>(i),
            frameBuffer,
            static_cast<ULONG>(bufferSize));
        QCAP_FREE_VIDEO_GPUDIRECT_PREVIEW_BUFFER(
            m_device,
            frameBuffer,
            static_cast<ULONG>(bufferSize));
    }

    m_gpuDirectBuffers.clear();
    m_gpuDirectMappedDevicePointers.clear();
    m_gpuDirectStatus.buffersBound = false;
    m_gpuDirectStatus.buffersAllocated = false;
    m_gpuDirectStatus.queueConfigured = false;
    m_gpuDirectStatus.buffersCudaRegistered = false;
    m_gpuDirectStatus.buffersCudaMapped = false;
    m_gpuDirectStatus.queueSize = 0;
    m_gpuDirectStatus.bufferSize = 0;
    m_gpuDirectActive = false;
}

bool QCAPCapture::TryMapGpuDirectBuffersToCuda() {
    const std::size_t bufferSize = m_gpuDirectStatus.bufferSize;
    if (bufferSize == 0 || m_gpuDirectBuffers.empty()) {
        return false;
    }

    m_gpuDirectMappedDevicePointers.assign(m_gpuDirectBuffers.size(), nullptr);

    for (std::size_t i = 0; i < m_gpuDirectBuffers.size(); ++i) {
        BYTE* frameBuffer = m_gpuDirectBuffers[i];
        if (frameBuffer == nullptr) {
            m_gpuDirectStatus.message = "CUDA mapping skipped because a GPUDirect preview buffer pointer is null.";
            UnmapGpuDirectBuffersFromCuda();
            return false;
        }

        const cudaError_t registerResult = cudaHostRegister(
            frameBuffer,
            bufferSize,
            cudaHostRegisterMapped | cudaHostRegisterPortable);
        if (registerResult != cudaSuccess && registerResult != cudaErrorHostMemoryAlreadyRegistered) {
            m_gpuDirectStatus.message = FormatCudaError("cudaHostRegister(GPUDirect preview buffer)", registerResult);
            cudaGetLastError();
            UnmapGpuDirectBuffersFromCuda();
            return false;
        }

        void* mappedDevicePointer = nullptr;
        const cudaError_t devicePointerResult = cudaHostGetDevicePointer(&mappedDevicePointer, frameBuffer, 0);
        if (devicePointerResult != cudaSuccess) {
            m_gpuDirectStatus.message = FormatCudaError("cudaHostGetDevicePointer(GPUDirect preview buffer)", devicePointerResult);
            cudaGetLastError();
            UnmapGpuDirectBuffersFromCuda();
            return false;
        }

        m_gpuDirectMappedDevicePointers[i] = mappedDevicePointer;
    }

    m_gpuDirectStatus.buffersCudaRegistered = true;
    m_gpuDirectStatus.buffersCudaMapped = true;
    return true;
}

void QCAPCapture::UnmapGpuDirectBuffersFromCuda() {
    for (std::size_t i = 0; i < m_gpuDirectBuffers.size(); ++i) {
        BYTE* frameBuffer = m_gpuDirectBuffers[i];
        if (frameBuffer == nullptr) {
            continue;
        }

        const cudaError_t unregisterResult = cudaHostUnregister(frameBuffer);
        if (unregisterResult != cudaSuccess) {
            if (unregisterResult != cudaErrorHostMemoryNotRegistered) {
                cudaGetLastError();
            }
            continue;
        }
    }

    m_gpuDirectMappedDevicePointers.clear();
    m_gpuDirectStatus.buffersCudaRegistered = false;
    m_gpuDirectStatus.buffersCudaMapped = false;
}

bool QCAPCapture::RebindGpuDirectIfNeeded() {
    bool shouldRebind = false;
    std::string reason;
    {
        std::lock_guard<std::mutex> lock(m_frameMutex);
        shouldRebind = m_gpuDirectRebindPending &&
            !m_shuttingDown &&
            m_config.requestGpuDirect &&
            m_device != nullptr;
        if (shouldRebind) {
            reason = m_gpuDirectRebindReason;
            m_gpuDirectRebindPending = false;
            m_gpuDirectRebindReason.clear();
        }
    }

    if (!shouldRebind) {
        return m_gpuDirectActive;
    }

    std::cout << "[QCAP] Rebinding GPUDirect buffers after format/size drift";
    if (!reason.empty()) {
        std::cout << ": " << reason;
    }
    std::cout << std::endl;

    const bool wasRunning = m_device != nullptr;
    {
        std::lock_guard<std::mutex> lock(m_frameMutex);
        ++m_runtimeStats.gpuDirectRebindAttempts;
    }
    if (wasRunning) {
        QCAP_STOP(m_device);
    }

    TeardownGpuDirect();
    {
        std::lock_guard<std::mutex> lock(m_frameMutex);
        ResetFrameState();
    }
    const bool rebound = TrySetupGpuDirect();

    if (wasRunning) {
        const QRESULT runResult = QCAP_RUN(m_device);
        if (runResult != QCAP_RT_OK) {
            SetLastError("QCAP_RUN failed after GPUDirect rebind", runResult);
            m_gpuDirectActive = false;
            m_gpuDirectStatus.message = "GPUDirect rebind attempted, but QCAP_RUN failed afterwards.";
            std::lock_guard<std::mutex> lock(m_frameMutex);
            ++m_runtimeStats.gpuDirectRebindFailures;
            return false;
        }
    }

    if (!rebound) {
        if (m_gpuDirectStatus.message.empty()) {
            m_gpuDirectStatus.message = "GPUDirect rebind failed; capture remains on compatibility path.";
        }
        std::lock_guard<std::mutex> lock(m_frameMutex);
        ++m_runtimeStats.gpuDirectRebindFailures;
        std::cout << "[QCAP] GPUDirect rebind failed, staying on compatibility path: "
            << m_gpuDirectStatus.message << std::endl;
        return false;
    }

    {
        std::lock_guard<std::mutex> lock(m_frameMutex);
        ++m_runtimeStats.gpuDirectRebindSuccesses;
        m_awaitingFreshFrameAfterRebind = true;
        m_lastDeliveredSequence = m_frameInfo.sequence;
    }
    std::cout << "[QCAP] GPUDirect rebind completed queue=" << m_gpuDirectStatus.queueSize
        << " bufferSize=" << m_gpuDirectStatus.bufferSize << std::endl;
    return true;
}

bool QCAPCapture::Capture(cv::Mat& output, int timeoutMs) {
    if (!m_initialized) {
        SetLastError("QCAP capture is not initialized");
        return false;
    }

    if (!WaitForFrame(timeoutMs)) {
        std::lock_guard<std::mutex> lock(m_frameMutex);
        ++m_runtimeStats.captureTimeouts;
        return false;
    }

    {
        std::lock_guard<std::mutex> lock(m_frameMutex);
        ++m_runtimeStats.captureSuccesses;
    }
    return ConvertFrameToMat(output);
}

const unsigned char* QCAPCapture::CaptureBGR(int timeoutMs) {
    if (!m_initialized) {
        SetLastError("QCAP capture is not initialized");
        return nullptr;
    }

    if (!WaitForFrame(timeoutMs)) {
        std::lock_guard<std::mutex> lock(m_frameMutex);
        ++m_runtimeStats.captureTimeouts;
        return nullptr;
    }

    std::lock_guard<std::mutex> lock(m_frameMutex);
    if (m_frameInfo.colorSpaceType != QCAP_COLORSPACE_TYPE_BGR24) {
        SetLastError("CaptureBGR requires QCAP BGR24 output");
        return nullptr;
    }
    if (!EnsureCpuShadowBufferLocked()) {
        return nullptr;
    }

    return m_frameBuffer.empty() ? nullptr : m_frameBuffer.data();
}

bool QCAPCapture::GetLatestFrameView(QCAPFrameView& frameView, int timeoutMs) {
    frameView = QCAPFrameView{};

    if (!m_initialized) {
        SetLastError("QCAP capture is not initialized");
        return false;
    }

    if (!WaitForFrame(timeoutMs)) {
        std::lock_guard<std::mutex> lock(m_frameMutex);
        ++m_runtimeStats.frameViewTimeouts;
        return false;
    }

    RebindGpuDirectIfNeeded();

    std::lock_guard<std::mutex> lock(m_frameMutex);
    if (!m_hasFrame) {
        SetLastError("No QCAP frame is available yet");
        return false;
    }

    const bool gpuDirectFrameUsable =
        m_gpuDirectActive &&
        m_frameInfo.colorSpaceType == QCAP_COLORSPACE_TYPE_BGR24 &&
        m_lastGpuDirectBufferIndex >= 0 &&
        static_cast<std::size_t>(m_lastGpuDirectBufferIndex) < m_gpuDirectMappedDevicePointers.size() &&
        m_gpuDirectMappedDevicePointers[static_cast<std::size_t>(m_lastGpuDirectBufferIndex)] != nullptr &&
        (m_gpuDirectStatus.bufferSize == 0 || m_frameInfo.bufferSize <= m_gpuDirectStatus.bufferSize);
    const bool needsCpuShadowCopy =
        !gpuDirectFrameUsable &&
        (!m_config.requestGpuDirect || m_config.keepCpuShadowCopy);
    if (needsCpuShadowCopy && !EnsureCpuShadowBufferLocked()) {
        return false;
    }

    frameView.cpuData = m_hasCpuShadowCopy && !m_frameBuffer.empty() ? m_frameBuffer.data() : nullptr;
    frameView.sourceBuffer = m_lastSourceFrameBuffer;
    frameView.bufferSize = m_frameInfo.bufferSize;
    frameView.colorSpaceType = m_frameInfo.colorSpaceType;
    frameView.width = m_frameInfo.width;
    frameView.height = m_frameInfo.height;
    frameView.frameRate = m_frameInfo.frameRate;
    frameView.sampleTime = m_frameInfo.sampleTime;
    frameView.sequence = m_frameInfo.sequence;
    frameView.sourceBufferMatchesGpuDirect = gpuDirectFrameUsable;
    if (gpuDirectFrameUsable &&
        m_lastGpuDirectBufferIndex >= 0 &&
        static_cast<std::size_t>(m_lastGpuDirectBufferIndex) < m_gpuDirectMappedDevicePointers.size()) {
        frameView.sourceDevicePointer = m_gpuDirectMappedDevicePointers[static_cast<std::size_t>(m_lastGpuDirectBufferIndex)];
        frameView.sourceBufferCudaMapped = frameView.sourceDevicePointer != nullptr;
    }
    frameView.cpuShadowCopyAvailable = frameView.cpuData != nullptr;
    frameView.gpuDirectBufferIndex = m_lastGpuDirectBufferIndex;
    frameView.imageUpsideDown = IsFrameImageUpsideDown();
    ++m_runtimeStats.frameViewSuccesses;
    return true;
}

void QCAPCapture::DebugForceGpuDirectRebind() {
    std::lock_guard<std::mutex> lock(m_frameMutex);
    MarkGpuDirectRebindPendingLocked(
        "Debug-triggered GPUDirect rebind request.",
        m_gpuDirectStatus.bufferSize);
}

QCAPGpuDirectStatus QCAPCapture::GetGpuDirectStatus() const {
    std::lock_guard<std::mutex> lock(m_frameMutex);
    return m_gpuDirectStatus;
}

QCAPRuntimeStats QCAPCapture::GetRuntimeStats() const {
    std::lock_guard<std::mutex> lock(m_frameMutex);
    return m_runtimeStats;
}

QCAPFrameInfo QCAPCapture::GetFrameInfo() const {
    std::lock_guard<std::mutex> lock(m_frameMutex);
    return m_frameInfo;
}

void QCAPCapture::PrintInputDiagnostics(const char* context) const {
    const PVOID device = m_device;
    if (device == nullptr) {
        std::cout << "[QCAP] input diagnostics skipped (device is null)" << std::endl;
        return;
    }

    std::ostringstream prefix;
    prefix << "[QCAP]";
    if (context != nullptr && context[0] != '\0') {
        prefix << " [" << context << "]";
    }

    ULONG currentInput = 0;
    const QRESULT inputResult = QCAP_GET_VIDEO_INPUT(device, &currentInput);
    if (inputResult == QCAP_RT_OK) {
        std::cout << prefix.str() << " getVideoInput ok requested="
            << DescribeQcapInput(m_config.videoInput) << "(" << m_config.videoInput << ")"
            << " active=" << DescribeQcapInput(currentInput) << "(" << currentInput << ")";
        if (currentInput != m_config.videoInput) {
            std::cout << " note=vendor-active-input-differs-from-request";
        }
        std::cout << std::endl;
    }
    else {
        std::cout << prefix.str() << " getVideoInput failed result=0x"
            << std::hex << std::uppercase << static_cast<unsigned int>(inputResult)
            << std::dec << std::nouppercase << std::endl;
    }

    ULONG colorSpaceType = 0;
    ULONG width = 0;
    ULONG height = 0;
    BOOL interleaved = FALSE;
    double frameRate = 0.0;
    const QRESULT formatResult = QCAP_GET_VIDEO_CURRENT_INPUT_FORMAT(
        device,
        &colorSpaceType,
        &width,
        &height,
        &interleaved,
        &frameRate);
    if (formatResult == QCAP_RT_OK) {
        std::cout << prefix.str() << " currentInputFormat ok colorspace="
            << DescribeQcapColorSpace(colorSpaceType) << "(" << colorSpaceType << ")"
            << " size=" << width << "x" << height
            << " interleaved=" << (interleaved ? "yes" : "no")
            << " frameRate=" << frameRate << std::endl;
    }
    else {
        std::cout << prefix.str() << " currentInputFormat failed result=0x"
            << std::hex << std::uppercase << static_cast<unsigned int>(formatResult)
            << std::dec << std::nouppercase << std::endl;
    }
}

void QCAPCapture::ResetRuntimeTelemetry() {
    std::lock_guard<std::mutex> lock(m_frameMutex);
    m_runtimeStats = QCAPRuntimeStats{};
    m_gpuDirectStatus.callbackUsesBoundBuffer = false;
    m_gpuDirectStatus.previewFramesObserved = 0;
    m_gpuDirectStatus.matchedPreviewFrames = 0;
    m_gpuDirectStatus.lastBoundBufferIndex = -1;
    m_consecutivePreviewWaitTimeouts = 0;
    m_hasLastPreviewArrival = false;
    m_totalPreviewGapMs = 0.0;
}

bool QCAPCapture::HasCpuShadowBuffer() const {
    std::lock_guard<std::mutex> lock(m_frameMutex);
    return m_hasCpuShadowCopy && !m_frameBuffer.empty();
}

bool QCAPCapture::IsFrameImageUpsideDown() {
    return kQcapBgrPreviewImageUpsideDown;
}

void QCAPCapture::Release() {
    {
        std::lock_guard<std::mutex> lock(m_frameMutex);
        m_shuttingDown = true;
    }
    m_frameCondition.notify_all();

    if (m_device != nullptr) {
        QCAP_STOP(m_device);
    }

    TeardownGpuDirect();

    if (m_device != nullptr) {
        QCAP_DESTROY(m_device);
        m_device = nullptr;
    }

    {
        std::lock_guard<std::mutex> lock(m_frameMutex);
        ResetFrameState();
        m_frameBuffer.clear();
        m_shuttingDown = false;
    }

    m_initialized = false;
    m_gpuDirectActive = false;
    m_gpuDirectRebindPending = false;
    m_awaitingFreshFrameAfterRebind = false;
    m_lastDeliveredSequence = 0;
    m_consecutivePreviewWaitTimeouts = 0;
    m_nextAutoResyncAllowedTime = std::chrono::steady_clock::time_point::min();
    m_hasLastPreviewArrival = false;
    m_totalPreviewGapMs = 0.0;
    m_requestedGpuDirectBufferSize = 0;
    m_gpuDirectRebindReason.clear();
    m_enumeratedDevices.clear();
}

bool QCAPCapture::WaitForFrame(int timeoutMs) {
    std::unique_lock<std::mutex> lock(m_frameMutex);
    if (m_shuttingDown) {
        SetLastError("QCAP capture is shutting down");
        return false;
    }
    ++m_runtimeStats.waitCalls;
    const std::uint64_t currentSequence = m_frameInfo.sequence;

    if (!m_hasFrame || currentSequence == m_lastDeliveredSequence || m_awaitingFreshFrameAfterRebind) {
        const bool received = m_frameCondition.wait_for(
            lock,
            std::chrono::milliseconds(timeoutMs),
            [this, currentSequence]() {
                return m_shuttingDown ||
                    (m_hasFrame && m_frameInfo.sequence != currentSequence && !m_awaitingFreshFrameAfterRebind);
            });

        if (!received) {
            ++m_runtimeStats.waitTimeouts;
            ++m_consecutivePreviewWaitTimeouts;
            const double callbackAgeMs = m_hasLastPreviewArrival
                ? std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - m_lastPreviewArrivalTime).count()
                : -1.0;
            m_runtimeStats.lastCallbackAgeMs = callbackAgeMs;
            std::ostringstream oss;
            oss << "Timed out waiting for a QCAP preview frame";
            if (callbackAgeMs >= 0.0) {
                oss << " (lastCallbackAgeMs=" << std::fixed << std::setprecision(3) << callbackAgeMs << ")";
            }
            oss << " previewCallbacks=" << m_runtimeStats.previewCallbacksObserved
                << " formatChanges=" << m_runtimeStats.formatChangesObserved
                << " noSignal=" << m_runtimeStats.noSignalEventsObserved
                << " signalRemoved=" << m_runtimeStats.signalRemovedEventsObserved;
            if (m_runtimeStats.noSignalEventsObserved > 0) {
                oss << " note=vendor-no-signal";
            }
            else if (m_runtimeStats.signalRemovedEventsObserved > 0) {
                oss << " note=vendor-signal-removed";
            }
            else if (m_runtimeStats.previewCallbacksObserved == 0) {
                oss << " note=no-preview-callback-yet";
            }
            SetLastError(oss.str());
            lock.unlock();
            if (AttemptAutoResyncAfterPreviewStall(callbackAgeMs)) {
                return WaitForFrame(timeoutMs);
            }
            return false;
        }
    }

    if (m_shuttingDown) {
        SetLastError("QCAP capture is shutting down");
        return false;
    }

    m_consecutivePreviewWaitTimeouts = 0;
    m_lastDeliveredSequence = m_frameInfo.sequence;
    return true;
}

bool QCAPCapture::AttemptAutoResyncAfterPreviewStall(double callbackAgeMs) {
    if (m_device == nullptr || !m_initialized || m_shuttingDown) {
        return false;
    }

    double effectiveFrameRate = 0.0;
    std::uint32_t consecutiveTimeouts = 0;
    {
        std::lock_guard<std::mutex> lock(m_frameMutex);
        if (!m_config.autoResyncOnPreviewStall) {
            return false;
        }
        effectiveFrameRate = m_frameInfo.frameRate > 0.0 ? m_frameInfo.frameRate : m_config.frameRate;
        consecutiveTimeouts = m_consecutivePreviewWaitTimeouts;
    }
    const double expectedFrameMs = effectiveFrameRate > 0.0 ? (1000.0 / effectiveFrameRate) : 0.0;
    const double dynamicThresholdMs = expectedFrameMs > 0.0
        ? (std::max)(kAutoResyncMinThresholdMs, expectedFrameMs * kAutoResyncExpectedFrameMultiplier)
        : kAutoResyncMinThresholdMs;
    if (callbackAgeMs < dynamicThresholdMs) {
        return false;
    }
    if (consecutiveTimeouts < kAutoResyncRequiredTimeouts) {
        return false;
    }

    const auto now = std::chrono::steady_clock::now();
    {
        std::lock_guard<std::mutex> lock(m_frameMutex);
        if (now < m_nextAutoResyncAllowedTime) {
            return false;
        }
        m_nextAutoResyncAllowedTime = now + std::chrono::seconds(3);
        ++m_runtimeStats.captureResyncAttempts;
        ResetFrameState();
        m_consecutivePreviewWaitTimeouts = 0;
    }

    std::cout << "[QCAP] Auto-resync triggered after preview stall"
        << " callbackAgeMs=" << callbackAgeMs
        << " thresholdMs=" << dynamicThresholdMs
        << " consecutiveTimeouts=" << consecutiveTimeouts
        << " cooldownMs=3000"
        << std::endl;

    const QRESULT stopResult = QCAP_STOP(m_device);
    if (stopResult != QCAP_RT_OK) {
        std::lock_guard<std::mutex> lock(m_frameMutex);
        ++m_runtimeStats.captureResyncFailures;
        SetLastError("QCAP_STOP failed during auto-resync", stopResult);
        return false;
    }

    const QRESULT runResult = QCAP_RUN(m_device);
    if (runResult != QCAP_RT_OK) {
        std::lock_guard<std::mutex> lock(m_frameMutex);
        ++m_runtimeStats.captureResyncFailures;
        SetLastError("QCAP_RUN failed during auto-resync", runResult);
        return false;
    }

    {
        std::lock_guard<std::mutex> lock(m_frameMutex);
        ++m_runtimeStats.captureResyncSuccesses;
        m_awaitingFreshFrameAfterRebind = true;
    }

    return true;
}

bool QCAPCapture::ConvertFrameToMat(cv::Mat& output) {
    std::lock_guard<std::mutex> lock(m_frameMutex);

    if (!m_hasFrame) {
        SetLastError("No QCAP frame is available yet");
        return false;
    }
    if (!EnsureCpuShadowBufferLocked()) {
        return false;
    }

    const int width = m_frameInfo.width;
    const int height = m_frameInfo.height;

    switch (m_frameInfo.colorSpaceType) {
    case QCAP_COLORSPACE_TYPE_BGR24:
        output = cv::Mat(height, width, CV_8UC3);
        std::memcpy(output.data, m_frameBuffer.data(), m_frameBuffer.size());
        if (IsFrameImageUpsideDown()) {
            cv::flip(output, output, 0);
        }
        return true;

    case QCAP_COLORSPACE_TYPE_RGB24:
    {
        cv::Mat rgb(height, width, CV_8UC3);
        std::memcpy(rgb.data, m_frameBuffer.data(), m_frameBuffer.size());
        if (IsFrameImageUpsideDown()) {
            cv::flip(rgb, rgb, 0);
        }
        cv::cvtColor(rgb, output, cv::COLOR_RGB2BGR);
        return true;
    }

    case QCAP_COLORSPACE_TYPE_YUY2:
    {
        cv::Mat yuy2(height, width, CV_8UC2, const_cast<unsigned char*>(m_frameBuffer.data()));
        cv::cvtColor(yuy2, output, cv::COLOR_YUV2BGR_YUY2);
        if (IsFrameImageUpsideDown()) {
            cv::flip(output, output, 0);
        }
        return true;
    }

    case QCAP_COLORSPACE_TYPE_UYVY:
    {
        cv::Mat uyvy(height, width, CV_8UC2, const_cast<unsigned char*>(m_frameBuffer.data()));
        cv::cvtColor(uyvy, output, cv::COLOR_YUV2BGR_UYVY);
        if (IsFrameImageUpsideDown()) {
            cv::flip(output, output, 0);
        }
        return true;
    }

    case QCAP_COLORSPACE_TYPE_NV12:
    {
        cv::Mat nv12(height + height / 2, width, CV_8UC1, const_cast<unsigned char*>(m_frameBuffer.data()));
        cv::cvtColor(nv12, output, cv::COLOR_YUV2BGR_NV12);
        if (IsFrameImageUpsideDown()) {
            cv::flip(output, output, 0);
        }
        return true;
    }

    case QCAP_COLORSPACE_TYPE_I420:
    {
        cv::Mat i420(height + height / 2, width, CV_8UC1, const_cast<unsigned char*>(m_frameBuffer.data()));
        cv::cvtColor(i420, output, cv::COLOR_YUV2BGR_I420);
        if (IsFrameImageUpsideDown()) {
            cv::flip(output, output, 0);
        }
        return true;
    }

    case QCAP_COLORSPACE_TYPE_YV12:
    {
        cv::Mat yv12(height + height / 2, width, CV_8UC1, const_cast<unsigned char*>(m_frameBuffer.data()));
        cv::cvtColor(yv12, output, cv::COLOR_YUV2BGR_YV12);
        if (IsFrameImageUpsideDown()) {
            cv::flip(output, output, 0);
        }
        return true;
    }

    default:
        SetLastError("Unsupported QCAP output color space for cv::Mat conversion");
        return false;
    }
}

bool QCAPCapture::ConvertFrameToPreviewMat(cv::Mat& output, int maxWidth) {
    output.release();

    std::lock_guard<std::mutex> lock(m_frameMutex);

    if (!m_hasFrame) {
        SetLastError("No QCAP frame is available yet");
        return false;
    }
    const int width = m_frameInfo.width;
    const int height = m_frameInfo.height;
    if (width <= 0 || height <= 0) {
        SetLastError("QCAP preview conversion requires a valid frame size");
        return false;
    }

    int targetWidth = width;
    int targetHeight = height;
    if (maxWidth > 0 && width > maxWidth) {
        const double scale = static_cast<double>(maxWidth) / static_cast<double>(width);
        targetWidth = maxWidth;
        targetHeight = (std::max)(1, static_cast<int>(std::round(static_cast<double>(height) * scale)));
    }

    switch (m_frameInfo.colorSpaceType) {
    case QCAP_COLORSPACE_TYPE_BGR24:
    {
        const unsigned char* sourceData = m_hasCpuShadowCopy && !m_frameBuffer.empty()
            ? m_frameBuffer.data()
            : m_lastSourceFrameBuffer;
        if (sourceData == nullptr) {
            SetLastError("QCAP BGR preview conversion has no source buffer");
            return false;
        }
        cv::Mat source(height, width, CV_8UC3, const_cast<unsigned char*>(sourceData));
        if (targetWidth != width || targetHeight != height) {
            cv::resize(source, output, cv::Size(targetWidth, targetHeight), 0.0, 0.0, cv::INTER_AREA);
        }
        else {
            output = source.clone();
        }
        if (IsFrameImageUpsideDown()) {
            cv::flip(output, output, 0);
        }
        return true;
    }

    case QCAP_COLORSPACE_TYPE_RGB24:
    {
        const unsigned char* sourceData = m_hasCpuShadowCopy && !m_frameBuffer.empty()
            ? m_frameBuffer.data()
            : m_lastSourceFrameBuffer;
        if (sourceData == nullptr) {
            SetLastError("QCAP RGB preview conversion has no source buffer");
            return false;
        }
        cv::Mat source(height, width, CV_8UC3, const_cast<unsigned char*>(sourceData));
        cv::Mat resizedRgb;
        if (targetWidth != width || targetHeight != height) {
            cv::resize(source, resizedRgb, cv::Size(targetWidth, targetHeight), 0.0, 0.0, cv::INTER_AREA);
        }
        else {
            resizedRgb = source.clone();
        }
        if (IsFrameImageUpsideDown()) {
            cv::flip(resizedRgb, resizedRgb, 0);
        }
        cv::cvtColor(resizedRgb, output, cv::COLOR_RGB2BGR);
        return true;
    }

    default:
    {
        if (!EnsureCpuShadowBufferLocked()) {
            return false;
        }

        cv::Mat converted;
        switch (m_frameInfo.colorSpaceType) {
        case QCAP_COLORSPACE_TYPE_YUY2:
        {
            cv::Mat yuy2(height, width, CV_8UC2, const_cast<unsigned char*>(m_frameBuffer.data()));
            cv::cvtColor(yuy2, converted, cv::COLOR_YUV2BGR_YUY2);
            break;
        }
        case QCAP_COLORSPACE_TYPE_UYVY:
        {
            cv::Mat uyvy(height, width, CV_8UC2, const_cast<unsigned char*>(m_frameBuffer.data()));
            cv::cvtColor(uyvy, converted, cv::COLOR_YUV2BGR_UYVY);
            break;
        }
        case QCAP_COLORSPACE_TYPE_NV12:
        {
            cv::Mat nv12(height + height / 2, width, CV_8UC1, const_cast<unsigned char*>(m_frameBuffer.data()));
            cv::cvtColor(nv12, converted, cv::COLOR_YUV2BGR_NV12);
            break;
        }
        case QCAP_COLORSPACE_TYPE_I420:
        {
            cv::Mat i420(height + height / 2, width, CV_8UC1, const_cast<unsigned char*>(m_frameBuffer.data()));
            cv::cvtColor(i420, converted, cv::COLOR_YUV2BGR_I420);
            break;
        }
        case QCAP_COLORSPACE_TYPE_YV12:
        {
            cv::Mat yv12(height + height / 2, width, CV_8UC1, const_cast<unsigned char*>(m_frameBuffer.data()));
            cv::cvtColor(yv12, converted, cv::COLOR_YUV2BGR_YV12);
            break;
        }
        default:
            SetLastError("Unsupported QCAP output color space for preview conversion");
            return false;
        }

        if (IsFrameImageUpsideDown()) {
            cv::flip(converted, converted, 0);
        }
        if (targetWidth != width || targetHeight != height) {
            cv::resize(converted, output, cv::Size(targetWidth, targetHeight), 0.0, 0.0, cv::INTER_AREA);
        }
        else {
            output = converted;
        }
        return true;
    }
    }
}

bool QCAPCapture::EnsureCpuShadowBufferLocked() {
    if (!m_hasFrame) {
        SetLastError("No QCAP frame is available yet");
        return false;
    }

    if (m_hasCpuShadowCopy && !m_frameBuffer.empty()) {
        return true;
    }

    if (m_lastSourceFrameBuffer == nullptr || m_frameInfo.bufferSize == 0) {
        SetLastError("QCAP CPU shadow copy is unavailable because there is no source preview buffer yet");
        return false;
    }

    if (m_frameBuffer.size() != m_frameInfo.bufferSize) {
        m_frameBuffer.resize(m_frameInfo.bufferSize);
    }

    std::memcpy(m_frameBuffer.data(), m_lastSourceFrameBuffer, m_frameInfo.bufferSize);
    m_hasCpuShadowCopy = true;
    return true;
}

int QCAPCapture::FindGpuDirectBufferIndex(const BYTE* frameBuffer) const {
    if (frameBuffer == nullptr) {
        return -1;
    }

    for (std::size_t i = 0; i < m_gpuDirectBuffers.size(); ++i) {
        if (m_gpuDirectBuffers[i] == frameBuffer) {
            return static_cast<int>(i);
        }
    }

    return -1;
}

void QCAPCapture::MarkGpuDirectRebindPendingLocked(const std::string& reason, std::size_t requiredBufferSize) {
    if (!m_config.requestGpuDirect || m_shuttingDown) {
        return;
    }

    m_gpuDirectActive = false;
    m_gpuDirectRebindPending = true;
    m_gpuDirectStatus.callbackUsesBoundBuffer = false;
    m_gpuDirectStatus.lastBoundBufferIndex = -1;
    if (requiredBufferSize != 0) {
        m_requestedGpuDirectBufferSize = requiredBufferSize;
    }
    m_gpuDirectRebindReason = reason;
    m_gpuDirectStatus.message = reason + " Rebind is pending.";
}

void QCAPCapture::ResetFrameState() {
    m_hasFrame = false;
    m_hasCpuShadowCopy = false;
    m_frameInfo = QCAPFrameInfo{};
    m_lastSourceFrameBuffer = nullptr;
    m_lastGpuDirectBufferIndex = -1;
    m_awaitingFreshFrameAfterRebind = false;
}

void QCAPCapture::SetLastError(const std::string& message, QRESULT result) {
    std::ostringstream oss;
    oss << message;
    if (result != QCAP_RT_OK) {
        oss << " (0x" << std::uppercase << std::hex << static_cast<unsigned int>(result) << ")";
    }
    m_lastError = oss.str();
}

void QCAPCapture::HandleFormatChanged(ULONG width, ULONG height, BOOL interleaved, double frameRate) {
    std::lock_guard<std::mutex> lock(m_frameMutex);
    if (m_shuttingDown) {
        return;
    }

    ++m_runtimeStats.formatChangesObserved;

    m_frameInfo.width = static_cast<int>(width);
    m_frameInfo.height = static_cast<int>(height);
    m_frameInfo.frameRate = frameRate;
    m_frameInfo.bufferSize = CalculateFrameBufferSize(
        m_frameInfo.colorSpaceType,
        m_frameInfo.width,
        m_frameInfo.height);
    m_hasFrame = false;
    m_hasCpuShadowCopy = false;
    m_frameBuffer.clear();
    m_lastSourceFrameBuffer = nullptr;
    m_lastGpuDirectBufferIndex = -1;

    const std::size_t expectedBufferSize = CalculateFrameBufferSize(
        m_frameInfo.colorSpaceType,
        m_frameInfo.width,
        m_frameInfo.height);
    if (m_gpuDirectActive &&
        expectedBufferSize != 0 &&
        m_gpuDirectStatus.bufferSize != 0 &&
        expectedBufferSize != m_gpuDirectStatus.bufferSize) {
        MarkGpuDirectRebindPendingLocked(
            "GPUDirect zero-copy paused after format change because the negotiated frame size no longer matches the bound preview buffer size.",
            expectedBufferSize);
    }

    std::cout << "[QCAP] Format changed: "
        << width << "x" << height
        << (interleaved ? " interleaved" : " progressive")
        << " @" << frameRate << " fps"
        << " (requested " << m_config.frameRate << ")"
        << std::endl;
}

void QCAPCapture::HandleNoSignalDetected(ULONG videoInput, ULONG audioInput) {
    std::lock_guard<std::mutex> lock(m_frameMutex);
    if (m_shuttingDown) {
        return;
    }

    ++m_runtimeStats.noSignalEventsObserved;
    std::ostringstream oss;
    oss << "QCAP reported no signal (videoInput=" << videoInput
        << ", audioInput=" << audioInput << ")";
    m_lastError = oss.str();
    std::cout << "[QCAP] " << m_lastError << std::endl;
}

void QCAPCapture::HandleSignalRemoved(ULONG videoInput, ULONG audioInput) {
    std::lock_guard<std::mutex> lock(m_frameMutex);
    if (m_shuttingDown) {
        return;
    }

    ++m_runtimeStats.signalRemovedEventsObserved;
    std::ostringstream oss;
    oss << "QCAP reported signal removed (videoInput=" << videoInput
        << ", audioInput=" << audioInput << ")";
    m_lastError = oss.str();
    std::cout << "[QCAP] " << m_lastError << std::endl;
}

QRETURN QCAPCapture::HandleVideoPreview(double sampleTime, BYTE* frameBuffer, ULONG frameBufferLen) {
    if (frameBuffer == nullptr || frameBufferLen == 0) {
        return QCAP_RT_OK;
    }

    {
        std::lock_guard<std::mutex> lock(m_frameMutex);
        if (m_shuttingDown || !m_initialized) {
            return QCAP_RT_OK;
        }
        ++m_runtimeStats.previewCallbacksObserved;
        const auto now = std::chrono::steady_clock::now();
        int gpuDirectBufferIndex = FindGpuDirectBufferIndex(frameBuffer);
        if (gpuDirectBufferIndex >= 0 &&
            m_gpuDirectStatus.bufferSize != 0 &&
            static_cast<std::size_t>(frameBufferLen) > m_gpuDirectStatus.bufferSize) {
            gpuDirectBufferIndex = -1;
            MarkGpuDirectRebindPendingLocked(
                "GPUDirect zero-copy paused because the callback frame length exceeded the bound preview buffer size.",
                static_cast<std::size_t>(frameBufferLen));
        }
        if (m_hasLastPreviewArrival) {
            const double gapMs = std::chrono::duration<double, std::milli>(now - m_lastPreviewArrivalTime).count();
            ++m_runtimeStats.callbackGapSamples;
            m_runtimeStats.lastCallbackGapMs = gapMs;
            m_runtimeStats.maxCallbackGapMs = (std::max)(m_runtimeStats.maxCallbackGapMs, gapMs);
            m_totalPreviewGapMs += gapMs;
            m_runtimeStats.avgCallbackGapMs =
                m_totalPreviewGapMs / static_cast<double>(m_runtimeStats.callbackGapSamples);

            const double expectedFrameMs = m_frameInfo.frameRate > 0.0
                ? (1000.0 / m_frameInfo.frameRate)
                : (m_config.frameRate > 0.0 ? 1000.0 / m_config.frameRate : 0.0);
            if (expectedFrameMs > 0.0) {
                if (gapMs > expectedFrameMs * 2.0) {
                    ++m_runtimeStats.callbackGapOver2xExpected;
                }
                if (gapMs > expectedFrameMs * 4.0) {
                    ++m_runtimeStats.callbackGapOver4xExpected;
                }
            }
        }
        m_lastPreviewArrivalTime = now;
        m_hasLastPreviewArrival = true;
        m_runtimeStats.lastCallbackAgeMs = 0.0;
        m_hasFrame = true;
        m_frameInfo.sampleTime = sampleTime;
        m_frameInfo.bufferSize = frameBufferLen;
        ++m_frameInfo.sequence;
        m_lastSourceFrameBuffer = frameBuffer;
        m_lastGpuDirectBufferIndex = gpuDirectBufferIndex;
        m_consecutivePreviewWaitTimeouts = 0;
        if (!m_config.keepCpuShadowCopy) {
            m_hasCpuShadowCopy = false;
        }
        else {
            if (m_frameBuffer.size() != m_frameInfo.bufferSize) {
                m_frameBuffer.resize(m_frameInfo.bufferSize);
            }
            std::memcpy(m_frameBuffer.data(), frameBuffer, m_frameInfo.bufferSize);
            m_hasCpuShadowCopy = true;
        }
        if (m_awaitingFreshFrameAfterRebind) {
            m_awaitingFreshFrameAfterRebind = false;
            ++m_runtimeStats.freshFramesAfterGpuDirectRebind;
            ++m_runtimeStats.callbackGapResyncs;
        }

        ++m_gpuDirectStatus.previewFramesObserved;
        if (gpuDirectBufferIndex >= 0) {
            ++m_gpuDirectStatus.matchedPreviewFrames;
            m_gpuDirectStatus.callbackUsesBoundBuffer = true;
            m_gpuDirectStatus.lastBoundBufferIndex = gpuDirectBufferIndex;
        }
    }

    m_frameCondition.notify_one();
    return QCAP_RT_OK;
}

std::size_t QCAPCapture::CalculateFrameBufferSize(ULONG colorSpaceType, int width, int height) {
    switch (colorSpaceType) {
    case QCAP_COLORSPACE_TYPE_BGR24:
    case QCAP_COLORSPACE_TYPE_RGB24:
        return BytesForPackedFrame(width, height, 3);

    case QCAP_COLORSPACE_TYPE_ARGB32:
    case QCAP_COLORSPACE_TYPE_ABGR32:
        return BytesForPackedFrame(width, height, 4);

    case QCAP_COLORSPACE_TYPE_YUY2:
    case QCAP_COLORSPACE_TYPE_UYVY:
        return BytesForPackedFrame(width, height, 2);

    case QCAP_COLORSPACE_TYPE_NV12:
    case QCAP_COLORSPACE_TYPE_I420:
    case QCAP_COLORSPACE_TYPE_YV12:
        return BytesForPackedFrame(width, height, 3) / 2;

    default:
        return 0;
    }
}

QRETURN QCAPCapture::OnFormatChanged(
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
    PVOID pUserData) {
    (void)pDevice;
    (void)nVideoInput;
    (void)nAudioInput;
    (void)nAudioChannels;
    (void)nAudioBitsPerSample;
    (void)nAudioSampleFrequency;

    QCAPCapture* self = static_cast<QCAPCapture*>(pUserData);
    if (self == nullptr) {
        return QCAP_RT_OK;
    }

    self->HandleFormatChanged(nVideoWidth, nVideoHeight, bVideoIsInterleaved, dVideoFrameRate);
    return QCAP_RT_OK;
}

QRETURN QCAPCapture::OnVideoPreview(
    PVOID pDevice,
    double dSampleTime,
    BYTE* pFrameBuffer,
    ULONG nFrameBufferLen,
    PVOID pUserData) {
    (void)pDevice;

    QCAPCapture* self = static_cast<QCAPCapture*>(pUserData);
    if (self == nullptr) {
        return QCAP_RT_OK;
    }

    return self->HandleVideoPreview(dSampleTime, pFrameBuffer, nFrameBufferLen);
}

QRETURN QCAPCapture::OnNoSignalDetected(
    PVOID pDevice,
    ULONG nVideoInput,
    ULONG nAudioInput,
    PVOID pUserData) {
    (void)pDevice;

    QCAPCapture* self = static_cast<QCAPCapture*>(pUserData);
    if (self == nullptr) {
        return QCAP_RT_OK;
    }

    self->HandleNoSignalDetected(nVideoInput, nAudioInput);
    return QCAP_RT_OK;
}

QRETURN QCAPCapture::OnSignalRemoved(
    PVOID pDevice,
    ULONG nVideoInput,
    ULONG nAudioInput,
    PVOID pUserData) {
    (void)pDevice;

    QCAPCapture* self = static_cast<QCAPCapture*>(pUserData);
    if (self == nullptr) {
        return QCAP_RT_OK;
    }

    self->HandleSignalRemoved(nVideoInput, nAudioInput);
    return QCAP_RT_OK;
}
