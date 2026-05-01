#include "DetectorBackend.h"
#include "TensorRTDetector.h"
#include "TensorRTYoloDetector.h"

#include <Windows.h>

#include <filesystem>
#include <memory>
#include <sstream>
#include <string>
#include <utility>

namespace {

std::string BackendKindToString(InferenceBackendKind backend) {
    switch (backend) {
    case InferenceBackendKind::TensorRTYolo:
        return "TensorRTYolo";
    case InferenceBackendKind::TensorRTLegacy:
        return "TensorRTLegacy";
    case InferenceBackendKind::TensorRTPreferred:
    default:
        return "TensorRTPreferred";
    }
}

std::string TryFindRuntimeDll(const char* dllName) {
    char buffer[MAX_PATH] = {};
    const DWORD length = SearchPathA(nullptr, dllName, nullptr, MAX_PATH, buffer, nullptr);
    if (length == 0 || length >= MAX_PATH) {
        return {};
    }
    return std::string(buffer, length);
}

std::string DetectTensorRtRuntimePath() {
    const char* dllNames[] = {
        "nvinfer_10.dll",
        "nvinfer_9.dll",
        "nvinfer_8.dll",
        "nvinfer.dll",
    };

    for (const char* dllName : dllNames) {
        const std::string path = TryFindRuntimeDll(dllName);
        if (!path.empty()) {
            return path;
        }
    }

    return {};
}

bool FileExists(const std::string& path) {
    if (path.empty()) {
        return false;
    }

    std::error_code ec;
    return std::filesystem::exists(path, ec);
}

std::filesystem::path GetExecutableDirectory() {
    wchar_t modulePath[MAX_PATH] = {};
    const DWORD length = ::GetModuleFileNameW(nullptr, modulePath, MAX_PATH);
    if (length == 0 || length >= MAX_PATH) {
        return {};
    }

    return std::filesystem::path(modulePath).parent_path();
}

std::string WideToAnsiString(const std::wstring& value) {
    if (value.empty()) {
        return {};
    }

    const int length = ::WideCharToMultiByte(
        CP_ACP,
        0,
        value.c_str(),
        static_cast<int>(value.size()),
        nullptr,
        0,
        nullptr,
        nullptr);
    if (length <= 0) {
        return {};
    }

    std::string result(static_cast<std::size_t>(length), '\0');
    const int converted = ::WideCharToMultiByte(
        CP_ACP,
        0,
        value.c_str(),
        static_cast<int>(value.size()),
        result.data(),
        length,
        nullptr,
        nullptr);
    if (converted <= 0) {
        return {};
    }

    return result;
}

std::string PathToTensorRtString(const std::filesystem::path& path) {
    const std::wstring widePath = path.native();

    bool asciiOnly = true;
    for (const wchar_t ch : widePath) {
        if (ch > 0x7F) {
            asciiOnly = false;
            break;
        }
    }

    if (asciiOnly) {
        std::string asciiPath;
        asciiPath.reserve(widePath.size());
        for (const wchar_t ch : widePath) {
            asciiPath.push_back(static_cast<char>(ch));
        }
        return asciiPath;
    }

    std::wstring shortPath(MAX_PATH, L'\0');
    DWORD shortLength = ::GetShortPathNameW(widePath.c_str(), shortPath.data(), static_cast<DWORD>(shortPath.size()));
    if (shortLength > 0) {
        if (shortLength >= shortPath.size()) {
            shortPath.resize(shortLength + 1, L'\0');
            shortLength = ::GetShortPathNameW(widePath.c_str(), shortPath.data(), static_cast<DWORD>(shortPath.size()));
        }

        if (shortLength > 0 && shortLength < shortPath.size()) {
            shortPath.resize(shortLength);
            const std::string shortAnsi = WideToAnsiString(shortPath);
            if (!shortAnsi.empty()) {
                return shortAnsi;
            }
        }
    }

    const std::string ansiPath = WideToAnsiString(widePath);
    if (!ansiPath.empty()) {
        return ansiPath;
    }

    return path.string();
}

std::string ResolveExistingInputPath(const std::string& path) {
    if (path.empty()) {
        return {};
    }

    const std::filesystem::path rawPath(path);
    if (rawPath.is_absolute()) {
        return PathToTensorRtString(rawPath);
    }

    if (FileExists(path)) {
        return path;
    }

    const auto executableCandidate = GetExecutableDirectory() / rawPath;
    if (FileExists(PathToTensorRtString(executableCandidate))) {
        return PathToTensorRtString(executableCandidate);
    }

    return path;
}

std::string ResolveEnginePath(const std::string& path) {
    if (path.empty()) {
        return {};
    }

    const std::filesystem::path rawPath(path);
    if (rawPath.is_absolute()) {
        return PathToTensorRtString(rawPath);
    }

    if (FileExists(path)) {
        return path;
    }

    const auto executableCandidate = GetExecutableDirectory() / rawPath;
    if (FileExists(PathToTensorRtString(executableCandidate))) {
        return PathToTensorRtString(executableCandidate);
    }

    if (!GetExecutableDirectory().empty()) {
        return PathToTensorRtString(executableCandidate);
    }

    return path;
}

DetectorSettings ResolveDetectorSettings(const DetectorSettings& settings, DetectorInitReport& report) {
    DetectorSettings resolved = settings;
    resolved.modelPath = ResolveExistingInputPath(settings.modelPath);
    resolved.enginePath = ResolveEnginePath(settings.enginePath);
    report.resolvedModelPath = resolved.modelPath;
    report.resolvedEnginePath = resolved.enginePath;
    report.tensorRtRoi = resolved.tensorRtRoi;
    return resolved;
}

bool IsTensorRtCompatibleQcapFrameView(const QCAPFrameView& frameView) {
    return frameView.colorSpaceType == QCAP_COLORSPACE_TYPE_BGR24 &&
        frameView.width > 0 &&
        frameView.height > 0 &&
        (frameView.sourceDevicePointer != nullptr || frameView.cpuData != nullptr);
}

class TensorRTYoloDetectorBackend final : public IObjectDetector {
public:
    bool Initialize(const DetectorSettings& settings) override {
        if (!m_impl.Initialize(settings)) {
            m_lastError = m_impl.GetLastError();
            return false;
        }

        m_lastError.clear();
        return true;
    }

    std::vector<DetectionObject> DetectBGR(
        const unsigned char* imageData,
        int width,
        int height,
        float nms,
        float conf) override {
        return m_impl.DetectBGR(imageData, width, height, nms, conf);
    }

    bool SupportsQcapFrameView() const override {
        return true;
    }

    bool CanConsumeQcapFrameView(const QCAPFrameView& frameView) const override {
        return IsTensorRtCompatibleQcapFrameView(frameView);
    }

    std::vector<DetectionObject> DetectQcapFrameView(
        const QCAPFrameView& frameView,
        float nms,
        float conf) override {
        if (!IsTensorRtCompatibleQcapFrameView(frameView)) {
            return {};
        }

        return m_impl.DetectQcapFrameView(frameView, nms, conf);
    }

    void ReleaseResources() override {
        m_impl.ReleaseResources();
    }

    const char* GetBackendName() const override {
        return "TensorRT-YOLO";
    }

    const std::string& GetLastError() const override {
        return m_lastError;
    }

    DetectorFrameStats GetLastFrameStats() const override {
        return m_impl.GetLastFrameStats();
    }

private:
    TensorRTYoloDetector m_impl;
    std::string m_lastError;
};

class TensorRTLegacyDetectorBackend final : public IObjectDetector {
public:
    bool Initialize(const DetectorSettings& settings) override {
        if (!m_impl.Initialize(settings)) {
            m_lastError = m_impl.GetLastError();
            return false;
        }

        m_lastError.clear();
        return true;
    }

    std::vector<DetectionObject> DetectBGR(
        const unsigned char* imageData,
        int width,
        int height,
        float nms,
        float conf) override {
        return m_impl.DetectBGR(imageData, width, height, nms, conf);
    }

    bool SupportsQcapFrameView() const override {
        return true;
    }

    bool CanConsumeQcapFrameView(const QCAPFrameView& frameView) const override {
        return IsTensorRtCompatibleQcapFrameView(frameView);
    }

    std::vector<DetectionObject> DetectQcapFrameView(
        const QCAPFrameView& frameView,
        float nms,
        float conf) override {
        if (!IsTensorRtCompatibleQcapFrameView(frameView)) {
            return {};
        }

        return m_impl.DetectQcapFrameView(frameView, nms, conf);
    }

    void ReleaseResources() override {
        m_impl.ReleaseResources();
    }

    const char* GetBackendName() const override {
        return "TensorRT-Legacy";
    }

    const std::string& GetLastError() const override {
        return m_lastError;
    }

    DetectorFrameStats GetLastFrameStats() const override {
        return m_impl.GetLastFrameStats();
    }

private:
    TensorRTDetector m_impl;
    std::string m_lastError;
};

bool TryInitializeTensorRTYolo(
    const DetectorSettings& settings,
    std::unique_ptr<IObjectDetector>& detector,
    DetectorInitReport& report) {
    auto candidate = std::make_unique<TensorRTYoloDetectorBackend>();
    report.tensorRtRuntimePath = DetectTensorRtRuntimePath();
    report.tensorRtRuntimeDetected = !report.tensorRtRuntimePath.empty();
    report.tensorRtEngineDetected = FileExists(settings.enginePath);

    if (!candidate->Initialize(settings)) {
        report.message = candidate->GetLastError();
        return false;
    }

    detector = std::move(candidate);
    report.activeBackend = "TensorRT-YOLO";
    report.message = "TensorRT-YOLO backend initialized successfully.";
    return true;
}

bool TryInitializeTensorRTLegacy(
    const DetectorSettings& settings,
    std::unique_ptr<IObjectDetector>& detector,
    DetectorInitReport& report) {
    auto candidate = std::make_unique<TensorRTLegacyDetectorBackend>();
    report.tensorRtRuntimePath = DetectTensorRtRuntimePath();
    report.tensorRtRuntimeDetected = !report.tensorRtRuntimePath.empty();
    report.tensorRtEngineDetected = FileExists(settings.enginePath);

    if (!candidate->Initialize(settings)) {
        report.message = candidate->GetLastError();
        return false;
    }

    detector = std::move(candidate);
    report.activeBackend = "TensorRT-Legacy";
    report.message = "Legacy TensorRT backend initialized successfully.";
    return true;
}

} // namespace

bool InitializeDetectorWithFallback(
    const DetectorSettings& settings,
    std::unique_ptr<IObjectDetector>& detector,
    DetectorInitReport& report) {
    detector.reset();
    report = DetectorInitReport{};
    report.requestedBackend = BackendKindToString(settings.backend);
    const DetectorSettings resolvedSettings = ResolveDetectorSettings(settings, report);

    switch (resolvedSettings.backend) {
    case InferenceBackendKind::TensorRTYolo:
        return TryInitializeTensorRTYolo(resolvedSettings, detector, report);

    case InferenceBackendKind::TensorRTLegacy:
        return TryInitializeTensorRTLegacy(resolvedSettings, detector, report);

    case InferenceBackendKind::TensorRTPreferred:
    default:
    {
        DetectorInitReport tensorRtReport;
        tensorRtReport.requestedBackend = report.requestedBackend;
        tensorRtReport.resolvedModelPath = report.resolvedModelPath;
        tensorRtReport.resolvedEnginePath = report.resolvedEnginePath;
        tensorRtReport.tensorRtRoi = report.tensorRtRoi;
        if (TryInitializeTensorRTYolo(resolvedSettings, detector, tensorRtReport)) {
            report = tensorRtReport;
            return true;
        }

        report = tensorRtReport;
        report.requestedBackend = BackendKindToString(settings.backend);
        report.usedFallback = false;
        report.tensorRtRuntimeDetected = tensorRtReport.tensorRtRuntimeDetected;
        report.tensorRtEngineDetected = tensorRtReport.tensorRtEngineDetected;
        report.tensorRtRuntimePath = tensorRtReport.tensorRtRuntimePath;
        report.resolvedModelPath = tensorRtReport.resolvedModelPath;
        report.resolvedEnginePath = tensorRtReport.resolvedEnginePath;
        report.message = tensorRtReport.message + " TensorRTPreferred is strict: no DirectML/ONNX fallback is allowed on the snowball mainline.";
        return false;
    }
    }
}
