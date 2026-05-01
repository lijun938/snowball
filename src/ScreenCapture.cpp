#include "ScreenCapture.h"

#include <Windows.h>

#include <iostream>

namespace {

bool InitializeQcapWithSeh(QCAPCapture& capture, const QCAPCaptureConfig& config, unsigned long& exceptionCode) {
    exceptionCode = 0;
    __try {
        return capture.Initialize(config);
    }
    __except (EXCEPTION_EXECUTE_HANDLER) {
        exceptionCode = GetExceptionCode();
        return false;
    }
}

} // namespace

ScreenCapture::ScreenCapture()
    : m_currentMode(CaptureMode::GDI)
    , m_initialized(false)
    , m_hasQcapConfig(false)
    , m_width(0)
    , m_height(0)
    , m_x(0)
    , m_y(0)
{
}

ScreenCapture::~ScreenCapture() {
    Release();
}

bool ScreenCapture::Initialize(
    CaptureMode mode,
    int width,
    int height,
    int x,
    int y,
    HWND hwnd,
    const std::string& obsIp,
    int obsPort
) {
    Release();

    m_width = width;
    m_height = height;
    m_x = x;
    m_y = y;

    bool success = false;

    switch (mode) {
    case CaptureMode::GDI:
        success = InitializeGDI(width, height, x, y, hwnd);
        if (success) {
            m_currentMode = CaptureMode::GDI;
        }
        break;

    case CaptureMode::OBS:
        success = InitializeOBS(obsIp, obsPort);
        if (success) {
            m_currentMode = CaptureMode::OBS;
        }
        break;

    case CaptureMode::QCAP:
    {
        QCAPCaptureConfig config;
        config.width = width;
        config.height = height;
        config.attachedWindow = hwnd;
        config.outputColorSpace = QCAP_COLORSPACE_TYPE_BGR24;
        success = InitializeQCAPInternal(config);
        if (success) {
            m_qcapConfig = config;
            m_hasQcapConfig = true;
            m_currentMode = CaptureMode::QCAP;
        }
        break;
    }
    }

    if (success) {
        m_initialized = true;
        const char* modeName =
            m_currentMode == CaptureMode::GDI ? "GDI" :
            m_currentMode == CaptureMode::OBS ? "OBS" :
            "QCAP";
        std::cout << "截图模块初始化成功，模式: " << modeName << std::endl;
    }
    else {
        std::cerr << "截图模块初始化失败" << std::endl;
    }

    return success;
}

bool ScreenCapture::InitializeQCAP(const QCAPCaptureConfig& config) {
    Release();

    m_width = config.width;
    m_height = config.height;
    m_x = 0;
    m_y = 0;

    const bool success = InitializeQCAPInternal(config);
    if (success) {
        m_qcapConfig = config;
        m_hasQcapConfig = true;
        m_currentMode = CaptureMode::QCAP;
        m_initialized = true;
        std::cout << "截图模块初始化成功，模式: QCAP" << std::endl;
    }
    else {
        std::cerr << "QCAP 初始化失败" << std::endl;
    }

    return success;
}

bool ScreenCapture::InitializeGDI(int width, int height, int x, int y, HWND hwnd) {
    try {
        m_gdiCapture = std::make_unique<GDIScreenCapture>();

        bool success = false;
        if (hwnd != nullptr) {
            success = m_gdiCapture->SetWindow(hwnd);
        }

        if (x != 0 || y != 0) {
            success = m_gdiCapture->InitializeRegion(x, y, width, height);
        }
        else {
            success = m_gdiCapture->Initialize(width, height);
        }

        if (!success) {
            m_gdiCapture.reset();
            return false;
        }

        return true;
    }
    catch (const std::exception& e) {
        std::cerr << "GDI 初始化失败: " << e.what() << std::endl;
        m_gdiCapture.reset();
        return false;
    }
}

bool ScreenCapture::InitializeOBS(const std::string& ip, int port) {
    try {
        // 创建新的 Obs 实例
        m_obsCapture = std::make_unique<Obs>();
        bool success = m_obsCapture->Init(ip, port);

        if (!success) {
            m_obsCapture.reset();
            return false;
        }

        return true;
    }
    catch (const std::exception& e) {
        std::cerr << "OBS 初始化失败: " << e.what() << std::endl;
        m_obsCapture.reset();
        return false;
    }
}

bool ScreenCapture::InitializeQCAPInternal(const QCAPCaptureConfig& config) {
    try {
        m_qcapCapture = std::make_unique<QCAPCapture>();
        unsigned long sehExceptionCode = 0;
        const bool success = InitializeQcapWithSeh(*m_qcapCapture, config, sehExceptionCode);
        if (!success) {
            std::cerr << "QCAP 初始化失败: " << m_qcapCapture->GetLastError() << std::endl;
            m_qcapCapture.reset();
            return false;
        }

        return true;
    }
    catch (const std::exception& e) {
        std::cerr << "QCAP 初始化异常: " << e.what() << std::endl;
        m_qcapCapture.reset();
        return false;
    }
    catch (...) {
        std::cerr << "QCAP initialization raised a non-standard exception. This usually means a vendor runtime DLL is still missing." << std::endl;
        m_qcapCapture.reset();
        return false;
    }
}

bool ScreenCapture::SetCaptureMode(CaptureMode mode) {
    if (!m_initialized) {
        return false;
    }

    if (mode == m_currentMode) {
        return true;
    }

    // 需要重新初始化
    if (mode == CaptureMode::QCAP && m_hasQcapConfig) {
        return InitializeQCAP(m_qcapConfig);
    }

    return Initialize(mode, m_width, m_height, m_x, m_y);
}

bool ScreenCapture::Capture(cv::Mat& output, int timeoutMs) {
    if (!m_initialized) {
        std::cerr << "截图模块未初始化" << std::endl;
        return false;
    }

    switch (m_currentMode) {
    case CaptureMode::GDI:
    {
        if (!m_gdiCapture) {
            return false;
        }
        const unsigned char* bgrData = m_gdiCapture->CaptureBGR();
        if (bgrData == nullptr) {
            return false;
        }
        return ConvertGDIToMat(bgrData, output);
    }

    case CaptureMode::OBS:
    {
        if (m_obsCapture) {
            return m_obsCapture->Capture(output);
        }
        return false;
    }

    case CaptureMode::QCAP:
    {
        if (m_qcapCapture) {
            return m_qcapCapture->Capture(output, timeoutMs);
        }
        return false;
    }

    default:
        return false;
    }
}

const unsigned char* ScreenCapture::CaptureBGR(int timeoutMs) {
    if (!m_initialized) {
        return nullptr;
    }

    if (m_currentMode == CaptureMode::GDI && m_gdiCapture) {
        return m_gdiCapture->CaptureBGR();
    }

    if (m_currentMode == CaptureMode::QCAP && m_qcapCapture) {
        return m_qcapCapture->CaptureBGR(timeoutMs);
    }

    return nullptr;
}

const unsigned char* ScreenCapture::CaptureBMP() {
    if (!m_initialized || m_currentMode != CaptureMode::GDI) {
        return nullptr;
    }

    return m_gdiCapture->CaptureBMP();
}

bool ScreenCapture::SetGDIRegion(int x, int y, int width, int height) {
    if (!m_initialized || m_currentMode != CaptureMode::GDI) {
        return false;
    }

    m_x = x;
    m_y = y;
    m_width = width;
    m_height = height;

    return m_gdiCapture->SetRegion(x, y, width, height);
}

bool ScreenCapture::SetGDIWindow(HWND hwnd) {
    if (!m_initialized || m_currentMode != CaptureMode::GDI) {
        return false;
    }

    return m_gdiCapture->SetWindow(hwnd);
}

void ScreenCapture::Release() {
    if (m_gdiCapture) {
        m_gdiCapture->Release();
        m_gdiCapture.reset();
    }

    if (m_obsCapture) {
        m_obsCapture->Release();
        m_obsCapture.reset();
    }

    if (m_qcapCapture) {
        m_qcapCapture->Release();
        m_qcapCapture.reset();
    }

    m_initialized = false;
    m_tempBuffer.clear();
}

bool ScreenCapture::ConvertGDIToMat(const unsigned char* bgrData, cv::Mat& output) {
    if (bgrData == nullptr) {
        return false;
    }

    int width = m_gdiCapture->GetWidth();
    int height = m_gdiCapture->GetHeight();

    // 创建 Mat 对象 (GDI 返回的是 24 位 BGR 数据)
    output = cv::Mat(height, width, CV_8UC3);

    // 复制数据
    memcpy(output.data, bgrData, width * height * 3);

    // GDI 截图是上下翻转的, 需要翻转回来
    cv::flip(output, output, 0);

    return true;
}
