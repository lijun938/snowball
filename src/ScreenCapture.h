#pragma once
#include "gdi.h"
#include "obs.h"
#include "QCAPCapture.h"

#include <memory>
#include <opencv2/opencv.hpp>
#include <vector>

//============================================================================
// 截图模式枚举
//============================================================================
enum class CaptureMode {
    GDI,        // 使用 GDI 截图
    OBS,        // 使用 OBS 截图
    QCAP,       // 使用天创 710N1 / QCAP 采集
};

//============================================================================
// 统一截图管理类
//============================================================================
class ScreenCapture {
public:
    ScreenCapture();
    ~ScreenCapture();

    // 初始化接口
    // 参数说明:
    // mode: 截图模式 (GDI 或 OBS)
    // width, height: 截图区域大小
    // x, y: 截图区域位置 (仅 GDI 模式有效)
    // hwnd: 目标窗口句柄 (仅 GDI 模式有效, nullptr 表示全屏)
    // obsIp, obsPort: OBS 服务器地址和端口 (仅 OBS 模式有效)
    bool Initialize(
        CaptureMode mode = CaptureMode::GDI,
        int width = 1920,
        int height = 1080,
        int x = 0,
        int y = 0,
        HWND hwnd = nullptr,
        const std::string& obsIp = "0.0.0.0",
        int obsPort = 7788
    );

    // 使用天创 QCAP 初始化
    bool InitializeQCAP(const QCAPCaptureConfig& config);

    // 设置截图模式
    bool SetCaptureMode(CaptureMode mode);

    // 获取当前截图模式
    CaptureMode GetCaptureMode() const { return m_currentMode; }

    // 截图接口
    // output: 输出的 OpenCV Mat 对象 (BGR 格式)
    bool Capture(cv::Mat& output, int timeoutMs = 30);

    // 获取原始 BGR 数据 (仅 GDI 模式, 返回 24 位 BGR 数据)
    const unsigned char* CaptureBGR(int timeoutMs = 30);

    // 获取原始 BMP 数据 (仅 GDI 模式)
    const unsigned char* CaptureBMP();

    // 更新 GDI 截图区域
    bool SetGDIRegion(int x, int y, int width, int height);

    // 设置 GDI 目标窗口
    bool SetGDIWindow(HWND hwnd);

    // 释放资源
    void Release();

    // 获取截图尺寸
    int GetWidth() const { return m_width; }
    int GetHeight() const { return m_height; }

    // 检查是否已初始化
    bool IsInitialized() const { return m_initialized; }

    // 获取 QCAP 采集实例，后续 TensorRT / GPUDirect 接入会直接复用它
    QCAPCapture* GetQCAPCapture() { return m_qcapCapture.get(); }
    const QCAPCapture* GetQCAPCapture() const { return m_qcapCapture.get(); }

private:
    // 初始化 GDI 截图
    bool InitializeGDI(int width, int height, int x, int y, HWND hwnd);

    // 初始化 OBS 截图
    bool InitializeOBS(const std::string& ip, int port);

    // 初始化 QCAP 截图
    bool InitializeQCAPInternal(const QCAPCaptureConfig& config);

    // GDI BGR 转 OpenCV Mat
    bool ConvertGDIToMat(const unsigned char* bgrData, cv::Mat& output);

    std::unique_ptr<GDIScreenCapture> m_gdiCapture;
    std::unique_ptr<Obs> m_obsCapture;
    std::unique_ptr<QCAPCapture> m_qcapCapture;

    CaptureMode m_currentMode;
    bool m_initialized;
    bool m_hasQcapConfig;

    int m_width;
    int m_height;
    int m_x;
    int m_y;
    QCAPCaptureConfig m_qcapConfig;

    // GDI 模式下的临时缓冲区
    std::vector<unsigned char> m_tempBuffer;
};
