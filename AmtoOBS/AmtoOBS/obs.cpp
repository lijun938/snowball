#include "obs.h"
#include <algorithm>
#include <iostream>

// 高精度睡眠函数，可以让出CPU时间片
void yield_sleep_us(double duration_us)
{
    LARGE_INTEGER start, now, frequency;
    QueryPerformanceFrequency(&frequency);
    QueryPerformanceCounter(&start);

    while (true)
    {
        QueryPerformanceCounter(&now);

        // 计算经过的时间（单位：微秒）
        double elapsed_us = (double)(now.QuadPart - start.QuadPart) * 1000000.0 / (double)frequency.QuadPart;

        if (elapsed_us >= duration_us)
        {
            break;
        }
        else
        {
            SwitchToThread();
        }
    }
}

Obs::Obs()
    : m_initialized(false)
    , m_connected(false)
    , m_receivedPackets(0)
    , m_decodedFrames(0)
{
    std::cout << "Obs 实例已创建" << std::endl;
}

Obs::~Obs() {
    Release();
    std::cout << "Obs 实例已销毁" << std::endl;
}

bool Obs::Init(const std::string& ipAddress, int port) {
    if (m_initialized) {
        std::cout << "Obs 已经初始化" << std::endl;
        return true;
    }

    m_pServer.Attach(HP_Create_UdpServer(this));

    if (!m_pServer.IsValid()) {
        std::cerr << "创建 UDP 服务器失败" << std::endl;
        return false;
    }

    m_pServer->SetMaxDatagramSize(65535);
    m_pServer->SetPostReceiveCount(256);
    m_pServer->SetReuseAddressPolicy(RAP_ADDR_AND_PORT);
    m_pServer->SetWorkerThreadCount(1);

    if (!m_pServer->Start(nullptr, (USHORT)port)) {
        std::cerr << "UDP 服务器启动失败，端口：" << port
            << "，错误代码：" << m_pServer->GetLastError() << std::endl;
        m_pServer.Detach();
        return false;
    }

    m_initialized = true;
    std::cout << "UDP 服务器已启动，端口：" << port << "，监听所有网络接口" << std::endl;
    return true;
}

bool Obs::Release() {
    if (!m_initialized) {
        return true;
    }

    if (m_pServer.IsValid() && m_pServer->HasStarted()) {
        m_pServer->Stop();
    }
    m_pServer.Detach();

    {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_imageBuffer.clear();
    }

    m_initialized = false;
    m_connected = false;

    std::cout << "OBS 接收器已停止，已解码 " << m_decodedFrames << " 帧" << std::endl;
    return true;
}

cv::Mat Obs::TryDecodeLatestFrame() {
    const BYTE start_marker[2] = { 0xFF, 0xD8 };  // JPEG 开始标记
    const BYTE end_marker[2] = { 0xFF, 0xD9 };    // JPEG 结束标记

    std::lock_guard<std::mutex> lock(m_mutex);

    // 查找最后一个 JPEG 开始标记
    auto last_start_it = std::find_end(m_imageBuffer.begin(), m_imageBuffer.end(), start_marker, start_marker + 2);

    if (last_start_it == m_imageBuffer.end()) {
        // 如果缓冲区过大且没有有效数据，清空缓冲区
        if (m_imageBuffer.size() > 5 * 1024 * 1024) {
            m_imageBuffer.clear();
        }
        return cv::Mat();
    }

    // 从开始标记后查找结束标记
    auto first_end_it = std::search(last_start_it + 2, m_imageBuffer.end(), end_marker, end_marker + 2);

    if (first_end_it != m_imageBuffer.end()) {
        // 找到完整的 JPEG 帧
        std::vector<BYTE> frame_data(last_start_it, first_end_it + 2);
        m_imageBuffer.erase(m_imageBuffer.begin(), first_end_it + 2);

        // 解码 JPEG 数据
        cv::Mat frame = cv::imdecode(frame_data, cv::IMREAD_COLOR);
        if (!frame.empty()) {
            m_decodedFrames++;
            return frame;
        }
    }
    else {
        // 未找到结束标记，清理开始标记之前的数据
        m_imageBuffer.erase(m_imageBuffer.begin(), last_start_it);
    }

    return cv::Mat();
}

bool Obs::Capture(cv::Mat& output) {
    if (!m_initialized) {
        return false;
    }

    cv::Mat frame = TryDecodeLatestFrame();
    if (!frame.empty()) {
        output = frame;
        return true;
    }
    else {
        yield_sleep_us(200);  // 等待 200 微秒
        return false;
    }
}

// ========== HP-Socket 回调函数 ==========

EnHandleResult Obs::OnPrepareListen(IUdpServer* pSender, SOCKET soListen) {
    // 设置接收缓冲区大小为 4MB
    int recvBufSize = 4 * 1024 * 1024;
    setsockopt(soListen, SOL_SOCKET, SO_RCVBUF, (char*)&recvBufSize, sizeof(recvBufSize));
    return HR_OK;
}

EnHandleResult Obs::OnAccept(IUdpServer* pSender, CONNID dwConnID, UINT_PTR pSockAddr) {
    m_connected = true;
    std::cout << "客户端已连接，连接ID：" << dwConnID << std::endl;
    return HR_OK;
}

EnHandleResult Obs::OnReceive(IUdpServer* pSender, CONNID dwConnID, const BYTE* pData, int iLength) {
    m_receivedPackets++;

    std::lock_guard<std::mutex> lock(m_mutex);
    // 将接收到的数据追加到缓冲区
    m_imageBuffer.insert(m_imageBuffer.end(), pData, pData + iLength);

    // 限制缓冲区大小，防止内存溢出
    const size_t MAX_BUFFER = 5 * 1024 * 1024;  // 5MB
    if (m_imageBuffer.size() > MAX_BUFFER) {
        // 保留最近的 2MB 数据
        m_imageBuffer.erase(m_imageBuffer.begin(), m_imageBuffer.begin() + (m_imageBuffer.size() - (2 * 1024 * 1024)));
    }

    return HR_OK;
}

EnHandleResult Obs::OnSend(IUdpServer* pSender, CONNID dwConnID, const BYTE* pData, int iLength) {
    return HR_OK;
}

EnHandleResult Obs::OnClose(IUdpServer* pSender, CONNID dwConnID, EnSocketOperation enOperation, int iErrorCode) {
    m_connected = false;
    std::cout << "客户端已断开连接，连接ID：" << dwConnID << std::endl;
    return HR_OK;
}

EnHandleResult Obs::OnShutdown(IUdpServer* pSender) {
    m_connected = false;
    std::cout << "服务器已关闭" << std::endl;
    return HR_OK;
}

EnHandleResult Obs::OnHandShake(IUdpServer* pSender, CONNID dwConnID) {
    return HR_OK;
}