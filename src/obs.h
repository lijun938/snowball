#pragma once
#include "HPSocket/HPSocket.h"
#include <Windows.h>
#include <string>
#include <vector>
#include <mutex>
#include <atomic>
#include <opencv2/opencv.hpp>

class Obs : public CUdpServerListener {
public:
    // 构造函数和析构函数
    Obs();
    ~Obs();

    // 删除拷贝和移动
    Obs(const Obs&) = delete;
    Obs& operator=(const Obs&) = delete;
    Obs(Obs&&) = delete;
    Obs& operator=(Obs&&) = delete;

    // 初始化和释放函数
    bool Init(const std::string& ipAddress = "0.0.0.0", int port = 7788);
    bool Capture(cv::Mat& output);
    bool Release();

    // 获取状态信息
    bool IsInitialized() const { return m_initialized; }
    bool IsConnected() const { return m_connected; }
    uint32_t GetReceivedPackets() const { return m_receivedPackets; }
    uint32_t GetDecodedFrames() const { return m_decodedFrames; }

private:
    // HP-Socket 回调函数
    virtual EnHandleResult OnPrepareListen(IUdpServer* pSender, SOCKET soListen) override;
    virtual EnHandleResult OnAccept(IUdpServer* pSender, CONNID dwConnID, UINT_PTR pSockAddr) override;
    virtual EnHandleResult OnReceive(IUdpServer* pSender, CONNID dwConnID, const BYTE* pData, int iLength) override;
    virtual EnHandleResult OnSend(IUdpServer* pSender, CONNID dwConnID, const BYTE* pData, int iLength) override;
    virtual EnHandleResult OnClose(IUdpServer* pSender, CONNID dwConnID, EnSocketOperation enOperation, int iErrorCode) override;
    virtual EnHandleResult OnShutdown(IUdpServer* pSender) override;
    virtual EnHandleResult OnHandShake(IUdpServer* pSender, CONNID dwConnID) override;

    // 尝试从累积的缓冲区中解码最新的一帧
    cv::Mat TryDecodeLatestFrame();

    CUdpServerPtr m_pServer;

    std::mutex m_mutex;
    std::vector<BYTE> m_imageBuffer;

    std::atomic<bool> m_initialized;
    std::atomic<bool> m_connected;

    // 统计信息
    std::atomic<uint32_t> m_receivedPackets;
    std::atomic<uint32_t> m_decodedFrames;
};