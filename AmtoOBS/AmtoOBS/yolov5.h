#pragma once
#include "DetectionTypes.h"

#include <onnxruntime_cxx_api.h>
#include <onnxruntime_c_api.h>
#include <dml_provider_factory.h>
#include <windows.h>
#include <iostream>
#include <vector>
#include <algorithm>
#include <string>
#include <cmath>

// 检测对象结构
// LetterBox信息结构
struct LetterBoxInfo {
    float scale;           // 缩放比例
    float pad_w;          // 水平padding
    float pad_h;          // 垂直padding
    int new_unpad_w;      // 缩放后未填充的宽度
    int new_unpad_h;      // 缩放后未填充的高度
};

// 内部检测对象结构
struct DMLObject {
    float x;
    float y;
    float width;
    float height;
    int label;
    float prob;
};

// YoloV5 DirectML检测器类
class YoloV5DetectorDML {
public:
    YoloV5DetectorDML();
    ~YoloV5DetectorDML();

    // 初始化接口
    bool InitializePath(const char* weightsPath, int device = 0, int numThreads = 1);
    bool InitializeData(const unsigned char* modelData, size_t modelLen, int device = 0, int numThreads = 1);

    // 检测接口
    std::vector<DetectionObject> Detect(const unsigned char* imageData, int width, int height, float nms = 0.45f, float conf = 0.25f);
    std::vector<DetectionObject> DetectBMP(const unsigned char* bmpData, size_t bmpSize, float nms = 0.45f, float conf = 0.25f);
    std::vector<DetectionObject> DetectBGR(const unsigned char* imageData, int width, int height, float nms = 0.45f, float conf = 0.25f);

    // 资源管理
    void ReleaseResources();
    void ResetState();

    // 获取结果
    int GetObjectCount() const { return static_cast<int>(m_detectionResults.size()); }
    const std::vector<DetectionObject>& GetDetectionResults() const { return m_detectionResults; }

private:
    // ONNX Runtime相关
    OrtEnv* m_env = nullptr;
    OrtSessionOptions* m_session_options = nullptr;
    OrtSession* m_session = nullptr;
    OrtMemoryInfo* m_memory_info = nullptr;
    OrtAllocator* m_allocator = nullptr;
    OrtValue* m_input_tensor = nullptr;
    OrtValue* m_output_tensor = nullptr;
    const OrtApi* m_ort = OrtGetApiBase()->GetApi(ORT_API_VERSION);

    // 模型参数
    const char* m_input_name[1] = { "images" };
    const char* m_output_name[1] = { "output" };
    std::vector<int64_t> m_input_dims;
    std::vector<int64_t> m_output_dims;
    size_t m_input_tensor_size = 1;
    int m_input_dim = 0;
    float* m_blob = nullptr;
    float* m_output_data = nullptr;

    // 检测结果
    std::vector<DetectionObject> m_detectionResults;
    std::vector<DMLObject> m_rawDetections;

    // 图像信息
    int m_original_width = 0;
    int m_original_height = 0;

    // 内部方法
    bool CheckStatus(OrtStatus* status, int line);
    std::wstring String2WString(const std::string& s);
    bool AutoDetectIONames();
    bool ParseModelInfo();
    bool ParseInput();
    bool ParseOutput();

    // 图像预处理
    bool LetterBoxPreProcess(const unsigned char* imageData, int width, int height, LetterBoxInfo& letterbox_info);

    // YoloV5特定处理
    void GenerateProposals(float* output, std::vector<DMLObject>& proposals, float conf);
    std::vector<DMLObject> NMSBoxes(std::vector<DMLObject>& objects, float threshold);
    void ScaleBoxes(std::vector<DMLObject>& objects, const LetterBoxInfo& letterbox_info);
    float CalculateIOU(const DMLObject& a, const DMLObject& b);

    // BMP处理
    bool IsBMPFile(const unsigned char* data, size_t dataSize);
    bool ExtractBMPPixelData(const unsigned char* bmpData, size_t bmpSize,
        std::vector<unsigned char>& pixelData, int& width, int& height);
};
