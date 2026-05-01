#define NOMINMAX
#include "yolov5.h"

// BMP结构体定义
#pragma pack(push, 1)
struct BMPFileHeader {
    uint16_t bfType;
    uint32_t bfSize;
    uint16_t bfReserved1;
    uint16_t bfReserved2;
    uint32_t bfOffBits;
};
struct BMPInfoHeader {
    uint32_t biSize;
    int32_t  biWidth;
    int32_t  biHeight;
    uint16_t biPlanes;
    uint16_t biBitCount;
    uint32_t biCompression;
    uint32_t biSizeImage;
    int32_t  biXPelsPerMeter;
    int32_t  biYPelsPerMeter;
    uint32_t biClrUsed;
    uint32_t biClrImportant;
};
#pragma pack(pop)

YoloV5DetectorDML::YoloV5DetectorDML() {
}

YoloV5DetectorDML::~YoloV5DetectorDML() {
    ReleaseResources();
}

bool YoloV5DetectorDML::InitializePath(const char* weightsPath, int device, int numThreads) {
    // 创建环境
    if (!CheckStatus(m_ort->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "YoloV5DML", &m_env), __LINE__)) {
        return false;
    }

    // 创建会话选项
    if (!CheckStatus(m_ort->CreateSessionOptions(&m_session_options), __LINE__)) {
        return false;
    }

    // 设置图优化级别
    if (!CheckStatus(m_ort->SetSessionGraphOptimizationLevel(m_session_options, ORT_ENABLE_BASIC), __LINE__)) {
        return false;
    }

    // 禁用内存模式
    if (!CheckStatus(m_ort->DisableMemPattern(m_session_options), __LINE__)) {
        return false;
    }

    // 设置执行模式
    m_ort->SetSessionExecutionMode(m_session_options, ORT_SEQUENTIAL);

    // 设置线程数
    if (!CheckStatus(m_ort->SetIntraOpNumThreads(m_session_options, numThreads), __LINE__)) {
        return false;
    }

    // 添加DirectML执行提供者
    OrtSessionOptionsAppendExecutionProvider_DML(m_session_options, device);

    // 创建会话
    if (!CheckStatus(m_ort->CreateSession(m_env, String2WString(weightsPath).c_str(), m_session_options, &m_session), __LINE__)) {
        return false;
    }

    // 获取分配器
    if (!CheckStatus(m_ort->GetAllocatorWithDefaultOptions(&m_allocator), __LINE__)) {
        return false;
    }

    // 自动检测输入输出名称
    if (!AutoDetectIONames()) {
        return false;
    }

    // 解析模型信息
    if (!ParseModelInfo()) {
        return false;
    }

    return true;
}

bool YoloV5DetectorDML::InitializeData(const unsigned char* modelData, size_t modelLen, int device, int numThreads) {
    // 创建环境
    if (!CheckStatus(m_ort->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "YoloV5DML", &m_env), __LINE__)) {
        return false;
    }

    // 创建会话选项
    if (!CheckStatus(m_ort->CreateSessionOptions(&m_session_options), __LINE__)) {
        return false;
    }

    // 设置图优化级别
    if (!CheckStatus(m_ort->SetSessionGraphOptimizationLevel(m_session_options, ORT_ENABLE_BASIC), __LINE__)) {
        return false;
    }

    // 禁用内存模式
    if (!CheckStatus(m_ort->DisableMemPattern(m_session_options), __LINE__)) {
        return false;
    }

    // 设置执行模式
    m_ort->SetSessionExecutionMode(m_session_options, ORT_SEQUENTIAL);

    // 设置线程数
    if (!CheckStatus(m_ort->SetIntraOpNumThreads(m_session_options, numThreads), __LINE__)) {
        return false;
    }

    // 添加DirectML执行提供者
    OrtSessionOptionsAppendExecutionProvider_DML(m_session_options, device);

    // 使用内存中的模型数据创建会话
    if (!CheckStatus(m_ort->CreateSessionFromArray(m_env, modelData, modelLen, m_session_options, &m_session), __LINE__)) {
        return false;
    }

    // 获取分配器
    if (!CheckStatus(m_ort->GetAllocatorWithDefaultOptions(&m_allocator), __LINE__)) {
        return false;
    }

    // 自动检测输入输出名称
    if (!AutoDetectIONames()) {
        return false;
    }

    // 解析模型信息
    if (!ParseModelInfo()) {
        return false;
    }

    return true;
}

std::wstring YoloV5DetectorDML::String2WString(const std::string& s) {
    std::string strLocale = setlocale(LC_ALL, "");
    const char* chSrc = s.c_str();
    size_t nDestSize = mbstowcs(NULL, chSrc, 0) + 1;
    wchar_t* wchDest = new wchar_t[nDestSize];
    wmemset(wchDest, 0, nDestSize);
    mbstowcs(wchDest, chSrc, nDestSize);
    std::wstring wstrResult = wchDest;
    delete[] wchDest;
    setlocale(LC_ALL, strLocale.c_str());
    return wstrResult;
}

bool YoloV5DetectorDML::CheckStatus(OrtStatus* status, int line) {
    if (status != NULL) {
        const char* msg = m_ort->GetErrorMessage(status);
        m_ort->ReleaseStatus(status);
        return false;
    }
    return true;
}

bool YoloV5DetectorDML::AutoDetectIONames() {
    if (!m_session || !m_allocator) {
        return false;
    }

    // 获取输入节点数量
    size_t num_input_nodes = 0;
    if (!CheckStatus(m_ort->SessionGetInputCount(m_session, &num_input_nodes), __LINE__)) {
        return false;
    }

    if (num_input_nodes > 0) {
        char* input_name_temp = nullptr;
        if (!CheckStatus(m_ort->SessionGetInputName(m_session, 0, m_allocator, &input_name_temp), __LINE__)) {
            return false;
        }

        if (input_name_temp) {
            static char input_name_buffer[256];
            strncpy(input_name_buffer, input_name_temp, sizeof(input_name_buffer) - 1);
            input_name_buffer[sizeof(input_name_buffer) - 1] = '\0';
            m_input_name[0] = input_name_buffer;
            m_ort->AllocatorFree(m_allocator, input_name_temp);
        }
    }

    // 获取输出节点数量
    size_t num_output_nodes = 0;
    if (!CheckStatus(m_ort->SessionGetOutputCount(m_session, &num_output_nodes), __LINE__)) {
        return false;
    }

    if (num_output_nodes > 0) {
        char* output_name_temp = nullptr;
        if (!CheckStatus(m_ort->SessionGetOutputName(m_session, 0, m_allocator, &output_name_temp), __LINE__)) {
            return false;
        }

        if (output_name_temp) {
            static char output_name_buffer[256];
            strncpy(output_name_buffer, output_name_temp, sizeof(output_name_buffer) - 1);
            output_name_buffer[sizeof(output_name_buffer) - 1] = '\0';
            m_output_name[0] = output_name_buffer;
            m_ort->AllocatorFree(m_allocator, output_name_temp);
        }
    }

    return true;
}

bool YoloV5DetectorDML::ParseModelInfo() {
    if (!ParseInput()) {
        return false;
    }

    if (!ParseOutput()) {
        return false;
    }

    if (!CheckStatus(m_ort->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &m_memory_info), __LINE__)) {
        return false;
    }

    return true;
}

bool YoloV5DetectorDML::ParseInput() {
    size_t input_num = 0;
    size_t shape_size = 0;
    char* input_names_temp = nullptr;
    OrtTypeInfo* input_typeinfo = nullptr;
    const OrtTensorTypeAndShapeInfo* Input_tensor_info = nullptr;

    if (!CheckStatus(m_ort->SessionGetInputCount(m_session, &input_num), __LINE__)) {
        return false;
    }

    for (size_t i = 0; i < input_num; i++) {
        if (!CheckStatus(m_ort->SessionGetInputName(m_session, i, m_allocator, &input_names_temp), __LINE__)) {
            return false;
        }

        if (strcmp(*(m_input_name), input_names_temp) != 0) {
            continue;
        }

        if (!CheckStatus(m_ort->SessionGetInputTypeInfo(m_session, i, &input_typeinfo), __LINE__)) {
            return false;
        }

        if (!CheckStatus(m_ort->CastTypeInfoToTensorInfo(input_typeinfo, &Input_tensor_info), __LINE__)) {
            return false;
        }

        if (!CheckStatus(m_ort->GetDimensionsCount(Input_tensor_info, &shape_size), __LINE__)) {
            return false;
        }

        std::vector<int64_t> temp;
        temp.resize(shape_size);
        if (!CheckStatus(m_ort->GetDimensions(Input_tensor_info, temp.data(), shape_size), __LINE__)) {
            return false;
        }

        if (temp.empty() || temp[2] != temp[3]) {
            return false;
        }

        m_input_tensor_size = 1;
        for (size_t j = 0; j < shape_size; j++) {
            m_input_tensor_size *= temp[j];
        }

        m_input_dims.assign(temp.begin(), temp.end());
        m_input_dim = temp[3];
        m_blob = new float[m_input_dim * m_input_dim * 3];

        m_ort->AllocatorFree(m_allocator, input_names_temp);
        return true;
    }

    return false;
}

bool YoloV5DetectorDML::ParseOutput() {
    size_t shape_size = 0;
    size_t num_output_nodes = 0;
    char* output_names_temp = nullptr;
    OrtTypeInfo* output_typeinfo = nullptr;
    const OrtTensorTypeAndShapeInfo* output_tensor_info = nullptr;

    if (!CheckStatus(m_ort->SessionGetOutputCount(m_session, &num_output_nodes), __LINE__)) {
        return false;
    }

    for (size_t i = 0; i < num_output_nodes; i++) {
        if (!CheckStatus(m_ort->SessionGetOutputName(m_session, i, m_allocator, &output_names_temp), __LINE__)) {
            return false;
        }

        if (strcmp(*(m_output_name), output_names_temp) != 0) {
            continue;
        }

        if (!CheckStatus(m_ort->SessionGetOutputTypeInfo(m_session, i, &output_typeinfo), __LINE__)) {
            return false;
        }

        if (!CheckStatus(m_ort->CastTypeInfoToTensorInfo(output_typeinfo, &output_tensor_info), __LINE__)) {
            return false;
        }

        if (!CheckStatus(m_ort->GetDimensionsCount(output_tensor_info, &shape_size), __LINE__)) {
            return false;
        }

        std::vector<int64_t> temp;
        temp.resize(shape_size);
        if (!CheckStatus(m_ort->GetDimensions(output_tensor_info, temp.data(), shape_size), __LINE__)) {
            return false;
        }

        if (temp.empty()) {
            return false;
        }

        m_output_dims.assign(temp.begin(), temp.end());
        m_ort->AllocatorFree(m_allocator, output_names_temp);
        return true;
    }

    return false;
}

bool YoloV5DetectorDML::LetterBoxPreProcess(const unsigned char* imageData, int width, int height, LetterBoxInfo& letterbox_info) {
    m_original_width = width;
    m_original_height = height;

    // 计算缩放比例
    float scale = std::min(static_cast<float>(m_input_dim) / width,
        static_cast<float>(m_input_dim) / height);

    int new_unpad_w = static_cast<int>(std::round(width * scale));
    int new_unpad_h = static_cast<int>(std::round(height * scale));

    float dw = m_input_dim - new_unpad_w;
    float dh = m_input_dim - new_unpad_h;
    dw /= 2.0f;
    dh /= 2.0f;

    int pad_left = static_cast<int>(std::round(dw - 0.1f));
    int pad_top = static_cast<int>(std::round(dh - 0.1f));

    letterbox_info.scale = scale;
    letterbox_info.pad_w = pad_left;
    letterbox_info.pad_h = pad_top;
    letterbox_info.new_unpad_w = new_unpad_w;
    letterbox_info.new_unpad_h = new_unpad_h;

    // 初始化为灰色背景
    const size_t total_pixels = m_input_dim * m_input_dim;
    float* r_ptr = m_blob;
    float* g_ptr = r_ptr + total_pixels;
    float* b_ptr = g_ptr + total_pixels;

    const float gray_value = 114.0f / 255.0f;
    for (size_t i = 0; i < total_pixels; ++i) {
        r_ptr[i] = g_ptr[i] = b_ptr[i] = gray_value;
    }

    // 计算坐标映射
    const float x_ratio = static_cast<float>(width) / new_unpad_w;
    const float y_ratio = static_cast<float>(height) / new_unpad_h;
    const float norm = 1.0f / 255.0f;

    // 最近邻插值处理
    for (int dst_y = 0; dst_y < new_unpad_h; ++dst_y) {
        float src_y = (dst_y + 0.5f) * y_ratio - 0.5f;
        int nearest_y = static_cast<int>(src_y + 0.5f);
        nearest_y = std::max(0, std::min(nearest_y, height - 1));

        const unsigned char* src_row = imageData + nearest_y * width * 3;
        const int dst_row_start = (dst_y + pad_top) * m_input_dim + pad_left;

        for (int dst_x = 0; dst_x < new_unpad_w; ++dst_x) {
            float src_x = (dst_x + 0.5f) * x_ratio - 0.5f;
            int nearest_x = static_cast<int>(src_x + 0.5f);
            nearest_x = std::max(0, std::min(nearest_x, width - 1));

            const unsigned char* src_pixel = src_row + nearest_x * 3;
            const int dst_idx = dst_row_start + dst_x;

            // BGR转RGB并归一化
            r_ptr[dst_idx] = src_pixel[2] * norm;
            g_ptr[dst_idx] = src_pixel[1] * norm;
            b_ptr[dst_idx] = src_pixel[0] * norm;
        }
    }

    return true;
}

void YoloV5DetectorDML::GenerateProposals(float* output, std::vector<DMLObject>& proposals, float conf) {
    proposals.clear();

    for (int box_idx = 0; box_idx < m_output_dims[1]; ++box_idx) {
        const int base_pos = box_idx * m_output_dims[2];
        float objectness = output[base_pos + 4];

        if (objectness > conf) {
            float x_center = output[base_pos + 0];
            float y_center = output[base_pos + 1];
            float width = output[base_pos + 2];
            float height = output[base_pos + 3];

            for (int cls_idx = 5; cls_idx < m_output_dims[2]; ++cls_idx) {
                float cls_prob = output[base_pos + cls_idx];
                float score = objectness * cls_prob;

                if (score > conf) {
                    DMLObject obj;
                    obj.x = x_center - width * 0.5f;
                    obj.y = y_center - height * 0.5f;
                    obj.width = width;
                    obj.height = height;
                    obj.label = cls_idx - 5;
                    obj.prob = score;
                    proposals.emplace_back(obj);
                }
            }
        }
    }
}

float YoloV5DetectorDML::CalculateIOU(const DMLObject& a, const DMLObject& b) {
    float cleft = std::max(a.x, b.x);
    float ctop = std::max(a.y, b.y);
    float cright = std::min(a.x + a.width, b.x + b.width);
    float cbottom = std::min(a.y + a.height, b.y + b.height);

    float c_area = std::max(cright - cleft, 0.0f) * std::max(cbottom - ctop, 0.0f);
    if (c_area == 0.0f) return 0.0f;

    float a_area = std::max(0.0f, a.width) * std::max(0.0f, a.height);
    float b_area = std::max(0.0f, b.width) * std::max(0.0f, b.height);
    return c_area / (a_area + b_area - c_area);
}

std::vector<DMLObject> YoloV5DetectorDML::NMSBoxes(std::vector<DMLObject>& objects, float threshold) {
    if (objects.empty()) return {};

    // 按置信度降序排序
    std::stable_sort(objects.begin(), objects.end(),
        [](const DMLObject& a, const DMLObject& b) {
            return a.prob > b.prob;
        });

    std::vector<DMLObject> result;
    result.reserve(objects.size());
    std::vector<bool> suppressed(objects.size(), false);

    for (size_t i = 0; i < objects.size(); ++i) {
        if (suppressed[i]) continue;

        result.emplace_back(objects[i]);

        for (size_t j = i + 1; j < objects.size(); ++j) {
            if (suppressed[j] || objects[i].label != objects[j].label) continue;

            float iou = CalculateIOU(objects[i], objects[j]);
            if (iou >= threshold) {
                suppressed[j] = true;
            }
        }
    }

    return result;
}

void YoloV5DetectorDML::ScaleBoxes(std::vector<DMLObject>& objects, const LetterBoxInfo& letterbox_info) {
    for (auto& obj : objects) {
        // 减去padding
        obj.x -= letterbox_info.pad_w;
        obj.y -= letterbox_info.pad_h;

        // 除以缩放因子
        float scale_factor = letterbox_info.scale;
        if (scale_factor > 0.0f) {
            obj.x /= scale_factor;
            obj.y /= scale_factor;
            obj.width /= scale_factor;
            obj.height /= scale_factor;
        }

        // 边界检查
        obj.x = std::max(0.0f, std::min(obj.x, static_cast<float>(m_original_width - 1)));
        obj.y = std::max(0.0f, std::min(obj.y, static_cast<float>(m_original_height - 1)));
        obj.width = std::max(1.0f, std::min(obj.width, static_cast<float>(m_original_width) - obj.x));
        obj.height = std::max(1.0f, std::min(obj.height, static_cast<float>(m_original_height) - obj.y));
    }
}

std::vector<DetectionObject> YoloV5DetectorDML::Detect(const unsigned char* imageData, int width, int height, float nms, float conf) {
    m_detectionResults.clear();
    m_rawDetections.clear();

    // LetterBox预处理
    LetterBoxInfo letterbox_info;
    if (!LetterBoxPreProcess(imageData, width, height, letterbox_info)) {
        return m_detectionResults;
    }

    // 创建输入张量
    m_ort->CreateTensorWithDataAsOrtValue(m_memory_info, m_blob, m_input_tensor_size * sizeof(float),
        m_input_dims.data(), m_input_dims.size(),
        ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &m_input_tensor);

    // 运行推理
    m_ort->Run(m_session, NULL, m_input_name, &m_input_tensor, 1, m_output_name, 1, &m_output_tensor);

    // 获取输出数据
    void* outputData;
    m_ort->GetTensorMutableData(m_output_tensor, &outputData);
    m_output_data = static_cast<float*>(outputData);

    // 生成检测提案
    GenerateProposals(m_output_data, m_rawDetections, conf);

    // NMS处理
    m_rawDetections = NMSBoxes(m_rawDetections, nms);

    // 坐标还原
    ScaleBoxes(m_rawDetections, letterbox_info);

    // 释放张量
    if (m_input_tensor) {
        m_ort->ReleaseValue(m_input_tensor);
        m_input_tensor = nullptr;
    }
    if (m_output_tensor) {
        m_ort->ReleaseValue(m_output_tensor);
        m_output_tensor = nullptr;
    }

    // 转换为最终格式
    for (const auto& obj : m_rawDetections) {
        DetectionObject det;
        det.bbox.x = obj.x;
        det.bbox.y = obj.y;
        det.bbox.width = obj.width;
        det.bbox.height = obj.height;
        det.label = obj.label;
        det.prob = obj.prob;
        m_detectionResults.push_back(det);
    }

    return m_detectionResults;
}

bool YoloV5DetectorDML::IsBMPFile(const unsigned char* data, size_t dataSize) {
    if (dataSize < sizeof(BMPFileHeader)) return false;
    const BMPFileHeader* header = reinterpret_cast<const BMPFileHeader*>(data);
    return header->bfType == 0x4D42;
}

bool YoloV5DetectorDML::ExtractBMPPixelData(const unsigned char* bmpData, size_t bmpSize,
    std::vector<unsigned char>& pixelData, int& width, int& height) {
    if (bmpSize < sizeof(BMPFileHeader) + sizeof(BMPInfoHeader)) return false;

    const BMPFileHeader* fileHeader = reinterpret_cast<const BMPFileHeader*>(bmpData);
    const BMPInfoHeader* infoHeader = reinterpret_cast<const BMPInfoHeader*>(bmpData + sizeof(BMPFileHeader));

    if (fileHeader->bfType != 0x4D42 || infoHeader->biBitCount != 24) return false;

    width = infoHeader->biWidth;
    height = abs(infoHeader->biHeight);
    int rowSize = ((width * 3 + 3) / 4) * 4;
    const unsigned char* srcPixels = bmpData + fileHeader->bfOffBits;

    pixelData.resize(width * height * 3);
    bool isTopDown = infoHeader->biHeight < 0;

    for (int y = 0; y < height; y++) {
        int srcRow = isTopDown ? y : (height - 1 - y);
        const unsigned char* srcRowData = srcPixels + srcRow * rowSize;
        for (int x = 0; x < width; x++) {
            int srcIndex = x * 3;
            int dstIndex = (y * width + x) * 3;
            pixelData[dstIndex] = srcRowData[srcIndex];
            pixelData[dstIndex + 1] = srcRowData[srcIndex + 1];
            pixelData[dstIndex + 2] = srcRowData[srcIndex + 2];
        }
    }
    return true;
}

std::vector<DetectionObject> YoloV5DetectorDML::DetectBMP(const unsigned char* bmpData, size_t bmpSize, float nms, float conf) {
    std::vector<unsigned char> pixelData;
    int width = 0, height = 0;
    if (IsBMPFile(bmpData, bmpSize) && ExtractBMPPixelData(bmpData, bmpSize, pixelData, width, height)) {
        return Detect(pixelData.data(), width, height, nms, conf);
    }
    return {};
}

std::vector<DetectionObject> YoloV5DetectorDML::DetectBGR(const unsigned char* imageData, int width, int height, float nms, float conf) {
    return Detect(imageData, width, height, nms, conf);
}

void YoloV5DetectorDML::ReleaseResources() {
    if (m_input_tensor) {
        m_ort->ReleaseValue(m_input_tensor);
        m_input_tensor = nullptr;
    }
    if (m_output_tensor) {
        m_ort->ReleaseValue(m_output_tensor);
        m_output_tensor = nullptr;
    }
    if (m_session) {
        m_ort->ReleaseSession(m_session);
        m_session = nullptr;
    }
    if (m_session_options) {
        m_ort->ReleaseSessionOptions(m_session_options);
        m_session_options = nullptr;
    }
    if (m_memory_info) {
        m_ort->ReleaseMemoryInfo(m_memory_info);
        m_memory_info = nullptr;
    }
    if (m_env) {
        m_ort->ReleaseEnv(m_env);
        m_env = nullptr;
    }
    if (m_blob) {
        delete[] m_blob;
        m_blob = nullptr;
    }
}

void YoloV5DetectorDML::ResetState() {
    m_detectionResults.clear();
    m_rawDetections.clear();
}