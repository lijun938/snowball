# Snowball 源码目录

本文件夹包含项目的完整 C++ 源码。

## 核心源文件

| 文件 | 说明 |
|------|------|
| main.cpp | 程序入口，主循环逻辑 |
| AppConfig.h/cpp | 配置文件(config.ini)及命令行参数解析 |
| LiveControlPanel.h/cpp | Win32 GUI 控制面板 |
| MovementController.h/cpp | 追踪、目标选择、滤波、运动预测、摇杆输出 |
| DetectorBackend.h/cpp | 检测器后端接口和初始化 |
| ScreenCapture.h/cpp | 屏幕采集抽象 (GDI/OBS/QCAP) |
| TensorRTYoloDetector.h/cpp | TensorRT YOLO 检测后端 |
| TensorRTDetector.h/cpp | TensorRT Legacy 检测后端 |
| TensorRTGpuPreprocess.cu/h | GPU 预处理 CUDA 核函数 |
| TensorRTGpuPostprocess.cu/h | GPU 后处理 CUDA 核函数 |
| QCAPCapture.h/cpp | 天创采集卡封装 |
| gdi.h/cpp | GDI 截图 |
| obs.h/cpp | OBS 截图 |
| yolov5.h/cpp | YOLOv5 后处理 |
| DetectionTypes.h | 检测数据结构 |

## 编译方式

使用 Visual Studio 2022 打开 `snowball.sln` 解决方案文件。

选择 Release | x64 配置，按 F7 编译。

## 依赖

- CUDA Toolkit
- TensorRT
- OpenCV 4.x
- ONNX Runtime (DirectML)
- 天创 QCAP SDK（仅 QCAP 模式需要）
