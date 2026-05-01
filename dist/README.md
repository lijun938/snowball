# Snowball 使用说明

## 环境要求

- Windows 10/11 64位
- NVIDIA 显卡（支持 CUDA，推荐 GTX 1060 及以上）
- CUDA Toolkit 已安装
- TensorRT 运行时（nvinfer DLL 已包含在本目录）
- 采集卡（二选一）：
  - 天创恒达 710N1 或兼容 QCAP 的型号
  - 任意 USB/PCI HDMI 采集卡（使用 OpenCV 模式）
- TitanTwo 设备 + Gtuner IV 软件

## 快速开始

### 第一步：配置采集卡

编辑 `config.ini` 的 `[capture]` 段：

**天创 710N1 采集卡：**
```ini
[capture]
mode=qcap
device=SC0710 PCI
input=hdmi
```

**通用 USB/HDMI 采集卡：**
```ini
[capture]
mode=opencv
opencv_index=0
```
> `opencv_index` 从 0 开始编号，如果画面不对可尝试 1、2...

### 第二步：配置 TitanTwo GCV 路径

编辑 `config.ini` 中的 `titan_two_gcv_path`：

```ini
titan_two_gcv_path=D:\你的路径\gtuner_cv_workspace\titan_two_gcv.bin
```

同时确保 Gtuner IV 中 GCV Python 脚本的 JSON 配置文件 (`amtoobs_titan_two_gcv.json`) 中 `packet_path` 与上面路径**完全一致**。

### 第三步：配置 Gtuner IV

1. 打开 Gtuner IV
2. 在「计算机视觉」面板中，设置 CV 工作区路径
3. 选择 `amtoobs_titan_two_gcv` 作为 GCV 脚本
4. 在设备储存槽中加载 `amtoobs_titan_two_gamepad.gpc`
5. 启动 GCV 脚本

### 第四步：运行

双击 `snowball.exe` 启动。

## 文件说明

```
snowball.exe                   主程序
config.ini                     配置文件（需按自己环境修改）
apex-yolov8n-trtyolo.engine    TensorRT 推理引擎
gtuner_cv_workspace/           Gtuner GCV 工作区
  amtoobs_titan_two_gcv.py       GCV Python 脚本
  amtoobs_titan_two_gcv.json     GCV 配置（需配置 packet_path）
  amtoobs_titan_two_gamepad.gpc  GPC 手柄脚本
  titan_two_gcv.bin              通信文件（运行时自动生成）
*.dll                          运行时依赖库
```

## 调参指南

优先调节的参数（按顺序）：

| 参数 | 建议值 | 说明 |
|------|--------|------|
| vertical_bias | -0.15 ~ -0.25 | 瞄准偏上比例，负值瞄头 |
| deadzone | 4-8 | 死区像素，防微抖 |
| stick_curve | 0.3-1.3 | 大于1柔和，小于1灵敏 |
| stick_response_boost | 1.5-3.0 | 响应倍率 |
| aim_lock_bonus | 1.0-2.0 | 目标锁定粘性 |
| aim_prediction_ms | 30-60 | 运动预测时间 (ms) |

## 常见问题

**Q: 双击 exe 闪退**
- 检查采集卡是否连接
- 检查 config.ini 中 device 名称是否正确
- 检查 CUDA/TensorRT 环境

**Q: 检测到目标但不瞄准**
- 确认 Gtuner IV 的 GCV 脚本已启动
- 确认 config.ini 的 `titan_two_gcv_path` 与 Gtuner JSON 中的 `packet_path` 完全一致
- 确认 TitanTwo 设备已连接

**Q: 如何更换模型**
- 将新的 .engine 文件放在 exe 同目录
- 修改 config.ini 中 `engine=新文件名.engine`
- 重启程序

**Q: 使用通用采集卡没画面**
- 确认 config.ini 中 `mode=opencv`
- 尝试不同的 `opencv_index`（0, 1, 2...）
- 确认采集卡驱动已正确安装
