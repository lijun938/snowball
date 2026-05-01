# Snowball

基于 TensorRT + YOLO 的实时目标检测与辅助瞄准系统，支持 TitanTwo 手柄输出。

---

## 硬件需求

| 硬件 | 说明 |
|------|------|
| **NVIDIA 显卡** | GTX 1060 及以上，需支持 CUDA 和 TensorRT |
| **采集卡**（二选一） | ① 天创恒达 710N1 (SC0710 PCI)，使用 QCAP 模式<br>② 任意 USB/PCI 采集卡，使用 OpenCV 模式 |
| **TitanTwo 控制器** | 用于手柄摇杆信号输出 |
| **虚拟摄像头**（可选） | OBS 虚拟摄像头，用于串流预览 |

## 软件环境

| 软件 | 版本/说明 |
|------|-----------|
| **Windows 10/11** | 64-bit |
| **CUDA Toolkit** | 11.x 或 12.x（与显卡驱动匹配） |
| **TensorRT** | 8.x / 10.x（需与 CUDA 版本匹配） |
| **Gtuner IV** | TitanTwo 配套软件，运行 GCV Python 脚本 |
| **Visual Studio 2022** | 仅编译源码时需要（Community 版即可） |
| **OpenCV 4.x** | 已包含在 dist 运行时 DLL 中 |

## 目录结构

```
snowball/
├── dist/               # 可直接运行的发行包
│   ├── snowball.exe    # 主程序
│   ├── config.ini      # 配置文件（需按自己环境修改）
│   ├── *.engine        # TensorRT 模型引擎文件
│   └── *.dll           # 运行时依赖库
├── src/                # 完整源码副本（可独立编译）
│   ├── snowball.sln    # Visual Studio 解决方案
│   ├── snowball.vcxproj
│   ├── *.cpp / *.h     # C++ 源码
│   └── config.ini      # 源码目录的配置副本
├── AmtoOBS/            # 原始 Visual Studio 工程目录
│   ├── snowball.sln
│   └── AmtoOBS/        # 工程源文件
├── README.md           # 本文档
└── .gitignore
```

## 快速开始（使用预编译 EXE）

### 1. 配置采集卡

编辑 `dist/config.ini` 中的 `[capture]` 段：

**天创 710N1 采集卡：**
```ini
[capture]
mode=qcap
device=SC0710 PCI
```

**通用 USB 采集卡 / HDMI 采集卡：**
```ini
[capture]
mode=opencv
opencv_index=0
; opencv_index 从 0 开始，多设备时尝试 0、1、2...
```

### 2. 配置 TitanTwo GCV 路径

编辑 `dist/config.ini` 中的 `titan_two_gcv_path`，指向 Gtuner IV 工作区：

```ini
titan_two_gcv_path=D:\你的路径\gtuner_cv_workspace\titan_two_gcv.bin
```

同时确保 Gtuner IV 中 GCV Python 脚本的 JSON 配置文件 (`amtoobs_titan_two_gcv.json`) 中的 `packet_path` 与上面路径**完全一致**。

### 3. 配置模型

将 TensorRT 引擎文件（`.engine`）放入 `dist/` 目录，并在 `config.ini` 中设置：

```ini
[detector]
engine=你的模型.engine
```

### 4. 启动

1. 在 Gtuner IV 中启动 GCV Python 脚本
2. 双击 `dist/snowball.exe`
3. 在 UI 控制面板中调整参数

## 编译指南（从源码构建）

### 前置条件

- Visual Studio 2022（安装 C++ 桌面开发工作负载）
- CUDA Toolkit + TensorRT（配置好系统 PATH）
- OpenCV 4.x（配置好 include/lib 路径）
- QCAP SDK（仅 QCAP 模式需要）

### 编译步骤

```powershell
# 使用 MSBuild 编译
$msbuild = "C:\Program Files\Microsoft Visual Studio\2022\Community\MSBuild\Current\Bin\amd64\MSBuild.exe"
& $msbuild "AmtoOBS\snowball.sln" /p:Configuration=Release /p:Platform=x64

# 编译产物在 AmtoOBS\x64\Release\snowball.exe
# 复制到 dist 目录
Copy-Item "AmtoOBS\x64\Release\snowball.exe" -Destination "dist\" -Force
```

### 同步源码副本

```powershell
Copy-Item "AmtoOBS\AmtoOBS\*.cpp" -Destination "src\" -Force
Copy-Item "AmtoOBS\AmtoOBS\*.h"   -Destination "src\" -Force
Copy-Item "AmtoOBS\AmtoOBS\config.ini" -Destination "src\" -Force
```

## 关键配置说明

### config.ini 参数速查

| 参数 | 推荐值 | 说明 |
|------|--------|------|
| `vertical_bias` | -0.15 ~ -0.25 | 垂直偏移，负值偏头部 |
| `deadzone` | 4 ~ 8 | 死区像素，防微抖 |
| `stick_curve` | 0.3 ~ 1.3 | 摇杆曲线，>1 柔和 <1 灵敏 |
| `stick_response_boost` | 1.5 ~ 3.0 | 响应倍率 |
| `aim_track_confirm_frames` | 1 ~ 3 | 检测多少帧后锁定目标 |
| `aim_track_lost_frames` | 5 ~ 15 | 丢失多少帧后释放目标 |
| `aim_lock_bonus` | 1.0 ~ 2.0 | 目标锁定粘性 |
| `aim_prediction_ms` | 30 ~ 60 | 运动预测时间 (ms) |

### TitanTwo + Gtuner IV 配置

1. 将 TitanTwo 连接电脑
2. 打开 Gtuner IV → Device Monitor
3. 运行 GCV Python 脚本（`amtoobs_titan_two_gcv.py`）
4. 确认 JSON 配置中 `packet_path` 与 `config.ini` 中 `titan_two_gcv_path` 一致
5. 启动 `snowball.exe`

## License

开源项目，仅供学习交流使用。
