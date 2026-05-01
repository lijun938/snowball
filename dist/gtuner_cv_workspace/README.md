# TitanTwo GCV 工作区

本文件夹包含 TitanTwo + Gtuner IV 所需的全部脚本。

## 文件说明

| 文件 | 说明 |
|------|------|
| `amtoobs_titan_two_gcv.py` | GCV Python 脚本，Gtuner IV 的「计算机视觉」模块加载此脚本 |
| `amtoobs_titan_two_gcv.json` | GCV 配置，定义通信文件路径等参数 |
| `amtoobs_titan_two_gamepad.gpc` | GPC 手柄脚本，需烧录到 TitanTwo 设备槽位 |
| `titan_two_gcv.bin` | 通信文件（运行时由 snowball.exe 自动生成，无需手动创建） |

## 安装步骤

### 1. 安装 Gtuner IV

从 [Gtuner IV 官网](https://www.consoletuner.com/downloads/) 下载并安装。

### 2. 烧录 GPC 脚本到 TitanTwo

1. 打开 Gtuner IV
2. 连接 TitanTwo 设备
3. 打开 `amtoobs_titan_two_gamepad.gpc`
4. 点击「编译」然后「烧录到设备」，选择一个空闲槽位

### 3. 配置 GCV（计算机视觉）

1. 在 Gtuner IV 中打开「计算机视觉」面板
2. 设置 CV 工作区路径为**本文件夹的绝对路径**
3. 选择 `amtoobs_titan_two_gcv` 作为 GCV 脚本
4. 点击「启动」

### 4. 配置通信路径

**重要：snowball.exe 和 GCV 脚本必须读写同一个 `titan_two_gcv.bin` 文件。**

有两种方式：

**方式 A：GCV 工作区就在 dist 目录下（推荐）**

如果你把 Gtuner IV 的 CV 工作区直接设为本文件夹（`dist/gtuner_cv_workspace/`），则 `config.ini` 中设置：

```ini
titan_two_gcv_path=gtuner_cv_workspace\titan_two_gcv.bin
```

此时两者自动使用同一个文件，无需额外配置。

**方式 B：GCV 工作区在其他位置**

如果你把脚本复制到了其他目录，需要用绝对路径：

1. 修改 `config.ini`：
```ini
titan_two_gcv_path=D:\你的路径\gtuner_cv_workspace\titan_two_gcv.bin
```

2. 修改 `amtoobs_titan_two_gcv.json`：
```json
{
    "packet_path": "D:\\你的路径\\gtuner_cv_workspace\\titan_two_gcv.bin"
}
```

确保两边路径**完全一致**。

### 5. 验证

1. 启动 Gtuner IV 的 GCV 脚本
2. 启动 `snowball.exe`
3. 如果检测到目标且手柄有输出，说明配置成功

## GPC 脚本说明

`amtoobs_titan_two_gamepad.gpc` 的工作原理：

- 保持原始手柄直通，不影响正常操作
- 通过 GCV 协议接收 snowball.exe 发送的瞄准数据
- 仅在按下触发键（默认 ADS 或开火键）时叠加瞄准辅助
- 可通过修改 GPC 头部的宏定义自定义触发条件：
  - `AMTOOBS_TRIGGER_ADS` — ADS 触发按键
  - `AMTOOBS_TRIGGER_FIRE` — 开火触发按键
  - `AMTOOBS_TRIGGER_REQUIRE_BOTH` — 是否要求同时按下两个键
  - `AMTOOBS_TRIGGER_THRESHOLD` — 触发阈值（模拟量）
