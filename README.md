# 批量音频/视频转写工具

这是一个基于 `faster-whisper` 的高性能批量转写工具，支持将音频和视频文件快速转换为纯文本 (.txt) 和 SRT 字幕文件，并针对 GPU 进行了深度优化。

## 🚀 功能特点

- **批量处理**：支持单个文件转写，或对整个文件夹中的所有媒体文件进行批量处理。
- **GPU 加速**：专为 NVIDIA GPU 优化（特别是针对 RTX 50 系列 / CUDA 13 环境进行了 DLL 路径修复）。
- **双重输出**：每个处理的文件将同时生成：
  - `.txt` 文件：仅包含纯净的转写文本（无时间戳）。
  - `.srt` 文件：包含精确时间戳的标准字幕文件。
- **实时进度**：基于音频实际时长的可视化进度条，准确掌握处理进度。
- **多格式支持**：支持 `.mp4`, `.mp3`, `.wav`, `.m4a`, `.flac` 等常用格式。
- **智能回退**：在检测到 GPU 不可用时，会自动回退至 CPU (int8) 模式运行。

## 🛠️ 环境准备

### 1. 系统要求
- **Python 3.8+**
- **NVIDIA GPU**（强烈建议，以获得极速转写体验；否则将使用 CPU）。
- **FFmpeg**：用于音频提取和时长检测。
  - 脚本会自动在系统环境变量 PATH 中寻找 `ffmpeg`，若未找到，请将 `ffmpeg.exe` 和 `ffprobe.exe` 直接放置在脚本所在目录下。

### 2. 依赖安装
安装必要的 Python 库：
```bash
pip install faster-whisper tqdm
```

## 📖 使用说明

在终端中运行脚本，并传入文件路径或文件夹路径作为参数。

### 转写单个文件
```bash
python batch_transcribe.py \"你的视频文件.mp4\"
```

### 批量转写整个文件夹
```bash
python batch_transcribe.py \"C:/路径/到/你的/音频文件夹\"
```

## 📄 输出结果

例如，对于输入文件 `interview_01.mp4`，工具将生成：
- `interview_01.txt`：所有口语内容的纯净文本。
- `interview_01.srt`：带有精准时间轴的标准字幕文件。

## ⚙️ 技术细节

- **模型选择**：默认使用 `medium` 模型，在转写速度与识别准确率之间取得了最佳平衡。
- **CUDA 修复**：内置了针对 RTX 50 系列显卡常见的 DLL 发现问题的自动修复逻辑。
- **音频预处理**：所有输入格式都会通过 FFmpeg 统一转换为 16kHz 单声道 WAV 格式，以确保 Whisper 模型的最佳性能。

## ❓ 常见问题

- **FFmpeg 报错**：如果提示 `FileNotFoundError: ffmpeg not found`，请确保 `ffmpeg.exe` 和 `ffprobe.exe` 已放置在 `batch_transcribe.py` 同级目录下。
- **GPU 未激活**：请检查是否正确安装了与显卡匹配的 CUDA Toolkit 和 cuDNN 库。
