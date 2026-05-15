# 启智平台纯终端运行指南

## 📌 重要说明

启智平台只有终端，没有 Web 界面，所以**跳过交互式点选步骤**，直接使用手动创建的锚点文件即可。

---

## 🚀 完整运行流程

### 步骤 1: 准备数据

将你的视频数据放到 `data/` 目录下（例如 `data/simple_sorting_0409/videos/`）

### 步骤 2: 预览第一帧并确定锚点坐标

```bash
# 从你的数据集中选一个视频，提取第一帧
python3 scripts/preview_first_frame.py --video data/simple_sorting_0409/videos/chunk-000/observation.images.wrist/episode_000000.mp4
```

这会生成 `first_frame_512p.jpg`，你可以：
- 下载这个图片到本地
- 用图片查看器打开
- 确定要追踪物体的中心坐标 (x, y)，范围是 0-512

### 步骤 3: 创建锚点文件

假设你确定的坐标是 (256, 256)：

```bash
python3 scripts/create_anchor_manual.py --x 256 --y 256
```

这会创建 `results/anchor_point.json`

### 步骤 4: 运行自动批处理

```bash
python3 batch_process_auto.py --video_dir data/simple_sorting_0409/videos --limit 200
```

### 步骤 5: 生成筛选后的数据集

```bash
python3 scripts/generate_filtered_dataset.py
```

---

## 📝 快速命令总结

```bash
# 1. 预览第一帧
python3 scripts/preview_first_frame.py --video <你的视频路径>

# 2. 创建锚点 (替换 x, y 为你确定的坐标)
python3 scripts/create_anchor_manual.py --x 256 --y 256

# 3. 自动批处理
python3 batch_process_auto.py --video_dir <你的视频目录>

# 4. 生成筛选数据集
python3 scripts/generate_filtered_dataset.py
```

---

## ⚠️ 注意事项

1. **坐标是 512p 分辨率下的**：`preview_first_frame.py` 输出的图片已经缩放到 512p，直接看这个图上的坐标即可
2. **锚点要在物体中心**：尽量选在你要追踪的物体中心位置
3. **权重文件**：确保 external 目录下的模型权重已经下载好（.gitignore 会忽略权重，需要单独下载）

---

## ☁️ OBS 使用注意事项

### RHOS 环境配置

在 RHOS 实验室服务器上，`obsutil` 已安装在 `/usr/local/bin/obsutil`。

由于 home 目录是只读文件系统，**必须修改配置文件或使用命令行参数**：

### 方法 1：修改配置文件（推荐，一次修改永久有效）

编辑 `~/.obsutilconfig`，修改以下配置项：

```ini
sdkLogPath=/data/lihong-project/qihang/projects/robot_tracking_integration/.obsutil_log/obssdk.log
utilLogPath=/data/lihong-project/qihang/projects/robot_tracking_integration/.obsutil_log/obsutil.log
defaultTempFileDir=/data/lihong-project/qihang/projects/robot_tracking_integration
```

### 方法 2：使用命令行参数（每次都要加）

下载文件时必须加上 `-cpd` 参数指定检查点目录：

```bash
obsutil cp obs://sai.liyl/lihong/文件名 data/ -cpd ./.obsutil_checkpoint
```

### 上传文件到 OBS

同样需要注意检查点目录问题：

```bash
cd data
tar -czf 你的文件名.tar.gz 你的目录/
obsutil cp 你的文件名.tar.gz obs://sai.liyl/lihong/ -cpd ./.obsutil_checkpoint
```

### OBS 上现有文件列表

| 文件名 | 大小 |
|--------|------|
| `510_erase_board_350_lerobot.zip` | 2.49GB |
| `512_stiring_vision_only_50000.tar.gz` | 11.59GB |
| `513_screw_350_vision_30000.tar.gz` | 11.59GB |
| `513_screw_lerobot_490.zip` | 3.35GB |
| `513_screw_vision_90000.tar.gz` | 11.59GB |
| `514_stiring_350_vision_30000.tar.gz` | 11.59GB |

### 查看 OBS 文件

```bash
obsutil ls obs://sai.liyl/lihong/
```
