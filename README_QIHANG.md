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
