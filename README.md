# Robot Tracking Integration (Robot Vision Tracking)

本项目旨在整合 **SAM 2** (用于目标分割与掩码传播) 和 **SpatialTracker V2 (SpaTracker)** (用于 3D 轨迹追踪)，实现高效、低显存占用的机器人视觉追踪流水线。

## 🚀 快速开始

### 1. 环境准备
确保已安装 `conda` 并在 `tracking` 环境下运行：
```bash
conda activate tracking
```

### 2. 即插即用 (迁移指南)
本项目支持“即插即用”模式。迁移到新服务器（如实验室 4090）时，只需确保以下库与本项目处于**同级目录**：
- `sam2/`
- `SpaTrackerV2/`
- `utils3d-main/`

代码会自动通过 `core/config.py` 识别这些路径。

### 3. 运行追踪
```bash
python main_pipeline.py
```
- 输入视频默认路径: `data/protein.mp4`
- 结果保存路径: `results/`

### 4. 可视化
- **2D 轨迹视频**: `python scripts/visualize_2d.py`
- **3D 交互可视化**: `python /path/to/SpaTrackerV2/tapip3d_viz.py results/result_tapip3d.npz`

## 🛠️ 核心优化 (针对 24GB 显存)
- **分段特征提取**: 强制 `fmaps_chunk_size=4`，大幅降低显存峰值。
- **滑动窗口压缩**: 后端追踪窗口 `S_wind=24`，解决 Transformer Attention OOM。
- **BA 绕过补丁**: 自动拦截并跳过 `pycolmap` 的 Bundle Adjustment，解决版本冲突并节省显存。
- **BF16 兼容性**: 自动为 `inverse` 和 `quantile` 提供 Float32 上采样补丁。

## 📂 项目结构
- `core/`: 核心逻辑补丁与包装类
- `utils/`: 采样与视频处理工具
- `scripts/`: 可视化与辅助脚本
- `data/`: 存放输入视频与临时帧
- `results/`: 追踪结果与可视化视频
