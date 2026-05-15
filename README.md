# Robot Tracking Integration & Quality Control Pipeline

本项目是一套为 **Diffusion Policy** 和 **pi0** 等多模态策略模型量身定制的 3D 数据增强与自动化筛选流水线。通过集成 **SAM 2** 和 **SpaTracker V2**，实现了从原始视频到高质量、物理一致的 3D 轨迹真值提取。

---

## 🚀 核心流水线 (End-to-End Pipeline)

本流水线的设计核心是 **“感知小脑验证大脑”**：利用几何与物理约束，自动从海量感知输出中筛选出 100% 正确的专家演示数据。

### 阶段 1：交互式锚点标定 (Human-in-the-loop)
1.  **启动一致性测试**：运行以下命令，在 Web 界面（默认端口 8080）对前 10 段视频进行点选。
    ```bash
    python3 test_batch_consistency.py
    ```
2.  **计算黄金锚点**：点选完成后，运行脚本分析重叠区域并生成锚点配置文件 `results/anchor_point.json`。
    ```bash
    python3 scripts/find_anchor.py
    ```

### 阶段 2：大规模自动批处理 (Auto-Batch Processing)
1.  **执行全自动追踪**：电脑将读取锚点，自动为目录下所有视频生成 3D 轨迹和评分。
    ```bash
    python3 batch_process_auto.py --video_dir data/simple_sorting_0409/videos --limit 200
    ```
    -   `--video_dir`: 必填，指定包含 `.mp4` 文件的原始视频根目录。
    -   `--anchor_json`: 可选，默认为 `results/anchor_point.json`。
    -   `--limit`: 可选，限制处理的视频数量。

### 阶段 3：数据校验与离线适配 (Quality Control & Meta-Fix)
1.  **生成统计报告**：分析所有视频的得分，查看通过率。
    ```bash
    python3 scripts/summarize_results.py
    ```
2.  **打包高质量数据（新版推荐）**：基于打分文件重新生成完整的数据集，自动处理所有索引对齐。
    ```bash
    python3 scripts/generate_filtered_dataset.py
    ```
    输出目录：`data/simple_sorting_0409_filtered_v4/`
    
    **旧版方式（不推荐）**：仅复制视频，需要额外修复
    ```bash
    python3 scripts/package_filtered_data.py
    python3 scripts/full_fix_v4.py --data_dir data/simple_sorting_0409_filtered
    ```
3.  **上传到 OBS**（可选）：打包并上传到对象存储供训练服务器下载
    ```bash
    cd data && tar -czf simple_sorting_0409_filtered_v4.tar.gz simple_sorting_0409_filtered_v4/
    obsutil cp simple_sorting_0409_filtered_v4.tar.gz obs://your-bucket/path/
    ```

---

## 🎯 质量评估与打分标准 (Quality Metrics)

本项目采用 **四项核心指标** 对追踪质量进行自动评估，任一指标触发即视为该 episode 数据不合格。这套标准是确保 3D 轨迹真值数据质量的关键。

### 四大核心指标

| 指标名称 | 含义 | 触发条件 |
|:--------|:-----|:---------|
| **`visibility_failure`** | 可见性失败 | 连续 5 帧以上，每帧平均可见性 < 0.4 |
| **`low_confidence`** | 低置信度 | 追踪器整体置信度均值 < 0.6 |
| **`reprojection_conflict`** | 重投影冲突 | 重投影误差 P95 > 20px（几何一致性校验失败） |
| **`tracking_jump`** | 跟踪跳跃 | 速度 P95 > 2.0 m/s（轨迹出现突变/跳变） |

### 指标计算逻辑详解

所有打分标准均基于 SpaTracker V2 输出的原始数据计算，代码实现位于 `utils/data_filter.py`。

> **📌 相机参数从何而来？**
> 
> 本项目采用 **VGGT (Visual Geometry Grounded Transformer)** 作为前端，直接从单目视频中预测相机参数，无需人工标定或外部输入：
> 
> ```python
> # 1. VGGT 前端预测几何信息
> vggt_model = VGGT4Track.from_pretrained("weights/SpatialTrackerV2_Front")
> predictions = vggt_model(video)  # 视频 -> 相机参数 + 深度
> 
> extrs = predictions["poses_pred"]  # 相机位姿 (T, 4, 4) 世界到相机
> intrs = predictions["intrs"]       # 相机内参 (T, 3, 3)
> depth = predictions["points_map"][..., 2]  # 深度图 (T, H, W)
> 
> # 2. SpaTracker 后端利用这些参数进行 3D 追踪
> results = spatracker_model.forward(
>     video=video,
>     depth=depth,
>     intrs=intrs,      # VGGT 预测的内参
>     extrs=extrs,      # VGGT 预测的位姿
>     queries=queries,
>     ...
> )
> ```
> 
> **关键特性**:
> - **自包含**: 仅需 RGB 视频，无需相机标定文件
> - **在线估计**: 每帧的相机内参和位姿都是 VGGT 实时预测的
> - **精度**: 基于大规模视觉几何预训练，精度接近传统 SLAM 方法

#### 1. visibility_failure (可见性失败)

**计算流程**:
```
SpaTracker V2 输出:
  - vis_pred: (T, N) 每个追踪点在每帧的可见性分数 [0, 1]
    T = 视频帧数, N = SpaTracker 追踪的网格点数量 (默认 756 个)

计算步骤:
  1. frame_vis_mean = mean(vis_pred, dim=1)  # 每帧对所有追踪点求平均可见性
  2. low_vis_mask = frame_vis_mean < 0.4     # 标记低可见性帧
  3. visibility_low_run = max_consecutive_true(low_vis_mask)  # 最长连续低可见帧数

触发条件: visibility_low_run >= 5
```

**关键细节**:
- **针对所有追踪点**: 不是 mask 内的点，而是 SpaTracker 在整个画面上追踪的 **756 个网格点**
- **每帧平均**: 先对单帧的所有点求平均，再检测连续低可见帧

**❓ SAM 2 分割的意义是什么？**

虽然 visibility 打分是针对 756 个网格点，但 **SAM 2 的作用至关重要**：

```
完整流水线:

用户点击/框选目标
        ↓
SAM 2 分割 → 生成目标 Mask
        ↓
从 Mask 中采样 256 个查询点 (sample_points_from_mask)
        ↓
SpaTracker 以这 256 个点为初始查询点进行追踪
        ↓
同时 SpaTracker 内部维护 756 个网格点进行全局追踪
```

**SAM 2 的核心作用**:

| 作用 | 说明 |
|:-----|:-----|
| **交互式目标指定** | 用户点哪里，SAM 就分割出对应的物体，无需人工标注 |
| **生成查询点** | 从 mask 中采样 256 个初始点，告诉 SpaTracker "追踪这个区域" |
| **限定追踪范围** | SpaTracker 会聚焦于用户指定的物体，而非盲目追踪整个画面 |

**为什么 visibility 不针对 mask 内的点？**
- SpaTracker 内部的 756 个网格点是**均匀分布在整个画面**的追踪点
- 这些点的 visibility 反映的是**整体追踪质量**（相机运动、遮挡情况）
- 即使查询点在目标上，如果背景网格点大面积不可见，说明追踪器已丢失整体画面感知

**物理意义**:
- 目标被遮挡、离开视野范围、或追踪器丢失目标
- 连续多帧不可见表明追踪质量已严重下降

**阈值参数**:
- `visibility_frame_mean_min = 0.4` (单帧可见性阈值)
- `visibility_low_run_len = 5` (连续帧数阈值)

#### 2. low_confidence (低置信度)

**计算流程**:
```
SpaTracker V2 输出:
  - conf_pred: (T, N) 每个追踪点的置信度分数 [0, 1]

计算步骤:
  1. mean_confidence = mean(conf_pred)  # 全序列所有点的平均置信度

触发条件: mean_confidence < 0.6
```

**物理意义**:
- 特征点纹理模糊、光照变化大、快速运动导致的追踪不稳定
- 置信度低表示追踪器对预测结果不确定

**阈值参数**:
- `confidence_mean_min = 0.6`

#### 3. reprojection_conflict (重投影冲突)

**计算流程**:
```
SpaTracker V2 输出:
  - track3d_pred: (T, N, 3) 3D 轨迹 (相机坐标系)
  - track2d_pred: (T, N, 2) 2D 轨迹 (像素坐标)
  - intrs: (T, 3, 3) 相机内参矩阵

计算步骤:
  1. 投影 3D 到 2D:
     u = fx * (X / Z) + cx
     v = fy * (Y / Z) + cy

  2. 计算重投影误差:
     reproj_err = sqrt((u - track2d_x)^2 + (v - track2d_y)^2)
     # 形状: (T, N) - 每帧每个点的误差

  3. 展平并取 P95 分位数:
     reproj_err_flat = reproj_err.flatten()  # (T*N,) 整段视频的所有误差
     reprojection_error_p95_px = quantile(reproj_err_flat, 0.95)

触发条件: reprojection_error_p95_px > 20.0
```

**关键细节**:
- **P95 含义**: 第 95 百分位数，即 95% 的误差值都小于此数值，只有 5% 的异常值比它大
- **统计范围**: **整段视频** (所有帧 + 所有追踪点)，不是单帧统计
- **为什么要 P95**: 排除个别异常跳变点的影响，反映整体几何一致性

**物理意义**:
- 3D 轨迹与 2D 观测之间的几何不一致
- 可能的成因：深度估计错误、相机姿态漂移、追踪点跳变
- 正常情况下，3D 投影应与 2D 观测完全吻合

**阈值参数**:
- `reprojection_error_p95_max_px = 20.0` (像素)

#### 4. tracking_jump (跟踪跳跃)

**计算流程**:
```
SpaTracker V2 输出:
  - track3d_pred: (T, N, 3) 3D 轨迹 (相机坐标系)
  - c2w_traj: (T, 4, 4) 相机到世界的变换矩阵
  - fps: 视频帧率

计算步骤:
  1. 转换到世界坐标系:
     coords_world = c2w_rotation @ track3d + c2w_translation

  2. 计算速度:
     dt = 1 / fps
     velocity = (coords_world[1:] - coords_world[:-1]) / dt
     speed = norm(velocity, dim=-1)  # 形状: (T-1, N) - 每帧间每个点的速度

  3. 展平并取 P95 分位数:
     speed_flat = speed.flatten()  # ((T-1)*N,) 整段视频的所有速度值
     speed_p95 = quantile(speed_flat, 0.95)

触发条件: speed_p95 > 2.0
```

**关键细节**:
- **P95 含义**: 第 95 百分位数，即 95% 的速度值都小于此数值，反映最极端的正常运动
- **统计范围**: **整段视频** (所有时间间隔 + 所有追踪点)，不是单帧统计
- **为什么要 P95**: 排除正常运动，专门捕捉异常的"瞬移"行为

**物理意义**:
- 物理运动的不连续性，目标在单帧内"瞬移"
- 可能的成因：追踪器丢失目标后重新捕获、特征点混淆、遮挡恢复
- 正常情况下，物体运动应连续平滑

**阈值参数**:
- `velocity_p95_max = 2.0` (米/秒)

### 阈值配置

所有阈值集中定义在 `utils/data_filter.py` 的 `FilterThresholds` 类中：

```python
@dataclass(frozen=True)
class FilterThresholds:
    visibility_frame_mean_min: float = 0.4    # 可见性阈值
    visibility_low_run_len: int = 5           # 连续帧数
    confidence_mean_min: float = 0.6          # 置信度阈值
    reprojection_error_p95_max_px: float = 3.0  # 重投影误差阈值 (实际使用 20.0)
    velocity_p95_max: float = 2.0             # 速度阈值
```

### 筛选结果示例

运行 `scripts/generate_filtered_dataset.py` 后，系统会输出如下报告：

```
总计: 191 个 episode
通过筛选: 137 个 (71.7%)
未通过: 54 个 (28.3%)

未通过指标统计:
  - visibility_failure: 19 个
  - reprojection_conflict: 25 个
  - tracking_jump: 13 个
  - low_confidence: 0 个
```

不合格数据会自动整理到 `data/simple_sorting_0409_rejected/`，保留原始 index 和详细报告。

---

## 🌐 可视化结果查看 (3D Visualization)

使用 TAPIP3D 风格的 3D 可视化工具查看追踪结果，直观检验轨迹质量和相机运动。

### 快速开始

```bash
# 可视化单个 episode
python3 scripts/visualize_tapip3d.py --episode 0

# 可视化所有 batch 中的 episode
python3 scripts/visualize_tapip3d.py --all

# 自定义帧率
python3 scripts/visualize_tapip3d.py --episode 0 --fps 8
```

### 如何用浏览器打开

脚本会生成独立的 HTML 文件，内置所有可视化数据：

```bash
# 方法 1: 直接双击打开（最简单）
# 在文件管理器中找到生成的 HTML 文件，双击即可在默认浏览器中打开

# 方法 2: 命令行打开
# Linux (Ubuntu)
xdg-open visualizations/episode_000000_viz.html

# macOS
open visualizations/episode_000000_viz.html

# Windows (WSL)
explorer.exe visualizations/episode_000000_viz.html

# 方法 3: Python 简易 HTTP 服务器（推荐用于远程服务器）
# 在服务器上启动 HTTP 服务
cd visualizations && python3 -m http.server 8080

# 然后在本地浏览器访问
# http://<服务器IP>:8080/episode_000000_viz.html

# 方法 4: VS Code Live Server 插件
# 右键点击 HTML 文件 → "Open with Live Server"
```

### 可视化界面说明

打开 HTML 后，你会看到 TAPIP3D 风格的交互式 3D 可视化界面：

| 区域 | 内容 |
|:-----|:-----|
| **左上** | RGB 原视频 |
| **右上** | 深度图可视化 |
| **下方** | **3D 场景**：相机轨迹（红线）+ 追踪点轨迹（彩色点云） |

**交互控制**：
- **鼠标左键拖拽**：旋转 3D 视角
- **鼠标右键拖拽**：平移视角
- **滚轮**：缩放
- **播放按钮**：播放/暂停视频
- **进度条**：拖动跳转到指定帧

### 对比合格 vs 不合格数据

```bash
# Episode 0: 合格数据 (所有指标通过)
python3 scripts/visualize_tapip3d.py --episode 0

# Episode 1: 不合格数据 (visibility_failure)
python3 scripts/visualize_tapip3d.py --episode 1
```

**观察重点**：
- ✅ **合格**：轨迹平滑连续，相机运动自然，点云稳定
- ❌ **visibility_failure**：点云突然消失或大面积不可见
- ❌ **tracking_jump**：轨迹出现明显"瞬移"或跳变
- ❌ **reprojection_conflict**：3D 点云与 2D 投影位置明显偏离

### 高级用法

```bash
# 可视化自定义 npz 文件
python3 scripts/visualize_tapip3d.py \
    --input results/auto_batch/episode_000000/result_tapip3d.npz \
    --output my_visualization.html

# 调整分辨率
python3 scripts/visualize_tapip3d.py --episode 0 --width 512 --height 384
```

### 输出文件位置

```
visualizations/
├── episode_000000_viz.html    # 合格示例
├── episode_000001_viz.html    # 不合格示例 (visibility_failure)
├── episode_000003_viz.html    # 不合格示例 (tracking_jump)
└── ...
```

> **注意**：每个 HTML 文件约 15-25MB，包含完整的视频、深度图和轨迹数据，无需外部依赖即可离线查看。

---

## 📂 命令行参数详解 (CLI Arguments)

### 1. `batch_process_auto.py` (主批处理)
| 参数 | 类型 | 默认值 | 说明 |
| :--- | :--- | :--- | :--- |
| `--video_dir` | str | **必填** | 视频数据集的根目录路径 |
| `--anchor_json` | str | `results/anchor_point.json` | 阶段 1 生成的点击坐标文件 |
| `--limit` | int | None | 仅处理前 N 个视频（用于快速测试） |

### 2. `test_first_video.py` (单段诊断)
| 参数 | 类型 | 默认值 | 说明 |
| :--- | :--- | :--- | :--- |
| `--video` | str | episode_000000.mp4 | 指定要诊断的视频路径 |
| `--box` | float | None | 手动输入坐标 `x1 y1 x2 y2` 跳过 Web 交互 |

### 3. `generate_filtered_dataset.py` (推荐：生成完整筛选数据集)
| 参数 | 类型 | 默认值 | 说明 |
| :--- | :--- | :--- | :--- |
| 无 | - | - | 脚本内置路径配置，直接运行 |

此脚本会自动：
- 读取 `results/auto_batch/` 下的打分文件
- 筛选通过四项指标检查的 episode
- 从原始数据复制并重命名视频和 parquet
- 更新 episode_index、frame_index、index 列
- 生成正确的 `info.json`、`episodes.jsonl`

### 4. `full_fix_v4.py` (Meta 数据完整修复)
| 参数 | 类型 | 默认值 | 说明 |
| :--- | :--- | :--- | :--- |
| `--data_dir` | str | **必填** | 筛选后数据的根目录路径 |

此脚本解决筛选后数据的所有元数据不一致问题：
- 对齐 parquet 行数与视频帧数
- 重新编号并同步所有 meta 文件
- **必须在训练前执行**

---

## 🛠️ 训练服务器执行指南 (Training Guide)

在断网环境（如创智服务器 GPU 区）启动训练，请**严格执行**以下步骤：

### 方案 A：直接下载已修复的数据包（推荐）

下载最新生成的完整数据集（`simple_sorting_0409_filtered_v4.tar.gz`）：

```bash
# 1. 下载数据包
cd /path/to/your/workspace
obsutil cp obs://sai.liyl/lihong/simple_sorting_0409_filtered_v4.tar.gz ./

# 2. 解压
tar -xzvf simple_sorting_0409_filtered_v4.tar.gz

# 3. 启动训练
export HF_HUB_OFFLINE=1

python -m lerobot.scripts.train \
    --dataset.repo_id=None \
    --dataset.root=/path/to/your/workspace/simple_sorting_0409_filtered_v4 \
    --policy.type=diffusion \
    --batch_size=256 \
    --policy.use_tactile=false \
    --steps=400000 \
    --wandb.mode="offline"
```

**历史数据包**（旧版，需要额外修复）：
```bash
obsutil cp obs://sai.liyl/lihong/simple_sorting_0409_filtered_fixed.tar ./
```

### 方案 B：本地修复后再训练

如果在训练服务器上有原始筛选数据但需要修复：

```bash
# 1. 运行完整修复脚本
python3 scripts/full_fix_v4.py --data_dir data/simple_sorting_0409_filtered

# 2. 启动训练
export HF_HUB_OFFLINE=1

python -m lerobot.scripts.train \
    --dataset.repo_id=None \
    --dataset.root=data/simple_sorting_0409_filtered \
    --policy.type=diffusion \
    --batch_size=256 \
    --policy.use_tactile=false \
    --steps=400000 \
    --wandb.mode="offline"
```

### ⚠️ 常见问题排查

| 错误类型 | 原因 | 解决方案 |
|---------|------|---------|
| `AssertionError: episode数量不匹配` | meta/info.json 未更新 | 运行 `full_fix_v4.py` |
| `ValueError: timestamps violate tolerance` | episodes.jsonl 的 length 与实际不符 | 运行 `full_fix_v4.py` |
| `RuntimeError: Invalid frame index` | parquet 行数与视频帧数不一致 | 运行 `full_fix_v4.py` |
| `FileNotFoundError: video not found` | 视频文件缺失 | 检查 package_filtered_data.py 是否完整执行 |

---

## 🗑️ 维护说明

### 核心模块
- 核心引擎位于 `core/`，打分标准位于 `utils/data_filter.py`。
- 若需修正重投影误差，请运行 `python3 scripts/recompute_scores.py`。

### 数据集生成流程

**推荐方式（一键生成完整数据集）**：
```bash
# 基于打分文件重新生成完整对齐的数据集
python3 scripts/generate_filtered_dataset.py

# 输出位置
data/simple_sorting_0409_filtered_v4/
├── data/chunk-000/          # 对齐后的 parquet 文件
├── videos/chunk-000/        # 视频文件
└── meta/                    # 正确的 meta 文件
```

**旧版方式（需要手动修复）**：
```bash
python3 scripts/package_filtered_data.py
python3 scripts/full_fix_v4.py --data_dir data/simple_sorting_0409_filtered
```

### 问题说明
旧版 `package_filtered_data.py` 只拷贝了视频文件，未拷贝 parquet，导致数据不一致。`generate_filtered_dataset.py` 会基于打分文件从原始数据重新提取，确保所有索引和帧数完全对齐。

---

## 🌐 启智平台运行指南

在启智平台（纯终端环境下运行，请查看 [README_QIHANG.md](README_QIHANG.md)。

### 快速链接：
- [启智平台纯终端运行指南](README_QIHANG.md) - 包含 OBS 使用注意事项、如何跳过交互式点选等内容。
