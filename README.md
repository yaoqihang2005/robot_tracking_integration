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
2.  **打包高质量数据**：将通过筛选（无跳变、重投影误差正常）的数据提取到独立目录。
    ```bash
    python3 scripts/package_filtered_data.py
    ```
3.  **Meta 数据修复 (离线训练必做)**：修正 LeRobot 索引，解决 `AssertionError`。
    ```bash
    python3 scripts/fix_meta.py
    ```

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

---

## 🛠️ 训练服务器执行指南 (Training Guide)

在断网环境（如创智服务器 GPU 区）启动训练，请**严格执行**以下步骤：

### 1. 补全元数据并同步
在数据源机器运行 `fix_meta.py` 后，使用 `obsutil` 同步：
```bash
# 在训练机上网区
./obsutil cp obs://sai.liyl/lihong/meta_fixed.tar ./
tar -xvf meta_fixed.tar -C path/to/Data/
```

### 2. 启动训练脚本
```bash
# 1. 屏蔽云端校验
export HF_HUB_OFFLINE=1

# 2. 运行训练 (注意 \ 后面不要有空格或注释)
python -m lerobot.scripts.train \
    --dataset.repo_id=None \
    --dataset.root=Data/handcap2603/simple_sorting_0409_filtered \
    --policy.type=diffusion \
    --batch_size=256 \
    --policy.use_tactile=true \
    --steps=250000 \
    --wandb.mode="offline"
```

---

## 🗑️ 维护说明
- 核心引擎位于 `core/`，打分标准位于 `utils/data_filter.py`。
- 若需修正重投影误差，请运行 `python3 scripts/recompute_scores.py`。
