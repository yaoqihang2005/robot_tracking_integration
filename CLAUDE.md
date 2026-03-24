# 项目规范 (Robot Tracking Integration)

## 环境准则
- **Conda 环境**: 必须在 `conda activate sam2` 激活后执行脚本。
- **路径约束**: SAM2 位于 `../SAM2`，SpaTracker 位于 `../SpaTracker`，不要移动它们。
- **禁止操作**: 禁止执行 `rm -rf` 或修改 `/data/` 下非本项目目录的其他文件夹。

## 开发风格
- **语言**: 优先使用中文注释和回复。
- **架构**: 遵循模块化原则，核心逻辑放 `core/`，工具类放 `utils/`。
- **硬件**: 只能使用 GPU 0，不要干扰其他同学的实验。

## Git 准则
- 每次修改代码后，请提醒我 `git add` 并生成 Commit Message，但由我手动执行 `git push`。