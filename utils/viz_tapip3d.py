#!/usr/bin/env python3
"""
使用 result_tapip3d.npz 进行可视化
这个格式是标准的，与 inference.py 输出一致
"""

import os
import sys
import numpy as np
import cv2
import torch
from pathlib import Path

# 添加 SpaTrackerV2 路径以使用 Visualizer
SPA_TRACKER_ROOT = Path("/data/lihong-project/qihang/projects/robot_tracking_integration/external/SpaTrackerV2")
sys.path.insert(0, str(SPA_TRACKER_ROOT))

from models.SpaTrackV2.utils.visualizer import Visualizer


def viz_tapip3d(npz_path, video_path=None, output_path=None):
    """
    使用 result_tapip3d.npz 进行可视化
    
    Args:
        npz_path: result_tapip3d.npz 文件路径
        video_path: 原始视频路径（可选，默认使用 npz 中的 video）
        output_path: 输出路径
    """
    npz_path = Path(npz_path)
    episode_id = npz_path.parent.name
    
    if output_path is None:
        output_path = npz_path.parent / f"{episode_id}_viz.mp4"
    
    print("=" * 60)
    print(f"🎬 TAPiP-3D 可视化: {episode_id}")
    print("=" * 60)
    
    # 加载数据
    print(f"\n📂 加载: {npz_path}")
    data = np.load(npz_path, allow_pickle=True)
    
    print("   可用 keys:", list(data.keys()))
    
    # 提取数据
    video = torch.from_numpy(data['video'])  # (T, 3, H, W)
    tracks = torch.from_numpy(data['coords'])  # (T, N, 3) - 3D坐标
    visibs = torch.from_numpy(data['visibs'])  # (T, N)
    
    T, N = tracks.shape[:2]
    print(f"   视频: {video.shape}")
    print(f"   轨迹: {tracks.shape}")
    print(f"   可见性: {visibs.shape}")
    
    # 如果有 intrinsics 和 extrinsics，可以投影 3D 到 2D
    # 这里简化处理，直接使用 tracks 的前两维作为 2D 投影
    tracks_2d = tracks[..., :2]  # (T, N, 2)
    
    # 归一化到视频尺寸
    H, W = video.shape[2], video.shape[3]
    print(f"   视频尺寸: {W}x{H}")
    
    # 创建 Visualizer
    visualizer = Visualizer(
        save_dir=str(npz_path.parent),
        fps=24,
        mode="rainbow",
        linewidth=2,
        tracks_leave_trace=5,
    )
    
    print(f"\n🎨 渲染...")
    
    # 调用 visualize
    visualizer.visualize(
        video=video[None],  # (1, T, 3, H, W)
        tracks=tracks_2d[None],  # (1, T, N, 2)
        visibility=visibs[None],  # (1, T, N)
        filename=episode_id,
        save_video=True,
    )
    
    print(f"\n✅ 完成: {output_path}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="TAPiP-3D 可视化")
    parser.add_argument("npz_path", type=str, help="result_tapip3d.npz 文件路径")
    parser.add_argument("--video", "-v", type=str, default=None, help="原始视频路径（可选）")
    parser.add_argument("--output", "-o", type=str, default=None, help="输出路径")
    
    args = parser.parse_args()
    
    viz_tapip3d(args.npz_path, args.video, args.output)
