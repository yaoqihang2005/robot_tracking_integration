#!/usr/bin/env python3
"""
基于 inference.py 的逻辑，从 result_tapip3d.npz 文件生成带点云标注的可视化视频。
"""

import os
import sys
import numpy as np
from pathlib import Path
try:
    import torch
except ModuleNotFoundError as e:
    raise ModuleNotFoundError(
        "缺少依赖 torch。请在包含 SpaTrackerV2 依赖的环境中运行，例如：conda run -n spatrack python utils/visualize_from_npz.py <npz_path>"
    ) from e

# 添加 SpaTrackerV2 路径以使用 Visualizer
SPA_TRACKER_ROOT = Path("/data/lihong-project/qihang/projects/robot_tracking_integration/external/SpaTrackerV2")
sys.path.insert(0, str(SPA_TRACKER_ROOT))

from models.SpaTrackV2.utils.visualizer import Visualizer


def project_tracks3d_to_2d(
    tracks_3d: torch.Tensor,
    intrinsics: torch.Tensor,
    extrinsics_w2c: torch.Tensor,
    image_hw: tuple[int, int],
    eps: float = 1e-6,
):
    """
    将 (T,N,3) world 坐标的轨迹，使用每帧的 intrinsics/extrinsics 投影到像素坐标 (T,N,2)。

    约定:
      - intrinsics: (T,3,3)
      - extrinsics_w2c: (T,4,4) world->camera
      - 输出 tracks_2d 为像素坐标 (x, y)
    """
    if tracks_3d.ndim != 3 or tracks_3d.shape[-1] != 3:
        raise ValueError(f"tracks_3d 期望形状 (T,N,3)，实际 {tuple(tracks_3d.shape)}")
    if intrinsics.ndim != 3 or intrinsics.shape[-2:] != (3, 3):
        raise ValueError(f"intrinsics 期望形状 (T,3,3)，实际 {tuple(intrinsics.shape)}")
    if extrinsics_w2c.ndim != 3 or extrinsics_w2c.shape[-2:] != (4, 4):
        raise ValueError(f"extrinsics 期望形状 (T,4,4)，实际 {tuple(extrinsics_w2c.shape)}")

    Tt = tracks_3d.shape[0]
    if intrinsics.shape[0] != Tt or extrinsics_w2c.shape[0] != Tt:
        raise ValueError(
            f"T 维度不一致: tracks={Tt}, intr={intrinsics.shape[0]}, extr={extrinsics_w2c.shape[0]}"
        )

    H, W = image_hw
    R = extrinsics_w2c[:, :3, :3]  # (T,3,3)
    t = extrinsics_w2c[:, :3, 3]  # (T,3)

    Xw = tracks_3d  # (T,N,3)
    Xc = torch.einsum("tij,tnj->tni", R, Xw) + t[:, None, :]  # (T,N,3)

    Z = Xc[..., 2]
    valid_z = torch.isfinite(Z) & (Z > eps)

    x = Xc[..., 0] / torch.clamp(Z, min=eps)
    y = Xc[..., 1] / torch.clamp(Z, min=eps)

    fx = intrinsics[:, 0, 0][:, None]
    fy = intrinsics[:, 1, 1][:, None]
    cx = intrinsics[:, 0, 2][:, None]
    cy = intrinsics[:, 1, 2][:, None]

    u = fx * x + cx
    v = fy * y + cy

    tracks_2d = torch.stack([u, v], dim=-1)  # (T,N,2)

    valid_xy = torch.isfinite(tracks_2d).all(dim=-1)
    valid_in_frame = (tracks_2d[..., 0] >= 0) & (tracks_2d[..., 0] < W) & (tracks_2d[..., 1] >= 0) & (tracks_2d[..., 1] < H)
    valid = valid_z & valid_xy & valid_in_frame

    tracks_2d = tracks_2d.clone()
    tracks_2d[~valid] = 0.0

    return tracks_2d, valid


def visualize_from_npz(npz_path):
    """
    从 result_tapip3d.npz 文件生成带点云标注的可视化视频。

    Args:
        npz_path: result_tapip3d.npz 文件路径
    Returns:
        True 成功，False 失败
    """
    npz_path = Path(npz_path)
    if not npz_path.exists():
        print(f"❌ 文件不存在: {npz_path}")
        return False

    episode_id = npz_path.parent.name

    # Visualizer 实际输出的文件名是 _pred_track.mp4
    if npz_path.name == "trajectory_3d.npz":
        output_stem = f"{episode_id}_traj2d"
    else:
        output_stem = episode_id
    output_file = npz_path.parent / f"{output_stem}_pred_track.mp4"

    print("=" * 60)
    print(f"🎬 TAPiP-3D 可视化: {episode_id} ({npz_path.name})")
    print("=" * 60)

    print(f"\n📂 加载: {npz_path}")
    data = np.load(npz_path, allow_pickle=True)

    if "video" in data:
        video_data_np = data["video"]
    else:
        sibling = npz_path.parent / "result_tapip3d.npz"
        if not sibling.exists():
            raise RuntimeError(
                "输入 npz 不包含 video，并且同目录也找不到 result_tapip3d.npz 来提供视频帧。"
            )
        sibling_data = np.load(sibling, allow_pickle=True)
        if "video" not in sibling_data:
            raise RuntimeError("result_tapip3d.npz 中缺少 video 字段，无法可视化。")
        video_data_np = sibling_data["video"]

    video = torch.from_numpy(video_data_np).float() * 255.0

    # 加载轨迹和可见性数据
    if "trajectories_2d" in data:
        tracks_2d = torch.from_numpy(data["trajectories_2d"]).float()
        if tracks_2d.ndim != 3 or tracks_2d.shape[-1] < 2:
            raise ValueError(f"trajectories_2d 期望形状 (T,N,>=2)，实际 {tuple(tracks_2d.shape)}")
        tracks_2d = tracks_2d[..., :2]

        vis_key = "visibility" if "visibility" in data else "visibs"
        if vis_key not in data:
            raise RuntimeError("npz 中缺少 visibility/visibs 字段，无法可视化。")
        visibs = torch.from_numpy(data[vis_key]).float()
        if visibs.ndim == 3 and visibs.shape[-1] == 1:
            visibs = visibs.squeeze(-1)
    else:
        tracks = torch.from_numpy(data["coords"])  # (T, N, 3)
        visibs = torch.from_numpy(data["visibs"])  # (T, N)
    
    # 撤销对 visibs 形状的 unsqueeze(-1) 操作，使其与 inference.py 保持一致
    # if visibs.ndim == 2:
    #     visibs = visibs.unsqueeze(-1) # 增加一个维度，变为 (T, N, 1)

    if "trajectories_2d" in data:
        T, N = tracks_2d.shape[:2]
    else:
        T, N = tracks.shape[:2]
    H, W = video.shape[2], video.shape[3]

    # 从 npz 中获取 fps，如果不存在则默认为 24
    fps = float(data.get("src_fps", 24.0)) or 24.0

    print(f"   视频帧数: {T}, 视频尺寸: {W}x{H}")
    if "trajectories_2d" in data:
        print(f"   轨迹(2D): {tracks_2d.shape}")
    else:
        print(f"   轨迹(3D): {tracks.shape}")
    print(f"   可见性 (原始): {visibs.shape}")

    if "trajectories_2d" not in data:
        intrs = torch.from_numpy(data["intrinsics"]).float() if "intrinsics" in data else None
        extrs = torch.from_numpy(data["extrinsics"]).float() if "extrinsics" in data else None
        if intrs is None or extrs is None:
            raise RuntimeError("npz 中缺少 intrinsics/extrinsics，无法把 coords(3D) 投影到像素坐标。")

        tracks = tracks.float()
        tracks_2d, valid_proj = project_tracks3d_to_2d(
            tracks_3d=tracks,
            intrinsics=intrs,
            extrinsics_w2c=extrs,
            image_hw=(H, W),
        )

        visibs = visibs.float()
        if visibs.ndim == 3 and visibs.shape[-1] == 1:
            visibs = visibs.squeeze(-1)
        if visibs.ndim != 2:
            raise ValueError(f"visibs 期望形状 (T,N) 或 (T,N,1)，实际 {tuple(visibs.shape)}")
        visibs = visibs * valid_proj.float()
    else:
        if tracks_2d.shape[0] != video.shape[0]:
            raise ValueError(
                f"T 维度不一致: trajectories_2d T={tracks_2d.shape[0]}, video T={video.shape[0]}"
            )
        valid_in_frame = (tracks_2d[..., 0] >= 0) & (tracks_2d[..., 0] < W) & (tracks_2d[..., 1] >= 0) & (tracks_2d[..., 1] < H)
        valid = torch.isfinite(tracks_2d).all(dim=-1) & valid_in_frame
        tracks_2d = tracks_2d.clone()
        tracks_2d[~valid] = 0.0
        if visibs.ndim != 2:
            raise ValueError(f"visibility/visibs 期望形状 (T,N) 或 (T,N,1)，实际 {tuple(visibs.shape)}")
        visibs = visibs * valid.float()

    # 创建 Visualizer，严格参考 inference.py 的参数
    visualizer = Visualizer(
        save_dir=str(npz_path.parent),
        fps=10,  # 严格按照 inference.py 的设置
        mode="rainbow",
        linewidth=2,
        tracks_leave_trace=5,
        grayscale=True,  # 严格按照 inference.py 的设置
        pad_value=0,
    )

    print(f"\n🎨 渲染...")

    # 调用 visualize，传入正确缩放的视频数据
    visualizer.visualize(
        video=video[None],  # (1, T, 3, H, W)
        tracks=tracks_2d[None],  # (1, T, N, 2)
        visibility=visibs[None],  # (1, T, N)
        filename=output_stem,
        save_video=True,
    )

    print(f"✅ 完成: {output_file}")
    return True


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="从 result_tapip3d.npz 生成可视化视频")
    parser.add_argument("npz_path", nargs="?", default=None, help="单个 result_tapip3d.npz 文件路径")
    parser.add_argument("--all", "-a", action="store_true", help="批量处理所有 episode")
    parser.add_argument("--results-dir", "-r", type=str,
                        default="results/auto_batch_510_erase_board_350_lerobot",
                        help="批量处理的结果目录")

    args = parser.parse_args()

    if args.all:
        results_dir = Path(args.results_dir)
        if not results_dir.exists():
            print(f"❌ 结果目录不存在: {results_dir}")
            sys.exit(1)

        print(f"🔍 扫描结果目录: {results_dir}")
        episodes = sorted(
            [d for d in results_dir.iterdir() if d.is_dir() and d.name.startswith("episode_")],
            key=lambda x: x.name
        )
        print(f"   找到 {len(episodes)} 个 episode")

        success_count = 0
        fail_count = 0
        for ep_dir in episodes:
            npz_file = ep_dir / "result_tapip3d.npz"
            result = visualize_from_npz(str(npz_file))
            if result:
                success_count += 1
            else:
                fail_count += 1

        print("\n" + "=" * 60)
        print(f"📊 批量处理完成:")
        print(f"   成功: {success_count}")
        print(f"   失败: {fail_count}")
        if success_count + fail_count > 0:
            print(f"   跳过已存在: {len(episodes) - success_count - fail_count}")
        print("=" * 60)
    else:
        if args.npz_path is None:
            parser.error("请指定 npz_path 或使用 --all 批量处理")
        visualize_from_npz(args.npz_path)
