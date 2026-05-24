#!/usr/bin/env python3
"""
正确切分 LeRobot 数据集的长视频（使用 index 作为全局帧索引）
"""

import os
import cv2
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm


def get_video_frame_counts(video_dir):
    video_files = sorted([f for f in os.listdir(video_dir) if f.endswith(".mp4")])
    frame_counts = {}
    for vf in video_files:
        cap = cv2.VideoCapture(os.path.join(video_dir, vf))
        frame_counts[vf] = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
    return frame_counts


def build_frame_mapping(frame_counts):
    mapping = {}
    current_global_frame = 0
    
    video_files = sorted(frame_counts.keys())
    for vf in video_files:
        num_frames = frame_counts[vf]
        for local_frame in range(num_frames):
            mapping[current_global_frame] = (vf, local_frame)
            current_global_frame += 1
    
    return mapping, current_global_frame


def split_videos_correctly(
    data_root="data/510_erase_board_350_lerobot",
    video_key="observation.images.wrist",
    output_dir="data/510_erase_board_350_lerobot/episode_videos",
    output_chunk="chunk-000",
    fps=24.0,
):
    data_path = Path(data_root)
    main_df = pd.read_parquet(data_path / "data/chunk-000/file-000.parquet")
    
    video_dir = data_path / "videos" / video_key / "chunk-000"
    frame_counts = get_video_frame_counts(str(video_dir))
    
    print("=" * 80)
    print("长视频文件信息:")
    print("=" * 80)
    for vf, cnt in sorted(frame_counts.items()):
        print(f"  {vf}: {cnt} frames")
    
    frame_mapping, total_global_frames = build_frame_mapping(frame_counts)
    print(f"\n总全局帧数 (index): {total_global_frames}")
    
    episode_index = main_df["episode_index"].to_numpy()
    global_frame_index = main_df["index"].to_numpy()
    
    unique_eps = np.unique(episode_index)
    ep_global_ranges = {}
    
    for ep in unique_eps:
        mask = episode_index == ep
        ep_global_indices = global_frame_index[mask]
        ep_global_ranges[ep] = (int(ep_global_indices[0]), int(ep_global_indices[-1]))
    
    print(f"\n找到 {len(unique_eps)} 个 episodes")
    print(f"第一个 episode: {ep_global_ranges[unique_eps[0]]}")
    print(f"最后一个 episode: {ep_global_ranges[unique_eps[-1]]}")
    
    output_root = Path(output_dir)
    output_video_dir = output_root / output_chunk / video_key
    os.makedirs(output_video_dir, exist_ok=True)
    
    print("\n开始切分...")
    
    video_caps = {}
    for vf in frame_counts.keys():
        video_caps[vf] = cv2.VideoCapture(str(video_dir / vf))
    
    try:
        for ep in tqdm(unique_eps, desc="切分 episodes"):
            start_global, end_global = ep_global_ranges[ep]
            num_frames_ep = end_global - start_global + 1
            
            first_vf, first_local = frame_mapping[start_global]
            cap = video_caps[first_vf]
            cap.set(cv2.CAP_PROP_POS_FRAMES, first_local)
            ret, frame = cap.read()
            if not ret:
                print(f"  警告: Episode {ep} 无法读取第一帧")
                continue
            
            height, width = frame.shape[:2]
            
            output_path = str(output_video_dir / f"episode_{ep:06d}.mp4")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            
            out.write(frame)
            
            for global_frame in range(start_global + 1, end_global + 1):
                vf, local_frame = frame_mapping[global_frame]
                
                cap = video_caps[vf]
                cap.set(cv2.CAP_PROP_POS_FRAMES, local_frame)
                ret, frame = cap.read()
                
                if ret:
                    out.write(frame)
                else:
                    print(f"  警告: Episode {ep}, Global frame {global_frame} 读取失败")
            
            out.release()
            
    finally:
        for cap in video_caps.values():
            cap.release()
    
    print("\n" + "=" * 80)
    print(f"✅ 切分完成！输出到: {output_video_dir}")
    print("=" * 80)
    
    return str(output_root)


if __name__ == "__main__":
    import argparse
    import datetime

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="data/510_erase_board_350_lerobot")
    parser.add_argument("--video_key", type=str, default="observation.images.wrist")
    parser.add_argument("--output_dir", type=str, default="data/510_erase_board_350_lerobot/episode_videos")
    parser.add_argument("--output_chunk", type=str, default="chunk-000")
    parser.add_argument("--fps", type=float, default=24.0)
    parser.add_argument("--no_backup", action="store_true")
    args = parser.parse_args()

    if os.path.exists(args.output_dir) and not args.no_backup:
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        old_dir = f"{args.output_dir}_old_{ts}"
        os.rename(args.output_dir, old_dir)
        print(f"已备份旧目录到: {old_dir}")

    output_root = split_videos_correctly(
        data_root=args.data_root,
        video_key=args.video_key,
        output_dir=args.output_dir,
        output_chunk=args.output_chunk,
        fps=args.fps,
    )
    
    print("\n验证切分结果:")
    video_files = sorted(list(Path(output_root).glob("**/*.mp4")))
    for vf in video_files[:10]:
        cap = cv2.VideoCapture(str(vf))
        num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        print(f"  {vf.relative_to(Path(output_root))}: {num_frames} frames")
