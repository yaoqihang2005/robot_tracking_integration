#!/usr/bin/env python3
"""
完整修复筛选后数据的 Meta 信息
解决 parquet 行数与视频帧数不匹配的问题

Usage:
    python3 scripts/full_fix_v4.py --data_dir data/simple_sorting_0409_filtered
"""

import argparse
import json
import os
import cv2
import pandas as pd
from pathlib import Path


def get_video_frame_count(video_path):
    """获取视频帧数"""
    cap = cv2.VideoCapture(str(video_path))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return frame_count


def fix_parquet_files(data_dir):
    """修复 parquet 文件行数，使其与视频帧数匹配"""
    data_dir = Path(data_dir)
    video_dir = data_dir / "videos/chunk-000/observation.images.wrist"
    parquet_dir = data_dir / "data/chunk-000"
    
    print("=== Step 1: 修复 parquet 行数 ===")
    fixed_count = 0
    
    for i in range(1000):  # 假设最多1000个episode
        ep_str = f"episode_{i:06d}"
        video_path = video_dir / f"{ep_str}.mp4"
        parquet_path = parquet_dir / f"{ep_str}.parquet"
        
        if not video_path.exists():
            break
            
        if not parquet_path.exists():
            print(f"  ⚠️  {ep_str}: parquet 不存在")
            continue
        
        frame_count = get_video_frame_count(video_path)
        df = pd.read_parquet(parquet_path)
        parquet_rows = len(df)
        
        if frame_count != parquet_rows:
            if frame_count > parquet_rows:
                # 需要补充行（重复最后一行）
                last_row = df.iloc[-1:].copy()
                repeat_count = frame_count - parquet_rows
                df = pd.concat([df] + [last_row] * repeat_count, ignore_index=True)
                action = "补充"
            else:
                # 需要截断
                df = df.iloc[:frame_count].copy()
                action = "截断"
            
            # 重置索引
            df['frame_index'] = range(len(df))
            df.to_parquet(parquet_path, index=False)
            fixed_count += 1
            print(f"  {ep_str}: {action} {parquet_rows} -> {frame_count} 行")
    
    print(f"✅ 修复了 {fixed_count} 个 parquet 文件\n")
    return i


def fix_episodes_jsonl(data_dir, num_episodes):
    """修复 episodes.jsonl"""
    data_dir = Path(data_dir)
    meta_path = data_dir / "meta/episodes.jsonl"
    video_dir = data_dir / "videos/chunk-000/observation.images.wrist"
    
    print("=== Step 2: 修复 episodes.jsonl ===")
    
    new_lines = []
    total_frames = 0
    
    for i in range(num_episodes):
        ep_str = f"episode_{i:06d}"
        video_path = video_dir / f"{ep_str}.mp4"
        
        if not video_path.exists():
            break
            
        frame_count = get_video_frame_count(video_path)
        
        data = {
            "episode_index": i,
            "tasks": ["peg_in_hole"],
            "length": frame_count
        }
        new_lines.append(json.dumps(data) + '\n')
        total_frames += frame_count
    
    with open(meta_path, 'w') as f:
        f.writelines(new_lines)
    
    print(f"✅ episodes.jsonl: {len(new_lines)} 个 episodes, {total_frames} 总帧数\n")
    return total_frames


def fix_info_json(data_dir, num_episodes, total_frames):
    """修复 info.json"""
    data_dir = Path(data_dir)
    info_path = data_dir / "meta/info.json"
    
    print("=== Step 3: 修复 info.json ===")
    
    with open(info_path, 'r') as f:
        info = json.load(f)
    
    info['total_episodes'] = num_episodes
    info['total_frames'] = total_frames
    info['total_videos'] = num_episodes * 3  # 3个视角
    info['splits']['train'] = f"0:{num_episodes}"
    
    with open(info_path, 'w') as f:
        json.dump(info, f, indent=4)
    
    print(f"✅ info.json: total_episodes={num_episodes}, total_frames={total_frames}\n")


def fix_episodes_stats_jsonl(data_dir, num_episodes):
    """修复或创建 episodes_stats.jsonl"""
    data_dir = Path(data_dir)
    stats_path = data_dir / "meta/episodes_stats.jsonl"
    
    print("=== Step 4: 修复 episodes_stats.jsonl ===")
    
    # 如果存在就删除（stats可以后续重新计算）
    if stats_path.exists():
        stats_path.unlink()
        print("  已删除旧的 episodes_stats.jsonl")
    
    # 创建空的 stats 文件（训练时会自动计算）
    with open(stats_path, 'w') as f:
        pass
    
    print("✅ episodes_stats.jsonl 已清空（训练时自动计算）\n")


def main():
    parser = argparse.ArgumentParser(description='修复筛选后数据的 Meta 信息')
    parser.add_argument('--data_dir', type=str, required=True, 
                        help='筛选后数据的根目录，如 data/simple_sorting_0409_filtered')
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    
    if not data_dir.exists():
        print(f"❌ 目录不存在: {data_dir}")
        return
    
    print(f"开始修复: {data_dir}\n")
    print("=" * 60)
    
    # Step 1: 修复 parquet
    num_episodes = fix_parquet_files(data_dir)
    
    # Step 2: 修复 episodes.jsonl
    total_frames = fix_episodes_jsonl(data_dir, num_episodes)
    
    # Step 3: 修复 info.json
    fix_info_json(data_dir, num_episodes, total_frames)
    
    # Step 4: 修复 episodes_stats.jsonl
    fix_episodes_stats_jsonl(data_dir, num_episodes)
    
    print("=" * 60)
    print("🎉 所有修复完成！")
    print(f"📊 共 {num_episodes} 个 episodes, {total_frames} 帧")
    print("💡 现在可以直接用于训练了")


if __name__ == "__main__":
    main()
