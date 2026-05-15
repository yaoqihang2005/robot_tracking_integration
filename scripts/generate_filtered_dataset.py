#!/usr/bin/env python3
"""
基于打分文件生成完整的筛选后数据集
- 读取 results/auto_batch/ 下的 quality_scores.npz
- 复制通过筛选的视频和 parquet
- 重新编号并更新索引
- 生成正确的 meta 文件
"""

import os
import glob
import shutil
import json
import numpy as np
import pandas as pd
import cv2
from tqdm import tqdm
from pathlib import Path


def load_passed_episodes(results_dir):
    """读取打分文件，返回通过的 episode 原始编号列表"""
    score_files = glob.glob(os.path.join(results_dir, "*/quality_scores.npz"))
    passed_episodes = []
    
    for sf in sorted(score_files):
        try:
            data = np.load(sf, allow_pickle=True)
            # 筛选条件：四项指标全部通过
            if not (data["visibility_failure"].item() or 
                    data["low_confidence"].item() or 
                    data["reprojection_conflict"].item() or 
                    data["tracking_jump"].item()):
                # 从路径提取 episode 编号
                ep_id = int(os.path.basename(os.path.dirname(sf)).replace("episode_", ""))
                passed_episodes.append(ep_id)
        except Exception as e:
            print(f"⚠️ 读取 {sf} 失败: {e}")
    
    return sorted(passed_episodes)


def get_video_frame_count(video_path):
    """获取视频帧数"""
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return frame_count


def copy_and_fix_parquet(src_parquet, dst_parquet, new_episode_idx, start_global_idx):
    """复制 parquet 并更新 episode_index 和 index 列"""
    df = pd.read_parquet(src_parquet)
    
    # 获取视频帧数
    video_frames = len(df)  # 假设 parquet 行数和原始视频匹配
    
    # 更新 episode_index
    df['episode_index'] = new_episode_idx
    
    # 重新生成 index（全局索引）
    df['index'] = range(start_global_idx, start_global_idx + len(df))
    
    # 确保 timestamp 从 0 开始
    if 'timestamp' in df.columns:
        df['timestamp'] = df['timestamp'] - df['timestamp'].iloc[0]
    
    # 确保 frame_index 从 0 开始
    if 'frame_index' in df.columns:
        df['frame_index'] = range(len(df))
    
    # 保存
    os.makedirs(os.path.dirname(dst_parquet), exist_ok=True)
    df.to_parquet(dst_parquet, index=False)
    
    return len(df)


def copy_videos(src_video_dir, dst_video_dir, old_episode_idx, new_episode_idx):
    """复制所有视角的视频文件"""
    old_name = f"episode_{old_episode_idx:06d}.mp4"
    new_name = f"episode_{new_episode_idx:06d}.mp4"
    
    copied_files = []
    
    # 查找所有视角的视频
    search_pattern = os.path.join(src_video_dir, "**/" + old_name)
    src_files = glob.glob(search_pattern, recursive=True)
    
    for src_path in src_files:
        # 计算相对路径
        rel_path = os.path.relpath(src_path, src_video_dir)
        # 替换文件名
        dst_rel_path = rel_path.replace(old_name, new_name)
        dst_path = os.path.join(dst_video_dir, dst_rel_path)
        
        os.makedirs(os.path.dirname(dst_path), exist_ok=True)
        shutil.copy2(src_path, dst_path)
        copied_files.append(dst_path)
    
    return copied_files


def generate_meta_files(output_dir, episode_info_list, total_frames, original_stats_dict):
    """生成 meta 文件"""
    meta_dir = os.path.join(output_dir, "meta")
    os.makedirs(meta_dir, exist_ok=True)
    
    # 1. info.json - 完全复制原始格式
    info = {
        "codebase_version": "v2.1",
        "robot_type": None,
        "total_episodes": len(episode_info_list),
        "total_frames": total_frames,
        "total_tasks": 1,
        "total_videos": len(episode_info_list) * 3,  # wrist + 2 tactile
        "total_chunks": 1,
        "chunks_size": 1000,
        "fps": 24,
        "splits": {"train": f"0:{len(episode_info_list)}"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": {
            "start_pose": {
                "dtype": "float32",
                "shape": [6],
                "names": ["x", "y", "z", "qx", "qy", "qz"]
            },
            "end_pose": {
                "dtype": "float32",
                "shape": [6],
                "names": ["x", "y", "z", "qx", "qy", "qz"]
            },
            "action": {
                "dtype": "float32",
                "shape": [10],
                "names": ["x", "y", "z", "qx", "qy", "qz", "gripper", "none1", "none2", "none3"]
            },
            "observation.state": {
                "dtype": "float32",
                "shape": [10],
                "names": ["x", "y", "z", "qx", "qy", "qz", "gripper", "none1", "none2", "none3"]
            },
            "observation.images.wrist": {
                "dtype": "video",
                "shape": [480, 640, 3],
                "names": ["height", "width", "channels"],
                "info": {
                    "video.height": 480,
                    "video.width": 640,
                    "video.codec": "h264",
                    "video.pix_fmt": "yuv420p",
                    "video.is_depth_map": False,
                    "video.fps": 24,
                    "video.channels": 3,
                    "has_audio": False
                }
            },
            "observation.tactiles.left": {
                "dtype": "video",
                "shape": [224, 224, 3],
                "names": ["height", "width", "channels"],
                "info": {
                    "video.height": 224,
                    "video.width": 224,
                    "video.codec": "h264",
                    "video.pix_fmt": "yuv420p",
                    "video.is_depth_map": False,
                    "video.fps": 24,
                    "video.channels": 3,
                    "has_audio": False
                }
            },
            "observation.tactiles.right": {
                "dtype": "video",
                "shape": [224, 224, 3],
                "names": ["height", "width", "channels"],
                "info": {
                    "video.height": 224,
                    "video.width": 224,
                    "video.codec": "h264",
                    "video.pix_fmt": "yuv420p",
                    "video.is_depth_map": False,
                    "video.fps": 24,
                    "video.channels": 3,
                    "has_audio": False
                }
            },
            "observation.forces.left": {"dtype": "float32", "shape": [1], "names": None},
            "observation.forces.right": {"dtype": "float32", "shape": [1], "names": None},
            "timestamp": {"dtype": "float32", "shape": [1], "names": None},
            "frame_index": {"dtype": "int64", "shape": [1], "names": None},
            "episode_index": {"dtype": "int64", "shape": [1], "names": None},
            "index": {"dtype": "int64", "shape": [1], "names": None},
            "task_index": {"dtype": "int64", "shape": [1], "names": None}
        }
    }
    
    with open(os.path.join(meta_dir, "info.json"), "w") as f:
        json.dump(info, f, indent=2)
    
    # 2. episodes.jsonl
    with open(os.path.join(meta_dir, "episodes.jsonl"), "w") as f:
        for ep_info in episode_info_list:
            f.write(json.dumps(ep_info) + "\n")
    
    # 3. episodes_stats.jsonl - 从原始数据复制并更新 episode_index
    with open(os.path.join(meta_dir, "episodes_stats.jsonl"), "w") as f:
        for ep_info in episode_info_list:
            new_idx = ep_info["episode_index"]
            old_idx = ep_info["original_index"]
            # 获取原始 stats
            if old_idx in original_stats_dict:
                stats_entry = {
                    "episode_index": new_idx,
                    "stats": original_stats_dict[old_idx]
                }
                f.write(json.dumps(stats_entry) + "\n")
    
    # 4. tasks.jsonl (LeRobot v2.0 需要)
    tasks = [{"task_index": 0, "task": "Sort blocks"}]
    with open(os.path.join(meta_dir, "tasks.jsonl"), "w") as f:
        for task in tasks:
            f.write(json.dumps(task) + "\n")
    
    print(f"✅ Meta 文件已生成: {meta_dir}")


def load_original_stats(original_data_root):
    """加载原始数据集的 episodes_stats"""
    stats_file = os.path.join(original_data_root, "meta/episodes_stats.jsonl")
    stats_dict = {}
    if os.path.exists(stats_file):
        with open(stats_file, "r") as f:
            for line in f:
                entry = json.loads(line.strip())
                ep_idx = entry["episode_index"]
                stats_dict[ep_idx] = entry["stats"]
    return stats_dict


def main():
    # 配置路径
    RESULTS_DIR = "results/auto_batch"
    ORIGINAL_DATA = "data/simple_sorting_0409"
    OUTPUT_DATA = "data/simple_sorting_0409_filtered_v4"
    
    print("=" * 60)
    print("基于打分文件生成筛选数据集")
    print("=" * 60)
    
    # 1. 读取通过的 episode 列表
    print("\n📊 读取打分文件...")
    passed_episodes = load_passed_episodes(RESULTS_DIR)
    print(f"✅ 通过筛选的 episode: {len(passed_episodes)} 个")
    print(f"   原始编号: {passed_episodes[:10]}...")
    
    # 2. 加载原始 stats
    print("\n📊 加载原始 stats...")
    original_stats = load_original_stats(ORIGINAL_DATA)
    print(f"✅ 加载了 {len(original_stats)} 个原始 stats")
    
    # 3. 创建输出目录
    os.makedirs(OUTPUT_DATA, exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DATA, "data/chunk-000"), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DATA, "videos/chunk-000"), exist_ok=True)
    
    # 4. 处理每个通过的 episode
    print("\n📦 复制并处理数据...")
    episode_info_list = []
    global_idx = 0
    total_frames = 0
    
    for new_idx, old_idx in enumerate(tqdm(passed_episodes, desc="处理 episode")):
        # 复制 parquet
        src_parquet = os.path.join(ORIGINAL_DATA, f"data/chunk-000/episode_{old_idx:06d}.parquet")
        dst_parquet = os.path.join(OUTPUT_DATA, f"data/chunk-000/episode_{new_idx:06d}.parquet")
        
        if os.path.exists(src_parquet):
            # 复制并修复 parquet
            frame_count = copy_and_fix_parquet(src_parquet, dst_parquet, new_idx, global_idx)
            
            # 复制视频
            src_video_dir = os.path.join(ORIGINAL_DATA, "videos/chunk-000")
            dst_video_dir = os.path.join(OUTPUT_DATA, "videos/chunk-000")
            copied_videos = copy_videos(src_video_dir, dst_video_dir, old_idx, new_idx)
            
            # 记录 episode 信息
            episode_info = {
                "episode_index": new_idx,
                "original_index": old_idx,
                "length": frame_count,
                "url": f"https://huggingface.co/datasets/lerobot/dataset/resolve/main/episode_{new_idx:06d}.parquet"
            }
            episode_info_list.append(episode_info)
            
            global_idx += frame_count
            total_frames += frame_count
        else:
            print(f"⚠️ 未找到原始 parquet: {src_parquet}")
    
    # 5. 生成 meta 文件
    print("\n📝 生成 meta 文件...")
    generate_meta_files(OUTPUT_DATA, episode_info_list, total_frames, original_stats)
    
    # 6. 验证
    print("\n🔍 验证结果...")
    print(f"   总 episodes: {len(episode_info_list)}")
    print(f"   总 frames: {total_frames}")
    
    print("\n" + "=" * 60)
    print(f"✅ 数据集生成完成: {OUTPUT_DATA}")
    print("=" * 60)


if __name__ == "__main__":
    main()
