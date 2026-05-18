#!/usr/bin/env python3
"""
可视化 TAPIP3D 格式的追踪结果
基于 SpaTrackerV2 的 process_point_cloud_data 函数
"""

import os
import sys
import json
import struct
import zlib
import base64
import time
import numpy as np
import cv2
from pathlib import Path
from einops import rearrange

# 添加 SpaTrackerV2 路径
SPA_TRACKER_PATH = "/data/lihong-project/qihang/projects/SpaTrackerV2"
sys.path.insert(0, SPA_TRACKER_PATH)

# 模板文件路径
VIZ_TEMPLATE_PATH = os.path.join(SPA_TRACKER_PATH, "_viz", "viz_template.html")
OUTPUT_DIR = "/data/lihong-project/qihang/projects/robot_tracking_integration/visualizations"


def compress_and_write(filename, header, blob):
    """压缩并写入二进制数据"""
    header_bytes = json.dumps(header).encode("utf-8")
    header_len = struct.pack("<I", len(header_bytes))
    with open(filename, "wb") as f:
        f.write(header_len)
        f.write(header_bytes)
        f.write(blob)


def process_point_cloud_data(npz_file, output_html_path, width=256, height=192, fps=4):
    """
    将 TAPIP3D 格式的 npz 文件转换为可视化 HTML
    
    Args:
        npz_file: 输入的 npz 文件路径
        output_html_path: 输出的 HTML 文件路径
        width: 输出视频宽度
        height: 输出视频高度
        fps: 帧率
    """
    fixed_size = (width, height)
    
    data = np.load(npz_file)
    extrinsics = data["extrinsics"]
    intrinsics = data["intrinsics"]
    trajs = data["coords"]
    T, C, H, W = data["video"].shape
    
    # 计算 FOV
    fx = intrinsics[0, 0, 0]
    fy = intrinsics[0, 1, 1]
    fov_y = 2 * np.arctan(H / (2 * fy)) * (180 / np.pi)
    fov_x = 2 * np.arctan(W / (2 * fx)) * (180 / np.pi)
    original_aspect_ratio = (W / fx) / (H / fy)
    
    # 处理 RGB 视频
    rgb_video = (rearrange(data["video"], "T C H W -> T H W C") * 255).astype(np.uint8)
    rgb_video = np.stack([cv2.resize(frame, fixed_size, interpolation=cv2.INTER_AREA)
                          for frame in rgb_video])
    
    # 处理深度图
    depth_video = data["depths"].astype(np.float32)
    if "confs_depth" in data.keys():
        confs = (data["confs_depth"].astype(np.float32) > 0.5).astype(np.float32)
        depth_video = depth_video * confs
    depth_video = np.stack([cv2.resize(frame, fixed_size, interpolation=cv2.INTER_NEAREST)
                            for frame in depth_video])
    
    # 调整内参
    scale_x = fixed_size[0] / W
    scale_y = fixed_size[1] / H
    intrinsics = intrinsics.copy()
    intrinsics[:, 0, :] *= scale_x
    intrinsics[:, 1, :] *= scale_y
    
    # 深度归一化
    min_depth = float(depth_video.min()) * 0.8
    max_depth = float(depth_video.max()) * 1.5
    depth_normalized = (depth_video - min_depth) / (max_depth - min_depth)
    depth_int = (depth_normalized * ((1 << 16) - 1)).astype(np.uint16)
    
    # 打包深度到 RGB
    depths_rgb = np.zeros((T, fixed_size[1], fixed_size[0], 3), dtype=np.uint8)
    depths_rgb[:, :, :, 0] = (depth_int & 0xFF).astype(np.uint8)
    depths_rgb[:, :, :, 1] = ((depth_int >> 8) & 0xFF).astype(np.uint8)
    
    # 标准化相机位姿（相对于第一帧）
    first_frame_inv = np.linalg.inv(extrinsics[0])
    normalized_extrinsics = np.array([first_frame_inv @ ext for ext in extrinsics])
    
    # 标准化轨迹（转换到第一帧坐标系）
    normalized_trajs = np.zeros_like(trajs)
    for t in range(T):
        homogeneous_trajs = np.concatenate([trajs[t], np.ones((trajs.shape[1], 1))], axis=1)
        transformed_trajs = (first_frame_inv @ homogeneous_trajs.T).T
        normalized_trajs[t] = transformed_trajs[:, :3]
    
    # 打包所有数据
    arrays = {
        "rgb_video": rgb_video,
        "depths_rgb": depths_rgb,
        "intrinsics": intrinsics,
        "extrinsics": normalized_extrinsics,
        "inv_extrinsics": np.linalg.inv(normalized_extrinsics),
        "trajectories": normalized_trajs.astype(np.float32),
        "cameraZ": 0.0
    }
    
    # 序列化数据
    header = {}
    blob_parts = []
    offset = 0
    for key, arr in arrays.items():
        arr = np.ascontiguousarray(arr)
        arr_bytes = arr.tobytes()
        header[key] = {
            "dtype": str(arr.dtype),
            "shape": arr.shape,
            "offset": offset,
            "length": len(arr_bytes)
        }
        blob_parts.append(arr_bytes)
        offset += len(arr_bytes)
    
    # 压缩数据
    raw_blob = b"".join(blob_parts)
    compressed_blob = zlib.compress(raw_blob, level=9)
    
    # 元数据
    header["meta"] = {
        "depthRange": [min_depth, max_depth],
        "totalFrames": int(T),
        "resolution": fixed_size,
        "baseFrameRate": fps,
        "numTrajectoryPoints": normalized_trajs.shape[1],
        "fov": float(fov_y),
        "fov_x": float(fov_x),
        "original_aspect_ratio": float(original_aspect_ratio),
        "fixed_aspect_ratio": float(fixed_size[0]/fixed_size[1])
    }
    
    # 创建临时 bin 文件并编码
    temp_bin_path = output_html_path.replace('.html', '_temp.bin')
    compress_and_write(temp_bin_path, header, compressed_blob)
    
    with open(temp_bin_path, "rb") as f:
        encoded_blob = base64.b64encode(f.read()).decode("ascii")
    os.unlink(temp_bin_path)
    
    # 读取模板并嵌入数据
    with open(VIZ_TEMPLATE_PATH) as f:
        html_template = f.read()
    
    html_out = html_template.replace(
        "<head>",
        f"<head>\n<script>window.embeddedBase64 = `{encoded_blob}`;</script>"
    )
    
    with open(output_html_path, 'w') as f:
        f.write(html_out)
    
    return output_html_path


def visualize_episode(episode_idx, fps=4):
    """
    可视化指定 episode 的 TAPIP3D 结果
    
    Args:
        episode_idx: episode 索引号 (如 0, 1, 2...)
        fps: 可视化帧率
    """
    episode_name = f"episode_{episode_idx:06d}"
    npz_path = f"/data/lihong-project/qihang/projects/robot_tracking_integration/results/auto_batch/{episode_name}/result_tapip3d.npz"
    
    if not os.path.exists(npz_path):
        print(f"❌ 文件不存在: {npz_path}")
        return None
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_html = os.path.join(OUTPUT_DIR, f"{episode_name}_viz.html")
    
    print(f"🔄 正在处理 {episode_name}...")
    print(f"   输入: {npz_path}")
    
    process_point_cloud_data(npz_path, output_html, width=256, height=192, fps=fps)
    
    print(f"✅ 完成!")
    print(f"   输出: {output_html}")
    print(f"   用浏览器打开即可查看 3D 可视化")
    
    return output_html


def visualize_all_in_batch(fps=4):
    """可视化 batch 中所有有结果的 episode"""
    batch_dir = "/data/lihong-project/qihang/projects/robot_tracking_integration/results/auto_batch"
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 获取所有 episode
    episodes = []
    for name in sorted(os.listdir(batch_dir)):
        episode_dir = os.path.join(batch_dir, name)
        if os.path.isdir(episode_dir):
            npz_path = os.path.join(episode_dir, "result_tapip3d.npz")
            if os.path.exists(npz_path):
                try:
                    idx = int(name.split("_")[1])
                    episodes.append((idx, name, npz_path))
                except:
                    pass
    
    print(f"📊 找到 {len(episodes)} 个可可视化的 episode")
    print(f"   输出目录: {OUTPUT_DIR}\n")
    
    for i, (idx, name, npz_path) in enumerate(episodes):
        output_html = os.path.join(OUTPUT_DIR, f"{name}_viz.html")
        print(f"[{i+1}/{len(episodes)}] 处理 {name}...", end=" ", flush=True)
        
        try:
            process_point_cloud_data(npz_path, output_html, width=256, height=192, fps=fps)
            print("✅")
        except Exception as e:
            print(f"❌ Error: {e}")
    
    print(f"\n✨ 全部完成! 共生成 {len(episodes)} 个可视化文件")
    print(f"   在 {OUTPUT_DIR} 目录下")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="可视化 TAPIP3D 追踪结果")
    parser.add_argument("--episode", "-e", type=int, default=None,
                        help="可视化指定 episode (如: --episode 0)")
    parser.add_argument("--all", "-a", action="store_true",
                        help="可视化所有 batch 中的 episode")
    parser.add_argument("--fps", "-f", type=int, default=4,
                        help="可视化帧率 (默认: 4)")
    parser.add_argument("--input", "-i", type=str, default=None,
                        help="直接指定 npz 文件路径")
    parser.add_argument("--output", "-o", type=str, default=None,
                        help="输出 HTML 路径 (仅与 --input 配合使用)")
    
    args = parser.parse_args()
    
    if args.input:
        # 直接处理指定文件
        if not os.path.exists(args.input):
            print(f"❌ 文件不存在: {args.input}")
            return
        
        output = args.output or os.path.join(OUTPUT_DIR, "custom_viz.html")
        os.makedirs(os.path.dirname(output), exist_ok=True)
        
        print(f"🔄 处理自定义文件...")
        process_point_cloud_data(args.input, output, width=256, height=192, fps=args.fps)
        print(f"✅ 输出: {output}")
    
    elif args.all:
        visualize_all_in_batch(fps=args.fps)
    
    elif args.episode is not None:
        visualize_episode(args.episode, fps=args.fps)
    
    else:
        # 默认可视化 episode 0
        print("用法示例:")
        print(f"  python3 {sys.argv[0]} --episode 0       # 可视化 episode 0")
        print(f"  python3 {sys.argv[0]} --all              # 可视化所有 episode")
        print(f"  python3 {sys.argv[0]} --input path/to/result.npz  # 可视化指定文件")
        print("\n现在默认可视化 episode 0...")
        visualize_episode(0, fps=args.fps)


if __name__ == "__main__":
    main()
