#!/usr/bin/env python3
"""
预览视频第一帧并保存为图片，方便确定锚点坐标
使用方法:
    python3 scripts/preview_first_frame.py --video path/to/video.mp4 --output first_frame.jpg
"""

import argparse
import cv2
import os


def preview_first_frame(video_path: str, output_path: str = "first_frame_512p.jpg"):
    """
    提取视频第一帧，缩放到 512p 并保存
    
    Args:
        video_path: 视频文件路径
        output_path: 输出图片路径
    """
    if not os.path.exists(video_path):
        print(f"❌ 找不到视频文件: {video_path}")
        return False
    
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        print(f"❌ 无法读取视频: {video_path}")
        return False
    
    h, w = frame.shape[:2]
    scale = 512.0 / max(h, w)
    new_h, new_w = int(h * scale), int(w * scale)
    resized_frame = cv2.resize(frame, (new_w, new_h))
    
    cv2.imwrite(output_path, resized_frame)
    
    print(f"✅ 第一帧已保存: {output_path}")
    print(f"📊 原始分辨率: {w}x{h}")
    print(f"📊 缩放后 (512p): {new_w}x{new_h}")
    print(f"📊 缩放比例: {scale:.4f}")
    print("\n下一步:")
    print("1. 下载并打开图片查看")
    print("2. 确定要追踪的物体中心坐标 (x, y)")
    print("3. 运行: python3 scripts/create_anchor_manual.py --x <x坐标> --y <y坐标>")
    
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="预览视频第一帧 (缩放到 512p)")
    parser.add_argument("--video", type=str, required=True, help="视频文件路径")
    parser.add_argument("--output", type=str, default="first_frame_512p.jpg", help="输出图片路径")
    
    args = parser.parse_args()
    
    preview_first_frame(args.video, args.output)
