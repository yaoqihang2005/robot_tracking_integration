#!/usr/bin/env python3
"""
纯终端环境下手动创建锚点文件
使用方法:
    python3 scripts/create_anchor_manual.py --x 256 --y 256
"""

import argparse
import json
import os


def create_anchor(x: float, y: float, output_path: str = "results/anchor_point.json"):
    """
    创建锚点配置文件
    
    Args:
        x: 512p 分辨率下的 x 坐标
        y: 512p 分辨率下的 y 坐标
        output_path: 输出文件路径
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    config = {
        "anchor_point": [x, y]
    }
    
    with open(output_path, "w") as f:
        json.dump(config, f, indent=2)
    
    print(f"✅ 锚点文件已创建: {output_path}")
    print(f"📍 锚点坐标 (512p): ({x:.2f}, {y:.2f})")
    print("\n现在你可以运行:")
    print("python3 batch_process_auto.py --video_dir data/your_video_dir")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="手动创建锚点配置文件 (纯终端环境)")
    parser.add_argument("--x", type=float, required=True, help="512p 分辨率下的 x 坐标 (0-512)")
    parser.add_argument("--y", type=float, required=True, help="512p 分辨率下的 y 坐标 (0-512)")
    parser.add_argument("--output", type=str, default="results/anchor_point.json", help="输出文件路径")
    
    args = parser.parse_args()
    
    create_anchor(args.x, args.y, args.output)
