#!/usr/bin/env python3
"""
把不合格的 episode 整理到一个文件夹里
"""

import os
import shutil
import json
import glob
from pathlib import Path

def collect_rejected_episodes(results_root, output_dir="results/rejected_episodes"):
    """
    把不合格的 episode 整理到一个文件夹里
    
    Args:
        results_root: 结果目录
        output_dir: 输出目录
    """
    
    # 1. 读取日志文件
    log_file = os.path.join(results_root, "filter_log.jsonl")
    if not os.path.exists(log_file):
        print(f"❌ 找不到日志文件: {log_file}")
        return
    
    print(f"📂 读取日志文件: {log_file}")
    
    # 2. 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 3. 读取日志
    rejected_episodes = []
    with open(log_file, "r") as f:
        for line in f:
            entry = json.loads(line.strip())
            if not entry["passed"]:
                rejected_episodes.append(entry)
    
    print(f"🚩 找到 {len(rejected_episodes)} 个不合格的 episode")
    
    # 4. 整理到输出目录
    print(f"📦 整理到: {output_dir}")
    
    # 创建 rejected info 文件
    rejected_info = []
    
    for entry in rejected_episodes:
        ep_name = entry["episode"]
        src_dir = os.path.join(results_root, ep_name)
        
        if os.path.exists(src_dir):
            dst_dir = os.path.join(output_dir, ep_name)
            
            # 复制目录
            shutil.copytree(src_dir, dst_dir, dirs_exist_ok=True)
            print(f"  ✅ {ep_name}")
            
            # 记录信息
            rejected_info.append(entry)
    
    # 保存 rejected info
    with open(os.path.join(output_dir, "rejected_info.jsonl"), "w") as f:
        for entry in rejected_info:
            f.write(json.dumps(entry) + "\n")
    
    # 保存 summary
    summary = {
        "total_rejected": len(rejected_episodes),
        "rejected_by": {
            "visibility_failure": sum(1 for e in rejected_info if e["visibility_failure"]),
            "low_confidence": sum(1 for e in rejected_info if e["low_confidence"]),
            "tracking_jump": sum(1 for e in rejected_info if e["tracking_jump"]),
        },
        "rejected_episodes": [e["episode"] for e in rejected_info]
    }
    
    with open(os.path.join(output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    
    print("\n" + "="*60)
    print(f"✅ 整理完成！")
    print(f"📂 输出目录: {output_dir}")
    print(f"🚩 不合格 episode: {len(rejected_episodes)}")
    print(f"   - 遮挡/丢失: {summary['rejected_by']['visibility_failure']}")
    print(f"   - 低置信度: {summary['rejected_by']['low_confidence']}")
    print(f"   - 轨迹跳变: {summary['rejected_by']['tracking_jump']}")
    print("="*60)

if __name__ == "__main__":
    import sys
    results_dir = "results/auto_batch_510_erase_board_350_lerobot"
    output_dir = "results/rejected_episodes_510_erase_board_350"
    if len(sys.argv) > 1:
        results_dir = sys.argv[1]
    if len(sys.argv) > 2:
        output_dir = sys.argv[2]
    collect_rejected_episodes(results_dir, output_dir)
