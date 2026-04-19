import os
import glob
import json
import numpy as np
import argparse
from main_pipeline import main_pipeline

def run_auto_batch():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_dir", type=str, required=True, help="视频所在目录")
    parser.add_argument("--anchor_json", type=str, default="results/anchor_point.json", help="锚点配置文件")
    parser.add_argument("--limit", type=int, default=None, help="处理视频数量限制")
    args = parser.parse_args()

    # 1. 加载锚点 (512p 坐标)
    if not os.path.exists(args.anchor_json):
        print(f"❌ 找不到锚点文件: {args.anchor_json}，请先运行 scripts/find_anchor.py")
        return
        
    with open(args.anchor_json, "r") as f:
        config = json.load(f)
        anchor_512p = config["anchor_point"] # [x, y]
    
    print(f"\n" + "="*60)
    print(f"🚀 启动自动批处理模式")
    print(f"📍 使用锚点 (512p): {anchor_512p}")
    print("="*60)

    # 2. 查找视频
    video_files = glob.glob(os.path.join(args.video_dir, "**/*.mp4"), recursive=True)
    video_files = [v for v in video_files if "images.wrist" in v]
    video_files = sorted(video_files)
    
    if args.limit:
        video_files = video_files[:args.limit]

    print(f"找到 {len(video_files)} 条视频，准备开始全自动处理...")

    for i, v_path in enumerate(video_files):
        video_id = os.path.basename(v_path).replace(".mp4", "")
        output_subdir = os.path.join("results", "auto_batch", video_id)
        
        # 检查是否已处理过
        if os.path.exists(os.path.join(output_subdir, "quality_scores.npz")):
            print(f"[{i+1}/{len(video_files)}] 跳过已完成: {video_id}")
            continue

        print(f"\n" + "-"*40)
        print(f"[{i+1}/{len(video_files)}] 正在处理: {v_path}")
        
        # 还原坐标到原图分辨率
        # 注意：main_pipeline 内部会处理缩放，但它期望输入的是原始分辨率坐标
        # 或者我们可以修改 main_pipeline 让它直接接受 512p 坐标
        # 目前 main_pipeline 内部逻辑是：
        # scale = 512.0 / max(h, w)
        # scaled_points = [[p[0] * scale, p[1] * scale] for p in points]
        # 所以我们需要把 anchor_512p 还原回原始分辨率
        
        import cv2
        cap = cv2.VideoCapture(v_path)
        ret, frame = cap.read()
        cap.release()
        if not ret: continue
        
        h, w = frame.shape[:2]
        scale = 512.0 / max(h, w)
        
        anchor_orig = [anchor_512p[0] / scale, anchor_512p[1] / scale]
        
        try:
            # 使用单点进行 SAM 2 分割
            main_pipeline(
                video_path=v_path, 
                points=[anchor_orig], 
                labels=[1], # 正样本点
                output_dir=output_subdir
            )
            print(f"✅ 自动处理成功: {video_id}")
        except Exception as e:
            print(f"❌ 自动处理失败 {video_id}: {e}")

if __name__ == "__main__":
    # 示例运行命令: 
    # python3 batch_process_auto.py --video_dir data/simple_sorting_0409/videos
    run_auto_batch()
