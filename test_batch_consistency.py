import os
import glob
import json
import cv2
import numpy as np
from batch_process import get_box_interactively

def compute_iou(mask1, mask2):
    """计算两个 Mask 的 IoU"""
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    if union == 0: return 0
    return intersection / union

def test_batch_consistency(video_dir, num_videos=10):
    """
    对前 N 段视频进行人工点选，记录 Mask 并分析区域一致性。
    """
    video_files = glob.glob(os.path.join(video_dir, "**/*.mp4"), recursive=True)
    video_files = [v for v in video_files if "images.wrist" in v]
    video_files = sorted(video_files)[:num_videos]
    
    if not video_files:
        print(f"❌ 在 {video_dir} 下未找到足够视频。")
        return

    print(f"\n" + "="*60)
    print(f"🚀 启动区域一致性测试: 处理前 {len(video_files)} 段视频")
    print(f"📊 目的: 分析 SAM 2 生成的 Mask 在不同视频中的重合度")
    print("="*60)

    masks = []
    video_ids = []
    output_dir = "results/consistency_test"
    os.makedirs(output_dir, exist_ok=True)

    for i, v_path in enumerate(video_files):
        video_id = os.path.basename(v_path)
        print(f"\n[{i+1}/{len(video_files)}] 正在点选: {video_id}")
        
        selection = get_box_interactively(v_path)
        if selection is None or 'mask' not in selection:
            print(f"⚠️ 视频 {video_id} 未完成 Mask 生成，跳过。")
            continue
            
        mask = selection['mask'] # 512p binary mask
        masks.append(mask)
        video_ids.append(video_id)
        
        # 保存单个 Mask 预览
        mask_save_path = os.path.join(output_dir, f"mask_{video_id}.png")
        cv2.imwrite(mask_save_path, (mask * 255).astype(np.uint8))
        
        print(f"✅ 已记录 Mask，面积: {mask.sum()} 像素")

    if len(masks) < 2:
        print("❌ 数据不足，无法分析一致性。")
        return

    # --- 1. 计算 IoU 矩阵 ---
    print("\n--- 正在计算 IoU 矩阵 ---")
    iou_matrix = np.zeros((len(masks), len(masks)))
    for i in range(len(masks)):
        for j in range(i, len(masks)):
            iou = compute_iou(masks[i], masks[j])
            iou_matrix[i, j] = iou_matrix[j, i] = iou

    # --- 2. 计算平均 Mask (Heatmap) ---
    print("--- 正在生成热力图 (Overlap Heatmap) ---")
    mean_mask = np.mean(np.array(masks), axis=0) # 0.0 ~ 1.0
    heatmap = (mean_mask * 255).astype(np.uint8)
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    
    heatmap_path = os.path.join(output_dir, "mask_heatmap.png")
    cv2.imwrite(heatmap_path, heatmap_color)

    # --- 3. 输出报告 ---
    print("\n" + "="*60)
    print(f"🏁 一致性分析报告")
    print(f"📁 结果保存至: {output_dir}")
    print(f"🔥 热力图已保存: {heatmap_path}")
    print("-" * 30)
    
    avg_iou = np.mean(iou_matrix[np.triu_indices(len(masks), k=1)])
    print(f"📈 平均成对 IoU (Mean Pairwise IoU): {avg_iou:.4f}")
    
    # 查找一致性最差的对
    if len(masks) >= 2:
        min_iou = 1.0
        worst_pair = (0, 1)
        for i in range(len(masks)):
            for j in range(i+1, len(masks)):
                if iou_matrix[i, j] < min_iou:
                    min_iou = iou_matrix[i, j]
                    worst_pair = (i, j)
        print(f"📉 最差一致性: {min_iou:.4f} (Between {video_ids[worst_pair[0]]} and {video_ids[worst_pair[1]]})")
    
    print("="*60)

if __name__ == "__main__":
    VIDEO_DIR = "data/simple_sorting_0409/videos"
    if not os.path.exists(VIDEO_DIR):
        VIDEO_DIR = "robot_tracking_integration/data/simple_sorting_0409/videos"
    test_batch_consistency(VIDEO_DIR, num_videos=10)
