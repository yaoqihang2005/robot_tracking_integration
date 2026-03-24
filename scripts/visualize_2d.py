import os
import sys
import numpy as np
import cv2
import torch
import torch.nn.functional as F

# 1. 动态关联外部路径
SPA_ROOT = "/data/lihong-project/qihang/projects/SpaTrackerV2"
if SPA_ROOT not in sys.path: sys.path.append(SPA_ROOT)

from models.SpaTrackV2.utils.visualizer import Visualizer

def visualize_2d():
    print("正在加载结果...")
    results_path = "results/trajectory_3d.npz"
    if not os.path.exists(results_path):
        print(f"错误: 未找到结果文件 {results_path}")
        return

    data = np.load(results_path)
    # track2d_pred: (T, N, 3) -> 需要转为 (B, T, N, 2)
    # 取前两维 (x, y)
    track2d_pred = torch.from_numpy(data['trajectories_2d'])[None, ..., :2].float()
    # vis_pred: (T, N, 1) -> 需要转为 (B, T, N, 1)
    vis_pred = torch.from_numpy(data['visibility'])[None].float()
    scale = float(data['resolution_scale'])
    
    # 加载原始视频帧
    frames_dir = "data/temp_frames"
    if not os.path.exists(frames_dir):
        print(f"错误: 未找到帧目录 {frames_dir}")
        return
        
    frame_files = sorted([f for f in os.listdir(frames_dir) if f.endswith(".jpg")])
    frames = []
    for f in frame_files:
        img = cv2.imread(os.path.join(frames_dir, f))
        frames.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    
    video_np = np.stack(frames, axis=0) # (T, H, W, C)
    video_tensor = torch.from_numpy(video_np).permute(0, 3, 1, 2).float() # (T, C, H, W)
    
    # 实例化可视化器
    vis_dir = "results/viz"
    os.makedirs(vis_dir, exist_ok=True)
    viser = Visualizer(save_dir=vis_dir, grayscale=False, fps=10, tracks_leave_trace=10)
    
    print("正在生成可视化视频...")
    # viser.visualize 期望 video: (B, T, C, H, W), tracks: (B, T, N, 2), visibility: (B, T, N, 1)
    # 注意: tracks 坐标必须与 video 分辨率对齐。
    # 这里的 video 是 512p (已缩放)，而 track2d_pred 在之前被还原到了 512p。
    viser.visualize(
        video=video_tensor[None], 
        tracks=track2d_pred, 
        visibility=vis_pred,
        filename="tracking_result_2d"
    )
    
    print(f"可视化完成！视频已保存至: {vis_dir}/tracking_result_2d.mp4")

if __name__ == "__main__":
    visualize_2d()
