import os
import torch
import numpy as np
import cv2
import gc

# 优化显存分配策略，防止碎片化
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from core.sam_helper import SAM2Helper
from core.tracker_helper import TrackerHelper
from utils.sampler import sample_points_from_mask
from utils.video_utils import video_to_frames
from core.config import check_env

def main_pipeline(video_path, box):
    """
    针对 OOM 优化的整合主流水线
    video_path: 视频文件路径
    box: 第一帧的目标 Box [x1, y1, x2, y2]
    """
    # 环境自检
    if not check_env():
        print("❌ 环境检查未通过，请确保依赖库路径正确。")
        return
        
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # --- 阶段 0: 视频预处理与缩放 ---
    frames_dir = "data/temp_frames"
    if not os.path.exists(frames_dir):
        os.makedirs(frames_dir, exist_ok=True)
    
    print(f"正在转换并缩放视频帧: {video_path}")
    cap = cv2.VideoCapture(video_path)
    frame_idx = 0
    frames_list = []
    scale = 1.0
    
    while True:
        ret, frame = cap.read()
        if not ret: break
        
        # 强制降采样：长边限制在 512
        h, w = frame.shape[:2]
        if frame_idx == 0:
            scale = 512.0 / max(h, w)
            new_h, new_w = int(h * scale), int(w * scale)
            print(f"原始分辨率: {w}x{h} -> 缩放后: {new_w}x{new_h} (Scale: {scale:.2f})")
        
        resized_frame = cv2.resize(frame, (int(w * scale), int(h * scale)))
        cv2.imwrite(os.path.join(frames_dir, f"{frame_idx:05d}.jpg"), resized_frame)
        # 转换为 RGB 并存入列表
        frames_list.append(cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB))
        frame_idx += 1
    cap.release()

    # 同步缩放 Box 坐标
    scaled_box = box * scale
    print(f"原始 Box: {box} -> 缩放后 Box: {scaled_box}")

    # --- 阶段 1: SAM 2 掩码传播 ---
    print("\n--- 阶段 1: SAM 2 掩码传播 (显存隔离模式) ---")
    sam_handler = SAM2Helper()
    video_masks = sam_handler.get_video_masks(frames_dir, scaled_box)
    
    # 【关键】立即释放 SAM 2 显存
    print("正在释放 SAM 2 资源...")
    del sam_handler
    gc.collect()
    torch.cuda.empty_cache()

    # --- 阶段 2: 高密度点采样 ---
    print("\n--- 阶段 2: 高密度点采样 (256 点) ---")
    first_frame_mask = video_masks[0]
    # 采样点数进一步下调至 256 以确保显存安全
    query_points = sample_points_from_mask(first_frame_mask, num_samples=256)

    # --- 阶段 3: SpaTracker V2 3D 追踪 ---
    print("\n--- 阶段 3: SpaTracker V2 3D 追踪 ---")
    tracker_handler = TrackerHelper()
    
    video_np = np.stack(frames_list, axis=0) # (T, H, W, C)
    video_tensor = torch.from_numpy(video_np).permute(0, 3, 1, 2).float() # (T, C, H, W)

    # 运行追踪
    outputs = tracker_handler.track_points(
        video_tensor=video_tensor,
        query_points=query_points
    )
    
    # 解包结果 (V2 9项输出)
    # c2w_traj, intrs_out, point_map, unc_metric, track3d_pred, track2d_pred, vis_pred, conf_pred, video_org
    c2w_traj, intrs_out, point_map, unc_metric, track3d_pred, track2d_pred, vis_pred, conf_pred, _ = outputs

    # --- 阶段 4: 保存结果 ---
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    
    # 保存基础结果
    output_path = os.path.join(results_dir, "trajectory_3d.npz")
    print(f"正在保存结果至 {output_path}...")
    np.savez(output_path, 
             trajectories_3d=track3d_pred.cpu().numpy(),
             trajectories_2d=track2d_pred.cpu().numpy(),
             camera_poses=c2w_traj.cpu().numpy(),
             intrinsics=intrs_out.cpu().numpy(),
             visibility=vis_pred.cpu().numpy(),
             resolution_scale=scale)

    # 额外保存 tapip3d 格式结果（用于 3D 可视化）
    tapip_path = os.path.join(results_dir, "result_tapip3d.npz")
    print(f"正在保存 tapip3d 格式结果至 {tapip_path}...")
    
    # 转换坐标到世界空间
    # track3d_pred: (T, N, 6) -> 前 3 维是 (x, y, z)
    # c2w_traj: (T, 4, 4)
    t3d = track3d_pred.cpu()[..., :3] # 只取 xyz
    c2w = c2w_traj.cpu()
    world_coords = (torch.einsum("tij,tnj->tni", c2w[:,:3,:3], t3d) + c2w[:,:3,3][:,None,:]).numpy()
    
    # 获取深度图 (point_map 是 (T, H, W, 3))
    depth_map = point_map[..., 2].cpu().numpy()
    
    np.savez(tapip_path,
             coords=world_coords,
             extrinsics=torch.inverse(c2w).numpy(), # 保存 w2c
             intrinsics=intrs_out.cpu().numpy(),
             depths=depth_map,
             video=video_tensor.cpu().numpy() / 255.0, # (T, 3, H, W)
             visibs=vis_pred.squeeze(-1).cpu().numpy(), # (T, N)
             confs=conf_pred.squeeze(-1).cpu().numpy()) # (T, N)
    
    print(f"\n✅ 全流程运行成功！")
    print(f" - 2D 可视化: 请运行 `python scripts/visualize_2d.py` 查看结果视频")
    print(f" - 3D 可视化: 请运行 `python /data/lihong-project/qihang/projects/SpaTrackerV2/tapip3d_viz.py {tapip_path}`")

if __name__ == "__main__":
    # 调试示例
    VIDEO_PATH = "data/protein.mp4" 
    # 请确保此 Box 在原始分辨率下是正确的
    BOX = np.array([100, 100, 300, 300], dtype=np.float32) 
    
    if os.path.exists(VIDEO_PATH):
        main_pipeline(VIDEO_PATH, BOX)
    else:
        print(f"⚠️ 未找到视频文件: {VIDEO_PATH}，请检查路径。")
