import sys
import os
import torch
import numpy as np
import torch.nn.functional as F
import gc
import functools

# 1. 动态关联外部路径
from core.config import SPA_ROOT, UTILS3D_ROOT, SAM2_ROOT, SPA_OFFLINE_CHECKPOINT, SPA_FRONT_CHECKPOINT

# 严格按照你截图中能跑通的顺序挂载路径
for path in [SPA_ROOT, UTILS3D_ROOT]:
    if path not in sys.path:
        sys.path.append(path)

# 2. 导入官方库
import utils3d
from models.SpaTrackV2.models.SpaTrack import SpaTrack2
from models.SpaTrackV2.models.predictor import Predictor as SpaTrackerPredictor 
from models.SpaTrackV2.models.vggt4track.models.vggt_moe import VGGT4Track
from models.SpaTrackV2.models.vggt4track.utils.load_fn import preprocess_image

# --- 关键补丁 A：精度自适应 (解决 BFloat16 冲突) ---
_original_torch_inverse = torch.inverse
def _safe_torch_inverse(input, *args, **kwargs):
    if input.dtype in [torch.bfloat16, torch.float16]:
        return _original_torch_inverse(input.float(), *args, **kwargs).to(input.dtype)
    return _original_torch_inverse(input, *args, **kwargs)
torch.inverse = _safe_torch_inverse
if hasattr(torch.linalg, 'inv'):
    torch.linalg.inv = _safe_torch_inverse

_original_torch_quantile = torch.quantile
def _safe_torch_quantile(input, *args, **kwargs):
    if input.dtype in [torch.bfloat16, torch.float16]:
        return _original_torch_quantile(input.float(), *args, **kwargs).to(input.dtype)
    return _original_torch_quantile(input, *args, **kwargs)
torch.quantile = _safe_torch_quantile
print("✅ 已成功注入精度自适应补丁 (inverse, quantile)")

# --- 关键补丁 B：拦截底层模型，强制开启分段特征提取 (解决 OOM) ---
_original_forward_stream = SpaTrack2.forward_stream
@functools.wraps(_original_forward_stream)
def _patched_forward_stream(self, video, queries, *args, **kwargs):
    # 强制启用分段特征提取 (fmaps_chunk_size=4)
    kwargs['fmaps_chunk_size'] = 4
    # 强制关闭 Bundle Adjustment (query_no_BA=True) 以节省显存
    kwargs['query_no_BA'] = True
    return _original_forward_stream(self, video, queries, *args, **kwargs)
SpaTrack2.forward_stream = _patched_forward_stream

# 深度补丁：对 TrackRefiner3D 的 extract_img_feat 也进行拦截
from models.SpaTrackV2.models.tracker3D.TrackRefiner import TrackRefiner3D
_original_extract = TrackRefiner3D.extract_img_feat
@functools.wraps(_original_extract)
def _patched_extract(self, video, fmaps_chunk_size=200):
    return _original_extract(self, video, fmaps_chunk_size=4)
TrackRefiner3D.extract_img_feat = _patched_extract

# --- 关键补丁 D：绕过 pycolmap BA (解决版本冲突与显存) ---
import models.SpaTrackV2.models.tracker3D.spatrack_modules.ba as ba_module
import models.SpaTrackV2.models.tracker3D.TrackRefiner as tr_module

def _patched_ba_pycolmap(world_tracks, intrs, c2w_traj, visb, tracks2d, image_size, **kwargs):
    """
    绕过 pycolmap 的 Bundle Adjustment。
    由于 pycolmap 版本不兼容或显存限制，直接返回输入的原始姿态。
    """
    # 转换输入维度以对齐输出格式
    # world_tracks: (B, 1, K, 3) -> (K, 3)
    # c2w_traj: (B, T, 4, 4) -> (B*T, 4, 4)
    # intrs: (B, T, 3, 3) -> (B*T, 3, 3)
    B, T, _, _ = c2w_traj.shape
    K = world_tracks.shape[2]
    
    # 返回: (c2w_traj_glob, world_tracks_refine, intrinsics)
    return c2w_traj.view(B*T, 4, 4).detach(), world_tracks.view(K, 3).detach(), intrs.view(B*T, 3, 3).detach()

# 全局替换 BA 函数，防止 TrackRefiner 调用本地引用的旧函数
ba_module.ba_pycolmap = _patched_ba_pycolmap
if hasattr(tr_module, 'ba_pycolmap'):
    tr_module.ba_pycolmap = _patched_ba_pycolmap

print("✅ 已成功注入全局 BA 优化补丁 (Skip_BA)")

def patch_utils3d():
    """复刻官方补丁逻辑"""
    if not hasattr(utils3d, 'torch'): utils3d.torch = utils3d
    
    def torch_point_map_to_normal_map(point, mask=None):
        x = point
        if x.shape[1] != 3:
            if x.shape[-1] == 3: x = x.permute(0, 3, 1, 2)
            elif x.shape[2] == 3: x = x.permute(0, 2, 1, 3)
            else: return torch.zeros_like(x), torch.ones_like(x[...,0], dtype=torch.bool)
        p = F.pad(x, (1, 1, 1, 1), mode='replicate')
        n1 = torch.cross(p[:,:,0:-2,1:-1]-p[:,:,1:-1,1:-1], p[:,:,1:-1,0:-2]-p[:,:,1:-1,1:-1], dim=1)
        n2 = torch.cross(p[:,:,1:-1,0:-2]-p[:,:,1:-1,1:-1], p[:,:,2:,1:-1]-p[:,:,1:-1,1:-1], dim=1)
        n3 = torch.cross(p[:,:,2:,1:-1]-p[:,:,1:-1,1:-1], p[:,:,1:-1,2:], dim=1)
        n4 = torch.cross(p[:,:,1:-1,2:]-p[:,:,1:-1,1:-1], p[:,:,0:-2,1:-1]-p[:,:,1:-1,1:-1], dim=1)
        normals = n1 + n2 + n3 + n4
        normal_map = normals / (torch.linalg.norm(normals, dim=1, keepdim=True) + 1e-12)
        return (normal_map, mask.squeeze(1).bool()) if mask is not None else normal_map

    def torch_depth_map_edge(depth, atol=None, rtol=None, kernel_size=3, mask=None):
        orig_shape = depth.shape
        d = depth.unsqueeze(1) if depth.ndim == 3 else depth.clone()
        if mask is not None: d[~mask.view(d.shape).bool()] = float('nan')
        padding = kernel_size // 2
        max_v = F.max_pool2d(d, kernel_size, 1, padding)
        min_v = -F.max_pool2d(-d, kernel_size, 1, padding)
        diff = max_v - min_v
        edge = torch.zeros_like(d, dtype=torch.bool)
        if atol is not None: edge |= (diff > atol)
        if rtol is not None: edge |= ((diff / torch.where(d==0, torch.ones_like(d)*1e-6, d)).nan_to_num_(0) > rtol)
        return edge.view(orig_shape)

    utils3d.torch.points_to_normals = torch_point_map_to_normal_map
    utils3d.torch.depth_edge = torch_depth_map_edge
    print("✅ 已成功注入 utils3d.torch 补丁 (points_to_normals, depth_edge)")

patch_utils3d()

class TrackerHelper:
    def __init__(self, checkpoint=SPA_OFFLINE_CHECKPOINT,
                 front_checkpoint=SPA_FRONT_CHECKPOINT):
        self.device = torch.device("cuda:0")
        print(f"正在初始化 SpaTracker (借鉴官方 inference.py)...")
        self.vggt_model = VGGT4Track.from_pretrained(front_checkpoint)
        self.vggt_model.to(self.device)
        self.vggt_model.eval()
        
        self.model = SpaTrackerPredictor.from_pretrained(checkpoint)
        self.model.to(self.device)
        self.model.eval()
        
        # --- 关键补丁 C：运行时窗口优化 (针对 24GB 显存) ---
        # 强制缩小滑动窗口 S_wind (原值 200 过大，导致 Attention 矩阵 OOM)
        # S_wind=24 与默认 chunk_size 对齐，可显著降低 Transformer 显存峰值
        self.model.S_wind = 24 
        self.model.overlap = 4
        print(f"✅ 已应用运行时窗口优化 (S_wind={self.model.S_wind}, overlap={self.model.overlap})")

    def track_points(self, video_tensor, query_points):
        """
        video_tensor: (T, C, H, W) 原始分辨率 (512p)
        query_points: (N, 3) 原始分辨率下的 (time, x, y)
        """
        with torch.no_grad():
            # 1. 几何预测阶段 (VGGT)
            print("Step 3.1: 运行 VGGT 分段预测...")
            # 这里的 v_input 会被 preprocess_image 改变分辨率 (通常是 width=518, height=14的倍数)
            orig_h, orig_w = video_tensor.shape[2:]
            v_input = preprocess_image(video_tensor.to(self.device)/255.0).to(self.device)
            new_h, new_w = v_input.shape[2:]
            
            # 计算坐标缩放比例
            scale_w = new_w / orig_w
            scale_h = new_h / orig_h
            print(f"分辨率调整: {orig_w}x{orig_h} -> {new_w}x{new_h} (Scale: {scale_w:.2f}, {scale_h:.2f})")
            
            # 缩放查询点坐标
            query_points_scaled = query_points.copy()
            query_points_scaled[:, 1] *= scale_w # x
            query_points_scaled[:, 2] *= scale_h # y

            all_res = []
            for i in range(0, len(v_input), 5):
                chunk = v_input[i:i+5][None] # (1, 5, 3, H, W)
                with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                    all_res.append(self.vggt_model(chunk))
            
            extrs = torch.cat([r["poses_pred"].squeeze(0) for r in all_res], dim=0)
            intrs = torch.cat([r["intrs"].squeeze(0) for r in all_res], dim=0)
            depth = torch.cat([r["points_map"][..., 2] for r in all_res], dim=0) # (T, H, W)
            
            print("正在释放前端显存...")
            del self.vggt_model; gc.collect(); torch.cuda.empty_cache()

            # 2. 追踪阶段 (后端)
            print("Step 3.2: 运行后端追踪 (利用 Autocast + 内部 Patch)...")
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                results = self.model.forward(
                    video=v_input, 
                    queries=query_points_scaled, # 使用缩放后的点
                    depth=depth,
                    intrs=intrs,
                    extrs=extrs,
                    stage=1,
                    support_frame=0,
                    replace_ratio=0.2,
                    fps=1
                )
            
            # 3. 结果还原：将 2D 轨迹还原回 512p 分辨率
            # results[5] 是 track2d_pred (B, T, N, 2)
            results = list(results)
            # 对 2D 轨迹进行缩放还原 (x, y)
            results[5] = results[5].clone()
            results[5][..., 0] /= scale_w
            results[5][..., 1] /= scale_h
            
            # 对 3D 轨迹进行缩放还原 (z 轴也要缩放吗？不，3D 坐标是世界坐标，
            # 但如果 VGGT 输出的深度是相对于缩放后的分辨率，则需要校正)
            # SpaTracker 的 3D 预测通常是基于内参还原的，如果内参也按比例缩放了，3D 应该是对的。
            # 这里先打印一下 3D 轨迹的范围，看看是否合理
            print(f"3D 轨迹范围: min={results[0].min():.2f}, max={results[0].max():.2f}")
            
            return tuple(results)