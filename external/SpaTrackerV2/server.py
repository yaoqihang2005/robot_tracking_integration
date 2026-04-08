import torch
import numpy as np
import zmq
import os
import traceback
import utils3d
import torch.nn.functional as F
from rich import print

def patch_utils3d():
    if not hasattr(utils3d, 'torch'):
        utils3d.torch = utils3d

    def torch_point_map_to_normal_map(point, mask=None):
        """
        全自动维度对齐的 PyTorch 法线计算
        """
        x = point
        
        # --- 核心修复：自动寻找坐标轴 (Size=3 的那一维) ---
        if x.shape[1] != 3:
            if x.shape[-1] == 3: # 情况 A: [B, H, W, 3]
                x = x.permute(0, 3, 1, 2)
            elif x.shape[2] == 3: # 情况 B: [B, H, 3, W] (极少见但可能)
                x = x.permute(0, 2, 1, 3)
            else:
                # 如果前四维都没找到 3，打印出来看看
                print(f"[CRITICAL] 无法在维度 {x.shape} 中找到坐标轴(3)")
                # 强行返回一个保底值，不让服务器崩
                B, _, H, W = x.shape if x.ndim==4 else (x.shape[0], 3, 336, 448)
                return torch.zeros((B, 3, H, W), device=x.device), torch.ones((B, H, W), device=x.device, dtype=torch.bool)

        B, C, H, W = x.shape
        # 此时 x 确定是 (B, 3, H, W)
        
        # 计算差分
        p = F.pad(x, (1, 1, 1, 1), mode='replicate')
        up    = p[:, :, 0:-2, 1:-1] - p[:, :, 1:-1, 1:-1]
        left  = p[:, :, 1:-1, 0:-2] - p[:, :, 1:-1, 1:-1]
        down  = p[:, :, 2:  , 1:-1] - p[:, :, 1:-1, 1:-1]
        right = p[:, :, 1:-1, 2:  ] - p[:, :, 1:-1, 1:-1]

        # 叉乘计算法线
        n1 = torch.cross(up, left, dim=1)
        n2 = torch.cross(left, down, dim=1)
        n3 = torch.cross(down, right, dim=1)
        n4 = torch.cross(right, up, dim=1)

        # 计算完 normals 后
        normals = n1 + n2 + n3 + n4
        norm = torch.linalg.norm(normals, dim=1, keepdim=True) + 1e-12
        normal_map = normals / norm

        if mask is not None:
            # 这里的核心：如果 mask 是 [6, 1, 336, 448]，
            # SpaTrack 期望返回的 normals_mask 是 [6, 336, 448] (去掉 Channel)
            if mask.ndim == 4:
                mask = mask.squeeze(1)
            return normal_map, mask.bool()
        
        return normal_map
    # 注入补丁
    utils3d.torch.points_to_normals = torch_point_map_to_normal_map
    
    # 复用之前写好的 PyTorch 版 depth_edge
    def torch_depth_map_edge(depth, atol=None, rtol=None, kernel_size=3, mask=None):
        if depth.ndim == 3: depth = depth.unsqueeze(1)
        d = depth.clone()
        if mask is not None:
            m = mask.view(d.shape) if mask.shape != d.shape else mask
            d[~m.bool()] = float('nan')
        padding = kernel_size // 2
        max_v = F.max_pool2d(d, kernel_size=kernel_size, stride=1, padding=padding)
        min_v = -F.max_pool2d(-d, kernel_size=kernel_size, stride=1, padding=padding)
        diff = max_v - min_v
        edge = torch.zeros_like(d, dtype=torch.bool)
        if atol is not None: edge |= (diff > atol)
        if rtol is not None:
            rel_diff = diff / torch.where(d == 0, torch.tensor(1e-6, device=d.device), d)
            edge |= (rel_diff.nan_to_num_(0) > rtol)
        return edge.squeeze(1)

    utils3d.torch.depth_edge = torch_depth_map_edge
    print("[bold green]✅ 终极修复：法线与深度边缘已全部重构为 Batch-Safe PyTorch 版本[/bold green]")

patch_utils3d()

# ==========================================
# 2. 模型加载与适配器
# ==========================================
from models.SpaTrackV2.models.predictor import Predictor
from models.SpaTrackV2.models.vggt4track.models.vggt_moe import VGGT4Track

PORT = 5555
curr_dir = os.path.dirname(os.path.abspath(__file__))
FRONT_PATH = os.path.join(curr_dir, "weights/SpatialTrackerV2_Front")
ONLINE_PATH = os.path.join(curr_dir, "weights/SpatialTrackerV2-Online")

def create_server():
    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind(f"tcp://*:{PORT}")
    
    print("[blue]加载基础模型 (VGGT)...[/blue]")
    base_model = VGGT4Track.from_pretrained(FRONT_PATH).eval().to("cuda")

    import torch
    
    
    def specialized_infer_adapter(images_4d):
        num_requested = images_4d.shape[0]
        
        # 1. 严格参考 inference.py 第 69 行：使用 autocast 和除以 255
        with torch.no_grad():
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                predictions = base_model(images_4d[None].cuda() / 255.0)

        # 2. 核心：将 inference.py 中的字段，映射给 SpaTrack.py 要求的键名
        # 没有任何额外逻辑，只是为了解决 Key 冲突
        return {
            # 解决 KeyError: 'mask_prob' (参考 inference.py 60行)
            'mask_prob': predictions['unc_metric'].squeeze(0)[:num_requested],
            
            # 解决 KeyError: 'intrinsics' (参考 inference.py 58行)
            'intrinsics': predictions['intrs'].squeeze(0)[:num_requested],
            
            # 解决 SpaTrack 内部对 depth 的需求 (参考 inference.py 60行 points_map[..., 2])
            'depth': predictions['points_map'][..., 2:3].squeeze(0).permute(0, 3, 1, 2)[:num_requested],
            
            # 解决可能的 KeyError: 'unc'
            'unc': predictions['unc_metric'].squeeze(0)[:num_requested]
        }
    
    torch.cuda.empty_cache()

    base_model.infer = specialized_infer_adapter
    
    print("[blue]加载预测器 (Online)...[/blue]")

    model = Predictor.from_pretrained(ONLINE_PATH)
    model.spatrack.base_model = base_model 
    model.eval().to("cuda")
    
    print(f"[bold green]🚀 服务器就绪，监听 {PORT}[/bold green]")

    while True:
        try:
            message = socket.recv_pyobj()
            video_data = message['video']
            video_tensor = torch.from_numpy(video_data).float().to("cuda")
            if video_tensor.ndim == 5: video_tensor = video_tensor.squeeze(0)
            
            # 这里的 video_tensor 形状通常是 [T, 3, H, W]
            T_count, _, H_orig, W_orig = video_tensor.shape
            
            # 输入图像缩放
            video_input = F.interpolate(video_tensor, size=(336, 448), mode='bilinear')
            
            queries_np = message['queries']
            if torch.is_tensor(queries_np): queries_np = queries_np.cpu().numpy()
            if queries_np.ndim == 3: queries_np = queries_np.squeeze(0)
            
            # 缩放 query 坐标到 336x448 空间
            queries_rescaled = queries_np.copy()
            queries_rescaled[:, 1] *= (336 / H_orig)
            queries_rescaled[:, 2] *= (448 / W_orig)
            
            with torch.no_grad():
                # outputs 包含 (tracks_2d, visible, ..., tracks_3d)
                outputs = model.spatrack.forward_stream(
                    video_input, 
                    queries=queries_rescaled, 
                    T_org=T_count,
                    chunk_size=4, 
                    it_num=1
                )
            
            # outputs[4] 是 tracks_3d, 形状通常为 [1, T, N, 3]
            track3d = outputs[4].squeeze(0).cpu().numpy()
            
            socket.send_pyobj({
                "status": "success", 
                "track3d": track3d
            })
            print(f"[green]✔ 推理完成 (帧数: {T_count}, 点数: {queries_np.shape[0]})[/green]")

        except Exception as e:
            traceback.print_exc()
            socket.send_pyobj({"status": "error", "message": str(e)})

        import torch

        torch.cuda.empty_cache()

if __name__ == "__main__":
    create_server()