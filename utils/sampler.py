import numpy as np

def sample_points_from_mask(mask, num_samples=1024, grid_size=32):
    """
    使用 Grid 采样 + Mask 过滤，确保点在物体表面均匀分布
    mask: (H, W) 二值掩码，True 代表目标区域
    num_samples: 最终需要的点数 (默认 1024)
    grid_size: 初始网格密度
    """
    H, W = mask.shape
    
    # 1. 生成均匀网格点 (y, x)
    y = np.linspace(0, H - 1, grid_size)
    x = np.linspace(0, W - 1, grid_size)
    xv, yv = np.meshgrid(x, y)
    grid_points = np.stack([yv.flatten(), xv.flatten()], axis=1).astype(np.int32)

    # 2. 过滤掉不在 Mask 内的点
    # 确保索引不越界
    grid_points[:, 0] = np.clip(grid_points[:, 0], 0, H - 1)
    grid_points[:, 1] = np.clip(grid_points[:, 1], 0, W - 1)
    mask_points = grid_points[mask[grid_points[:, 0], grid_points[:, 1]] > 0]

    # 3. 兜底策略：如果网格点太稀疏没踩中 Mask，则直接从 Mask 像素中选
    if len(mask_points) < 10:
        y_idx, x_idx = np.where(mask > 0)
        if len(y_idx) == 0:
            print("⚠️ 警告: Mask 为空，无法采样。")
            return np.zeros((num_samples, 3), dtype=np.float32)
        mask_points = np.stack([y_idx, x_idx], axis=1)
        
    # 4. 随机重采样到指定数量
    if len(mask_points) < num_samples:
        replace = True
    else:
        replace = False
        
    idx = np.random.choice(len(mask_points), num_samples, replace=replace)
    sampled_points = mask_points[idx]

    # 5. 格式化为 SpaTracker 要求的 (time_idx, x, y)
    # SpaTracker V2 内部通常处理 (B, N, 2) 的 (x, y) 坐标，这里返回 (N, 3) 方便主流程管理
    final_queries = np.zeros((num_samples, 3), dtype=np.float32)
    final_queries[:, 0] = 0  # 初始帧 time_idx = 0
    final_queries[:, 1] = sampled_points[:, 1] # x 坐标
    final_queries[:, 2] = sampled_points[:, 0] # y 坐标
    
    return final_queries
