import sys
import os

# 1. 动态关联外部路径
SPA_ROOT = "/data/lihong-project/qihang/projects/SpaTrackerV2"
UTILS3D_ROOT = "/data/lihong-project/qihang/projects/utils3d-main"

for path in [SPA_ROOT, UTILS3D_ROOT]:
    if path not in sys.path:
        sys.path.append(path)

try:
    # 假设 SpaTracker 的入口类名是 SpatialTracker
    # 这里需要根据你之前部署好的代码实际名称微调
    from models.SpaTrackV2.models.predictor import Predictor as SpaTrackerPredictor 
    print("✅ 成功关联外部 SpaTracker 库")
except ImportError as e:
    print(f"❌ 关联 SpaTracker 失败，请检查路径: {SPA_ROOT}")
    print(f"错误详情: {e}")

class TrackerHandler:
    def __init__(self, checkpoint="../checkpoints/spatracker_model.pth"):
        self.model = SpaTrackerPredictor(checkpoint=checkpoint)

    def track(self, video_tensor, query_points):
        """
        video_tensor: (T, C, H, W)
        query_points: (N, 3) -> (t, y, x)
        """
        return self.model.forward(video_tensor, queries=query_points)
