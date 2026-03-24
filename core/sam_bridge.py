import sys
import os
import torch

# 1. 动态关联外部路径 (根据你的服务器目录)
SAM2_ROOT = "/data/lihong-project/qihang/projects/sam2"
if SAM2_ROOT not in sys.path:
    sys.path.append(SAM2_ROOT)

# 确保能找到 sam2 内部的模块
if os.path.join(SAM2_ROOT, "sam2") not in sys.path:
    sys.path.append(os.path.join(SAM2_ROOT, "sam2"))

try:
    from sam2.build_sam import build_sam2_video_predictor
    print("✅ 成功关联外部 SAM 2 库")
except ImportError as e:
    print(f"❌ 关联 SAM 2 失败，请检查路径: {SAM2_ROOT}")
    print(f"错误详情: {e}")

class SAM2VideoHandler:
    def __init__(self, model_cfg="sam2_hiera_l.yaml", checkpoint="../checkpoints/sam2_hiera_large.pt"):
        """
        初始化 SAM 2 视频预测器
        checkpoint 建议放在你项目的 checkpoints 目录下，或者指向外部路径
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # 自动补全绝对路径防止相对路径失效
        if not os.path.isabs(checkpoint):
            checkpoint = os.path.abspath(os.path.join(os.path.dirname(__file__), checkpoint))
            
        self.predictor = build_sam2_video_predictor(model_cfg, checkpoint, device=self.device)

    def run_propagation(self, video_path, box):
        """
        video_path: 视频帧文件夹或视频文件路径
        box: [x1, y1, x2, y2] 格式的 numpy 数组
        """
        # 初始化状态
        inference_state = self.predictor.init_state(video_path=video_path)
        
        # 添加 Box 交互 (obj_id=1)
        _, out_obj_ids, out_mask_logits = self.predictor.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=0,
            obj_id=1,
            box=box,
        )
        
        # 记录结果的字典
        video_masks = {}
        
        # 开始全视频传播
        for out_frame_idx, out_obj_ids, out_mask_logits in self.predictor.propagate_in_video(inference_state):
            # 将 Logits 转为二值 Mask
            mask = (out_mask_logits[0] > 0.0).cpu().numpy()
            video_masks[out_frame_idx] = mask
            
        return video_masks

