import sys
import os
import torch
import numpy as np
from hydra import initialize_config_dir
from hydra.core.global_hydra import GlobalHydra

from core.config import SAM2_ROOT, SAM2_CHECKPOINT, SAM2_MODEL_CFG

# 1. 动态关联外部路径
if SAM2_ROOT not in sys.path:
    sys.path.append(SAM2_ROOT)

# 确保能找到 sam2 内部的模块
if os.path.join(SAM2_ROOT, "sam2") not in sys.path:
    sys.path.append(os.path.join(SAM2_ROOT, "sam2"))

from sam2.build_sam import build_sam2_video_predictor
print(f"✅ 成功关联外部 SAM 2 库: {SAM2_ROOT}")

class SAM2Helper:
    def __init__(self, model_cfg=SAM2_MODEL_CFG, 
                 checkpoint=SAM2_CHECKPOINT):
        """
        初始化 SAM 2 视频预测器
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        print(f"正在加载 SAM 2 模型: {checkpoint}")

        # 修复：当从外部项目调用时，Hydra 不知道 SAM 2 的配置路径。
        # 我们需要手动初始化，告诉它配置文件的位置。
        # 同时，为了防止二次初始化报错，先清空之前的 Hydra 实例。
        GlobalHydra.instance().clear()
        config_dir = os.path.abspath(os.path.join(SAM2_ROOT, "sam2", "configs"))
        with initialize_config_dir(version_base=None, config_dir=config_dir):
            # 现在 Hydra 知道去哪里找配置文件了
            self.predictor = build_sam2_video_predictor(model_cfg, checkpoint, device=self.device)

    def get_video_masks(self, video_path, box):
        """
        根据第一帧的 Box，生成全视频的 Masks
        video_path: 视频帧文件夹或视频文件路径
        box: [x1, y1, x2, y2] 格式的 numpy 数组
        """
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            # 初始化状态
            inference_state = self.predictor.init_state(video_path=video_path)
            
            # 添加 Box 交互 (obj_id=1)
            # SAM 2 的 add_new_points_or_box API
            _, out_obj_ids, out_mask_logits = self.predictor.add_new_points_or_box(
                inference_state=inference_state,
                frame_idx=0,
                obj_id=1,
                box=box,
            )
            
            video_masks = {}
            # 开始全视频传播
            for out_frame_idx, out_obj_ids, out_mask_logits in self.predictor.propagate_in_video(inference_state):
                # 将 Logits 转为二值 Mask (B, 1, H, W) -> (H, W)
                # SAM 2 内部阈值通常为 0.0
                mask = (out_mask_logits[0] > 0.0).cpu().numpy().squeeze()
                video_masks[out_frame_idx] = mask
            
            # 清理状态以释放显存
            self.predictor.reset_state(inference_state)
            torch.cuda.empty_cache()
            return video_masks