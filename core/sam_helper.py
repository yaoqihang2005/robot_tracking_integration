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

    def get_mask_from_points(self, video_path, points, labels):
        """
        根据点击的点生成第一帧的 Mask (交互模式)
        points: [[x, y], ...]
        labels: [1, 0, ...] (1: positive, 0: negative)
        """
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            inference_state = self.predictor.init_state(video_path=video_path)
            
            # 转换为 numpy 格式
            points_np = np.array(points, dtype=np.float32)
            labels_np = np.array(labels, dtype=np.int32)

            _, out_obj_ids, out_mask_logits = self.predictor.add_new_points_or_box(
                inference_state=inference_state,
                frame_idx=0,
                obj_id=1,
                points=points_np,
                labels=labels_np,
            )
            
            mask = (out_mask_logits[0] > 0.0).cpu().numpy().squeeze()
            
            # 注意：这里我们重置状态，因为这只是交互预览
            self.predictor.reset_state(inference_state)
            return mask

    def get_video_masks(self, video_path, points=None, labels=None, box=None):
        """
        根据第一帧的 Point 或 Box，生成全视频的 Masks
        video_path: 视频帧文件夹或视频文件路径
        points: [[x, y], ...]
        labels: [1, 0, ...]
        box: [x1, y1, x2, y2]
        """
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            # 初始化状态
            inference_state = self.predictor.init_state(video_path=video_path)
            
            if box is not None:
                # 添加 Box 交互 (obj_id=1)
                _, out_obj_ids, out_mask_logits = self.predictor.add_new_points_or_box(
                    inference_state=inference_state,
                    frame_idx=0,
                    obj_id=1,
                    box=box,
                )
            elif points is not None:
                # 添加 Point 交互
                points_np = np.array(points, dtype=np.float32)
                labels_np = np.array(labels, dtype=np.int32)
                _, out_obj_ids, out_mask_logits = self.predictor.add_new_points_or_box(
                    inference_state=inference_state,
                    frame_idx=0,
                    obj_id=1,
                    points=points_np,
                    labels=labels_np,
                )
            
            video_masks = {}
            # 开始全视频传播
            for out_frame_idx, out_obj_ids, out_mask_logits in self.predictor.propagate_in_video(inference_state):
                mask = (out_mask_logits[0] > 0.0).cpu().numpy().squeeze()
                video_masks[out_frame_idx] = mask
            
            self.predictor.reset_state(inference_state)
            torch.cuda.empty_cache()
            return video_masks