#!/usr/bin/env python3
"""
重新计算评分，不看重投影误差，但记录重投影误差到日志
"""

import os
import glob
import json
import numpy as np
import torch
from pathlib import Path
from utils.data_filter import compute_quality_scores, FilterThresholds

def recompute_and_log_scores(results_root):
    """
    重新计算评分，不看重投影误差，但记录重投影误差到日志
    """
    traj_files = sorted(glob.glob(os.path.join(results_root, "*/trajectory_3d.npz")))
    if not traj_files:
        print("❌ 未找到轨迹数据文件")
        return

    print(f"🚀 准备为 {len(traj_files)} 个 Episode 重新计算评分...")
    
    # 日志文件
    log_file = os.path.join(results_root, "filter_log.jsonl")
    print(f"📝 日志文件: {log_file}")
    
    passed_count = 0
    total_count = 0
    
    with open(log_file, "w") as log_f:
        for f in traj_files:
            episode_dir = os.path.dirname(f)
            episode_name = os.path.basename(episode_dir)
            
            try:
                data = np.load(f, allow_pickle=True)
                
                # 加载原始数据
                c2w_traj = torch.from_numpy(data["camera_poses"])
                intrs_out = torch.from_numpy(data["intrinsics"])
                track3d_pred = torch.from_numpy(data["trajectories_3d"])
                track2d_pred = torch.from_numpy(data["trajectories_2d"])
                vis_pred = torch.from_numpy(data["visibility"])
                conf_pred = torch.from_numpy(data["confidence"])
                dyn_pred = torch.from_numpy(data["dynamic_score"]) if "dynamic_score" in data and data["dynamic_score"].size > 0 else None
                fps = float(data["src_fps"])
                scale = float(data["resolution_scale"])
                
                dt = 1.0 / fps if fps > 0 else 1.0
                
                # --- 关键：把重投影误差阈值设得非常大，让它不会触发筛选
                scores = compute_quality_scores(
                    c2w_traj=c2w_traj,
                    intrs_out=intrs_out,
                    track3d_pred=track3d_pred,
                    track2d_pred=track2d_pred,
                    vis_pred=vis_pred,
                    conf_pred=conf_pred,
                    dyn_pred=dyn_pred,
                    dt=dt,
                    # 把重投影误差阈值设为 10000，基本不会触发
                    thresholds=FilterThresholds(reprojection_error_p95_max_px=10000.0),
                )
                
                # --- 人工修改：把 reprojection_conflict = False
                scores["flags"]["reprojection_conflict"] = False
                
                # --- 保存更新后的评分
                score_path = os.path.join(episode_dir, "quality_scores.npz")
                np.savez(
                    score_path,
                    mean_visibility=scores["mean_visibility"],
                    mean_confidence=scores["mean_confidence"],
                    dynamic_score_mean=scores["dynamic_score_mean"],
                    visibility_frame_mean=scores["visibility_frame_mean"],
                    visibility_low_run=scores["visibility_low_run"],
                    reprojection_error_p95_px=scores["reprojection_error_p95_px"],
                    reprojection_error_max_px=scores["reprojection_error_max_px"],
                    speed_p95=scores["speed_p95"],
                    speed_max=scores["speed_max"],
                    accel_p95=scores["accel_p95"],
                    visibility_failure=scores["flags"]["visibility_failure"],
                    low_confidence=scores["flags"]["low_confidence"],
                    reprojection_conflict=scores["flags"]["reprojection_conflict"],
                    tracking_jump=scores["flags"]["tracking_jump"],
                    src_fps=fps,
                    dt=dt,
                )
                
                # --- 记录到日志
                log_entry = {
                    "episode": episode_name,
                    "mean_visibility": float(scores["mean_visibility"]),
                    "mean_confidence": float(scores["mean_confidence"]),
                    "reprojection_error_p95_px": float(scores["reprojection_error_p95_px"]),
                    "reprojection_error_max_px": float(scores["reprojection_error_max_px"]),
                    "speed_p95": float(scores["speed_p95"]),
                    "visibility_failure": bool(scores["flags"]["visibility_failure"]),
                    "low_confidence": bool(scores["flags"]["low_confidence"]),
                    "tracking_jump": bool(scores["flags"]["tracking_jump"]),
                    "passed": not (
                        scores["flags"]["visibility_failure"] or 
                        scores["flags"]["low_confidence"] or 
                        scores["flags"]["tracking_jump"]
                    )
                }
                log_f.write(json.dumps(log_entry) + "\n")
                
                if log_entry["passed"]:
                    passed_count += 1
                total_count += 1
                
            except Exception as e:
                print(f"⚠️ 处理 {episode_dir} 失败: {e}")
    
    print(f"\n✅ 评分重算完成！")
    print(f"📊 总 episodes: {total_count}")
    print(f"✅ 通过筛选: {passed_count} ({passed_count/total_count*100:.1f}%")
    print(f"📝 日志已保存: {log_file}")

if __name__ == "__main__":
    import sys
    results_dir = "results/auto_batch_510_erase_board_350_lerobot"
    if len(sys.argv) > 1:
        results_dir = sys.argv[1]
    recompute_and_log_scores(results_dir)
