import numpy as np
import os
import torch
from utils.data_filter import FilterThresholds, compute_quality_scores, print_report

def generate_mock_results(output_dir="results"):
    """
    生成模拟的 3D 追踪数据用于测试筛选逻辑。
    """
    os.makedirs(output_dir, exist_ok=True)
    T = 30  # 30 帧
    N = 256 # 256 个点
    fps = 30.0
    dt = 1.0 / fps

    # 1. 模拟轨迹 (世界坐标)
    # 基础点云 + 线性位移
    base_points = np.random.randn(N, 3) * 0.5
    trajs_3d = []
    for t in range(T):
        # 模拟一点点运动
        move = np.array([0.1 * t * dt, 0, 0])
        trajs_3d.append(base_points + move)
    trajs_3d = np.stack(trajs_3d, axis=0) # (T, N, 3)

    # 2. 模拟置信度和可见性
    # 大部分点是好的，部分点在中间被遮挡
    confs = np.random.uniform(0.7, 0.95, (T, N))
    visibs = np.ones((T, N))
    
    # 制造一些“坏点”来触发 visibility_failure
    # 让前 10 个点在 10-20 帧全不可见
    visibs[10:20, :10] = 0.0
    # 让整体可见度在 15-22 帧大幅下降 (触发序列级失败)
    visibs[15:22, :] = 0.1 

    # 3. 模拟重投影误差
    # 正常误差 1.0 左右，部分帧制造大误差
    reproj_errors = np.random.uniform(0.5, 1.5, (T, N))
    reproj_errors[25:, :] = 5.0 # 触发 reprojection_conflict

    # 4. 模拟动态概率 (如果有)
    dyn_scores = np.random.uniform(0.1, 0.9, (T, N))

    # 5. 模拟相机和内参
    c2w = np.eye(4)[None].repeat(T, axis=0)
    intrs = np.array([[500, 0, 256], [0, 500, 256], [0, 0, 1]])[None].repeat(T, axis=0)

    # 构造 inputs 给 compute_quality_scores
    # 注意: compute_quality_scores 接收 torch.Tensor
    scores = compute_quality_scores(
        c2w_traj=torch.from_numpy(c2w),
        intrs_out=torch.from_numpy(intrs),
        track3d_pred=torch.from_numpy(trajs_3d),
        track2d_pred=torch.from_numpy(trajs_3d[..., :2]), # 简化
        vis_pred=torch.from_numpy(visibs[..., None]),
        conf_pred=torch.from_numpy(confs[..., None]),
        dyn_pred=torch.from_numpy(dyn_scores[..., None]),
        dt=dt,
        thresholds=FilterThresholds()
    )

    print("--- 模拟测试数据生成完成 ---")
    print_report(scores, FilterThresholds())

if __name__ == "__main__":
    generate_mock_results()
