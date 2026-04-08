import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def visualize_trace_perfect(npz_path, save_path='trajectory_cloud.png'):
    data = np.load(npz_path)
    coords = data['coords']
    T, N, _ = coords.shape
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # --- 修改点 1: 专门绘制起止位置的“半透明空间痕迹” ---
    # 绘制第一帧 (起始位置)
    ax.scatter(coords[0, :, 0], coords[0, :, 1], coords[0, :, 2], 
               color='blue', s=2, alpha=0.2, label='Start Pose')
    
    # 绘制最后一帧 (终止位置)
    ax.scatter(coords[-1, :, 0], coords[-1, :, 1], coords[-1, :, 2], 
               color='red', s=2, alpha=0.2, label='End Pose')
    
    # (可选) 如果中间过程也想要一点淡淡的痕迹，可以保持低频采样
    # step = T // 10
    # for t in range(step, T-step, step):
    #     ax.scatter(coords[t, :, 0], coords[t, :, 1], coords[t, :, 2], 
    #                color='gray', s=1, alpha=0.03)

    # --- 修改点 2: 确保轨迹处理成“单条线” ---
    # 计算质心轨迹
    center_line = np.mean(coords, axis=1) 
    
    # 用黑色的实线把这个轨迹串起来
    ax.plot(center_line[:, 0], center_line[:, 1], center_line[:, 2], 
            color='black', linewidth=2, label='Movement Trajectory', zorder=10)

    # --- 修改点 3: 视角设置 (相机坐标系通常 Y 是向下的) ---
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.view_init(elev=20, azim=45) # 稍微侧视，能更好地看出 3D 均匀度
    
    ax.legend()
    plt.savefig(save_path, dpi=300)
    print(f"✅ 符合学长要求的图已保存至: {save_path}")

if __name__ == "__main__":
    # 填入你跑出来的 result.npz 路径
    visualize_trace_perfect('temp_local/session_f43df004/results/result.npz')