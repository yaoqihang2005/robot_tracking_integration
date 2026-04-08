import os
import sys
from pathlib import Path

# 项目根目录 (robot_tracking_integration/)
PROJECT_ROOT = Path(__file__).parent.parent.absolute()

# --- 即插即用逻辑 ---
def get_path(env_key, relative_path, absolute_fallback):
    """
    优先级: 
    1. 环境变量
    2. 项目内部的 external/ 目录 (推荐)
    3. 项目同级目录 (即插即用)
    4. 原始绝对路径 (回退)
    """
    # 1. 检查环境变量
    env_path = os.getenv(env_key)
    if env_path and os.path.exists(env_path):
        return env_path
    
    # 2. 检查项目内部 external/ 目录
    internal_p = str(PROJECT_ROOT / "external" / relative_path)
    if os.path.exists(internal_p):
        return internal_p
    
    # 3. 检查项目同级目录 (即插即用)
    sibling_p = str(PROJECT_ROOT.parent / relative_path)
    if os.path.exists(sibling_p):
        return sibling_p
    
    # 4. 回退到原始绝对路径
    return absolute_fallback

# --- 外部依赖路径 ---
SAM2_ROOT = get_path("SAM2_ROOT", "sam2", "/data/lihong-project/qihang/projects/sam2")
SPA_ROOT = get_path("SPA_ROOT", "SpaTrackerV2", "/data/lihong-project/qihang/projects/SpaTrackerV2")
UTILS3D_ROOT = get_path("UTILS3D_ROOT", "utils3d-main", "/data/lihong-project/qihang/projects/utils3d-main")

# --- 权重路径 ---
# SAM 2 权重通常在 SAM2_ROOT/checkpoints/ 下
SAM2_CHECKPOINT = os.path.join(SAM2_ROOT, "checkpoints/sam2.1_hiera_large.pt")
SAM2_MODEL_CFG = "sam2.1/sam2.1_hiera_l.yaml"

# SpaTracker 权重在 SPA_ROOT/weights/ 下
SPA_OFFLINE_CHECKPOINT = os.path.join(SPA_ROOT, "weights/SpatialTrackerV2-Offline")
SPA_FRONT_CHECKPOINT = os.path.join(SPA_ROOT, "weights/SpatialTrackerV2_Front")

def check_env():
    """打印当前环境关联状态"""
    print("\n" + "="*50)
    print("🚀 机器人追踪集成环境初始化 (即插即用模式)")
    print(f"项目根目录: {PROJECT_ROOT}")
    print("-" * 50)
    
    deps = {
        "SAM 2": SAM2_ROOT,
        "SpaTracker V2": SPA_ROOT,
        "Utils3D": UTILS3D_ROOT
    }
    
    all_ok = True
    for name, path in deps.items():
        status = "✅ 已关联" if os.path.exists(path) else "❌ 未找到"
        if not os.path.exists(path): all_ok = False
        print(f"{name:<15}: {path} [{status}]")
    
    print("="*50 + "\n")
    return all_ok

if __name__ == "__main__":
    check_env()
