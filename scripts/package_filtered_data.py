import os
import glob
import shutil
import numpy as np
from tqdm import tqdm

def package_filtered_data(results_dir, original_data_root, output_dir):
    """
    将筛选通过的视频及其关联文件打包成原始格式。
    """
    score_files = glob.glob(os.path.join(results_dir, "*/quality_scores.npz"))
    if not score_files:
        print("❌ 未找到评分文件")
        return

    passed_episodes = []
    print("🔍 正在筛选高质量 Episode...")
    
    for f in score_files:
        try:
            data = np.load(f, allow_pickle=True)
            # 筛选条件：四项指标全部通过
            if not (data["visibility_failure"].item() or 
                    data["low_confidence"].item() or 
                    data["reprojection_conflict"].item() or 
                    data["tracking_jump"].item()):
                # 获取 episode ID, e.g., episode_000000
                episode_id = os.path.basename(os.path.dirname(f))
                passed_episodes.append(episode_id)
        except Exception as e:
            print(f"⚠️ 读取 {f} 失败: {e}")

    print(f"✅ 筛选完成：共 {len(passed_episodes)} 个高质量 Episode (总计 {len(score_files)} 个)")

    if not passed_episodes:
        print("❌ 没有符合条件的 Episode，停止打包。")
        return

    # --- 开始打包 ---
    print(f"📦 正在打包至 {output_dir}...")
    
    # 我们需要保持原来的目录结构：
    # videos/chunk-000/observation.images.wrist/episode_XXXXXX.mp4
    # videos/chunk-000/observation.tactiles.right/episode_XXXXXX.mp4
    # ... 以及 meta/ 和 data/ 中的相关文件（如果有的话）

    # 为了简单起见，我们主要打包 videos 目录下的所有关联视角
    for ep_id in tqdm(passed_episodes):
        # 在原始数据中搜索该 episode 的所有 mp4 文件
        search_pattern = os.path.join(original_data_root, "videos/**", f"{ep_id}.mp4")
        original_files = glob.glob(search_pattern, recursive=True)
        
        for src_path in original_files:
            # 计算相对路径，以便在输出目录重建结构
            rel_path = os.path.relpath(src_path, original_data_root)
            dst_path = os.path.join(output_dir, rel_path)
            
            # 创建目标文件夹
            os.makedirs(os.path.dirname(dst_path), exist_ok=True)
            
            # 使用符号链接 (Symlink) 节省空间，或者用 copy
            if os.path.exists(dst_path):
                continue
            
            # 这里建议用 copy2 保持元数据，或者用 symlink
            # 因为你要训练，copy 可能更稳妥
            shutil.copy2(src_path, dst_path)

    print(f"\n" + "="*60)
    print(f"🏁 打包完成！")
    print(f"📁 筛选后的数据集已准备就绪: {output_dir}")
    print(f"📊 包含 Episode 数量: {len(passed_episodes)}")
    print(f"💡 现在你可以直接指向该目录进行训练。")
    print("="*60)

if __name__ == "__main__":
    RESULTS_DIR = "results/auto_batch"
    ORIGINAL_DATA = "data/simple_sorting_0409"
    OUTPUT_DATA = "data/simple_sorting_0409_filtered"
    
    # 路径兼容性处理
    if not os.path.exists(ORIGINAL_DATA):
        ORIGINAL_DATA = "robot_tracking_integration/data/simple_sorting_0409"
        OUTPUT_DATA = "robot_tracking_integration/data/simple_sorting_0409_filtered"

    package_filtered_data(RESULTS_DIR, ORIGINAL_DATA, OUTPUT_DATA)
