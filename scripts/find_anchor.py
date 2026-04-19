import cv2
import numpy as np
import glob
import os

def find_optimal_anchor(mask_dir):
    mask_files = glob.glob(os.path.join(mask_dir, "mask_episode_*.png"))
    if not mask_files:
        print("❌ 未找到掩码文件")
        return None

    masks = []
    for f in mask_files:
        mask = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
        masks.append(mask > 127)

    # 计算交集 (所有视频都覆盖的区域)
    intersection = np.all(np.array(masks), axis=0)
    
    if not np.any(intersection):
        print("⚠️ 警告：所有视频之间没有共同的重叠区域！")
        # 如果没有交集，退而求其次寻找重叠最多的区域
        overlap_count = np.sum(np.array(masks), axis=0)
        max_overlap = np.max(overlap_count)
        print(f"ℹ️ 寻找最大重叠区域 (重叠视频数: {max_overlap})")
        intersection = (overlap_count == max_overlap)

    # 寻找连通域并取中心点
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(intersection.astype(np.uint8))
    
    if num_labels <= 1:
        print("❌ 无法找到有效的重叠区域")
        return None

    # 取面积最大的连通域的质心
    largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    anchor_pt = centroids[largest_label] # (x, y)
    
    print(f"✅ 找到最佳锚点: ({anchor_pt[0]:.2f}, {anchor_pt[1]:.2f})")
    return anchor_pt

if __name__ == "__main__":
    MASK_DIR = "results/consistency_test"
    anchor = find_optimal_anchor(MASK_DIR)
    if anchor is not None:
        # 记录到文件供后续使用
        with open("results/anchor_point.json", "w") as f:
            import json
            json.dump({"anchor_point": anchor.tolist()}, f)
