import os
import glob
import numpy as np

def summarize_results(results_dir):
    score_files = glob.glob(os.path.join(results_dir, "*/quality_scores.npz"))
    if not score_files:
        print("❌ 未找到评分文件")
        return

    total_videos = len(score_files)
    stats = {
        "mean_visibility": [],
        "mean_confidence": [],
        "reprojection_error_p95": [],
        "speed_p95": [],
        "visibility_failure": 0,
        "low_confidence": 0,
        "reprojection_conflict": 0,
        "tracking_jump": 0
    }

    passed_videos = 0

    for f in score_files:
        try:
            data = np.load(f, allow_pickle=True)
            
            # 基础指标
            stats["mean_visibility"].append(data["mean_visibility"].item())
            stats["mean_confidence"].append(data["mean_confidence"].item())
            stats["reprojection_error_p95"].append(data["reprojection_error_p95_px"].item())
            stats["speed_p95"].append(data["speed_p95"].item())

            # 错误标记
            is_failed = False
            if data["visibility_failure"].item():
                stats["visibility_failure"] += 1
                is_failed = True
            if data["low_confidence"].item():
                stats["low_confidence"] += 1
                is_failed = True
            if data["reprojection_conflict"].item():
                stats["reprojection_conflict"] += 1
                is_failed = True
            if data["tracking_jump"].item():
                stats["tracking_jump"] += 1
                is_failed = True
            
            if not is_failed:
                passed_videos += 1
        except Exception as e:
            print(f"⚠️ 读取 {f} 失败: {e}")

    # 计算平均值
    avg_stats = {k: np.mean(v) for k, v in stats.items() if isinstance(v, list)}

    print("\n" + "="*60)
    print(f"📊 批处理自动化追踪结果统计报告")
    print(f"📁 结果目录: {results_dir}")
    print(f"📹 总处理视频数: {total_videos}")
    print("-" * 30)
    print(f"✅ 通过筛选 (Clean Data): {passed_videos} ({passed_videos/total_videos*100:.1f}%)")
    print(f"❌ 被过滤 (Failed): {total_videos - passed_videos} ({(total_videos-passed_videos)/total_videos*100:.1f}%)")
    print("-" * 30)
    print(f"📈 指标平均值 (Mean Stats):")
    print(f" - 平均置信度 (Confidence): {avg_stats['mean_confidence']:.4f}")
    print(f" - 平均可见度 (Visibility): {avg_stats['mean_visibility']:.4f}")
    print(f" - 重投影误差 (Reproj Error): {avg_stats['reprojection_error_p95']:.2f} px")
    print(f" - 运动速度 (Speed P95): {avg_stats['speed_p95']:.4f}")
    print("-" * 30)
    print(f"🚩 错误分布 (Failure Distribution):")
    print(f" - 遮挡/丢失 (Visibility Failure): {stats['visibility_failure']}")
    print(f" - 低置信度 (Low Confidence): {stats['low_confidence']}")
    print(f" - 几何冲突 (Reproj Conflict): {stats['reprojection_conflict']}")
    print(f" - 轨迹跳变 (Tracking Jump): {stats['tracking_jump']}")
    print("="*60)

if __name__ == "__main__":
    summarize_results("results/auto_batch")
