import os
import cv2
import numpy as np
import torch
from main_pipeline import main_pipeline
from batch_process import get_box_interactively

def visualize_mask(video_path, box, output_dir):
    """
    Visualize SAM 2 mask on the first frame.
    """
    from core.sam_helper import SAM2Helper
    import gc

    print("\n[Debug] Generating SAM 2 mask for visualization...")
    
    # Pre-process first frame to 512p
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    if not ret: return
    
    h, w = frame.shape[:2]
    scale = 512.0 / max(h, w)
    resized_frame = cv2.resize(frame, (int(w * scale), int(h * scale)))
    scaled_box = box * scale
    
    # Create temp frame dir for SAM 2
    temp_dir = "data/debug_frames"
    os.makedirs(temp_dir, exist_ok=True)
    cv2.imwrite(os.path.join(temp_dir, "00000.jpg"), resized_frame)
    
    sam_handler = SAM2Helper()
    video_masks = sam_handler.get_video_masks(temp_dir, scaled_box)
    
    # Save the mask overlay on first frame
    first_mask = video_masks[0]
    overlay = resized_frame.copy()
    overlay[first_mask > 0] = overlay[first_mask > 0] * 0.5 + np.array([0, 255, 0], dtype=np.uint8) * 0.5
    
    # Draw the box
    x1, y1, x2, y2 = scaled_box.astype(int)
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 255), 2)
    
    debug_path = os.path.join(output_dir, "debug_sam_mask.jpg")
    cv2.imwrite(debug_path, overlay)
    print(f"✅ SAM 2 mask visualization saved to {debug_path}")
    
    del sam_handler
    gc.collect()
    torch.cuda.empty_cache()

def test_first_video(video_path=None, box=None, points=None, labels=None):
    if video_path is None:
        video_path = "data/simple_sorting_0409/videos/chunk-000/observation.images.wrist/episode_000000.mp4"
    
    if not os.path.exists(video_path):
        # Try relative to project root
        video_path = os.path.join("robot_tracking_integration", video_path)
        if not os.path.exists(video_path):
            print(f"❌ 找不到视频文件: {video_path}")
            return

    output_dir = "results/test_first_video"
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n" + "="*60)
    print(f"🚀 测试脚本: 追踪第一段视频 (点选模式)")
    print(f"🎥 视频路径: {video_path}")
    
    # Print video metadata
    cap = cv2.VideoCapture(video_path)
    if cap.isOpened():
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"📊 视频元数据: {w}x{h}, {fps:.2f} FPS, {count} 帧")
    cap.release()
    
    print("="*60)

    # 1. 获取交互式选择 (支持点选或框选)
    if box is None and points is None:
        print("\n[Step 1] 请在 Web 界面进行【交互式点选】...")
        # 预先提取第一帧
        cap = cv2.VideoCapture(video_path)
        ret, frame = cap.read()
        cap.release()
        if not ret: return
        
        temp_img = "data/first_frame_temp.jpg"
        cv2.imwrite(temp_img, frame)
        
        # 启动点选服务
        selection = get_box_interactively(video_path)
        if selection is None:
            print("❌ 未完成选择，测试终止。")
            return
        
        if isinstance(selection, dict):
            points = selection['points']
            labels = selection['labels']
            print(f"✅ 已获取 {len(points)} 个交互点")
        else:
            box = selection
            print(f"✅ 已获取初始框: {box}")

    # 2. 运行完整流水线
    print("\n[Step 2] 启动追踪流水线...")
    try:
        main_pipeline(video_path, box=box, points=points, labels=labels, output_dir=output_dir)
        print("\n" + "="*60)
        print(f"✅ 测试完成！结果保存在: {output_dir}")
        print(f"📊 质量评估结果: {os.path.join(output_dir, 'quality_scores.npz')}")
        print("="*60)
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, default=None, help="视频路径")
    parser.add_argument("--box", type=float, nargs=4, default=None, help="初始框 [x1 y1 x2 y2]")
    args = parser.parse_args()
    
    if args.box:
        box = np.array(args.box, dtype=np.float32)
    else:
        box = None
        
    test_first_video(args.video, box)
