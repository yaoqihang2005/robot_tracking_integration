import cv2
import os
from tqdm import tqdm

def video_to_frames(video_path, output_dir):
    """
    将视频文件转换为图片序列
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: 无法打开视频文件 {video_path}")
        return False
        
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"正在转换视频，共 {frame_count} 帧...")
    
    for i in tqdm(range(frame_count)):
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imwrite(os.path.join(output_dir, f"{i:05d}.jpg"), frame)
        
    cap.release()
    print(f"✅ 转换完成，图片保存至 {output_dir}")
    return True

def frames_to_video(frame_dir, output_video_path, fps=30):
    """
    将图片序列重新合成视频
    """
    images = [img for img in os.listdir(frame_dir) if img.endswith(".jpg")]
    images.sort()
    
    if not images:
        print("Error: 目录中没有图片文件")
        return False
        
    frame = cv2.imread(os.path.join(frame_dir, images[0]))
    height, width, layers = frame.shape
    
    video = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    
    print(f"正在合成视频 {output_video_path}...")
    for image in tqdm(images):
        video.write(cv2.imread(os.path.join(frame_dir, image)))
        
    video.release()
    print(f"✅ 视频合成完成: {output_video_path}")
    return True
