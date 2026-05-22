#!/usr/bin/env python3

import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.sam_helper import SAM2Helper
from utils.sampler import sample_points_from_mask


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _load_anchor_512p(anchor_json: Path) -> np.ndarray:
    with anchor_json.open("r") as f:
        cfg = json.load(f)
    anchor = np.array(cfg["anchor_point"], dtype=np.float32)
    if anchor.shape != (2,):
        raise ValueError(f"anchor_point 形状不正确: {anchor.shape}")
    return anchor


def _read_first_frame(video_path: Path) -> np.ndarray:
    cap = cv2.VideoCapture(str(video_path))
    ok, frame_bgr = cap.read()
    cap.release()
    if not ok or frame_bgr is None:
        raise RuntimeError(f"读取视频失败: {video_path}")
    return frame_bgr


def _write_resized_frames(video_path: Path, frames_dir: Path, target_long_side: int) -> tuple[float, int]:
    _ensure_dir(frames_dir)
    for p in frames_dir.glob("*.jpg"):
        p.unlink()

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    frame_idx = 0
    scale = 1.0

    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break
        h, w = frame_bgr.shape[:2]
        if frame_idx == 0:
            scale = float(target_long_side) / float(max(h, w))
        resized = cv2.resize(frame_bgr, (int(w * scale), int(h * scale)))
        cv2.imwrite(str(frames_dir / f"{frame_idx:05d}.jpg"), resized)
        frame_idx += 1

    cap.release()
    return fps, frame_idx


def _overlay_mask(frame_bgr: np.ndarray, mask: np.ndarray, color_bgr=(0, 255, 0), alpha: float = 0.5) -> np.ndarray:
    if mask.dtype != np.uint8:
        mask_u8 = (mask > 0).astype(np.uint8) * 255
    else:
        mask_u8 = mask
    mask_u8 = cv2.resize(mask_u8, (frame_bgr.shape[1], frame_bgr.shape[0]), interpolation=cv2.INTER_NEAREST)
    overlay = frame_bgr.copy()
    colored = np.zeros_like(frame_bgr, dtype=np.uint8)
    colored[:, :] = np.array(color_bgr, dtype=np.uint8)
    sel = mask_u8 > 0
    overlay[sel] = (overlay[sel].astype(np.float32) * (1 - alpha) + colored[sel].astype(np.float32) * alpha).astype(
        np.uint8
    )
    return overlay


def _draw_points(frame_bgr: np.ndarray, points_xy: np.ndarray, color_bgr=(0, 0, 255), radius: int = 2) -> np.ndarray:
    out = frame_bgr.copy()
    for x, y in points_xy:
        cv2.circle(out, (int(round(x)), int(round(y))), radius, color_bgr, -1)
    return out


def viz_step1_sam(video_path: str, anchor_json: str, out_dir: str, target_long_side: int = 256) -> None:
    video_path_p = Path(video_path)
    anchor_json_p = Path(anchor_json)
    out_dir_p = Path(out_dir)
    _ensure_dir(out_dir_p)

    anchor_512p = _load_anchor_512p(anchor_json_p)
    first_frame = _read_first_frame(video_path_p)
    h0, w0 = first_frame.shape[:2]

    scale_512 = 512.0 / float(max(h0, w0))
    anchor_orig = anchor_512p / float(scale_512)

    scale_target = float(target_long_side) / float(max(h0, w0))
    anchor_scaled = anchor_orig * float(scale_target)

    frames_dir = out_dir_p / "temp_frames"
    fps, frame_count = _write_resized_frames(video_path_p, frames_dir, target_long_side=target_long_side)

    frame0 = cv2.imread(str(frames_dir / "00000.jpg"))
    if frame0 is None:
        raise RuntimeError(f"读取缩放后首帧失败: {frames_dir / '00000.jpg'}")

    sam = SAM2Helper()
    mask0 = sam.get_mask_from_points(str(frames_dir), points=[anchor_scaled.tolist()], labels=[1])

    mask_path = out_dir_p / "sam_mask_frame00000.png"
    cv2.imwrite(str(mask_path), (mask0 > 0).astype(np.uint8) * 255)

    overlay = _overlay_mask(frame0, mask0, color_bgr=(0, 255, 0), alpha=0.5)
    overlay = _draw_points(overlay, np.array([anchor_scaled], dtype=np.float32), color_bgr=(0, 0, 255), radius=4)
    cv2.imwrite(str(out_dir_p / "sam_mask_overlay_frame00000.jpg"), overlay)

    query_points = sample_points_from_mask(mask0.astype(np.uint8), num_samples=256)
    qp_xy = query_points[:, 1:3]
    overlay_qp = _draw_points(overlay, qp_xy, color_bgr=(255, 0, 0), radius=2)
    cv2.imwrite(str(out_dir_p / "sam_mask_overlay_with_queries_frame00000.jpg"), overlay_qp)

    np.save(out_dir_p / "anchor_point_512p.npy", anchor_512p)
    np.save(out_dir_p / "anchor_point_orig.npy", anchor_orig)
    np.save(out_dir_p / "anchor_point_scaled.npy", anchor_scaled)
    np.save(out_dir_p / "query_points_256.npy", query_points)

    print("✅ Step1 可视化已保存")
    print(f"- mask: {mask_path}")
    print(f"- overlay: {out_dir_p / 'sam_mask_overlay_frame00000.jpg'}")
    print(f"- overlay+queries: {out_dir_p / 'sam_mask_overlay_with_queries_frame00000.jpg'}")
    print(f"- frames_dir: {frames_dir} (fps={fps}, frames={frame_count}, scale={scale_target:.4f})")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="可视化 pipeline 第 1 步：SAM 点选/分割 + 256 查询点采样")
    parser.add_argument("--video", required=True, type=str, help="输入视频路径（episode mp4）")
    parser.add_argument("--anchor_json", default="results/anchor_point.json", type=str, help="锚点配置文件（512p）")
    parser.add_argument("--out_dir", required=True, type=str, help="输出目录")
    parser.add_argument("--long_side", default=256, type=int, help="缩放后长边（需与 pipeline 一致）")

    args = parser.parse_args()
    viz_step1_sam(args.video, args.anchor_json, args.out_dir, target_long_side=args.long_side)
