from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch


@dataclass(frozen=True)
class FilterThresholds:
    visibility_frame_mean_min: float = 0.4
    visibility_low_run_len: int = 5
    confidence_mean_min: float = 0.6
    reprojection_error_p95_max_px: float = 3.0
    velocity_p95_max: float = 2.0


def _to_tn(x: torch.Tensor) -> torch.Tensor:
    if x.ndim == 4:
        return x[0]
    return x


def _to_tn1(x: torch.Tensor) -> torch.Tensor:
    x = _to_tn(x)
    if x.ndim == 3 and x.shape[-1] == 1:
        return x[..., 0]
    return x


def compute_reprojection_error_px(
    track3d_xyz_cam: torch.Tensor,
    track2d_xy_px: torch.Tensor,
    intrs: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    xyz = _to_tn(track3d_xyz_cam)
    xy = _to_tn(track2d_xy_px)
    k = _to_tn(intrs)

    x = xyz[..., 0]
    y = xyz[..., 1]
    z = xyz[..., 2].clamp_min(eps)

    fx = k[..., 0, 0]
    fy = k[..., 1, 1]
    cx = k[..., 0, 2]
    cy = k[..., 1, 2]

    u = fx[:, None] * (x / z) + cx[:, None]
    v = fy[:, None] * (y / z) + cy[:, None]

    du = u - xy[..., 0]
    dv = v - xy[..., 1]
    return torch.sqrt(du * du + dv * dv)


def compute_world_coords(
    track3d_xyz_cam: torch.Tensor,
    c2w: torch.Tensor,
) -> torch.Tensor:
    xyz = _to_tn(track3d_xyz_cam)
    c2w_tn = _to_tn(c2w)
    r = c2w_tn[:, :3, :3]
    t = c2w_tn[:, :3, 3]
    return torch.einsum("tij,tnj->tni", r, xyz) + t[:, None, :]


def compute_smoothness(
    coords_world: torch.Tensor,
    dt: float = 1.0,
    eps: float = 1e-12,
) -> Tuple[torch.Tensor, torch.Tensor]:
    p = coords_world
    v = (p[1:] - p[:-1]) / max(dt, eps)
    speed = torch.linalg.norm(v, dim=-1)
    a = (v[1:] - v[:-1]) / max(dt, eps)
    accel = torch.linalg.norm(a, dim=-1)
    return speed, accel


def _max_consecutive_true(mask_1d: torch.Tensor) -> int:
    if mask_1d.numel() == 0:
        return 0
    m = mask_1d.to(torch.int64)
    if int(m.max().item()) == 0:
        return 0
    dif = torch.diff(m, prepend=m[:1] * 0)
    starts = (dif == 1).nonzero(as_tuple=False).flatten()
    ends = (dif == -1).nonzero(as_tuple=False).flatten()
    if int(m[-1].item()) == 1:
        ends = torch.cat([ends, torch.tensor([m.numel()], device=m.device)])
    lengths = ends[: starts.numel()] - starts
    return int(lengths.max().item()) if lengths.numel() else 0


def compute_quality_scores(
    *,
    c2w_traj: torch.Tensor,
    intrs_out: torch.Tensor,
    track3d_pred: torch.Tensor,
    track2d_pred: torch.Tensor,
    vis_pred: torch.Tensor,
    conf_pred: torch.Tensor,
    dyn_pred: Optional[torch.Tensor] = None,
    dt: float = 1.0,
    thresholds: FilterThresholds = FilterThresholds(),
) -> Dict[str, Any]:
    vis = _to_tn1(vis_pred).float()
    conf = _to_tn1(conf_pred).float()
    frame_vis_mean = vis.mean(dim=1)
    low_vis_run = _max_consecutive_true(frame_vis_mean < thresholds.visibility_frame_mean_min)

    xyz_cam = _to_tn(track3d_pred)[..., :3].float()
    xy_2d = _to_tn(track2d_pred).float()
    intrs = _to_tn(intrs_out).float()

    reproj_err = compute_reprojection_error_px(xyz_cam, xy_2d, intrs)
    reproj_err_flat = reproj_err.flatten()
    reproj_p95 = float(torch.quantile(reproj_err_flat, 0.95).item()) if reproj_err_flat.numel() else 0.0
    reproj_max = float(reproj_err_flat.max().item()) if reproj_err_flat.numel() else 0.0

    coords_world = compute_world_coords(xyz_cam, c2w_traj.float())
    speed, accel = compute_smoothness(coords_world, dt=dt)
    speed_flat = speed.flatten()
    speed_p95 = float(torch.quantile(speed_flat, 0.95).item()) if speed_flat.numel() else 0.0
    speed_max = float(speed_flat.max().item()) if speed_flat.numel() else 0.0
    accel_flat = accel.flatten()
    accel_p95 = float(torch.quantile(accel_flat, 0.95).item()) if accel_flat.numel() else 0.0

    dyn_mean = None
    if dyn_pred is not None:
        dyn_mean = float(_to_tn1(dyn_pred).float().mean().item())

    mean_vis = float(vis.mean().item())
    mean_conf = float(conf.mean().item())

    flags = {
        "visibility_failure": low_vis_run >= thresholds.visibility_low_run_len,
        "low_confidence": mean_conf < thresholds.confidence_mean_min,
        "reprojection_conflict": reproj_p95 > thresholds.reprojection_error_p95_max_px,
        "tracking_jump": speed_p95 > thresholds.velocity_p95_max,
    }

    return {
        "mean_visibility": mean_vis,
        "mean_confidence": mean_conf,
        "dynamic_score_mean": dyn_mean,
        "visibility_frame_mean": frame_vis_mean.cpu().numpy(),
        "visibility_low_run": low_vis_run,
        "reprojection_error_p95_px": reproj_p95,
        "reprojection_error_max_px": reproj_max,
        "speed_p95": speed_p95,
        "speed_max": speed_max,
        "accel_p95": accel_p95,
        "flags": flags,
    }


def filter_trajectories_from_tapip(
    *,
    coords: np.ndarray,
    confs: np.ndarray,
    visibs: np.ndarray,
    conf_threshold: float = 0.8,
    vis_ratio_threshold: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray]:
    mean_confs = np.mean(confs, axis=0)
    conf_mask = mean_confs > conf_threshold
    vis_ratio = np.mean(visibs, axis=0)
    vis_mask = vis_ratio > vis_ratio_threshold
    final_mask = conf_mask & vis_mask
    return coords[:, final_mask, :], final_mask


def score_from_trajectory_npz(
    trajectory_npz_path: str,
    thresholds: FilterThresholds = FilterThresholds(),
) -> Dict[str, Any]:
    data = np.load(trajectory_npz_path)
    required = ["camera_poses", "intrinsics", "trajectories_3d", "trajectories_2d", "visibility", "confidence"]
    missing = [k for k in required if k not in data]
    if missing:
        raise KeyError(f"trajectory npz 缺少字段: {missing}")

    c2w_traj = torch.from_numpy(data["camera_poses"])
    intrs_out = torch.from_numpy(data["intrinsics"])
    track3d_pred = torch.from_numpy(data["trajectories_3d"])
    track2d_pred = torch.from_numpy(data["trajectories_2d"])
    vis_pred = torch.from_numpy(data["visibility"])
    conf_pred = torch.from_numpy(data["confidence"])

    dyn_pred = None
    if "dynamic_score" in data and data["dynamic_score"].size != 0:
        dyn_pred = torch.from_numpy(data["dynamic_score"])

    fps = float(data["src_fps"]) if "src_fps" in data else 0.0
    dt = 1.0 / fps if fps > 0 else 1.0

    scores = compute_quality_scores(
        c2w_traj=c2w_traj,
        intrs_out=intrs_out,
        track3d_pred=track3d_pred,
        track2d_pred=track2d_pred,
        vis_pred=vis_pred,
        conf_pred=conf_pred,
        dyn_pred=dyn_pred,
        dt=dt,
        thresholds=thresholds,
    )
    scores["src_fps"] = fps
    scores["dt"] = dt
    return scores


def load_quality_scores_npz(score_npz_path: str) -> Dict[str, Any]:
    data = np.load(score_npz_path)
    out: Dict[str, Any] = {}
    for k in data.files:
        v = data[k]
        if isinstance(v, np.ndarray) and v.shape == ():
            out[k] = v.item()
        else:
            out[k] = v
    return out


def summarize_rules(thresholds: FilterThresholds) -> Dict[str, Any]:
    return {
        "visibility": {
            "frame_mean_min": thresholds.visibility_frame_mean_min,
            "low_run_len": thresholds.visibility_low_run_len,
            "rule": f"连续 low_run_len 帧，frame_mean(visibility) < {thresholds.visibility_frame_mean_min} => visibility_failure",
        },
        "confidence": {
            "mean_min": thresholds.confidence_mean_min,
            "rule": f"mean(confidence) < {thresholds.confidence_mean_min} => low_confidence",
        },
        "reprojection_error": {
            "p95_max_px": thresholds.reprojection_error_p95_max_px,
            "rule": f"p95(reprojection_error_px) > {thresholds.reprojection_error_p95_max_px} => reprojection_conflict",
        },
        "smoothness": {
            "velocity_p95_max": thresholds.velocity_p95_max,
            "rule": f"p95(speed) > {thresholds.velocity_p95_max} => tracking_jump",
        },
    }


def decide_action(flags: Dict[str, bool]) -> str:
    if flags.get("visibility_failure") or flags.get("reprojection_conflict") or flags.get("tracking_jump"):
        return "DROP"
    if flags.get("low_confidence"):
        return "AUG_ONLY"
    return "KEEP"


def print_report(scores: Dict[str, Any], thresholds: FilterThresholds) -> None:
    rules = summarize_rules(thresholds)
    print("=== 评分规则 (Thresholds) ===")
    for group, info in rules.items():
        vals = {k: v for k, v in info.items() if k != "rule"}
        print(f"- {group}: {vals}")
        print(f"  {info['rule']}")

    print("\n=== 评分结果 (Scores) ===")
    print(f"- mean_visibility: {scores.get('mean_visibility'):.4f}")
    print(f"- mean_confidence: {scores.get('mean_confidence'):.4f}")
    print(f"- dynamic_score_mean: {scores.get('dynamic_score_mean')}")
    print(f"- visibility_low_run: {scores.get('visibility_low_run')}")
    print(f"- reprojection_error_p95_px: {scores.get('reprojection_error_p95_px'):.4f}")
    print(f"- reprojection_error_max_px: {scores.get('reprojection_error_max_px'):.4f}")
    print(f"- speed_p95: {scores.get('speed_p95'):.4f}")
    print(f"- speed_max: {scores.get('speed_max'):.4f}")
    print(f"- accel_p95: {scores.get('accel_p95'):.4f}")
    print(f"- src_fps: {scores.get('src_fps')}")
    print(f"- dt: {scores.get('dt')}")

    flags = scores.get("flags", {})
    print("\n=== 判定 (Flags) ===")
    for k in ["visibility_failure", "low_confidence", "reprojection_conflict", "tracking_jump"]:
        print(f"- {k}: {bool(flags.get(k, False))}")
    print(f"\n=== 结论 (Action) ===\n- {decide_action(flags)}")


def main() -> None:
    import argparse
    import os

    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--trajectory_npz", type=str, default="")
    parser.add_argument("--quality_npz", type=str, default="")
    args = parser.parse_args()

    thresholds = FilterThresholds()

    quality_npz = args.quality_npz
    if not quality_npz:
        quality_npz = os.path.join(args.results_dir, "quality_scores.npz")
    trajectory_npz = args.trajectory_npz
    if not trajectory_npz:
        trajectory_npz = os.path.join(args.results_dir, "trajectory_3d.npz")

    if os.path.exists(quality_npz):
        q = load_quality_scores_npz(quality_npz)
        scores = {
            "mean_visibility": float(q.get("mean_visibility", 0.0)),
            "mean_confidence": float(q.get("mean_confidence", 0.0)),
            "dynamic_score_mean": q.get("dynamic_score_mean", None),
            "visibility_low_run": int(q.get("visibility_low_run", 0)),
            "reprojection_error_p95_px": float(q.get("reprojection_error_p95_px", 0.0)),
            "reprojection_error_max_px": float(q.get("reprojection_error_max_px", 0.0)),
            "speed_p95": float(q.get("speed_p95", 0.0)),
            "speed_max": float(q.get("speed_max", 0.0)),
            "accel_p95": float(q.get("accel_p95", 0.0)),
            "src_fps": float(q.get("src_fps", 0.0)),
            "dt": float(q.get("dt", 1.0)),
            "flags": {
                "visibility_failure": bool(q.get("visibility_failure", False)),
                "low_confidence": bool(q.get("low_confidence", False)),
                "reprojection_conflict": bool(q.get("reprojection_conflict", False)),
                "tracking_jump": bool(q.get("tracking_jump", False)),
            },
        }
        print_report(scores, thresholds)
        return

    if os.path.exists(trajectory_npz):
        scores = score_from_trajectory_npz(trajectory_npz, thresholds=thresholds)
        print_report(scores, thresholds)
        return

    raise FileNotFoundError(f"未找到 {quality_npz} 或 {trajectory_npz}")


if __name__ == "__main__":
    main()
