#!/usr/bin/env python3
"""Compare OursV2 TCP and server-exact Real TCP from the same q/FK trace."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from frame_contract import (
    frame_contract_payload,
    matrix_to_pose_wxyz,
    pose_wxyz_to_matrix,
    sim_link6_pose_to_real_tcp_pose,
    write_frame_contract,
)


LINE_COLORS = {"ours": (255, 255, 0), "real": (0, 140, 255)}
AXIS_COLORS = [(0, 0, 255), (0, 200, 0), (255, 0, 0)]


def ours_ee_to_raw_sim_link6(
    pose: np.ndarray, config: dict[str, Any]
) -> np.ndarray:
    ours = pose_wxyz_to_matrix(pose)
    global_trans = np.asarray(config.get("global_trans_matrix", np.eye(3)), dtype=np.float64)
    delta = np.asarray(config.get("delta_matrix", np.eye(3)), dtype=np.float64)
    bias = float(config.get("gripper_bias", 0.12))
    link = np.eye(4, dtype=np.float64)
    link[:3, :3] = ours[:3, :3] @ np.linalg.inv(delta) @ np.linalg.inv(global_trans)
    link[:3, 3] = ours[:3, 3] - ours[:3, :3] @ np.array([bias - 0.12, 0.0, 0.0])
    return matrix_to_pose_wxyz(link)


def load_records(path: Path, robot_config: Path) -> tuple[list[dict], dict]:
    cfg = json.loads(robot_config.read_text())
    rows = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        out = {"record_index": rec.get("record_index", len(rows)), "stage": rec.get("stage", "")}
        for arm in ("left", "right"):
            ours_tcp = np.asarray(rec[f"current_{arm}_tcp_pose_world_wxyz"], dtype=np.float64)
            ours_ee = np.asarray(rec[f"current_{arm}_ee_pose_world_wxyz"], dtype=np.float64)
            arm_cfg = cfg[f"{arm}_embodiment_config"]
            sim_link6 = ours_ee_to_raw_sim_link6(ours_ee, arm_cfg)
            real_tcp = sim_link6_pose_to_real_tcp_pose(sim_link6)
            out[arm] = {
                "q_rad": rec.get(f"current_{arm}_arm_qpos_rad", []),
                "T_W_OursTCP_wxyz": ours_tcp.tolist(),
                "T_W_L6SIM_wxyz": sim_link6.tolist(),
                "T_W_RTCP_wxyz": real_tcp.tolist(),
                "p_W_delta_RTCP_minus_OursTCP_m": (real_tcp[:3] - ours_tcp[:3]).tolist(),
            }
        rows.append(out)
    if not rows:
        raise RuntimeError(f"No records in {path}")
    return rows, cfg


def draw_text_panel(height: int, width: int, frame: int, rows: list[dict]) -> np.ndarray:
    panel = np.full((height, width, 3), 247, dtype=np.uint8)
    contract = frame_contract_payload()
    lines = [
        "PiperCanonicalTCP-v1 | same q / same raw L6_SIM",
        "OursV2: p_W_OursTCP, legacy +0.12 m on local Ours +X",
        "Real: T_W_L6SIM @ Ry(+pi/2 exact) @ Ry(-1.57) @ Tx(0.19)",
        "local_RTCP axes: +X RED approach | +Y GREEN opening | +Z BLUE side",
        "Plot values below are WORLD X/Y/Z (not local components)",
        f"frame={frame}/{len(rows)-1} stage={rows[frame]['stage']}",
    ]
    y = 42
    for idx, line in enumerate(lines):
        cv2.putText(panel, line, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55 if idx else 0.7, (20, 20, 20), 1 if idx else 2, cv2.LINE_AA)
        y += 38
    for arm_idx, arm in enumerate(("left", "right")):
        item = rows[frame][arm]
        ours = np.asarray(item["T_W_OursTCP_wxyz"][:3])
        real = np.asarray(item["T_W_RTCP_wxyz"][:3])
        delta = (real - ours) * 1000.0
        text = f"{arm.upper()}  Ours={ours.round(4)} m  Real={real.round(4)} m  delta={delta.round(1)} mm"
        cv2.putText(panel, text, (20, y + arm_idx * 38), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (30, 30, 30), 1, cv2.LINE_AA)
    return panel


def draw_chart(values_ours: np.ndarray, values_real: np.ndarray, frame: int, axis: int, arm: str, width: int, height: int) -> np.ndarray:
    image = np.full((height, width, 3), 252, dtype=np.uint8)
    margin = (62, 18, 28, 28)
    left, top, right, bottom = margin
    cv2.rectangle(image, (left, top), (width-right, height-bottom), (180, 180, 180), 1)
    all_values = np.concatenate([values_ours[:, axis], values_real[:, axis]])
    lo, hi = float(np.min(all_values)), float(np.max(all_values))
    pad = max((hi-lo)*0.1, 0.005)
    lo -= pad; hi += pad
    n = len(values_ours)
    def points(values: np.ndarray) -> np.ndarray:
        xs = left + np.arange(n) * (width-left-right-1) / max(n-1, 1)
        ys = top + (hi-values[:, axis]) * (height-top-bottom-1) / max(hi-lo, 1e-9)
        return np.column_stack([xs, ys]).astype(np.int32)
    cv2.polylines(image, [points(values_ours)], False, LINE_COLORS["ours"], 2, cv2.LINE_AA)
    cv2.polylines(image, [points(values_real)], False, LINE_COLORS["real"], 2, cv2.LINE_AA)
    cursor_x = int(left + frame * (width-left-right-1) / max(n-1, 1))
    cv2.line(image, (cursor_x, top), (cursor_x, height-bottom), (40, 40, 40), 1)
    label = f"{arm.upper()} world {('X','Y','Z')[axis]}"
    cv2.putText(image, label, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.48, AXIS_COLORS[axis], 1, cv2.LINE_AA)
    cv2.putText(image, f"{lo:.3f}", (4, height-bottom), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (80,80,80), 1, cv2.LINE_AA)
    cv2.putText(image, f"{hi:.3f}", (4, top+5), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (80,80,80), 1, cv2.LINE_AA)
    return image


def render_video(rows: list[dict], source_video: Path | None, output: Path) -> None:
    cap = cv2.VideoCapture(str(source_video)) if source_video and source_video.is_file() else None
    fps = cap.get(cv2.CAP_PROP_FPS) if cap else 10.0
    if not fps or fps <= 0: fps = 10.0
    width, top_h, chart_h = 1280, 480, 105
    writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, top_h + chart_h*3))
    traces = {}
    for arm in ("left", "right"):
        traces[(arm,"ours")] = np.asarray([r[arm]["T_W_OursTCP_wxyz"][:3] for r in rows])
        traces[(arm,"real")] = np.asarray([r[arm]["T_W_RTCP_wxyz"][:3] for r in rows])
    last_frame = np.zeros((top_h, width//2, 3), dtype=np.uint8)
    for idx in range(len(rows)):
        if cap:
            ok, frame = cap.read()
            if ok:
                last_frame = cv2.resize(frame, (width//2, top_h))
        panel = draw_text_panel(top_h, width//2, idx, rows)
        canvas = np.vstack([np.hstack([last_frame, panel])] + [
            np.hstack([
                draw_chart(traces[("left","ours")], traces[("left","real")], idx, axis, "left", width//2, chart_h),
                draw_chart(traces[("right","ours")], traces[("right","real")], idx, axis, "right", width//2, chart_h),
            ]) for axis in range(3)
        ])
        writer.write(canvas)
    writer.release()
    if cap: cap.release()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pose-debug", type=Path, required=True)
    parser.add_argument("--robot-config", type=Path, required=True)
    parser.add_argument("--source-video", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--episode-id", type=int, required=True)
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        if (args.output_dir / "SUCCESS").is_file():
            print(f"[skip-existing-success] {args.output_dir}")
            return 0
        raise RuntimeError(f"Refusing non-empty output: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows, _ = load_records(args.pose_debug, args.robot_config)
    trace = args.output_dir / "joint_tcp_trace.jsonl"
    trace.write_text("".join(json.dumps(row, ensure_ascii=False)+"\n" for row in rows), encoding="utf-8")
    stats = {}
    for arm in ("left", "right"):
        delta = np.asarray([row[arm]["p_W_delta_RTCP_minus_OursTCP_m"] for row in rows])
        dist = np.linalg.norm(delta, axis=1)
        stats[arm] = {"mean_distance_m": float(dist.mean()), "max_distance_m": float(dist.max()), "mean_delta_world_m": delta.mean(axis=0).tolist()}
    (args.output_dir / "summary.json").write_text(json.dumps({"schema":"piper_canonical_tcp_v1.joint_compare.v2","task":args.task,"episode_id":args.episode_id,"records":len(rows),"stats":stats}, indent=2)+"\n")
    write_frame_contract(args.output_dir / "frame_contract.json", {"task":args.task,"episode_id":args.episode_id,"comparison":"same_q_ours_tcp_vs_real_tcp"})
    render_video(rows, args.source_video, args.output_dir / "joint_tcp_comparison.mp4")
    (args.output_dir / "SUCCESS").touch()
    print(json.dumps(stats, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
