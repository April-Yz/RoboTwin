#!/usr/bin/env python3
"""Compose Orientation/Fused/Top-score Real-TCP planner videos."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from vscode_video import ensure_vscode_mp4


STRATEGIES = (("orientation", "Orientation"), ("fused", "Fused 0.25 score + 0.75 rot"), ("top_score", "Top AnyGrasp score"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    paths = [args.episode_root / "eepose" / key / "head_cam_plan.mp4" for key, _ in STRATEGIES]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing: raise FileNotFoundError("Missing strategy videos: " + ", ".join(missing))
    if args.output.exists():
        print(f"[skip-existing] {args.output}")
        return 0
    caps = [cv2.VideoCapture(str(path)) for path in paths]
    fps = min([cap.get(cv2.CAP_PROP_FPS) or 10.0 for cap in caps])
    cell_w, cell_h, header_h, footer_h = 480, 360, 52, 78
    out_size = (cell_w*3, cell_h+header_h+footer_h)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(args.output), cv2.VideoWriter_fourcc(*"mp4v"), fps, out_size)
    last = [np.zeros((cell_h,cell_w,3),dtype=np.uint8) for _ in caps]
    while True:
        any_ok = False
        frames = []
        for idx, cap in enumerate(caps):
            ok, frame = cap.read()
            if ok:
                any_ok = True
                last[idx] = cv2.resize(frame, (cell_w,cell_h))
            frames.append(last[idx].copy())
        if not any_ok: break
        canvas = np.full((out_size[1],out_size[0],3), 245, dtype=np.uint8)
        for idx, ((_,label),frame) in enumerate(zip(STRATEGIES,frames)):
            x = idx*cell_w
            canvas[header_h:header_h+cell_h,x:x+cell_w] = frame
            cv2.putText(canvas,label,(x+16,34),cv2.FONT_HERSHEY_SIMPLEX,0.68,(20,20,20),2,cv2.LINE_AA)
        footer1 = "Target/current: T_W_RTCP | readback L6_SIM->L6_URDF exact Ry(+pi/2); server L6_URDF->RTCP Ry(-1.57), 0.19m"
        footer2 = "Preview CGRASP->RTCP local-axis remap is explicit; local_RTCP +X RED approach, +Y GREEN opening, +Z BLUE side; p in world"
        cv2.putText(canvas,footer1,(16,header_h+cell_h+30),cv2.FONT_HERSHEY_SIMPLEX,0.52,(20,20,20),1,cv2.LINE_AA)
        cv2.putText(canvas,footer2,(16,header_h+cell_h+59),cv2.FONT_HERSHEY_SIMPLEX,0.52,(20,20,20),1,cv2.LINE_AA)
        writer.write(canvas)
    writer.release()
    for cap in caps: cap.release()
    ensure_vscode_mp4(args.output)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
