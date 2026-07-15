#!/usr/bin/env python3
"""Plan the isolated OursV2/Canonical/Real joint and EE-pose comparison."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from curobo.geom.types import Pose as CuroboPose

from frame_contract import write_frame_contract
from real_control_contract import (
    SCHEMA,
    canonical_link6_target_from_real_tcp,
    canonical_rtcp_from_urdf_link6,
    oursv2_legacy_link6_target_from_real_tcp_numeric,
    oursv2_tcp_from_urdf_link6,
    rotation_error_rad,
)


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def rotations_to_wxyz(rotations: np.ndarray) -> np.ndarray:
    quat_xyzw = Rotation.from_matrix(rotations).as_quat()
    return np.column_stack([quat_xyzw[:, 3], quat_xyzw[:, :3]])


def batched_candidate(solver, goal: CuroboPose, seeds: np.ndarray | None):
    seed_tensor = None
    if seeds is not None:
        seed_tensor = torch.as_tensor(
            seeds, dtype=torch.float32, device=solver.tensor_args.device
        ).reshape(len(seeds), 1, 6)
    result = (
        solver.ik_solver.solve_batch(goal)
        if seed_tensor is None
        else solver.ik_solver.solve_batch(goal, seed_config=seed_tensor)
    )
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    success = result.success.detach().cpu().numpy().reshape(-1).astype(bool)
    solution = (
        result.solution.detach().cpu().numpy().reshape(len(success), -1, 6)[:, -1]
    )
    return success, solution.astype(np.float64)


def solve_arm(
    solver,
    target_positions: np.ndarray,
    target_rotations: np.ndarray,
    reference_q: np.ndarray,
    seed_perturbations: int,
    seed_scale: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    goal = CuroboPose(
        torch.as_tensor(
            target_positions, dtype=torch.float32, device=solver.tensor_args.device
        ),
        torch.as_tensor(
            rotations_to_wxyz(target_rotations),
            dtype=torch.float32,
            device=solver.tensor_args.device,
        ),
    )
    solver.ik_solver.position_threshold = float(solver.max_position_threshold)
    solver.ik_solver.rotation_threshold = float(solver.max_rotation_threshold)
    rng = np.random.RandomState(42)
    perturb = rng.normal(0.0, seed_scale, size=(seed_perturbations, 6))
    candidates: list[np.ndarray | None] = [reference_q]
    candidates.extend(reference_q + item[None, :] for item in perturb)
    candidates.append(None)

    count = len(reference_q)
    selected_q = reference_q.copy()
    selected_success = np.zeros(count, dtype=bool)
    selected_candidate = np.full(count, -1, dtype=np.int16)
    selected_l2 = np.full(count, np.inf, dtype=np.float64)
    selected_linf = np.full(count, np.inf, dtype=np.float64)
    for candidate_idx, seeds in enumerate(candidates):
        success, solution = batched_candidate(solver, goal, seeds)
        delta = solution - reference_q
        l2 = np.linalg.norm(delta, axis=1)
        linf = np.max(np.abs(delta), axis=1)
        better = success & (
            (~selected_success)
            | (l2 < selected_l2 - 1e-12)
            | (np.isclose(l2, selected_l2) & (linf < selected_linf))
        )
        selected_q[better] = solution[better]
        selected_success[better] = True
        selected_candidate[better] = candidate_idx
        selected_l2[better] = l2[better]
        selected_linf[better] = linf[better]
    return selected_q, selected_success, selected_candidate


def fk_series(fk, q_values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    count = q_values.shape[0]
    positions = np.empty((count, 2, 3), dtype=np.float64)
    rotations = np.empty((count, 2, 3, 3), dtype=np.float64)
    for frame_idx in range(count):
        for arm_idx in range(2):
            positions[frame_idx, arm_idx], rotations[frame_idx, arm_idx] = (
                fk.link6_pose(q_values[frame_idx, arm_idx])
            )
    return positions, rotations


def transform_series(
    positions: np.ndarray,
    rotations: np.ndarray,
    transform_fn,
    *extra,
) -> tuple[np.ndarray, np.ndarray]:
    out_positions = np.empty_like(positions)
    out_rotations = np.empty_like(rotations)
    for frame_idx in range(positions.shape[0]):
        for arm_idx in range(positions.shape[1]):
            out_positions[frame_idx, arm_idx], out_rotations[frame_idx, arm_idx] = (
                transform_fn(
                    positions[frame_idx, arm_idx],
                    rotations[frame_idx, arm_idx],
                    *extra,
                )
            )
    return out_positions, out_rotations


def error_arrays(
    actual_positions: np.ndarray,
    actual_rotations: np.ndarray,
    target_positions: np.ndarray,
    target_rotations: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    position_error = np.linalg.norm(actual_positions - target_positions, axis=2)
    rotation_error = np.empty(position_error.shape, dtype=np.float64)
    for frame_idx in range(position_error.shape[0]):
        for arm_idx in range(2):
            rotation_error[frame_idx, arm_idx] = rotation_error_rad(
                actual_rotations[frame_idx, arm_idx],
                target_rotations[frame_idx, arm_idx],
            )
    return position_error, rotation_error


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True)
    parser.add_argument("--episode", required=True)
    parser.add_argument("--episode-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--urdf", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--vis-script", type=Path, required=True)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--position-threshold", type=float, default=0.001)
    parser.add_argument("--rotation-threshold", type=float, default=0.02)
    parser.add_argument("--max-position-threshold", type=float, default=0.02)
    parser.add_argument("--ours-max-rotation-threshold", type=float, default=3.14)
    parser.add_argument("--canonical-max-rotation-threshold", type=float, default=0.12)
    parser.add_argument("--seed-perturbations", type=int, default=6)
    parser.add_argument("--seed-scale", type=float, default=0.05)
    return parser.parse_args()


def successful_mean(values: np.ndarray, success: np.ndarray, arm: int):
    selected = values[success[:, arm], arm]
    return float(np.mean(selected)) if selected.size else None


def main() -> int:
    args = parse_args()
    code_dir = Path("/home/zaijia001/ssd/RoboTwin/code_painting")
    if str(code_dir) not in sys.path:
        sys.path.insert(0, str(code_dir))
    from urdfik import URDFInverseKinematics

    start = time.time()
    vis = load_module("piper_pos_vis_real_control", args.vis_script)
    calibration = json.loads(args.calibration.read_text(encoding="utf-8"))
    embodiment = calibration["robot_config"]["left_embodiment_config"]
    global_transform = np.asarray(
        embodiment["global_trans_matrix"], dtype=np.float64
    )
    gripper_bias = float(embodiment["gripper_bias"])
    fk = vis.PiperFk(args.urdf)
    data = vis.load_episode(
        args.episode_dir, fk, global_transform, gripper_bias
    )
    count = len(data["main_times"])
    if args.max_frames > 0:
        count = min(count, args.max_frames)
    q_real = np.asarray(data["q_values"][:count], dtype=np.float64)
    real_positions = np.asarray(data["actual_positions"][:count], dtype=np.float64)
    real_rotations = np.asarray(data["actual_rotations"][:count], dtype=np.float64)

    direct_link_positions, direct_link_rotations = fk_series(fk, q_real)
    joint_ours_positions, joint_ours_rotations = transform_series(
        direct_link_positions,
        direct_link_rotations,
        oursv2_tcp_from_urdf_link6,
        global_transform,
        gripper_bias,
    )
    joint_canonical_positions, joint_canonical_rotations = transform_series(
        direct_link_positions,
        direct_link_rotations,
        canonical_rtcp_from_urdf_link6,
    )

    ours_target_positions, ours_target_rotations = transform_series(
        real_positions,
        real_rotations,
        oursv2_legacy_link6_target_from_real_tcp_numeric,
    )
    canonical_target_positions, canonical_target_rotations = transform_series(
        real_positions,
        real_rotations,
        canonical_link6_target_from_real_tcp,
    )

    planned_q = {}
    ik_success = {}
    selected_candidate = {}
    branches = {
        "oursv2": (
            ours_target_positions,
            ours_target_rotations,
            args.ours_max_rotation_threshold,
        ),
        "canonical": (
            canonical_target_positions,
            canonical_target_rotations,
            args.canonical_max_rotation_threshold,
        ),
    }
    for branch, (target_positions, target_rotations, max_rotation) in branches.items():
        branch_q = np.empty_like(q_real)
        branch_success = np.zeros((count, 2), dtype=bool)
        branch_candidate = np.full((count, 2), -1, dtype=np.int16)
        for arm_idx in range(2):
            solver = URDFInverseKinematics(
                urdf_file=args.urdf,
                base_link="base_link",
                ee_link="link6",
                position_threshold=args.position_threshold,
                rotation_threshold=args.rotation_threshold,
                max_position_threshold=args.max_position_threshold,
                max_rotation_threshold=max_rotation,
                num_seeds=1,
            )
            q, success, candidate = solve_arm(
                solver,
                target_positions[:, arm_idx],
                target_rotations[:, arm_idx],
                q_real[:, arm_idx],
                args.seed_perturbations,
                args.seed_scale,
            )
            branch_q[:, arm_idx] = q
            branch_success[:, arm_idx] = success
            branch_candidate[:, arm_idx] = candidate
            del solver
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        planned_q[branch] = branch_q
        ik_success[branch] = branch_success
        selected_candidate[branch] = branch_candidate

    achieved = {}
    for branch in branches:
        link_positions, link_rotations = fk_series(fk, planned_q[branch])
        rtcp_positions, rtcp_rotations = transform_series(
            link_positions, link_rotations, canonical_rtcp_from_urdf_link6
        )
        achieved[branch] = (
            link_positions,
            link_rotations,
            rtcp_positions,
            rtcp_rotations,
        )

    joint_ours_pos_error, joint_ours_rot_error = error_arrays(
        joint_ours_positions, joint_ours_rotations, real_positions, real_rotations
    )
    joint_canonical_pos_error, joint_canonical_rot_error = error_arrays(
        joint_canonical_positions,
        joint_canonical_rotations,
        real_positions,
        real_rotations,
    )
    eepose_errors = {}
    for branch in branches:
        eepose_errors[branch] = error_arrays(
            achieved[branch][2],
            achieved[branch][3],
            real_positions,
            real_rotations,
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    plan_path = args.output_dir / "control_plan.npz"
    np.savez_compressed(
        plan_path,
        schema=np.asarray(SCHEMA),
        task=np.asarray(args.task),
        episode=np.asarray(args.episode),
        main_times=np.asarray(data["main_times"][:count], dtype=np.float64),
        real_q=q_real,
        real_rtcp_positions=real_positions,
        real_rtcp_rotations=real_rotations,
        direct_link6_positions=direct_link_positions,
        direct_link6_rotations=direct_link_rotations,
        joint_oursv2_tcp_positions=joint_ours_positions,
        joint_oursv2_tcp_rotations=joint_ours_rotations,
        joint_canonical_rtcp_positions=joint_canonical_positions,
        joint_canonical_rtcp_rotations=joint_canonical_rotations,
        joint_oursv2_position_error_m=joint_ours_pos_error,
        joint_oursv2_rotation_error_rad=joint_ours_rot_error,
        joint_canonical_position_error_m=joint_canonical_pos_error,
        joint_canonical_rotation_error_rad=joint_canonical_rot_error,
        oursv2_link6_target_positions=ours_target_positions,
        oursv2_link6_target_rotations=ours_target_rotations,
        canonical_link6_target_positions=canonical_target_positions,
        canonical_link6_target_rotations=canonical_target_rotations,
        oursv2_planned_q=planned_q["oursv2"],
        canonical_planned_q=planned_q["canonical"],
        oursv2_ik_success=ik_success["oursv2"],
        canonical_ik_success=ik_success["canonical"],
        oursv2_selected_candidate=selected_candidate["oursv2"],
        canonical_selected_candidate=selected_candidate["canonical"],
        oursv2_achieved_link6_positions=achieved["oursv2"][0],
        oursv2_achieved_link6_rotations=achieved["oursv2"][1],
        canonical_achieved_link6_positions=achieved["canonical"][0],
        canonical_achieved_link6_rotations=achieved["canonical"][1],
        oursv2_achieved_rtcp_positions=achieved["oursv2"][2],
        oursv2_achieved_rtcp_rotations=achieved["oursv2"][3],
        canonical_achieved_rtcp_positions=achieved["canonical"][2],
        canonical_achieved_rtcp_rotations=achieved["canonical"][3],
        oursv2_eepose_position_error_m=eepose_errors["oursv2"][0],
        oursv2_eepose_rotation_error_rad=eepose_errors["oursv2"][1],
        canonical_eepose_position_error_m=eepose_errors["canonical"][0],
        canonical_eepose_rotation_error_rad=eepose_errors["canonical"][1],
    )
    for branch in branches:
        np.savez_compressed(
            args.output_dir / f"{branch}_renderer.npz",
            planned_q=planned_q[branch],
            ik_success=ik_success[branch],
        )

    arms = ["left", "right"]
    summary = {
        "schema": SCHEMA,
        "task": args.task,
        "episode": args.episode,
        "frames": count,
        "common_inputs": {
            "joint_control": "synchronized Piper real jointState q1-q6",
            "eepose_control": "synchronized Piper real arm/endPose T_B_RTCP",
        },
        "oursv2_eepose_semantics": (
            "numeric T_B_RTCP is sent unchanged as T_B_L6URDF target; "
            "target_retreat=0 and apply_global_trans_to_ik=0; no server tool inverse"
        ),
        "canonical_eepose_semantics": (
            "T_B_L6URDF = T_B_RTCP @ inv(Ry(-1.57) @ Tx(0.19))"
        ),
        "evaluation_semantics": (
            "both planned q traces are evaluated as physical Piper RTCP using "
            "T_B_L6URDF @ Ry(-1.57) @ Tx(0.19)"
        ),
        "ik_thresholds": {
            "max_position_m": args.max_position_threshold,
            "oursv2_max_rotation_rad": args.ours_max_rotation_threshold,
            "canonical_max_rotation_rad": args.canonical_max_rotation_threshold,
        },
        "joint_control_mean_position_error_m": {
            "oursv2": {
                arm: float(np.mean(joint_ours_pos_error[:, idx]))
                for idx, arm in enumerate(arms)
            },
            "canonical": {
                arm: float(np.mean(joint_canonical_pos_error[:, idx]))
                for idx, arm in enumerate(arms)
            },
        },
        "eepose_control": {},
        "elapsed_s": time.time() - start,
    }
    for branch in branches:
        summary["eepose_control"][branch] = {
            "success_rate": {
                arm: float(np.mean(ik_success[branch][:, idx]))
                for idx, arm in enumerate(arms)
            },
            "successful_mean_physical_rtcp_position_error_m": {
                arm: successful_mean(
                    eepose_errors[branch][0], ik_success[branch], idx
                )
                for idx, arm in enumerate(arms)
            },
            "successful_mean_physical_rtcp_rotation_error_rad": {
                arm: successful_mean(
                    eepose_errors[branch][1], ik_success[branch], idx
                )
                for idx, arm in enumerate(arms)
            },
        }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    write_frame_contract(
        args.output_dir / "frame_contract.json",
        {
            "experiment_schema": SCHEMA,
            "task": args.task,
            "episode": args.episode,
            "common_eepose_input": "T_B_RTCP from Piper arm/endPose",
            "oursv2_branch": summary["oursv2_eepose_semantics"],
            "canonical_branch": summary["canonical_eepose_semantics"],
        },
    )
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
