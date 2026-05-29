#!/usr/bin/env python3
"""Plan a two-keyframe RoboTwin demo from AnyGrasp candidates and hand gripper poses."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import sapien.core as sapien
from scipy.spatial.transform import Rotation as R
import trimesh

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent
os.chdir(PROJECT_ROOT)
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import render_hand_retarget_r1_npz as base
import render_hand_retarget_r1_npz_urdfik as urdfik_base
from render_object_pose_r1_npz import create_object_actor, get_camera_intrinsic_matrix, pose_wxyz_to_matrix
from replay_r1_h5 import ReplayRenderer, parse_optional_base_pose


R1_CONFIG = PROJECT_ROOT / "robot_config_R1.json"
R1_WRIST_CAMERA_LOCAL_QUAT_WXYZ = base.quat_multiply_wxyz(
    base.quat_from_euler_wxyz("xyz", [float(np.deg2rad(-10.0)), 0.0, 0.0], degrees=False),
    [0.5, 0.5, -0.5, 0.5],
)


@dataclass
class ObjectState:
    name: str
    mesh_file: Path
    pose_world_wxyz: np.ndarray
    pose_world_matrix: np.ndarray
    visible: bool
    visual_scale: np.ndarray = field(default_factory=lambda: np.ones(3, dtype=np.float64))
    collision_scale: np.ndarray = field(default_factory=lambda: np.ones(3, dtype=np.float64))
    actor: Optional[sapien.Entity] = None
    collision_bbox_actor: Optional[sapien.Entity] = None
    collision_groups_cache: Optional[List[List[int]]] = None
    collision_mode: str = "convex"
    collision_debug_info: Optional[Dict[str, object]] = None


@dataclass
class ObjectTrack:
    name: str
    mesh_file: Path
    frame_indices: np.ndarray
    pose_world_wxyz: np.ndarray
    pose_world_matrix: np.ndarray
    visible: np.ndarray
    actor: Optional[sapien.Entity] = None


@dataclass
class CandidatePose:
    candidate_idx: int
    score: float
    translation_cam: np.ndarray
    rotation_cam: np.ndarray
    width_m: float
    depth_m: float
    raw_pose_world_wxyz: np.ndarray
    raw_pose_world_matrix: np.ndarray
    pose_world_wxyz: np.ndarray
    pose_world_matrix: np.ndarray
    nearest_object: str
    nearest_object_distance_m: float
    rotation_distance_deg: float
    top_axis_up_dot: float
    original_top_axis_up_dot: float
    camera_up_flip_applied: int
    forward_axis_change_deg: float
    camera_up_selection_mode: str = "none"


@dataclass
class SelectedKeyframe:
    source_frame: int
    arm: str
    candidate: CandidatePose
    hand_rotation_cam: np.ndarray


@dataclass
class ArmSelectionResult:
    arm: str
    expected_object: Optional[str]
    selected_keyframes: List[SelectedKeyframe]
    ranked_candidates_per_frame: Dict[int, List[CandidatePose]]
    all_candidates_per_frame: Dict[int, List[CandidatePose]]
    diagnostics: Dict[int, Dict[str, int]]


@dataclass
class ArmDebugInfo:
    arm: str
    expected_object: Optional[str]
    ranked_candidates_per_frame: Dict[int, List[CandidatePose]]
    all_candidates_per_frame: Dict[int, List[CandidatePose]]
    diagnostics: Dict[int, Dict[str, int]]
    selected_keyframes: Optional[List[SelectedKeyframe]] = None


@dataclass
class DebugVisualBundle:
    keyframe_axis_actors: Dict[Tuple[int, str], sapien.Entity]
    common_candidate_actors: Dict[int, List[sapien.Entity]]
    arm_candidate_actors: Dict[str, Dict[int, List[sapien.Entity]]]
    arm_candidate_axis_actors: Dict[str, Dict[int, List[sapien.Entity]]]


@dataclass
class DebugExecutionState:
    writer: Optional[cv2.VideoWriter]
    left_wrist_writer: Optional[cv2.VideoWriter]
    right_wrist_writer: Optional[cv2.VideoWriter]
    selected_keyframes: List[SelectedKeyframe]
    common_candidates_per_frame: Dict[int, List[CandidatePose]]
    arm_display_candidates: Dict[str, Dict[int, List[CandidatePose]]]
    head_intrinsic: np.ndarray
    active_frame: Optional[int] = None
    active_frame_by_arm: Dict[str, int] = field(default_factory=dict)
    record_index: int = 0
    pose_debug_path: Optional[Path] = None
    metrics_debug_path: Optional[Path] = None
    replay_head_camera_pose_by_frame: Optional[Dict[int, np.ndarray]] = None
    object_tracks: Optional[Dict[str, ObjectTrack]] = None
    current_stage: Optional[str] = None
    target_pose_by_arm: Optional[Dict[str, np.ndarray]] = None
    target_object_by_arm: Optional[Dict[str, str]] = None
    goal_label_by_arm: Optional[Dict[str, str]] = None
    reach_error_pose_source: str = "tcp"
    show_selected_keyframe_axes: bool = True


@dataclass
class RankPreviewRecord:
    frame: int
    rank: int
    image_path: str
    left_candidate_idx: Optional[int]
    right_candidate_idx: Optional[int]


@dataclass
class SourcePreviewCompareRecord:
    frame: int
    image_path: str
    source_path: str
    source_kind: str


@dataclass
class ExecutionObjectReplayConfig:
    replay_frame_indices: np.ndarray
    object_tracks: Dict[str, ObjectTrack]
    start_frame: int
    end_frame: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Use AnyGrasp keyframes + hand orientation to plan a two-keyframe RoboTwin demo.")
    parser.add_argument("--anygrasp_dir", type=Path, required=True, help="Per-video AnyGrasp result dir, e.g. anygrasp_batch_results/d_pour_blue_1.")
    parser.add_argument("--replay_dir", type=Path, required=True, help="Per-video replay dir containing multi_object_world_poses.npz.")
    parser.add_argument("--hand_npz", type=Path, required=True, help="Per-video hand_detections_*.npz with gripper pose fields.")
    parser.add_argument("--output_dir", type=Path, required=True, help="Output dir for planned demo video and metadata.")
    parser.add_argument("--reuse_plan_summary_json", type=Path, default=None, help="Optional previous plan_summary.json. If set, skip AnyGrasp candidate recomputation and execute the already selected candidates from that JSON directly.")
    parser.add_argument("--reuse_preview_summary_json", type=Path, default=None, help="Optional previous preview summary.json from render_anygrasp_ranked_preview.py. If set, load top-ranked preview candidates directly instead of recomputing AnyGrasp selection.")
    parser.add_argument("--reuse_preview_frame_mode", choices=["legacy_1_max22rel", "annotated_json_keyframes"], default="legacy_1_max22rel", help="How to choose execution keyframes when --reuse_preview_summary_json is set. 'legacy_1_max22rel' keeps the old frame-1 + max(frame-22, resolved-relative-frame) rule. 'annotated_json_keyframes' uses the first two annotated keyframes recorded in preview frame_selection metadata and treats frame 0 as preview-only context.")
    parser.add_argument("--reuse_preview_candidate_group", choices=["orientation", "fused"], default="orientation", help="Which preview ranking list to use when --reuse_preview_summary_json is set.")
    parser.add_argument("--reuse_preview_top_rank", type=int, default=1, help="1-based rank to read from the preview summary candidate list when --reuse_preview_summary_json is set.")
    parser.add_argument("--keyframes", type=int, nargs=2, default=[1, 22], metavar=("GRASP_FRAME", "ACTION_FRAME"))
    parser.add_argument("--arm", choices=["auto", "left", "right"], default="auto")
    parser.add_argument("--execute_both_arms", type=int, default=1, help="If 1 and --arm auto, execute synchronized dual-arm stages and advance only when both arms satisfy reach checks.")
    parser.add_argument("--dual_stage_require_all_plans", type=int, default=1, help="If 1 in dual-arm mode, do not execute either arm in a stage unless both left/right plans are successful. This prevents one arm from moving alone when the other arm's IK fails.")
    parser.add_argument("--require_keyframe1_reached_before_close", type=int, default=0, help="If 1, skip gripper close and second-keyframe action unless the first-keyframe grasp stage reached the configured pose tolerance.")
    parser.add_argument("--require_keyframe1_reached_before_action", type=int, default=0, help="If 1, skip the second-keyframe action stage unless the first-keyframe grasp stage reached the configured pose tolerance.")
    parser.add_argument("--debug_stop_after_keyframe1", type=int, default=0, help="If 1, stop execution after the first-keyframe pregrasp/grasp stages. The gripper is not closed and the second-keyframe action is marked skipped. Use this to debug init-to-keyframe1 reachability in isolation.")
    parser.add_argument("--planner_backend", choices=["urdfik", "curobo"], default="urdfik")
    parser.add_argument(
        "--candidate_selection_mode",
        choices=["planner", "top_score_auto"],
        default="planner",
        help="How to choose the two keyframe candidates. 'planner' keeps the original orientation-first two-keyframe selection; 'top_score_auto' picks the highest AnyGrasp score candidate at the resolved frame-1 and resolved frame-2 for each arm after filtering.",
    )
    parser.add_argument(
        "--candidate_selection_relative_frame",
        type=int,
        default=None,
        help="Optional extra frame spec used only by --candidate_selection_mode top_score_auto. If set, resolve it against available grasp frames and use max(resolved_keyframe2, resolved_relative_frame) as the second execution keyframe.",
    )
    parser.add_argument(
        "--urdfik_trajectory_mode",
        choices=["joint_interp", "cartesian_interp_ik"],
        default="joint_interp",
        help="When planner_backend=urdfik: 'joint_interp' does one IK solve at the final target then interpolates in joint space; 'cartesian_interp_ik' first interpolates TCP poses in Cartesian space then solves IK waypoint by waypoint.",
    )
    parser.add_argument(
        "--urdfik_cartesian_interp_steps",
        type=int,
        default=8,
        help="Number of Cartesian TCP waypoints for urdfik_trajectory_mode=cartesian_interp_ik, including start and goal.",
    )
    parser.add_argument(
        "--urdfik_joint_interp_waypoints",
        type=int,
        default=10,
        help="Number of joint-space interpolation waypoints for urdfik_trajectory_mode=joint_interp (default 10). More waypoints = smoother motion.",
    )
    parser.add_argument(
        "--urdfik_cartesian_interp_auto_step_m",
        type=float,
        default=0.05,
        help="When --urdfik_cartesian_interp_steps=-1, translation threshold in meters used by auto waypoint mode. Smaller values create denser Cartesian interpolation.",
    )
    parser.add_argument("--urdfik_position_threshold_m", type=float, default=0.001, help="Initial URDF IK position success threshold in meters.")
    parser.add_argument("--urdfik_rotation_threshold_rad", type=float, default=0.02, help="Initial URDF IK rotation success threshold in radians.")
    parser.add_argument("--urdfik_max_position_threshold_m", type=float, default=None, help="Maximum relaxed URDF IK position threshold in meters. Default preserves the solver's old 2 mm cap.")
    parser.add_argument("--urdfik_max_rotation_threshold_rad", type=float, default=None, help="Maximum relaxed URDF IK rotation threshold in radians. Default preserves the solver's old 0.04 rad cap.")
    parser.add_argument("--urdfik_num_seeds", type=int, default=1, help="Number of CuRobo IK seeds used by URDF IK.")
    parser.add_argument("--execute_partial_cartesian_plan", type=int, default=0, help="If 1, cartesian_interp_ik plans that fail at an intermediate waypoint still execute the successfully solved waypoint prefix as a Partial plan. Diagnostic only; reached remains false unless the final target is reached.")
    parser.add_argument("--piper_urdfik_apply_global_trans_to_ik", type=int, default=0, help="Piper diagnostic only. If 1, additionally remove global_trans_matrix from the gripper target before URDFIK. Default 0 matches the direct Piper hand replay convention.")
    parser.add_argument("--left_target_object", type=str, default="cup")
    parser.add_argument("--right_target_object", type=str, default="bottle")
    parser.add_argument("--candidate_max_rotation_distance_deg", type=float, default=-1.0, help="If >= 0, drop planner candidates whose hand-orientation rotation distance exceeds this threshold before selection.")
    parser.add_argument("--candidate_object_max_distance_m", type=float, default=0.12)
    parser.add_argument("--enforce_target_object_constraint", type=int, default=1)
    parser.add_argument("--enforce_candidate_distance_constraint", type=int, default=1)
    parser.add_argument("--debug_candidate_top_k", type=int, default=5)
    parser.add_argument("--debug_show_all_candidates", type=int, default=1)
    parser.add_argument("--debug_common_candidate_top_k", type=int, default=0, help="How many raw-score candidates to show in green per keyframe. 0 hides them.")
    parser.add_argument("--debug_visualize_selected_keyframe_axes", type=int, default=1, help="If 1, show selected-keyframe axis actors in addition to the active target axes. Set 0 for target-axes-only viewer debugging.")
    parser.add_argument("--candidate_orientation_remap_label", type=str, default="identity")
    parser.add_argument("--candidate_post_rot_xyz_deg", type=float, nargs=3, default=[0.0, 0.0, 0.0])
    parser.add_argument("--candidate_keep_camera_up", type=int, default=0, help="If 1, keep the gripper/camera top side facing upward overall while preserving the original grasp direction. The planner only resolves the redundant roll about the gripper forward axis.")
    parser.add_argument("--candidate_camera_top_axis", choices=["y", "z"], default="z", help="Which local gripper axis should be treated as the camera/top direction when --candidate_keep_camera_up=1.")
    parser.add_argument("--candidate_target_local_x_offset_m", type=float, default=0.0, help="Additional translation applied to each AnyGrasp world target along target local +X before planning/visualization. In the default identity AnyGrasp frame this is the AnyGrasp finger-depth axis, which is not the same convention as direct hand replay's local +Z approach axis.")
    parser.add_argument("--candidate_target_local_z_offset_m", type=float, default=0.0, help="Additional translation applied to each AnyGrasp world target along target local +Z before planning/visualization. Use this with --candidate_orientation_remap_label swap_red_blue to reproduce the direct replay convention where blue local +Z is the approach/forward axis.")
    parser.add_argument("--manual_candidate", type=str, nargs=3, action="append", default=[], metavar=("FRAME", "ARM", "CANDIDATE_IDX"), help="Optional manual candidate override, e.g. --manual_candidate 1 left 5. Partial overrides only reorder debug display; full two-frame overrides for one arm drive selection directly.")
    parser.add_argument("--object_mesh_override", action="append", default=[], help="Repeatable mesh override in the form NAME=/abs/path/to/mesh.obj, e.g. cup=/.../blue_cup.obj")
    parser.add_argument("--robot_config", type=Path, default=R1_CONFIG)
    parser.add_argument("--image_width", type=int, default=640)
    parser.add_argument("--image_height", type=int, default=360)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--fovy_deg", type=float, default=90.0)
    parser.add_argument("--torso_qpos", type=float, nargs=4, default=base.DEFAULT_TORSO_QPOS.tolist())
    parser.add_argument("--robot_base_pose", type=float, nargs=7, default=None, metavar=("X", "Y", "Z", "QW", "QX", "QY", "QZ"))
    parser.add_argument("--open_gripper", type=float, default=1.0)
    parser.add_argument("--close_gripper", type=float, default=0.0)
    parser.add_argument("--approach_offset_m", type=float, default=0.08, help="Pregrasp-only retreat distance. The grasp target itself is unchanged; instead the planner first creates a separate pregrasp pose by moving backward from the target pose along --approach_axis.")
    parser.add_argument("--approach_axis", choices=["local_x", "local_z"], default="local_x", help="Target local axis used for pregrasp retreat. Default local_x preserves the legacy AnyGrasp planner behavior. local_z matches the direct hand replay convention after remapping AnyGrasp +X to replay +Z.")
    parser.add_argument("--settle_steps", type=int, default=4)
    parser.add_argument("--execute_interp_steps", type=int, default=24)
    parser.add_argument("--joint_command_scene_steps", type=int, default=2, help="Physics scene steps to advance after each commanded arm waypoint.")
    parser.add_argument("--joint_target_wait_steps", type=int, default=60, help="Maximum extra physics scene steps used after a stage trajectory to let the arm converge to the final commanded joint target.")
    parser.add_argument("--joint_target_wait_tol_rad", type=float, default=0.01, help="Per-joint absolute tolerance in radians used by the final post-trajectory convergence wait.")
    parser.add_argument("--print_execution_pose_every", type=int, default=0, help="If >0, print TCP/EE world positions every N executed trajectory steps. Useful for confirming that viewer motion is actually changing the robot pose.")
    parser.add_argument("--reach_pos_tol_m", type=float, default=0.03)
    parser.add_argument("--reach_rot_tol_deg", type=float, default=20.0)
    parser.add_argument("--reach_error_pose_source", choices=["tcp", "ee"], default="tcp", help="Which arm pose to use when computing reach error against the target.")
    parser.add_argument("--max_stage_replans", type=int, default=3)
    parser.add_argument("--replan_until_reached", type=int, default=1, help="If 1, keep replanning from the current state until the stage reaches tolerance. Use --replan_until_reached_max_attempts > 0 to impose an upper bound; <=0 means unbounded.")
    parser.add_argument("--replan_until_reached_max_attempts", type=int, default=0)
    parser.add_argument("--hold_frames_after_stage", type=int, default=2)
    parser.add_argument("--init_prefix_frames", type=int, default=0, help="Emit fixed init-pose frames before moving to keyframe-1; useful for downstream trimming.")
    parser.add_argument("--pause_after_keyframe1_seconds", type=float, default=0.0, help="After reaching keyframe-1 and closing the gripper, hold the robot at that pose for N seconds before planning/executing the next target.")
    parser.add_argument("--replay_objects_during_action", type=int, default=0, help="If 1, replay object tracks from keyframe-1 to keyframe-2 during the action stage instead of attaching selected objects to the TCP.")
    parser.add_argument("--replay_objects_ignore_collision", type=int, default=1, help="If 1, replayed objects are created as visual-only kinematic actors without collision.")
    parser.add_argument("--enable_grasp_action_object_collision", type=int, default=0, help="If 1, keep selected execution objects collision-capable but disable their collision before grasp. Collision is enabled only for the selected grasped objects during close_gripper/action, while the original no-collision mode remains available when this flag is 0.")
    parser.add_argument("--grasp_action_object_collision_start_stage", choices=["close_gripper", "grasp", "pregrasp"], default="close_gripper", help="When enable_grasp_action_object_collision=1, decide from which stage the selected execution objects should participate in collision. 'close_gripper' keeps the old behavior. 'grasp' enables collision before the grasp stage. 'pregrasp' keeps collision enabled from the beginning of execution.")
    parser.add_argument("--execution_object_collision_mode", choices=["convex", "solid_bbox"], default="convex", help="Collision shape used for execution objects when collision is enabled. 'convex' keeps the current convex mesh collision. 'solid_bbox' replaces collision with one solid axis-aligned box derived from the mesh bounds.")
    parser.add_argument("--execution_object_scale_override", action="append", default=[], help="Repeatable legacy execution-object scale override in the form NAME=S or NAME=SX,SY,SZ. When provided, the same scale is applied to both visual mesh and collision shape unless a more specific visual/collision override is also given.")
    parser.add_argument("--execution_object_visual_scale_override", action="append", default=[], help="Repeatable execution-object visual scale override in the form NAME=S or NAME=SX,SY,SZ.")
    parser.add_argument("--execution_object_collision_scale_override", action="append", default=[], help="Repeatable execution-object collision scale override in the form NAME=S or NAME=SX,SY,SZ.")
    parser.add_argument("--debug_collision_report", type=int, default=0, help="If 1, print detailed collision/contact debug info during gripper close, including finger-only vs finger+gripper-base contacts and collision-shape summaries.")
    parser.add_argument("--debug_visualize_object_collision_bbox", type=int, default=0, help="If 1, create a visual-only box actor for each execution object using its collision bbox (currently available for solid_bbox mode) so the collision primitive can be compared directly against the rendered mesh.")
    parser.add_argument("--gripper_contact_monitor_mode", choices=["fingers", "fingers_and_base", "all_robot_links"], default="fingers", help="Which robot links are allowed to trigger contact during close_gripper collision monitoring. 'fingers' keeps the old behavior. 'fingers_and_base' also monitors left/right_gripper_link. 'all_robot_links' monitors the full articulation link set for debugging.")
    parser.add_argument("--save_debug_preview", type=int, default=1)
    parser.add_argument("--debug_preview_fps", type=int, default=10)
    parser.add_argument("--debug_keyframe_hold_frames", type=int, default=12)
    parser.add_argument("--save_debug_execution_preview", type=int, default=1)
    parser.add_argument("--debug_execution_fps", type=int, default=10)
    parser.add_argument("--save_pose_debug", type=int, default=0, help="If 1, dump per-frame planner camera/TCP/EE/qpos/object poses to pose_debug.jsonl. pure_scene_output also enables this file automatically.")
    parser.add_argument("--pure_scene_output", type=int, default=0, help="If 1, keep head/third output videos clean: no overlay text, no candidate grippers, and no target-axis visualization. Also skip debug_selection_preview.mp4, keep head/left_wrist/right_wrist plan videos, and auto-save pose_debug.jsonl.")
    parser.add_argument("--debug_visualize_targets", type=int, default=1, help="If 1, show target axis actors. Original internal name: debug_visualize_targets.")
    parser.add_argument("--debug_visualize_ik_waypoints", type=int, default=0, help="If 1, visualize intermediate URDF IK Cartesian waypoints as point+forward-axis markers in viewer/debug output.")
    parser.add_argument("--debug_gripper_actor_forward_axis", choices=["local_x", "local_z"], default="local_x", help="Visual-only C-gripper actor forward axis. Use local_z for robot/replay-frame summaries where blue local +Z is the approach axis.")
    parser.add_argument("--debug_visualize_cameras", type=int, default=0, help="If 1, draw calibrated camera axes/frustums in rendered debug outputs.")
    parser.add_argument("--debug_camera_axis_length", type=float, default=0.16)
    parser.add_argument("--debug_camera_axis_thickness", type=float, default=0.006)
    parser.add_argument("--target_local_forward_retreat_m", type=float, default=0.0, help="Retreat replay target opposite the hand/gripper local forward axis before planning; 0 keeps legacy AnyGrasp planner behavior.")
    parser.add_argument("--save_rank_preview_images", type=int, default=1)
    parser.add_argument("--rank_preview_top_n", type=int, default=3, help="Save per-keyframe rank preview PNGs for left/right rank 1..N.")
    parser.add_argument("--debug_target_axis_length", type=float, default=0.08)
    parser.add_argument("--debug_target_axis_thickness", type=float, default=0.004)
    parser.add_argument("--head_only", type=int, default=1)
    parser.add_argument("--overlay_text", type=int, default=1)
    parser.add_argument("--third_person_view", type=int, default=0)
    parser.add_argument("--vscode_compatible_video", type=int, default=1, help="If 1, transcode head/third mp4 outputs to H.264 yuv420p faststart so VS Code can preview them reliably when ffmpeg is available.")
    parser.add_argument("--enable_viewer", type=int, default=0)
    parser.add_argument("--viewer_show_camera_frustums", type=int, default=0, help="If 1, keep SAPIEN viewer camera frustum lines visible. Original viewer plugin field: show_camera_linesets.")
    parser.add_argument("--viewer_frame_delay", type=float, default=0.0)
    parser.add_argument("--viewer_wait_at_end", type=int, default=0)
    parser.add_argument("--disable_table", type=int, default=1)
    parser.add_argument("--base_occluder_enable", type=int, default=0, help="If 1, add a visual-only box attached above the robot base to occlude the chassis in camera views. No collision is created.")
    parser.add_argument("--base_occluder_local_pos", type=float, nargs=3, default=[0.0, 0.0, 0.4], metavar=("X", "Y", "Z"))
    parser.add_argument("--base_occluder_half_size", type=float, nargs=3, default=[0.28, 0.32, 0.02], metavar=("HX", "HY", "HZ"))
    parser.add_argument("--base_occluder_color", type=float, nargs=3, default=[1.0, 1.0, 1.0], metavar=("R", "G", "B"))
    parser.add_argument("--lighting_mode", choices=["default", "front", "front_no_shadow"], default="front_no_shadow")
    parser.add_argument("--camera_cv_axis_mode", choices=sorted(base.CV_TO_WORLD_CAMERA_PRESETS.keys()), default="legacy_r1")
    parser.add_argument("--head_camera_local_pos", type=float, nargs=3, default=base.DEFAULT_HEAD_CAMERA_LOCAL_POS.tolist())
    parser.add_argument("--head_camera_local_quat_wxyz", type=float, nargs=4, default=base.DEFAULT_HEAD_CAMERA_LOCAL_QUAT_WXYZ.tolist())
    return parser.parse_args()


def candidate_pose_from_summary_entry(entry: Dict[str, object]) -> CandidatePose:
    raw_pose_world_wxyz = np.asarray(entry["raw_pose_world_wxyz"], dtype=np.float64).reshape(7)
    pose_world_wxyz = np.asarray(entry["pose_world_wxyz"], dtype=np.float64).reshape(7)
    translation_cam = np.asarray(entry.get("translation_cam", [0.0, 0.0, 0.0]), dtype=np.float64).reshape(3)
    rotation_cam = np.asarray(entry.get("rotation_cam", np.eye(3, dtype=np.float64).tolist()), dtype=np.float64).reshape(3, 3)
    return CandidatePose(
        candidate_idx=int(entry["candidate_idx"]),
        score=float(entry.get("score", 0.0)),
        translation_cam=translation_cam,
        rotation_cam=rotation_cam,
        width_m=float(entry.get("width_m", 0.08)),
        depth_m=float(entry.get("depth_m", 0.04)),
        raw_pose_world_wxyz=raw_pose_world_wxyz,
        raw_pose_world_matrix=pose_wxyz_to_matrix(raw_pose_world_wxyz),
        pose_world_wxyz=pose_world_wxyz,
        pose_world_matrix=pose_wxyz_to_matrix(pose_world_wxyz),
        nearest_object=str(entry.get("nearest_object", "")),
        nearest_object_distance_m=float(entry.get("nearest_object_distance_m", 0.0)),
        rotation_distance_deg=float(entry.get("rotation_distance_deg", 0.0)),
        top_axis_up_dot=float(entry.get("top_axis_up_dot", 0.0)),
        original_top_axis_up_dot=float(entry.get("original_top_axis_up_dot", 0.0)),
        camera_up_flip_applied=int(entry.get("camera_up_flip_applied", 0)),
        forward_axis_change_deg=float(entry.get("forward_axis_change_deg", 0.0)),
        camera_up_selection_mode=str(entry.get("camera_up_selection_mode", "none")),
    )


def selected_keyframe_from_summary_entry(entry: Dict[str, object]) -> SelectedKeyframe:
    return SelectedKeyframe(
        source_frame=int(entry["source_frame"]),
        arm=str(entry["arm"]),
        candidate=candidate_pose_from_summary_entry(entry),
        hand_rotation_cam=np.asarray(entry.get("hand_rotation_cam", np.eye(3, dtype=np.float64).tolist()), dtype=np.float64).reshape(3, 3),
    )


def preview_candidate_entry_to_pose(
    renderer: ReplayRenderer,
    args: argparse.Namespace,
    arm: str,
    frame: int,
    entry: Dict[str, object],
) -> CandidatePose:
    grasp = {
        "score": float(entry.get("anygrasp_score", 0.0)),
        "translation": np.asarray(entry["translation_cam"], dtype=np.float64).reshape(3),
        "rotation_matrix": np.asarray(entry["rotation_matrix"], dtype=np.float64).reshape(3, 3),
        "width": float(entry.get("width", 0.08)),
        "depth": float(entry.get("depth", 0.04)),
    }
    raw_pose_world_wxyz, raw_pose_world_matrix, pose_world_wxyz, pose_world_matrix, roll_debug = candidate_to_world_pose(renderer, args, grasp)
    return CandidatePose(
        candidate_idx=int(entry["candidate_idx"]),
        score=float(entry.get("anygrasp_score", 0.0)),
        translation_cam=np.asarray(entry["translation_cam"], dtype=np.float64).reshape(3),
        rotation_cam=np.asarray(entry["rotation_matrix"], dtype=np.float64).reshape(3, 3),
        width_m=float(entry.get("width", 0.08)),
        depth_m=float(entry.get("depth", 0.04)),
        raw_pose_world_wxyz=raw_pose_world_wxyz,
        raw_pose_world_matrix=raw_pose_world_matrix,
        pose_world_wxyz=pose_world_wxyz,
        pose_world_matrix=pose_world_matrix,
        nearest_object=str(entry.get("nearest_object", "")),
        nearest_object_distance_m=float(entry.get("nearest_object_distance_m", 0.0)),
        rotation_distance_deg=float(entry.get("rotation_distance_deg", 0.0)),
        top_axis_up_dot=top_axis_up_dot(pose_world_matrix[:3, :3], args.candidate_camera_top_axis),
        original_top_axis_up_dot=float(roll_debug["original_top_axis_up_dot"]),
        camera_up_flip_applied=int(roll_debug["camera_up_flip_applied"]),
        forward_axis_change_deg=float(roll_debug["forward_axis_change_deg"]),
        camera_up_selection_mode=f"preview_{args.reuse_preview_candidate_group}_rank_{int(args.reuse_preview_top_rank)}",
    )


def load_reused_plan_summary(
    path: Path,
    requested_arm_mode: str,
) -> Tuple[ArmSelectionResult, Dict[str, ArmDebugInfo], List[Tuple[str, List[SelectedKeyframe]]], Dict]:
    with path.open("r", encoding="utf-8") as f:
        summary = json.load(f)

    selected_by_arm_payload = summary.get("selected_candidates_by_executed_arm")
    execution_sequences: List[Tuple[str, List[SelectedKeyframe]]] = []
    if isinstance(selected_by_arm_payload, dict) and selected_by_arm_payload:
        for arm_name, items in selected_by_arm_payload.items():
            if not items:
                continue
            execution_sequences.append((str(arm_name), [selected_keyframe_from_summary_entry(item) for item in items]))
    else:
        primary_items = summary.get("selected_candidates", [])
        if not primary_items:
            raise ValueError(f"reuse_plan_summary_json has no selected_candidates: {path}")
        arm_name = str(primary_items[0]["arm"])
        execution_sequences = [(arm_name, [selected_keyframe_from_summary_entry(item) for item in primary_items])]

    if requested_arm_mode in ("left", "right"):
        execution_sequences = [item for item in execution_sequences if item[0] == requested_arm_mode]
        if not execution_sequences:
            raise ValueError(
                f"reuse_plan_summary_json={path} does not contain selected candidates for requested arm={requested_arm_mode}"
            )

    primary_arm_name = str(summary.get("selected_arm", execution_sequences[0][0]))
    primary_sequence = None
    for arm_name, seq in execution_sequences:
        if arm_name == primary_arm_name:
            primary_sequence = seq
            break
    if primary_sequence is None:
        primary_arm_name, primary_sequence = execution_sequences[0]

    keyframes = [int(item.source_frame) for item in primary_sequence]
    primary_expected_object = summary.get("expected_object_for_selected_arm")
    if primary_expected_object is None and primary_sequence:
        primary_expected_object = primary_sequence[0].candidate.nearest_object

    ranked_candidates_per_frame = {int(item.source_frame): [item.candidate] for item in primary_sequence}
    all_candidates_per_frame = {int(item.source_frame): [item.candidate] for item in primary_sequence}
    diagnostics_raw = summary.get("selection_diagnostics", {})
    diagnostics = {int(frame): info for frame, info in diagnostics_raw.items()} if isinstance(diagnostics_raw, dict) else {}

    selection_result = ArmSelectionResult(
        arm=primary_arm_name,
        expected_object=None if primary_expected_object is None else str(primary_expected_object),
        selected_keyframes=primary_sequence,
        ranked_candidates_per_frame=ranked_candidates_per_frame,
        all_candidates_per_frame=all_candidates_per_frame,
        diagnostics=diagnostics,
    )

    arm_debugs: Dict[str, ArmDebugInfo] = {}
    for arm_name, seq in execution_sequences:
        frame_map = {int(item.source_frame): [item.candidate] for item in seq}
        arm_debugs[arm_name] = ArmDebugInfo(
            arm=arm_name,
            expected_object=(seq[0].candidate.nearest_object if seq else None),
            ranked_candidates_per_frame=frame_map,
            all_candidates_per_frame=frame_map,
            diagnostics={int(item.source_frame): {"reused_from_plan_summary": 1} for item in seq},
            selected_keyframes=seq,
        )
    return selection_result, arm_debugs, execution_sequences, summary


def resolve_frames_from_preview_summary(
    preview_summary: Dict[str, object],
    frame_mode: str,
    requested_keyframes: Sequence[int],
    requested_relative_frame: Optional[int],
) -> Tuple[List[int], List[int], List[Tuple[int, int]], Optional[int]]:
    resolved_pairs_raw = preview_summary.get("resolved_frames", [])
    resolved_map: Dict[int, int] = {}
    for item in resolved_pairs_raw:
        if not isinstance(item, dict):
            continue
        resolved_map[int(item["requested"])] = int(item["resolved"])

    if str(frame_mode) == "annotated_json_keyframes":
        if requested_relative_frame is not None:
            raise ValueError("--candidate_selection_relative_frame is not used with --reuse_preview_frame_mode annotated_json_keyframes")
        frame_selection = preview_summary.get("frame_selection", {})
        annotated_keyframes = preview_frame_selection_keyframes(frame_selection)
        if len(annotated_keyframes) < 2:
            raise ValueError(
                "Preview summary does not contain at least two usable keyframes in frame_selection"
            )
        requested_from_preview = annotated_keyframes[:2]
        resolved_keyframe_pairs = [
            (int(requested), int(resolved_map.get(int(requested), int(requested))))
            for requested in requested_from_preview
        ]
        keyframes = [int(resolved) for _, resolved in resolved_keyframe_pairs]
        return requested_from_preview, keyframes, resolved_keyframe_pairs, None

    resolved_keyframe_pairs: List[Tuple[int, int]] = []
    for requested in [int(v) for v in requested_keyframes]:
        resolved = resolved_map.get(int(requested), int(requested))
        resolved_keyframe_pairs.append((int(requested), int(resolved)))

    keyframes = [int(resolved) for _, resolved in resolved_keyframe_pairs]
    resolved_relative_frame = None
    if requested_relative_frame is not None:
        requested_relative_frame = int(requested_relative_frame)
        resolved_relative_frame = int(resolved_map.get(requested_relative_frame, requested_relative_frame))
        if len(keyframes) >= 2:
            keyframes[1] = max(int(keyframes[1]), int(resolved_relative_frame))
            resolved_keyframe_pairs[1] = (int(requested_keyframes[1]), int(keyframes[1]))
    return [int(v) for v in requested_keyframes], keyframes, resolved_keyframe_pairs, resolved_relative_frame


def preview_frame_selection_keyframes(frame_selection: Dict[str, object]) -> List[int]:
    for key in ("effective_keyframes", "annotated_keyframes"):
        values = frame_selection.get(key, [])
        if isinstance(values, list) and len(values) >= 2:
            return [int(v) for v in values]
    return []


def preview_frame_selection_keyframes_for_arm(frame_selection: Dict[str, object], arm: str) -> List[int]:
    for key in ("effective_keyframes_by_arm", "annotated_keyframes_by_arm"):
        payload = frame_selection.get(key, {})
        if isinstance(payload, dict):
            values = payload.get(str(arm), [])
            if isinstance(values, list) and len(values) >= 2:
                return [int(v) for v in values]
    return preview_frame_selection_keyframes(frame_selection)


def load_reused_preview_summary(
    renderer: ReplayRenderer,
    args: argparse.Namespace,
    hand_data: Dict[str, np.ndarray],
    path: Path,
    requested_arm_mode: str,
    requested_keyframes: Sequence[int],
    requested_relative_frame: Optional[int],
) -> Tuple[ArmSelectionResult, Dict[str, ArmDebugInfo], List[Tuple[str, List[SelectedKeyframe]]], Dict, List[int], List[int], List[Tuple[int, int]], Optional[int]]:
    with path.open("r", encoding="utf-8") as f:
        summary = json.load(f)

    requested_keyframes_used, keyframes, resolved_keyframe_pairs, resolved_relative_frame = resolve_frames_from_preview_summary(
        preview_summary=summary,
        frame_mode=str(args.reuse_preview_frame_mode),
        requested_keyframes=requested_keyframes,
        requested_relative_frame=requested_relative_frame,
    )
    frame_entries = {
        int(item["frame"]): item for item in summary.get("frames", []) if isinstance(item, dict) and "frame" in item
    }
    resolved_map: Dict[int, int] = {}
    for item in summary.get("resolved_frames", []):
        if not isinstance(item, dict):
            continue
        resolved_map[int(item["requested"])] = int(item["resolved"])
    if int(args.reuse_preview_top_rank) <= 0:
        raise ValueError(f"reuse_preview_top_rank must be >= 1, got {args.reuse_preview_top_rank}")
    rank_index = int(args.reuse_preview_top_rank) - 1
    frame_selection = summary.get("frame_selection", {})

    candidate_arms = [requested_arm_mode] if requested_arm_mode in ("left", "right") else ["left", "right"]
    arm_debugs: Dict[str, ArmDebugInfo] = {}
    execution_sequences: List[Tuple[str, List[SelectedKeyframe]]] = []
    best_selection: Optional[ArmSelectionResult] = None
    best_metric: Optional[Tuple[float, float]] = None
    group_suffix = str(args.reuse_preview_candidate_group)

    for arm in candidate_arms:
        hand_rotations = np.asarray(hand_data[f"{arm}_gripper_rotation_matrix"], dtype=np.float64)
        hand_valid = np.asarray(hand_data[f"{arm}_gripper_valid"], dtype=bool)
        selected_keyframes: List[SelectedKeyframe] = []
        diagnostics: Dict[int, Dict[str, int]] = {}
        failed = False
        arm_requested_keyframes = keyframes
        if str(args.reuse_preview_frame_mode) == "annotated_json_keyframes" and isinstance(frame_selection, dict):
            arm_requested_keyframes = [
                int(resolved_map.get(int(frame), int(frame)))
                for frame in preview_frame_selection_keyframes_for_arm(frame_selection, arm)[:2]
            ]
        for frame in arm_requested_keyframes:
            frame_entry = frame_entries.get(int(frame))
            if frame_entry is None:
                failed = True
                diagnostics[int(frame)] = {"preview_frame_present": 0}
                break
            candidate_key = f"{arm}_{group_suffix}"
            ranked_entries = list(frame_entry.get("top_candidates", {}).get(candidate_key, []))
            if rank_index >= len(ranked_entries):
                failed = True
                diagnostics[int(frame)] = {
                    "preview_frame_present": 1,
                    "preview_candidate_count": int(len(ranked_entries)),
                    "requested_rank": int(args.reuse_preview_top_rank),
                }
                break
            ref_key = f"{arm}_reference_hand_frame"
            ref_frame = int(frame_entry.get(ref_key, int(frame)))
            if ref_frame < 0 or ref_frame >= hand_rotations.shape[0] or (hand_valid.shape[0] > ref_frame and not bool(hand_valid[ref_frame])):
                fallback_ref = nearest_valid_hand_frame(int(frame), hand_valid)
                ref_frame = int(frame if fallback_ref is None else fallback_ref)
            candidate = preview_candidate_entry_to_pose(
                renderer=renderer,
                args=args,
                arm=arm,
                frame=int(frame),
                entry=ranked_entries[rank_index],
            )
            selected_keyframes.append(
                SelectedKeyframe(
                    source_frame=int(frame),
                    arm=arm,
                    candidate=candidate,
                    hand_rotation_cam=np.asarray(hand_rotations[int(ref_frame)], dtype=np.float64),
                )
            )
            diagnostics[int(frame)] = {
                "preview_frame_present": 1,
                "preview_candidate_count": int(len(ranked_entries)),
                "selected_preview_rank": int(args.reuse_preview_top_rank),
                "reference_hand_frame": int(ref_frame),
            }
        if failed:
            arm_debugs[arm] = ArmDebugInfo(
                arm=arm,
                expected_object=expected_object_for_arm(args, arm),
                ranked_candidates_per_frame={},
                all_candidates_per_frame={},
                diagnostics=diagnostics,
                selected_keyframes=None,
            )
            continue

        selected_keyframes = postprocess_selected_keyframe_rolls(selected_keyframes, args)
        frame_map = {int(item.source_frame): [item.candidate] for item in selected_keyframes}
        selection = ArmSelectionResult(
            arm=arm,
            expected_object=expected_object_for_arm(args, arm),
            selected_keyframes=selected_keyframes,
            ranked_candidates_per_frame=frame_map,
            all_candidates_per_frame=frame_map,
            diagnostics=diagnostics,
        )
        execution_sequences.append((arm, selected_keyframes))
        arm_debugs[arm] = ArmDebugInfo(
            arm=arm,
            expected_object=selection.expected_object,
            ranked_candidates_per_frame=frame_map,
            all_candidates_per_frame=frame_map,
            diagnostics=diagnostics,
            selected_keyframes=selected_keyframes,
        )
        total_rotation = float(sum(item.candidate.rotation_distance_deg for item in selected_keyframes))
        total_score = float(sum(item.candidate.score for item in selected_keyframes))
        if str(args.candidate_selection_mode) == "top_score_auto":
            metric = (-total_score, total_rotation)
        else:
            metric = (total_rotation, -total_score)
        if best_metric is None or metric < best_metric:
            best_metric = metric
            best_selection = selection

    if best_selection is None:
        raise RuntimeError(
            f"Failed to reuse preview summary candidates from {path}. "
            f"group={args.reuse_preview_candidate_group} rank={args.reuse_preview_top_rank}"
        )
    return best_selection, arm_debugs, execution_sequences, summary, requested_keyframes_used, keyframes, resolved_keyframe_pairs, resolved_relative_frame


def build_renderer(args: argparse.Namespace) -> ReplayRenderer:
    renderer_cls = ReplayRenderer if args.planner_backend == "curobo" else urdfik_base.HandRetargetR1URDFIKRenderer
    attach_planner = args.planner_backend == "curobo"
    renderer_kwargs = dict(
        robot_config_path=args.robot_config,
        image_width=args.image_width,
        image_height=args.image_height,
        fovy_deg=args.fovy_deg,
        torso_qpos=args.torso_qpos,
        robot_base_pose_override=parse_optional_base_pose(args.robot_base_pose),
        third_person_view=bool(args.third_person_view),
        need_topp=False,
        link_cam_debug_enable=False,
        link_cam_axis_mode="none",
        link_cam_debug_rot_xyz_deg=[0.0, 0.0, 0.0],
        link_cam_debug_shift_fru=[0.0, 0.0, 0.0],
        camera_cv_axis_mode=args.camera_cv_axis_mode,
        head_camera_local_pos=args.head_camera_local_pos,
        head_camera_local_quat_wxyz=args.head_camera_local_quat_wxyz,
        wrist_camera_local_pos=base.DEFAULT_WRIST_CAMERA_LOCAL_POS,
        # For the R1 planner path, match galaxea_sim/robots/r1.py exactly.
        # R1 Pro carries an extra local z=-90 deg wrist rotation, but R1 does not.
        wrist_camera_local_quat_wxyz=R1_WRIST_CAMERA_LOCAL_QUAT_WXYZ,
        camera_debug_target="head",
        enable_viewer=bool(args.enable_viewer),
        viewer_frame_delay=args.viewer_frame_delay,
        viewer_wait_at_end=bool(args.viewer_wait_at_end),
        debug_mode=False,
        debug_force_orientation="none",
        # Expose target-axis rendering explicitly so pure/debug workflows can switch
        # between clean and instrumented output without patching this script.
        debug_visualize_targets=bool(args.debug_visualize_targets),
        debug_target_axis_length=args.debug_target_axis_length,
        debug_target_axis_thickness=args.debug_target_axis_thickness,
        debug_visualize_cameras=bool(args.debug_visualize_cameras),
        debug_camera_axis_length=args.debug_camera_axis_length,
        debug_camera_axis_thickness=args.debug_camera_axis_thickness,
        orientation_remap_label="identity",
        orientation_remap_matrix=np.eye(3, dtype=np.float64),
        stored_orientation_post_rot_xyz_deg=[0.0, 0.0, 0.0],
        target_local_forward_retreat_m=args.target_local_forward_retreat_m,
        target_world_offset_xyz=[0.0, 0.0, 0.0],
        left_target_world_offset_xyz=[0.0, 0.0, 0.0],
        right_target_world_offset_xyz=[0.0, 0.0, 0.0],
        target_world_z_offset=0.0,
        disable_table=bool(args.disable_table),
        base_occluder_enable=bool(args.base_occluder_enable),
        base_occluder_local_pos=args.base_occluder_local_pos,
        base_occluder_half_size=args.base_occluder_half_size,
        base_occluder_color=args.base_occluder_color,
        camera_sweep_enable=False,
        camera_sweep_steps_deg=[0.0],
        init_left_arm_joints=None,
        init_right_arm_joints=None,
        init_gripper_open=None,
        lighting_mode=args.lighting_mode,
        attach_planner=attach_planner,
        hide_robot=False,
    )
    if args.planner_backend == "urdfik":
        renderer_kwargs["urdfik_trajectory_mode"] = str(args.urdfik_trajectory_mode)
        renderer_kwargs["urdfik_joint_interp_waypoints"] = int(args.urdfik_joint_interp_waypoints)
        renderer_kwargs["urdfik_cartesian_interp_steps"] = int(args.urdfik_cartesian_interp_steps)
        renderer_kwargs["urdfik_cartesian_interp_auto_step_m"] = float(args.urdfik_cartesian_interp_auto_step_m)
        renderer_kwargs["urdfik_position_threshold_m"] = float(args.urdfik_position_threshold_m)
        renderer_kwargs["urdfik_rotation_threshold_rad"] = float(args.urdfik_rotation_threshold_rad)
        renderer_kwargs["urdfik_max_position_threshold_m"] = args.urdfik_max_position_threshold_m
        renderer_kwargs["urdfik_max_rotation_threshold_rad"] = args.urdfik_max_rotation_threshold_rad
        renderer_kwargs["urdfik_num_seeds"] = int(args.urdfik_num_seeds)
        renderer_kwargs["urdfik_execute_partial_cartesian_plan"] = bool(args.execute_partial_cartesian_plan)
        renderer_kwargs["urdfik_apply_global_trans_to_ik"] = bool(args.piper_urdfik_apply_global_trans_to_ik)
    renderer = renderer_cls(**renderer_kwargs)
    # SAPIEN viewer draws camera frustum lines through ControlWindow.show_camera_linesets.
    # Keep them off by default so viewer recordings match the intended clean video output.
    try:
        if getattr(renderer, "viewer", None) is not None:
            renderer.viewer.control_window.show_camera_linesets = bool(args.viewer_show_camera_frustums)
    except Exception:
        pass
    return renderer


def load_hand_data(hand_npz: Path) -> Dict[str, np.ndarray]:
    data = np.load(str(hand_npz), allow_pickle=True)
    return {key: np.asarray(data[key]) for key in data.files}


def load_anygrasp_candidates(anygrasp_dir: Path, source_frame: int) -> List[Dict]:
    grasp_json = anygrasp_dir / "grasps" / f"grasp_{int(source_frame):06d}.json"
    if not grasp_json.is_file():
        raise FileNotFoundError(f"Missing AnyGrasp grasp json: {grasp_json}")
    with grasp_json.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return list(data.get("grasps", []))


def list_available_grasp_frames(anygrasp_dir: Path) -> List[int]:
    grasp_dir = anygrasp_dir / "grasps"
    if not grasp_dir.is_dir():
        raise FileNotFoundError(f"Missing grasp directory: {grasp_dir}")
    frames: List[int] = []
    for path in grasp_dir.iterdir():
        name = path.name
        if not name.startswith("grasp_") or not name.endswith(".json"):
            continue
        try:
            frames.append(int(name[len("grasp_") : -len(".json")]))
        except ValueError:
            continue
    if not frames:
        raise RuntimeError(f"No grasp_*.json files found in {grasp_dir}")
    return sorted(frames)


def resolve_requested_frames(requested_frames: Sequence[int], available_frames: Sequence[int]) -> List[Tuple[int, int]]:
    available = [int(v) for v in available_frames]
    resolved: List[Tuple[int, int]] = []
    for requested in [int(v) for v in requested_frames]:
        if requested >= 0:
            resolved.append((requested, requested))
            continue
        if abs(int(requested)) > len(available):
            raise IndexError(
                f"Requested relative frame {requested} is out of range for {len(available)} available grasp frames."
            )
        resolved.append((requested, int(available[int(requested)])))
    return resolved


def parse_object_mesh_overrides(specs: Sequence[str]) -> Dict[str, Path]:
    overrides: Dict[str, Path] = {}
    for spec in specs:
        if "=" not in str(spec):
            raise ValueError(f"Invalid --object_mesh_override value '{spec}'. Expected NAME=/abs/path/to/mesh.obj")
        name, mesh_file = str(spec).split("=", 1)
        obj_name = name.strip()
        mesh_path = Path(mesh_file.strip()).resolve()
        if not obj_name:
            raise ValueError(f"Invalid --object_mesh_override value '{spec}'. Empty object name.")
        if not mesh_path.is_file():
            raise FileNotFoundError(f"Override mesh file does not exist: {mesh_path}")
        overrides[obj_name] = mesh_path
    return overrides


def parse_named_scale_overrides(specs: Sequence[str], flag_name: str) -> Dict[str, np.ndarray]:
    overrides: Dict[str, np.ndarray] = {}
    for spec in specs:
        if "=" not in str(spec):
            raise ValueError(f"Invalid {flag_name} value '{spec}'. Expected NAME=S or NAME=SX,SY,SZ")
        name, raw_scale = str(spec).split("=", 1)
        obj_name = name.strip()
        if not obj_name:
            raise ValueError(f"Invalid {flag_name} value '{spec}'. Empty object name.")
        values = [float(v.strip()) for v in raw_scale.split(",") if v.strip()]
        if len(values) == 1:
            scale = np.asarray([values[0], values[0], values[0]], dtype=np.float64)
        elif len(values) == 3:
            scale = np.asarray(values, dtype=np.float64)
        else:
            raise ValueError(f"Invalid {flag_name} value '{spec}'. Expected 1 or 3 numeric scale values.")
        if np.any(scale <= 0.0):
            raise ValueError(f"Invalid {flag_name} value '{spec}'. Scale values must be > 0.")
        overrides[obj_name] = scale
    return overrides


def load_object_states(
    replay_dir: Path,
    source_frame: int,
    mesh_overrides: Optional[Dict[str, Path]] = None,
    visual_scale_overrides: Optional[Dict[str, np.ndarray]] = None,
    collision_scale_overrides: Optional[Dict[str, np.ndarray]] = None,
) -> Dict[str, ObjectState]:
    pose_path = replay_dir / "multi_object_world_poses.npz"
    if not pose_path.is_file():
        raise FileNotFoundError(f"Missing multi_object_world_poses.npz: {pose_path}")
    data = np.load(str(pose_path), allow_pickle=True)
    frame_indices = np.asarray(data["selected_source_frame_indices"], dtype=np.int32).reshape(-1)
    matches = np.where(frame_indices == int(source_frame))[0]
    if matches.size == 0:
        raise KeyError(f"source_frame={source_frame} not found in {pose_path}")
    idx = int(matches[0])

    object_names = [str(name) for name in np.asarray(data["object_names"], dtype=object).tolist()]
    states: Dict[str, ObjectState] = {}
    for object_name in object_names:
        key = object_name
        pose_world_wxyz = np.asarray(data[f"{key}__pose_world_wxyz"], dtype=np.float64)[idx]
        pose_world_matrix = np.asarray(data[f"{key}__pose_world_matrix"], dtype=np.float64)[idx]
        visible = bool(np.asarray(data[f"{key}__visible"], dtype=bool)[idx])
        mesh_file = Path(str(np.asarray(data[f"{key}__mesh_file"], dtype=object).reshape(()))).resolve()
        if mesh_overrides and object_name in mesh_overrides:
            mesh_file = mesh_overrides[object_name]
        states[object_name] = ObjectState(
            name=object_name,
            mesh_file=mesh_file,
            pose_world_wxyz=pose_world_wxyz,
            pose_world_matrix=pose_world_matrix,
            visible=visible,
            visual_scale=(
                np.asarray(visual_scale_overrides[object_name], dtype=np.float64).reshape(3)
                if visual_scale_overrides and object_name in visual_scale_overrides
                else np.ones(3, dtype=np.float64)
            ),
            collision_scale=(
                np.asarray(collision_scale_overrides[object_name], dtype=np.float64).reshape(3)
                if collision_scale_overrides and object_name in collision_scale_overrides
                else np.ones(3, dtype=np.float64)
            ),
        )
    return states


def load_object_tracks(replay_dir: Path, mesh_overrides: Optional[Dict[str, Path]] = None) -> Tuple[np.ndarray, Dict[str, ObjectTrack]]:
    pose_path = replay_dir / "multi_object_world_poses.npz"
    if not pose_path.is_file():
        raise FileNotFoundError(f"Missing multi_object_world_poses.npz: {pose_path}")
    data = np.load(str(pose_path), allow_pickle=True)
    frame_indices = np.asarray(data["selected_source_frame_indices"], dtype=np.int32).reshape(-1)
    object_names = [str(name) for name in np.asarray(data["object_names"], dtype=object).tolist()]
    tracks: Dict[str, ObjectTrack] = {}
    for object_name in object_names:
        key = object_name
        mesh_file = Path(str(np.asarray(data[f"{key}__mesh_file"], dtype=object).reshape(()))).resolve()
        if mesh_overrides and object_name in mesh_overrides:
            mesh_file = mesh_overrides[object_name]
        tracks[object_name] = ObjectTrack(
            name=object_name,
            mesh_file=mesh_file,
            frame_indices=frame_indices.copy(),
            pose_world_wxyz=np.asarray(data[f"{key}__pose_world_wxyz"], dtype=np.float64),
            pose_world_matrix=np.asarray(data[f"{key}__pose_world_matrix"], dtype=np.float64),
            visible=np.asarray(data[f"{key}__visible"], dtype=bool),
        )
    return frame_indices, tracks


def load_replay_head_camera_poses(replay_dir: Path) -> Dict[int, np.ndarray]:
    pose_path = replay_dir / "multi_object_world_poses.npz"
    if not pose_path.is_file():
        raise FileNotFoundError(f"Missing multi_object_world_poses.npz: {pose_path}")
    data = np.load(str(pose_path), allow_pickle=True)
    if "head_camera_pose_world_wxyz" not in data.files:
        return {}
    frame_indices = np.asarray(data["selected_source_frame_indices"], dtype=np.int32).reshape(-1)
    head_poses = np.asarray(data["head_camera_pose_world_wxyz"], dtype=np.float64).reshape(-1, 7)
    return {int(frame): head_poses[idx] for idx, frame in enumerate(frame_indices.tolist())}


def _build_execution_collision_debug_info(
    mesh_file: Path,
    collision_mode: str,
    mesh_scale: Sequence[float],
) -> Dict[str, object]:
    scale = np.asarray(mesh_scale, dtype=np.float64).reshape(3)
    info: Dict[str, object] = {
        "requested_mode": str(collision_mode),
        "effective_mode": str(collision_mode),
        "mesh_file": str(mesh_file),
        "mesh_scale": np.round(scale, 6).tolist(),
    }
    if collision_mode == "solid_bbox":
        mesh_scene = trimesh.load(str(mesh_file), force="scene", process=False)
        bounds = np.asarray(mesh_scene.bounds, dtype=np.float64).reshape(2, 3)
        center = bounds.mean(axis=0) * scale
        half_size = np.maximum((bounds[1] - bounds[0]) * 0.5 * scale, 1e-4)
        info["center"] = np.round(center, 6).tolist()
        info["half_size"] = np.round(half_size, 6).tolist()
    return info


def _add_execution_collision(
    builder: sapien.ActorBuilder,
    mesh_file: Path,
    actor_name: str,
    collision_mode: str,
    mesh_scale: Sequence[float],
) -> Dict[str, object]:
    scale = np.asarray(mesh_scale, dtype=np.float64).reshape(3)
    if collision_mode == "solid_bbox":
        try:
            debug_info = _build_execution_collision_debug_info(mesh_file, collision_mode, scale)
            center = np.asarray(debug_info["center"], dtype=np.float64)
            half_size = np.asarray(debug_info["half_size"], dtype=np.float64)
            builder.add_box_collision(
                pose=sapien.Pose(center.tolist()),
                half_size=half_size.tolist(),
            )
            print(
                "[object-collision] "
                f"actor={actor_name} mode=solid_bbox mesh={mesh_file} scale={np.round(scale, 4).tolist()} "
                f"center={np.round(center, 4).tolist()} half_size={np.round(half_size, 4).tolist()}"
            )
            return debug_info
        except Exception as exc:
            print(
                "[object-collision] "
                f"actor={actor_name} requested=solid_bbox fallback=convex mesh={mesh_file} scale={np.round(scale, 4).tolist()} reason={exc}"
            )
            return {
                "requested_mode": str(collision_mode),
                "effective_mode": "convex",
                "mesh_file": str(mesh_file),
                "mesh_scale": np.round(scale, 6).tolist(),
                "fallback_reason": repr(exc),
            }
    builder.add_convex_collision_from_file(str(mesh_file), scale=scale.tolist())
    return {
        "requested_mode": str(collision_mode),
        "effective_mode": "convex",
        "mesh_file": str(mesh_file),
        "mesh_scale": np.round(scale, 6).tolist(),
    }


def create_execution_object_actor(
    scene: sapien.Scene,
    mesh_file: Path,
    actor_name: str,
    ignore_collision: bool,
    collision_mode: str = "convex",
    visual_scale: Optional[Sequence[float]] = None,
    collision_scale: Optional[Sequence[float]] = None,
) -> Tuple[sapien.Entity, Optional[Dict[str, object]]]:
    visual_scale_arr = np.asarray(visual_scale if visual_scale is not None else [1.0, 1.0, 1.0], dtype=np.float64).reshape(3)
    collision_scale_arr = np.asarray(collision_scale if collision_scale is not None else [1.0, 1.0, 1.0], dtype=np.float64).reshape(3)
    if not bool(ignore_collision):
        builder = scene.create_actor_builder()
        try:
            builder.add_visual_from_file(str(mesh_file), scale=visual_scale_arr.tolist())
        except Exception as exc:
            raise RuntimeError(f"Failed to load mesh visual: {mesh_file}") from exc
        try:
            debug_info = _add_execution_collision(builder, mesh_file, actor_name, str(collision_mode), collision_scale_arr)
            debug_info["visual_scale"] = np.round(visual_scale_arr, 6).tolist()
        except Exception as exc:
            raise RuntimeError(f"Failed to add execution collision for {mesh_file}") from exc
        return builder.build_kinematic(name=actor_name), debug_info
    builder = scene.create_actor_builder()
    try:
        builder.add_visual_from_file(str(mesh_file), scale=visual_scale_arr.tolist())
    except Exception as exc:
        raise RuntimeError(f"Failed to load mesh visual: {mesh_file}") from exc
    return builder.build_kinematic(name=actor_name), {
        "requested_mode": "none",
        "effective_mode": "none",
        "mesh_file": str(mesh_file),
        "visual_scale": np.round(visual_scale_arr, 6).tolist(),
        "mesh_scale": np.round(collision_scale_arr, 6).tolist(),
    }


def _get_actor_collision_shapes(actor: Optional[sapien.Entity]) -> List[object]:
    if actor is None:
        return []
    try:
        components = list(actor.get_components())
    except Exception:
        return []
    for component in components:
        getter = getattr(component, "get_collision_shapes", None)
        if getter is None:
            continue
        try:
            shapes = list(getter())
        except Exception:
            continue
        if shapes:
            return shapes
    return []


def snapshot_actor_collision_groups(actor: Optional[sapien.Entity]) -> Optional[List[List[int]]]:
    shapes = _get_actor_collision_shapes(actor)
    if not shapes:
        return None
    groups: List[List[int]] = []
    for shape in shapes:
        try:
            groups.append(list(shape.get_collision_groups()))
        except Exception:
            return None
    return groups


def set_actor_collision_groups(actor: Optional[sapien.Entity], groups_per_shape: Sequence[Sequence[int]]) -> bool:
    shapes = _get_actor_collision_shapes(actor)
    if not shapes or len(shapes) != len(groups_per_shape):
        return False
    try:
        for shape, groups in zip(shapes, groups_per_shape):
            shape.set_collision_groups([int(v) for v in groups])
    except Exception:
        return False
    return True


def set_actor_collision_enabled(actor: Optional[sapien.Entity], enabled: bool, cached_groups: Optional[List[List[int]]]) -> Optional[List[List[int]]]:
    shapes = _get_actor_collision_shapes(actor)
    if not shapes:
        return cached_groups
    if enabled:
        if cached_groups is not None:
            set_actor_collision_groups(actor, cached_groups)
        return cached_groups
    if cached_groups is None:
        cached_groups = snapshot_actor_collision_groups(actor)
    disabled_groups = [[0, 0, 0, 0] for _ in shapes]
    set_actor_collision_groups(actor, disabled_groups)
    return cached_groups


def nearest_track_index(replay_frame_indices: np.ndarray, source_frame: int) -> int:
    frame_indices = np.asarray(replay_frame_indices, dtype=np.int32).reshape(-1)
    if frame_indices.size == 0:
        raise ValueError("replay_frame_indices is empty.")
    return int(np.argmin(np.abs(frame_indices - int(source_frame))))


def update_execution_object_replay(config: ExecutionObjectReplayConfig, progress_01: float) -> int:
    alpha = float(np.clip(progress_01, 0.0, 1.0))
    source_frame = int(round(float(config.start_frame) + alpha * float(config.end_frame - config.start_frame)))
    idx = nearest_track_index(config.replay_frame_indices, source_frame)
    for track in config.object_tracks.values():
        if track.actor is None:
            continue
        if bool(track.visible[idx]):
            set_actor_pose(track.actor, track.pose_world_wxyz[idx])
        else:
            hide_actor(track.actor)
    return int(config.replay_frame_indices[idx])


def expected_object_for_arm(args: argparse.Namespace, arm: str) -> Optional[str]:
    if not bool(args.enforce_target_object_constraint):
        return None
    if arm == "left":
        value = str(args.left_target_object).strip()
    elif arm == "right":
        value = str(args.right_target_object).strip()
    else:
        value = ""
    return value or None


def rotation_distance_deg(rot_a: np.ndarray, rot_b: np.ndarray) -> float:
    delta = R.from_matrix(np.asarray(rot_a, dtype=np.float64).reshape(3, 3)).inv() * R.from_matrix(np.asarray(rot_b, dtype=np.float64).reshape(3, 3))
    return float(np.rad2deg(delta.magnitude()))


def shift_pose_along_local_axis(pose_world_wxyz: np.ndarray, delta_m: float, axis_index: int) -> np.ndarray:
    # Candidate-target compensation before planning.
    # This modifies the AnyGrasp target itself and is used to compensate for
    # fingertip-TCP vs wrist/endlink convention mismatch.
    pose_world_wxyz = np.asarray(pose_world_wxyz, dtype=np.float64).reshape(7)
    pose_world_matrix = pose_wxyz_to_matrix(pose_world_wxyz)
    pose_world_matrix[:3, 3] += pose_world_matrix[:3, int(axis_index)] * float(delta_m)
    quat = base.quat_xyzw_to_wxyz(R.from_matrix(base.orthonormalize_rotation(pose_world_matrix[:3, :3])).as_quat())
    return np.concatenate([pose_world_matrix[:3, 3], quat]).astype(np.float64)


def shift_pose_along_local_x(pose_world_wxyz: np.ndarray, delta_m: float) -> np.ndarray:
    return shift_pose_along_local_axis(pose_world_wxyz, delta_m, 0)


def shift_pose_along_local_z(pose_world_wxyz: np.ndarray, delta_m: float) -> np.ndarray:
    return shift_pose_along_local_axis(pose_world_wxyz, delta_m, 2)


def candidate_to_world_pose(
    renderer: ReplayRenderer,
    args: argparse.Namespace,
    grasp: Dict,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, float]]:
    rot_cam = np.asarray(grasp["rotation_matrix"], dtype=np.float64).reshape(3, 3)
    trans_cam = np.asarray(grasp["translation"], dtype=np.float64).reshape(3)
    remapped_rotation = base.orthonormalize_rotation(
        base.orthonormalize_rotation(rot_cam) @ args.candidate_post_rot_matrix @ args.candidate_orientation_remap_matrix
    )
    raw_pose_world_wxyz = renderer.camera_to_world_pose(trans_cam, remapped_rotation)
    raw_pose_world_matrix = pose_wxyz_to_matrix(raw_pose_world_wxyz)
    pose_world_wxyz = raw_pose_world_wxyz.copy()
    if abs(float(args.candidate_target_local_x_offset_m)) > 1e-12:
        pose_world_wxyz = shift_pose_along_local_x(pose_world_wxyz, float(args.candidate_target_local_x_offset_m))
    if abs(float(args.candidate_target_local_z_offset_m)) > 1e-12:
        pose_world_wxyz = shift_pose_along_local_z(pose_world_wxyz, float(args.candidate_target_local_z_offset_m))
    pose_world_matrix = pose_wxyz_to_matrix(pose_world_wxyz)
    original_rotation_world = raw_pose_world_matrix[:3, :3].copy()
    roll_debug = {
        "original_top_axis_up_dot": top_axis_up_dot(original_rotation_world, args.candidate_camera_top_axis),
        "camera_up_flip_applied": 0,
        "forward_axis_change_deg": 0.0,
    }
    if bool(args.candidate_keep_camera_up):
        pose_world_matrix[:3, :3], roll_debug = constrain_roll_keep_top_axis_up(
            pose_world_matrix[:3, :3],
            top_axis=args.candidate_camera_top_axis,
        )
        quat = base.quat_xyzw_to_wxyz(R.from_matrix(base.orthonormalize_rotation(pose_world_matrix[:3, :3])).as_quat())
        pose_world_wxyz = np.concatenate([pose_world_matrix[:3, 3], quat]).astype(np.float64)
    return raw_pose_world_wxyz, raw_pose_world_matrix, pose_world_wxyz, pose_world_matrix, roll_debug


def top_axis_up_dot(rotation_world: np.ndarray, top_axis: str) -> float:
    axis_idx = 1 if top_axis == "y" else 2
    axis_vec = np.asarray(rotation_world, dtype=np.float64).reshape(3, 3)[:, axis_idx]
    return float(np.dot(axis_vec, np.array([0.0, 0.0, 1.0], dtype=np.float64)))


def forward_axis_change_deg(rotation_a: np.ndarray, rotation_b: np.ndarray) -> float:
    x_a = np.asarray(rotation_a, dtype=np.float64).reshape(3, 3)[:, 0]
    x_b = np.asarray(rotation_b, dtype=np.float64).reshape(3, 3)[:, 0]
    dot = float(np.clip(np.dot(x_a, x_b) / max(np.linalg.norm(x_a) * np.linalg.norm(x_b), 1e-12), -1.0, 1.0))
    return float(np.rad2deg(np.arccos(dot)))


def constrain_roll_keep_top_axis_up(rotation_world: np.ndarray, top_axis: str) -> Tuple[np.ndarray, Dict[str, float]]:
    rot = base.orthonormalize_rotation(rotation_world)
    roll_flip_180 = np.diag([1.0, -1.0, -1.0]).astype(np.float64)
    rot_flipped = base.orthonormalize_rotation(rot @ roll_flip_180)
    base_dot = top_axis_up_dot(rot, top_axis)
    flipped_dot = top_axis_up_dot(rot_flipped, top_axis)
    if flipped_dot > base_dot + 1e-9:
        chosen = rot_flipped
        flip_applied = 1
    else:
        chosen = rot
        flip_applied = 0
    return chosen, {
        "original_top_axis_up_dot": float(base_dot),
        "camera_up_flip_applied": int(flip_applied),
        "forward_axis_change_deg": float(forward_axis_change_deg(rot, chosen)),
    }


def build_candidate_pose_variant(
    candidate: CandidatePose,
    args: argparse.Namespace,
    *,
    flip_roll_180: bool,
    selection_mode: str,
) -> CandidatePose:
    pose_world_wxyz = np.asarray(candidate.raw_pose_world_wxyz, dtype=np.float64).reshape(7).copy()
    if abs(float(args.candidate_target_local_x_offset_m)) > 1e-12:
        pose_world_wxyz = shift_pose_along_local_x(pose_world_wxyz, float(args.candidate_target_local_x_offset_m))
    if abs(float(args.candidate_target_local_z_offset_m)) > 1e-12:
        pose_world_wxyz = shift_pose_along_local_z(pose_world_wxyz, float(args.candidate_target_local_z_offset_m))
    pose_world_matrix = pose_wxyz_to_matrix(pose_world_wxyz)
    rot = base.orthonormalize_rotation(pose_world_matrix[:3, :3])
    if bool(flip_roll_180):
        rot = base.orthonormalize_rotation(rot @ np.diag([1.0, -1.0, -1.0]).astype(np.float64))
    pose_world_matrix[:3, :3] = rot
    quat = base.quat_xyzw_to_wxyz(R.from_matrix(rot).as_quat())
    pose_world_wxyz = np.concatenate([pose_world_matrix[:3, 3], quat]).astype(np.float64)
    return replace(
        candidate,
        pose_world_wxyz=pose_world_wxyz,
        pose_world_matrix=pose_world_matrix,
        top_axis_up_dot=top_axis_up_dot(rot, args.candidate_camera_top_axis),
        original_top_axis_up_dot=top_axis_up_dot(base.orthonormalize_rotation(candidate.raw_pose_world_matrix[:3, :3]), args.candidate_camera_top_axis),
        camera_up_flip_applied=int(bool(flip_roll_180)),
        forward_axis_change_deg=float(
            forward_axis_change_deg(
                base.orthonormalize_rotation(candidate.raw_pose_world_matrix[:3, :3]),
                rot,
            )
        ),
        camera_up_selection_mode=selection_mode,
    )


def choose_roll_variant_with_previous(
    previous_rotation_world: np.ndarray,
    candidate: CandidatePose,
    args: argparse.Namespace,
) -> CandidatePose:
    variants = [
        build_candidate_pose_variant(candidate, args, flip_roll_180=False, selection_mode="follow_previous_base"),
        build_candidate_pose_variant(candidate, args, flip_roll_180=True, selection_mode="follow_previous_flip180"),
    ]
    prev_rot = base.orthonormalize_rotation(previous_rotation_world)
    variants.sort(
        key=lambda cand: (
            rotation_distance_deg(prev_rot, cand.pose_world_matrix[:3, :3]),
            -float(cand.top_axis_up_dot),
        )
    )
    return variants[0]


def postprocess_selected_keyframe_rolls(
    selected_keyframes: Sequence[SelectedKeyframe],
    args: argparse.Namespace,
) -> List[SelectedKeyframe]:
    if not bool(args.candidate_keep_camera_up):
        return list(selected_keyframes)
    processed: List[SelectedKeyframe] = []
    previous_rotation_world: Optional[np.ndarray] = None
    for idx, item in enumerate(selected_keyframes):
        if idx == 0:
            chosen_rot, roll_debug = constrain_roll_keep_top_axis_up(
                item.candidate.pose_world_matrix[:3, :3],
                top_axis=args.candidate_camera_top_axis,
            )
            chosen_matrix = np.asarray(item.candidate.pose_world_matrix, dtype=np.float64).copy()
            chosen_matrix[:3, :3] = chosen_rot
            quat = base.quat_xyzw_to_wxyz(R.from_matrix(chosen_rot).as_quat())
            chosen_pose = np.concatenate([chosen_matrix[:3, 3], quat]).astype(np.float64)
            chosen_candidate = replace(
                item.candidate,
                pose_world_wxyz=chosen_pose,
                pose_world_matrix=chosen_matrix,
                top_axis_up_dot=top_axis_up_dot(chosen_rot, args.candidate_camera_top_axis),
                original_top_axis_up_dot=float(roll_debug["original_top_axis_up_dot"]),
                camera_up_flip_applied=int(roll_debug["camera_up_flip_applied"]),
                forward_axis_change_deg=float(roll_debug["forward_axis_change_deg"]),
                camera_up_selection_mode="keyframe1_keep_up",
            )
        else:
            chosen_candidate = choose_roll_variant_with_previous(previous_rotation_world, item.candidate, args)
        processed.append(
            SelectedKeyframe(
                source_frame=item.source_frame,
                arm=item.arm,
                candidate=chosen_candidate,
                hand_rotation_cam=item.hand_rotation_cam,
            )
        )
        previous_rotation_world = np.asarray(chosen_candidate.pose_world_matrix[:3, :3], dtype=np.float64)
    return processed


def nearest_object_name(candidate_world_matrix: np.ndarray, object_states: Dict[str, ObjectState]) -> Tuple[str, float]:
    pos = np.asarray(candidate_world_matrix[:3, 3], dtype=np.float64)
    best_name = ""
    best_dist = float("inf")
    for name, state in object_states.items():
        if not state.visible or not np.isfinite(state.pose_world_matrix).all():
            continue
        obj_pos = np.asarray(state.pose_world_matrix[:3, 3], dtype=np.float64)
        dist = float(np.linalg.norm(pos - obj_pos))
        if dist < best_dist:
            best_dist = dist
            best_name = name
    if not best_name:
        raise RuntimeError("No visible objects found for nearest-object matching.")
    return best_name, best_dist


def nearest_valid_hand_frame(frame: int, hand_valid: np.ndarray) -> Optional[int]:
    valid_indices = np.where(np.asarray(hand_valid, dtype=bool))[0]
    if valid_indices.size == 0:
        return None
    return int(valid_indices[np.argmin(np.abs(valid_indices - int(frame)))])


def build_ranked_candidates_for_arm(
    renderer: ReplayRenderer,
    args: argparse.Namespace,
    anygrasp_dir: Path,
    hand_data: Dict[str, np.ndarray],
    keyframes: Sequence[int],
    arm: str,
    object_states_per_frame: Dict[int, Dict[str, ObjectState]],
) -> Tuple[Optional[Dict[int, List[CandidatePose]]], Dict[int, Dict[str, int]], Dict[int, List[CandidatePose]]]:
    hand_valid = np.asarray(hand_data[f"{arm}_gripper_valid"], dtype=bool)
    hand_rotations = np.asarray(hand_data[f"{arm}_gripper_rotation_matrix"], dtype=np.float64)

    expected_object = expected_object_for_arm(args, arm)
    candidates_per_frame: Dict[int, List[CandidatePose]] = {}
    all_candidates_per_frame: Dict[int, List[CandidatePose]] = {}
    diagnostics: Dict[int, Dict[str, int]] = {}
    for frame in keyframes:
        ref_frame = int(frame) if bool(hand_valid[frame]) else nearest_valid_hand_frame(int(frame), hand_valid)
        if ref_frame is None:
            diagnostics[int(frame)] = {"hand_valid": 0, "reference_hand_frame": -1}
            return None, diagnostics, all_candidates_per_frame
        ref_rotation = np.asarray(hand_rotations[ref_frame], dtype=np.float64)
        all_candidates = []
        raw_candidates = []
        total = 0
        object_pass = 0
        distance_pass = 0
        for candidate_idx, grasp in enumerate(load_anygrasp_candidates(anygrasp_dir, frame)):
            total += 1
            raw_pose_world_wxyz, raw_pose_world_matrix, pose_world_wxyz, pose_world_matrix, roll_debug = candidate_to_world_pose(renderer, args, grasp)
            nearest_name, nearest_dist = nearest_object_name(pose_world_matrix, object_states_per_frame[frame])
            candidate = CandidatePose(
                candidate_idx=candidate_idx,
                score=float(grasp["score"]),
                translation_cam=np.asarray(grasp["translation"], dtype=np.float64).reshape(3),
                rotation_cam=np.asarray(grasp["rotation_matrix"], dtype=np.float64).reshape(3, 3),
                width_m=float(grasp.get("width", 0.08)),
                depth_m=float(grasp.get("depth", 0.04)),
                raw_pose_world_wxyz=raw_pose_world_wxyz,
                raw_pose_world_matrix=raw_pose_world_matrix,
                pose_world_wxyz=pose_world_wxyz,
                pose_world_matrix=pose_world_matrix,
                nearest_object=nearest_name,
                nearest_object_distance_m=nearest_dist,
                rotation_distance_deg=rotation_distance_deg(ref_rotation, grasp["rotation_matrix"]),
                top_axis_up_dot=top_axis_up_dot(pose_world_matrix[:3, :3], args.candidate_camera_top_axis),
                original_top_axis_up_dot=float(roll_debug["original_top_axis_up_dot"]),
                camera_up_flip_applied=int(roll_debug["camera_up_flip_applied"]),
                forward_axis_change_deg=float(roll_debug["forward_axis_change_deg"]),
            )
            raw_candidates.append(candidate)
            object_ok = expected_object is None or nearest_name == expected_object
            if not object_ok:
                continue
            object_pass += 1
            if bool(args.enforce_candidate_distance_constraint) and nearest_dist > float(args.candidate_object_max_distance_m):
                continue
            distance_pass += 1
            if float(args.candidate_max_rotation_distance_deg) >= 0.0 and float(candidate.rotation_distance_deg) > float(args.candidate_max_rotation_distance_deg):
                continue
            all_candidates.append(candidate)
        raw_candidates.sort(key=lambda cand: (cand.rotation_distance_deg, cand.nearest_object_distance_m, -cand.score))
        all_candidates.sort(key=lambda cand: (cand.rotation_distance_deg, cand.nearest_object_distance_m, -cand.score))
        all_candidates_per_frame[frame] = raw_candidates
        candidates_per_frame[frame] = all_candidates
        diagnostics[int(frame)] = {
            "hand_valid": int(bool(hand_valid[frame])),
            "reference_hand_frame": int(ref_frame),
            "total_candidates": total,
            "object_pass": object_pass,
            "distance_pass": distance_pass,
            "rotation_pass": len(all_candidates),
            "raw_candidates": len(raw_candidates),
            "final_ranked": len(all_candidates),
        }
        if not all_candidates:
            return None, diagnostics, all_candidates_per_frame
    return candidates_per_frame, diagnostics, all_candidates_per_frame


def parse_manual_candidate_overrides(specs: Sequence[Sequence[str]]) -> Dict[str, Dict[int, int]]:
    overrides: Dict[str, Dict[int, int]] = {"left": {}, "right": {}}
    for spec in specs:
        if len(spec) != 3:
            raise ValueError(f"Invalid --manual_candidate entry: {spec}")
        frame = int(spec[0])
        arm = str(spec[1]).strip().lower()
        candidate_idx = int(spec[2])
        if arm not in overrides:
            raise ValueError(f"Unsupported arm in --manual_candidate: {arm}")
        overrides[arm][frame] = candidate_idx
    return overrides


def find_candidate_by_idx(candidates: Sequence[CandidatePose], candidate_idx: int) -> Optional[CandidatePose]:
    for cand in candidates:
        if int(cand.candidate_idx) == int(candidate_idx):
            return cand
    return None


def reorder_candidates_with_manual_choice(candidates: Sequence[CandidatePose], candidate_idx: Optional[int]) -> List[CandidatePose]:
    ordered = list(candidates)
    if candidate_idx is None:
        return ordered
    chosen = find_candidate_by_idx(ordered, int(candidate_idx))
    if chosen is None:
        return ordered
    return [chosen] + [cand for cand in ordered if int(cand.candidate_idx) != int(candidate_idx)]


def select_top_score_candidates_for_frames(
    candidates_per_frame: Dict[int, List[CandidatePose]],
    hand_rotations: np.ndarray,
    keyframes: Sequence[int],
    arm: str,
    args: argparse.Namespace,
) -> Optional[List[SelectedKeyframe]]:
    selected: List[SelectedKeyframe] = []
    for frame in keyframes:
        frame_candidates = list(candidates_per_frame.get(int(frame), []))
        if not frame_candidates:
            return None
        frame_candidates.sort(
            key=lambda cand: (
                -float(cand.score),
                float(cand.rotation_distance_deg),
                float(cand.nearest_object_distance_m),
                int(cand.candidate_idx),
            )
        )
        chosen = frame_candidates[0]
        selected.append(
            SelectedKeyframe(
                source_frame=int(frame),
                arm=arm,
                candidate=chosen,
                hand_rotation_cam=np.asarray(hand_rotations[int(frame)], dtype=np.float64),
            )
        )
    return postprocess_selected_keyframe_rolls(selected, args)


def select_keyframes_for_arm(
    renderer: ReplayRenderer,
    args: argparse.Namespace,
    anygrasp_dir: Path,
    hand_data: Dict[str, np.ndarray],
    keyframes: Sequence[int],
    arm: str,
    object_states_per_frame: Dict[int, Dict[str, ObjectState]],
) -> Optional[ArmSelectionResult]:
    hand_rotations = np.asarray(hand_data[f"{arm}_gripper_rotation_matrix"], dtype=np.float64)
    candidates_per_frame, diagnostics, all_candidates_per_frame = build_ranked_candidates_for_arm(
        renderer=renderer,
        args=args,
        anygrasp_dir=anygrasp_dir,
        hand_data=hand_data,
        keyframes=keyframes,
        arm=arm,
        object_states_per_frame=object_states_per_frame,
    )
    if candidates_per_frame is None:
        return None

    manual_overrides = getattr(args, "manual_candidate_overrides", {}).get(arm, {})
    for frame in keyframes:
        frame = int(frame)
        manual_idx = manual_overrides.get(frame)
        candidates_per_frame[frame] = reorder_candidates_with_manual_choice(candidates_per_frame.get(frame, []), manual_idx)
        all_candidates_per_frame[frame] = reorder_candidates_with_manual_choice(all_candidates_per_frame.get(frame, []), manual_idx)
        diagnostics.setdefault(frame, {})["manual_candidate_idx"] = -1 if manual_idx is None else int(manual_idx)
        diagnostics[frame]["manual_candidate_found"] = int(find_candidate_by_idx(all_candidates_per_frame.get(frame, []), manual_idx) is not None) if manual_idx is not None else 0

    if all(int(frame) in manual_overrides for frame in keyframes):
        manual_selection: List[SelectedKeyframe] = []
        for frame in keyframes:
            frame = int(frame)
            chosen = find_candidate_by_idx(all_candidates_per_frame.get(frame, []), int(manual_overrides[frame]))
            if chosen is None:
                return None
            manual_selection.append(
                SelectedKeyframe(
                    source_frame=frame,
                    arm=arm,
                    candidate=chosen,
                    hand_rotation_cam=np.asarray(hand_rotations[frame], dtype=np.float64),
                )
            )
        manual_selection = postprocess_selected_keyframe_rolls(manual_selection, args)
        return ArmSelectionResult(
            arm=arm,
            expected_object=expected_object_for_arm(args, arm),
            selected_keyframes=manual_selection,
            ranked_candidates_per_frame=candidates_per_frame,
            all_candidates_per_frame=all_candidates_per_frame,
            diagnostics=diagnostics,
        )

    if str(args.candidate_selection_mode) == "top_score_auto":
        top_score_selection = select_top_score_candidates_for_frames(
            candidates_per_frame=candidates_per_frame,
            hand_rotations=hand_rotations,
            keyframes=keyframes,
            arm=arm,
            args=args,
        )
        if top_score_selection is None:
            return None
        for item in top_score_selection:
            diagnostics.setdefault(int(item.source_frame), {})["top_score_auto_candidate_idx"] = int(item.candidate.candidate_idx)
        return ArmSelectionResult(
            arm=arm,
            expected_object=expected_object_for_arm(args, arm),
            selected_keyframes=top_score_selection,
            ranked_candidates_per_frame=candidates_per_frame,
            all_candidates_per_frame=all_candidates_per_frame,
            diagnostics=diagnostics,
        )

    best_selection: Optional[List[SelectedKeyframe]] = None
    best_cost = float("inf")
    for first_candidate in candidates_per_frame[keyframes[0]]:
        object_name = first_candidate.nearest_object
        second_candidates = [cand for cand in candidates_per_frame[keyframes[1]] if cand.nearest_object == object_name]
        if not second_candidates:
            continue
        best_second = second_candidates[0]
        total_cost = (
            first_candidate.rotation_distance_deg
            + best_second.rotation_distance_deg
            + 10.0 * (first_candidate.nearest_object_distance_m + best_second.nearest_object_distance_m)
        )
        if total_cost < best_cost:
            best_cost = total_cost
            best_selection = [
                SelectedKeyframe(
                    source_frame=int(keyframes[0]),
                    arm=arm,
                    candidate=first_candidate,
                    hand_rotation_cam=np.asarray(hand_rotations[keyframes[0]], dtype=np.float64),
                ),
                SelectedKeyframe(
                    source_frame=int(keyframes[1]),
                    arm=arm,
                    candidate=best_second,
                    hand_rotation_cam=np.asarray(hand_rotations[keyframes[1]], dtype=np.float64),
                ),
            ]
    if best_selection is None:
        return None
    best_selection = postprocess_selected_keyframe_rolls(best_selection, args)
    return ArmSelectionResult(
        arm=arm,
        expected_object=expected_object_for_arm(args, arm),
        selected_keyframes=best_selection,
        ranked_candidates_per_frame=candidates_per_frame,
        all_candidates_per_frame=all_candidates_per_frame,
        diagnostics=diagnostics,
    )


def choose_keyframes(
    renderer: ReplayRenderer,
    args: argparse.Namespace,
    anygrasp_dir: Path,
    hand_data: Dict[str, np.ndarray],
    keyframes: Sequence[int],
    arm_mode: str,
    object_states_per_frame: Dict[int, Dict[str, ObjectState]],
) -> Tuple[ArmSelectionResult, Dict[str, ArmDebugInfo]]:
    candidate_arms = [arm_mode] if arm_mode in ("left", "right") else ["left", "right"]
    best: Optional[ArmSelectionResult] = None
    best_metric: Optional[Tuple[float, float]] = None
    diagnostic_lines: List[str] = []
    arm_debugs: Dict[str, ArmDebugInfo] = {}
    for arm in candidate_arms:
        selection = select_keyframes_for_arm(renderer, args, anygrasp_dir, hand_data, keyframes, arm, object_states_per_frame)
        if selection is None:
            expected_object = expected_object_for_arm(args, arm)
            ranked, diagnostics, all_candidates = build_ranked_candidates_for_arm(
                renderer=renderer,
                args=args,
                anygrasp_dir=anygrasp_dir,
                hand_data=hand_data,
                keyframes=keyframes,
                arm=arm,
                object_states_per_frame=object_states_per_frame,
            )
            arm_debugs[arm] = ArmDebugInfo(
                arm=arm,
                expected_object=expected_object,
                ranked_candidates_per_frame=ranked or {},
                all_candidates_per_frame=all_candidates,
                diagnostics=diagnostics,
                selected_keyframes=None,
            )
            diag_chunks = []
            for frame in keyframes:
                info = diagnostics.get(int(frame), {})
                diag_chunks.append(
                    f"frame={int(frame)} hand_valid={info.get('hand_valid', 0)} "
                    f"total={info.get('total_candidates', 0)} "
                    f"object_pass={info.get('object_pass', 0)} "
                    f"distance_pass={info.get('distance_pass', 0)} "
                    f"final={info.get('final_ranked', 0)}"
                )
            diagnostic_lines.append(
                f"arm={arm} expected_object={expected_object} " + " | ".join(diag_chunks)
            )
            continue
        arm_debugs[arm] = ArmDebugInfo(
            arm=arm,
            expected_object=selection.expected_object,
            ranked_candidates_per_frame=selection.ranked_candidates_per_frame,
            all_candidates_per_frame=selection.all_candidates_per_frame,
            diagnostics=selection.diagnostics,
            selected_keyframes=selection.selected_keyframes,
        )
        total_rotation = float(sum(item.candidate.rotation_distance_deg for item in selection.selected_keyframes))
        total_score = float(sum(item.candidate.score for item in selection.selected_keyframes))
        if str(args.candidate_selection_mode) == "top_score_auto":
            metric = (-total_score, total_rotation)
        else:
            metric = (total_rotation, -total_score)
        if best_metric is None or metric < best_metric:
            best_metric = metric
            best = selection
    if best is None:
        diag_msg = "; ".join(diagnostic_lines) if diagnostic_lines else "no diagnostics"
        raise RuntimeError(
            "Failed to find valid AnyGrasp candidates after object/distance/orientation filtering. "
            f"Diagnostics: {diag_msg}"
        )
    return best, arm_debugs


def set_actor_pose(actor: sapien.Entity, pose_world_wxyz: np.ndarray) -> None:
    pose_world_wxyz = np.asarray(pose_world_wxyz, dtype=np.float64).reshape(7)
    actor.set_pose(sapien.Pose(pose_world_wxyz[:3], base.normalize_quat_wxyz(pose_world_wxyz[3:])))


def hide_actor(actor: sapien.Entity) -> None:
    actor.set_pose(base.HIDDEN_DEBUG_POSE)


def create_debug_collision_bbox_actor(
    scene: sapien.Scene,
    name: str,
    center: Sequence[float],
    half_size: Sequence[float],
    color: Sequence[float] = (0.2, 1.0, 0.2),
) -> sapien.Entity:
    builder = scene.create_actor_builder()
    builder.add_box_visual(
        pose=sapien.Pose(np.asarray(center, dtype=np.float64).reshape(3).tolist()),
        half_size=np.asarray(half_size, dtype=np.float64).reshape(3).tolist(),
        material=list(color),
    )
    actor = builder.build_kinematic(name=name)
    hide_actor(actor)
    return actor


def set_object_state_pose(state: ObjectState, pose_world_wxyz: np.ndarray) -> None:
    if state.actor is not None:
        set_actor_pose(state.actor, pose_world_wxyz)
    if state.collision_bbox_actor is not None:
        set_actor_pose(state.collision_bbox_actor, pose_world_wxyz)


def hide_object_state(state: ObjectState) -> None:
    if state.actor is not None:
        hide_actor(state.actor)
    if state.collision_bbox_actor is not None:
        hide_actor(state.collision_bbox_actor)


def set_object_collision_for_names(object_states: Dict[str, ObjectState], object_names: Sequence[str], enabled: bool) -> None:
    for object_name in object_names:
        state = object_states.get(str(object_name))
        if state is None or state.actor is None:
            continue
        state.collision_groups_cache = set_actor_collision_enabled(
            state.actor,
            enabled=bool(enabled),
            cached_groups=state.collision_groups_cache,
        )


def _get_gripper_joint_positions(renderer: ReplayRenderer, arm: str) -> np.ndarray:
    if renderer.robot is None:
        return np.zeros(0, dtype=np.float64)
    joints = renderer.robot.left_gripper if arm == "left" else renderer.robot.right_gripper
    entity = renderer.robot.left_entity if arm == "left" else renderer.robot.right_entity
    active_joints = list(entity.get_active_joints()) if entity is not None else []
    full_qpos = np.asarray(entity.get_qpos(), dtype=np.float64).reshape(-1) if entity is not None else np.zeros(0, dtype=np.float64)
    positions: List[float] = []
    for joint_info in joints:
        joint = joint_info[0]
        qpos_value = 0.0
        cursor = 0
        for active_joint in active_joints:
            dof = int(active_joint.get_dof())
            if active_joint is joint:
                if dof > 0 and cursor + dof <= full_qpos.size:
                    qpos_value = float(full_qpos[cursor])
                break
            cursor += max(dof, 0)
        positions.append(qpos_value)
    return np.asarray(positions, dtype=np.float64)


def _get_gripper_link_entities(renderer: ReplayRenderer, arm: str) -> List[sapien.Entity]:
    if renderer.robot is None:
        return []
    joints = renderer.robot.left_gripper if arm == "left" else renderer.robot.right_gripper
    return [joint_info[0].child_link for joint_info in joints if getattr(joint_info[0], "child_link", None) is not None]


def _get_gripper_base_entity(renderer: ReplayRenderer, arm: str) -> Optional[sapien.Entity]:
    if renderer.robot is None:
        return None
    ee_joint = renderer.robot.left_ee if arm == "left" else renderer.robot.right_ee
    return getattr(ee_joint, "child_link", None)


def _get_all_robot_link_entities(renderer: ReplayRenderer, arm: str) -> List[sapien.Entity]:
    if renderer.robot is None:
        return []
    entity = renderer.robot.left_entity if arm == "left" else renderer.robot.right_entity
    if entity is None:
        return []
    try:
        return list(entity.get_links())
    except Exception:
        return []


def _get_contact_monitor_entities(renderer: ReplayRenderer, arm: str, monitor_mode: str) -> List[sapien.Entity]:
    finger_entities = list(_get_gripper_link_entities(renderer, arm))
    if monitor_mode == "all_robot_links":
        all_links = _get_all_robot_link_entities(renderer, arm)
        unique_entities: List[sapien.Entity] = []
        seen_ids = set()
        for entity in all_links:
            entity_id = id(entity)
            if entity_id in seen_ids:
                continue
            seen_ids.add(entity_id)
            unique_entities.append(entity)
        return unique_entities
    if monitor_mode == "fingers_and_base":
        base_entity = _get_gripper_base_entity(renderer, arm)
        if base_entity is not None:
            finger_entities.append(base_entity)
    unique_entities = []
    seen_ids = set()
    for entity in finger_entities:
        entity_id = id(entity)
        if entity_id in seen_ids:
            continue
        seen_ids.add(entity_id)
        unique_entities.append(entity)
    return unique_entities


def _entity_name(entity: Optional[sapien.Entity]) -> str:
    if entity is None:
        return "none"
    getter = getattr(entity, "get_name", None)
    if getter is None:
        return type(entity).__name__
    try:
        return str(getter())
    except Exception:
        return type(entity).__name__


def _collision_shape_type_name(shape: object) -> str:
    return type(shape).__name__


def _summarize_entity_collision(entity: Optional[sapien.Entity]) -> str:
    if entity is None:
        return "none"
    shapes = _get_actor_collision_shapes(entity)
    if not shapes:
        return f"{_entity_name(entity)}(shapes=0)"
    type_names = ",".join(sorted({_collision_shape_type_name(shape) for shape in shapes}))
    return f"{_entity_name(entity)}(shapes={len(shapes)},types={type_names})"


def _summarize_entities_collision(entities: Sequence[sapien.Entity]) -> str:
    if not entities:
        return "none"
    return "|".join(_summarize_entity_collision(entity) for entity in entities)


def _contact_pairs_involving_entities(
    renderer: ReplayRenderer,
    monitored_entities: Sequence[sapien.Entity],
    target_entity: Optional[sapien.Entity],
) -> List[str]:
    if target_entity is None:
        return []
    monitored_set = set(monitored_entities)
    if not monitored_set:
        return []
    pairs: List[str] = []
    for contact in renderer.scene.get_contacts():
        entities: List[Optional[sapien.Entity]] = []
        if hasattr(contact, "bodies") and contact.bodies:
            for body in contact.bodies:
                entities.append(getattr(body, "entity", None))
        else:
            entities.append(getattr(contact, "actor0", None))
            entities.append(getattr(contact, "actor1", None))
        if len(entities) != 2:
            continue
        e0, e1 = entities
        if e0 is target_entity and e1 in monitored_set:
            pairs.append(f"{_entity_name(e1)}<->{_entity_name(e0)}")
        elif e1 is target_entity and e0 in monitored_set:
            pairs.append(f"{_entity_name(e0)}<->{_entity_name(e1)}")
    return sorted(set(pairs))


def _contact_involves_entities(
    renderer: ReplayRenderer,
    gripper_entities: Sequence[sapien.Entity],
    target_entity: Optional[sapien.Entity],
) -> bool:
    return bool(_contact_pairs_involving_entities(renderer, gripper_entities, target_entity))


def _raw_contact_records_for_target(
    renderer: ReplayRenderer,
    target_entity: Optional[sapien.Entity],
) -> List[Dict[str, object]]:
    if target_entity is None:
        return []
    records: List[Dict[str, object]] = []
    for contact in renderer.scene.get_contacts():
        entities: List[Optional[sapien.Entity]] = []
        if hasattr(contact, "bodies") and contact.bodies:
            for body in contact.bodies:
                entities.append(getattr(body, "entity", None))
        else:
            entities.append(getattr(contact, "actor0", None))
            entities.append(getattr(contact, "actor1", None))
        if len(entities) != 2:
            continue
        e0, e1 = entities
        if e0 is not target_entity and e1 is not target_entity:
            continue
        records.append(
            {
                "pair": f"{_entity_name(e0)}<->{_entity_name(e1)}",
                "entity0": _entity_name(e0),
                "entity1": _entity_name(e1),
                "n_points": int(len(getattr(contact, "points", []) or [])),
            }
        )
    unique_records: List[Dict[str, object]] = []
    seen = set()
    for item in records:
        key = (str(item["pair"]), int(item["n_points"]))
        if key in seen:
            continue
        seen.add(key)
        unique_records.append(item)
    return unique_records


def _entity_pose_summary(entity: Optional[sapien.Entity]) -> str:
    if entity is None:
        return "none"
    try:
        pose = entity.get_pose()
    except Exception as exc:
        return f"{_entity_name(entity)}(pose_error={exc})"
    p = np.round(np.asarray(pose.p, dtype=np.float64), 4).tolist()
    q = np.round(base.normalize_quat_wxyz(np.asarray(pose.q, dtype=np.float64)), 4).tolist()
    return f"{_entity_name(entity)}(p={p},q={q})"


def _entity_pose_record(entity: Optional[sapien.Entity]) -> Optional[Dict[str, object]]:
    if entity is None:
        return None
    try:
        pose = entity.get_pose()
    except Exception as exc:
        return {"name": _entity_name(entity), "pose_error": repr(exc)}
    return {
        "name": _entity_name(entity),
        "position": np.round(np.asarray(pose.p, dtype=np.float64), 6).tolist(),
        "quat_wxyz": np.round(base.normalize_quat_wxyz(np.asarray(pose.q, dtype=np.float64)), 6).tolist(),
    }


def export_close_stage_snapshot(
    output_dir: Path,
    tag: str,
    renderer: ReplayRenderer,
    object_states: Dict[str, ObjectState],
    selected_objects_by_arm: Dict[str, str],
    arms: Sequence[str],
) -> Path:
    payload: Dict[str, object] = {
        "tag": str(tag),
        "arms": list(arms),
        "objects_by_arm": dict(selected_objects_by_arm),
        "arm_records": {},
    }
    for arm in arms:
        obj_name = selected_objects_by_arm.get(arm)
        state = object_states.get(str(obj_name)) if obj_name is not None else None
        finger_entities = _get_gripper_link_entities(renderer, arm)
        base_entity = _get_gripper_base_entity(renderer, arm)
        arm_record: Dict[str, object] = {
            "tcp_pose_world": np.round(renderer.get_current_tcp_pose(arm), 6).tolist(),
            "gripper_joint_qpos": np.round(_get_gripper_joint_positions(renderer, arm), 6).tolist(),
            "gripper_base": _entity_pose_record(base_entity),
            "finger_links": [_entity_pose_record(entity) for entity in finger_entities],
        }
        if state is not None:
            arm_record["object_actor_pose"] = _entity_pose_record(state.actor)
            arm_record["object_visual_pose_world"] = np.round(np.asarray(state.pose_world_wxyz, dtype=np.float64), 6).tolist()
            arm_record["object_collision_mode"] = str(state.collision_mode)
            arm_record["object_visual_scale"] = np.round(np.asarray(state.visual_scale, dtype=np.float64), 6).tolist()
            arm_record["object_collision_scale"] = np.round(np.asarray(state.collision_scale, dtype=np.float64), 6).tolist()
            arm_record["object_collision_debug_info"] = state.collision_debug_info
        payload["arm_records"][arm] = arm_record
    path = output_dir / f"close_stage_snapshot_{tag}.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[close-stage-snapshot] wrote {path}")
    return path


def close_grippers_progressively_with_collision_stop(
    renderer: ReplayRenderer,
    left_target: Optional[float],
    right_target: Optional[float],
    object_actor_by_arm: Dict[str, Optional[sapien.Entity]],
    *,
    object_debug_by_arm: Optional[Dict[str, Dict[str, object]]] = None,
    command_step: float = 0.05,
    settle_steps_per_iter: int = 6,
    max_iters: int = 40,
    stall_qpos_tol: float = 5e-5,
    contact_confirm_iters: int = 2,
    debug_collision_report: bool = False,
    gripper_contact_monitor_mode: str = "fingers",
) -> Dict[str, Dict[str, object]]:
    if renderer.robot is None:
        return {}

    state_by_arm: Dict[str, Dict[str, object]] = {}
    for arm, target in (("left", left_target), ("right", right_target)):
        if target is None:
            continue
        target = float(np.clip(target, 0.0, 1.0))
        current_cmd = float(renderer.robot.get_left_gripper_val() if arm == "left" else renderer.robot.get_right_gripper_val())
        state_by_arm[arm] = {
            "target": target,
            "current_cmd": current_cmd,
            "last_qpos": _get_gripper_joint_positions(renderer, arm),
            "contact_iters": 0,
            "had_base_contact": False,
            "had_raw_target_contact": False,
            "stopped": False,
            "reason": "target_reached",
            "gripper_entities": _get_gripper_link_entities(renderer, arm),
            "gripper_base_entity": _get_gripper_base_entity(renderer, arm),
            "monitor_entities": _get_contact_monitor_entities(renderer, arm, str(gripper_contact_monitor_mode)),
            "object_actor": object_actor_by_arm.get(arm),
        }

    if not state_by_arm:
        return {}

    if bool(debug_collision_report):
        for arm, state in state_by_arm.items():
            finger_entities = list(state["gripper_entities"])  # type: ignore[arg-type]
            base_entity = state["gripper_base_entity"]  # type: ignore[assignment]
            target_entity = state["object_actor"]  # type: ignore[assignment]
            raw_target_contacts = _raw_contact_records_for_target(renderer, target_entity)
            object_debug = (object_debug_by_arm or {}).get(arm, {})
            print(
                "[collision-debug-init] "
                f"arm={arm} "
                f"monitor_mode={gripper_contact_monitor_mode} "
                f"target={_summarize_entity_collision(target_entity)} "
                f"target_pose={_entity_pose_summary(target_entity)} "
                f"target_collision_debug={object_debug if object_debug else {'none': True}} "
                f"gripper_base={_summarize_entity_collision(base_entity)} "
                f"gripper_base_pose={_entity_pose_summary(base_entity)} "
                f"finger_links={_summarize_entities_collision(finger_entities)} "
                f"monitor_entities={_summarize_entities_collision(list(state['monitor_entities']))} "
                f"raw_target_contacts={raw_target_contacts if raw_target_contacts else ['none']}"
            )

    for iter_idx in range(max_iters):
        left_cmd = None
        right_cmd = None
        pending = False
        for arm, state in state_by_arm.items():
            if bool(state["stopped"]):
                continue
            pending = True
            current_cmd = float(state["current_cmd"])
            target = float(state["target"])
            delta = target - current_cmd
            if abs(delta) <= 1e-6:
                state["stopped"] = True
                state["reason"] = "target_reached"
                continue
            next_cmd = current_cmd + float(np.sign(delta)) * min(abs(delta), command_step)
            state["current_cmd"] = float(next_cmd)
            if arm == "left":
                left_cmd = float(next_cmd)
            else:
                right_cmd = float(next_cmd)
        if not pending:
            break

        renderer.set_grippers(left_cmd, right_cmd)
        if settle_steps_per_iter > 0:
            renderer.step_scene(steps=int(settle_steps_per_iter))

        for arm, state in state_by_arm.items():
            if bool(state["stopped"]):
                continue
            current_qpos = _get_gripper_joint_positions(renderer, arm)
            prev_qpos = np.asarray(state["last_qpos"], dtype=np.float64)
            qpos_delta = float(np.max(np.abs(current_qpos - prev_qpos))) if current_qpos.size and prev_qpos.size else 0.0
            state["last_qpos"] = current_qpos
            finger_entities = list(state["gripper_entities"])  # type: ignore[arg-type]
            base_entity = state["gripper_base_entity"]  # type: ignore[assignment]
            monitored_with_base = finger_entities + ([base_entity] if base_entity is not None else [])
            monitor_entities = list(state["monitor_entities"])  # type: ignore[arg-type]
            has_contact = _contact_involves_entities(
                renderer,
                monitor_entities,
                state["object_actor"],  # type: ignore[arg-type]
            )
            base_contact = _contact_involves_entities(
                renderer,
                monitored_with_base,
                state["object_actor"],  # type: ignore[arg-type]
            )
            state["contact_iters"] = int(state["contact_iters"]) + 1 if has_contact else 0
            state["had_base_contact"] = bool(state["had_base_contact"]) or bool(base_contact)
            raw_target_contacts = _raw_contact_records_for_target(
                renderer,
                state["object_actor"],  # type: ignore[arg-type]
            )
            state["had_raw_target_contact"] = bool(state["had_raw_target_contact"]) or bool(raw_target_contacts)

            if bool(debug_collision_report):
                finger_pairs = _contact_pairs_involving_entities(
                    renderer,
                    monitor_entities,
                    state["object_actor"],  # type: ignore[arg-type]
                )
                base_pairs = _contact_pairs_involving_entities(
                    renderer,
                    monitored_with_base,
                    state["object_actor"],  # type: ignore[arg-type]
                )
                target_pose_summary = _entity_pose_summary(state["object_actor"])  # type: ignore[arg-type]
                print(
                    "[collision-debug-step] "
                    f"arm={arm} iter={iter_idx + 1} cmd={float(state['current_cmd']):.3f} "
                    f"qpos_delta={qpos_delta:.6f} "
                    f"target_pose={target_pose_summary} "
                    f"monitor_contact={int(has_contact)} base_contact={int(base_contact)} raw_target_contact_total={len(raw_target_contacts)} "
                    f"monitor_pairs={finger_pairs if finger_pairs else ['none']} "
                    f"base_pairs={base_pairs if base_pairs else ['none']} "
                    f"raw_target_contacts={raw_target_contacts if raw_target_contacts else ['none']}"
                )

            target = float(state["target"])
            current_cmd = float(state["current_cmd"])
            if abs(target - current_cmd) <= 1e-6:
                state["stopped"] = True
                state["reason"] = "target_reached"
            elif has_contact and qpos_delta <= stall_qpos_tol and int(state["contact_iters"]) >= contact_confirm_iters:
                state["stopped"] = True
                state["reason"] = "contact_stall"

        if all(bool(state["stopped"]) for state in state_by_arm.values()):
            break

    result: Dict[str, Dict[str, object]] = {}
    for arm, state in state_by_arm.items():
        qpos = np.asarray(state["last_qpos"], dtype=np.float64)
        result[arm] = {
            "final_cmd": float(state["current_cmd"]),
            "target": float(state["target"]),
            "actual_qpos_max_abs": float(np.max(np.abs(qpos))) if qpos.size else 0.0,
            "contact_iters": int(state["contact_iters"]),
            "had_contact": bool(int(state["contact_iters"]) > 0),
            "had_base_contact": bool(state["had_base_contact"]),
            "had_raw_target_contact": bool(state["had_raw_target_contact"]),
            "reason": str(state["reason"]),
            "monitor_mode": str(gripper_contact_monitor_mode),
        }
    return result


def create_colored_axis_actor(
    scene: sapien.Scene,
    name: str,
    axis_length: float,
    thickness: float,
    colors: Tuple[Sequence[float], Sequence[float], Sequence[float]],
) -> sapien.Entity:
    builder = scene.create_actor_builder()
    axis_half = axis_length * 0.5
    builder.add_sphere_visual(radius=thickness * 1.8, material=[1.0, 1.0, 1.0])
    builder.add_box_visual(pose=sapien.Pose([axis_half, 0.0, 0.0]), half_size=[axis_half, thickness, thickness], material=list(colors[0]))
    builder.add_box_visual(pose=sapien.Pose([0.0, axis_half, 0.0]), half_size=[thickness, axis_half, thickness], material=list(colors[1]))
    builder.add_box_visual(pose=sapien.Pose([0.0, 0.0, axis_half]), half_size=[thickness, thickness, axis_half], material=list(colors[2]))
    actor = builder.build_kinematic(name=name)
    hide_actor(actor)
    return actor


def create_forward_marker_actor(
    scene: sapien.Scene,
    name: str,
    axis_length: float,
    thickness: float,
    color: Sequence[float],
) -> sapien.Entity:
    builder = scene.create_actor_builder()
    axis_half = axis_length * 0.5
    builder.add_sphere_visual(radius=thickness * 2.0, material=list(color))
    builder.add_box_visual(
        pose=sapien.Pose([axis_half, 0.0, 0.0]),
        half_size=[axis_half, thickness, thickness],
        material=list(color),
    )
    actor = builder.build_kinematic(name=name)
    hide_actor(actor)
    return actor


def _get_ik_waypoint_marker_store(renderer: ReplayRenderer) -> Dict[str, List[sapien.Entity]]:
    store = getattr(renderer, "_ik_waypoint_marker_actors", None)
    if store is None:
        store = {"left": [], "right": []}
        setattr(renderer, "_ik_waypoint_marker_actors", store)
    return store


def _get_ik_waypoint_endpoint_store(renderer: ReplayRenderer) -> Dict[str, List[sapien.Entity]]:
    store = getattr(renderer, "_ik_waypoint_endpoint_actors", None)
    if store is None:
        store = {"left": [], "right": []}
        setattr(renderer, "_ik_waypoint_endpoint_actors", store)
    return store


def _ensure_ik_waypoint_marker_actors(
    renderer: ReplayRenderer,
    args: argparse.Namespace,
    arm: str,
    count: int,
) -> List[sapien.Entity]:
    store = _get_ik_waypoint_marker_store(renderer)
    actors = store.setdefault(arm, [])
    axis_length = max(float(args.debug_target_axis_length) * 0.28, 0.015)
    thickness = max(float(args.debug_target_axis_thickness) * 0.48, 0.0015)
    color = [0.15, 0.75, 1.0] if arm == "left" else [1.0, 0.65, 0.05]
    while len(actors) < count:
        idx = len(actors)
        actors.append(
            create_forward_marker_actor(
                renderer.scene,
                f"debug_ik_waypoint_{arm}_{idx}",
                axis_length=axis_length,
                thickness=thickness,
                color=color,
            )
        )
    return actors


def _ensure_ik_waypoint_endpoint_actors(
    renderer: ReplayRenderer,
    args: argparse.Namespace,
    arm: str,
    count: int,
) -> List[sapien.Entity]:
    store = _get_ik_waypoint_endpoint_store(renderer)
    actors = store.setdefault(arm, [])
    axis_length = max(float(args.debug_target_axis_length) * 0.34, 0.02)
    thickness = max(float(args.debug_target_axis_thickness) * 0.56, 0.0018)
    color = [1.0, 0.2, 0.2]
    while len(actors) < count:
        idx = len(actors)
        actors.append(
            create_forward_marker_actor(
                renderer.scene,
                f"debug_ik_waypoint_endpoint_{arm}_{idx}",
                axis_length=axis_length,
                thickness=thickness,
                color=color,
            )
        )
    return actors


def clear_ik_waypoint_visuals(renderer: ReplayRenderer) -> None:
    for attr_name in ("_ik_waypoint_marker_actors", "_ik_waypoint_endpoint_actors"):
        store = getattr(renderer, attr_name, None)
        if not store:
            continue
        for actors in store.values():
            for actor in actors:
                hide_actor(actor)


def update_ik_waypoint_visuals(
    renderer: ReplayRenderer,
    args: argparse.Namespace,
    plans_by_arm: Optional[Dict[str, Optional[Dict]]],
) -> None:
    if not bool(getattr(args, "debug_visualize_ik_waypoints", 0)):
        clear_ik_waypoint_visuals(renderer)
        return
    if not plans_by_arm:
        clear_ik_waypoint_visuals(renderer)
        return

    store = _get_ik_waypoint_marker_store(renderer)
    endpoint_store = _get_ik_waypoint_endpoint_store(renderer)
    for arm in ("left", "right"):
        plan = plans_by_arm.get(arm) if arm in plans_by_arm else None
        tcp_waypoints = None if plan is None else plan.get("tcp_waypoints_world")
        if tcp_waypoints is None:
            for actor in store.get(arm, []):
                hide_actor(actor)
            for actor in endpoint_store.get(arm, []):
                hide_actor(actor)
            continue
        waypoint_arr = np.asarray(tcp_waypoints, dtype=np.float64).reshape(-1, 7)
        intermediate = waypoint_arr[1:-1]
        actors = _ensure_ik_waypoint_marker_actors(renderer, args, arm, int(intermediate.shape[0]))
        for actor, pose_world_wxyz in zip(actors, intermediate):
            actor.set_pose(sapien.Pose(pose_world_wxyz[:3], base.normalize_quat_wxyz(pose_world_wxyz[3:])))
        for actor in actors[int(intermediate.shape[0]):]:
            hide_actor(actor)
        endpoint_poses = waypoint_arr[[0, -1]] if waypoint_arr.shape[0] >= 2 else waypoint_arr
        endpoint_actors = _ensure_ik_waypoint_endpoint_actors(renderer, args, arm, int(endpoint_poses.shape[0]))
        for actor, pose_world_wxyz in zip(endpoint_actors, endpoint_poses):
            actor.set_pose(sapien.Pose(pose_world_wxyz[:3], base.normalize_quat_wxyz(pose_world_wxyz[3:])))
        for actor in endpoint_actors[int(endpoint_poses.shape[0]):]:
            hide_actor(actor)


def create_gripper_candidate_actor(
    scene: sapien.Scene,
    name: str,
    color: Sequence[float],
    marker_side: str = "none",
    scale: float = 1.0,
    opening_width_m: float = 0.04,
    forward_axis: str = "local_x",
) -> sapien.Entity:
    builder = scene.create_actor_builder()
    scale = float(scale)
    forward_is_z = str(forward_axis) == "local_z"
    body_half = [0.004 * scale, 0.026 * scale, 0.008 * scale] if forward_is_z else [0.008 * scale, 0.026 * scale, 0.004 * scale]
    finger_half = [0.0035 * scale, 0.0035 * scale, 0.018 * scale] if forward_is_z else [0.018 * scale, 0.0035 * scale, 0.0035 * scale]
    finger_gap = float(np.clip(opening_width_m, 0.0, 0.12)) * 0.5 + finger_half[1]
    if forward_is_z:
        builder.add_box_visual(pose=sapien.Pose([0.0, 0.0, -0.012 * scale]), half_size=body_half, material=list(color))
        builder.add_box_visual(pose=sapien.Pose([0.0, finger_gap, 0.012 * scale]), half_size=finger_half, material=list(color))
        builder.add_box_visual(pose=sapien.Pose([0.0, -finger_gap, 0.012 * scale]), half_size=finger_half, material=list(color))
    else:
        builder.add_box_visual(pose=sapien.Pose([-0.012 * scale, 0.0, 0.0]), half_size=body_half, material=list(color))
        builder.add_box_visual(pose=sapien.Pose([0.012 * scale, finger_gap, 0.0]), half_size=finger_half, material=list(color))
        builder.add_box_visual(pose=sapien.Pose([0.012 * scale, -finger_gap, 0.0]), half_size=finger_half, material=list(color))
    marker_color = [0.05, 0.05, 0.05]
    if marker_side == "left":
        builder.add_box_visual(pose=sapien.Pose([0.0, 0.0, 0.012]), half_size=[0.004, 0.004, 0.002], material=marker_color)
    elif marker_side == "right":
        builder.add_box_visual(pose=sapien.Pose([0.0, 0.0, -0.012]), half_size=[0.004, 0.004, 0.002], material=marker_color)
    actor = builder.build_kinematic(name=name)
    hide_actor(actor)
    return actor



def capture_head_bgr(renderer: ReplayRenderer) -> np.ndarray:
    head_rgb, _ = renderer.capture_camera(renderer.zed_camera)
    return cv2.cvtColor(head_rgb, cv2.COLOR_RGB2BGR)


def project_world_point_to_image(
    camera,
    intrinsic: np.ndarray,
    point_world: np.ndarray,
    image_width: int,
    image_height: int,
) -> Optional[Tuple[int, int]]:
    try:
        extrinsic = np.asarray(camera.get_extrinsic_matrix(), dtype=np.float64)
    except Exception:
        return None
    if extrinsic.shape == (3, 4):
        extrinsic_4x4 = np.eye(4, dtype=np.float64)
        extrinsic_4x4[:3, :4] = extrinsic
        extrinsic = extrinsic_4x4
    if extrinsic.shape != (4, 4):
        return None

    point_h = np.ones(4, dtype=np.float64)
    point_h[:3] = np.asarray(point_world, dtype=np.float64).reshape(3)
    point_cam = extrinsic @ point_h
    z = float(point_cam[2])
    if not np.isfinite(z) or z <= 1e-6:
        return None

    pixel = np.asarray(intrinsic, dtype=np.float64) @ point_cam[:3]
    if not np.isfinite(pixel).all() or abs(pixel[2]) <= 1e-6:
        return None
    u = int(round(float(pixel[0] / pixel[2])))
    v = int(round(float(pixel[1] / pixel[2])))
    if u < -32 or u >= image_width + 32 or v < -32 or v >= image_height + 32:
        return None
    return u, v


def draw_small_candidate_label(
    image_bgr: np.ndarray,
    text: str,
    pixel_xy: Tuple[int, int],
    color_bgr: Tuple[int, int, int],
    font_scale: float = 0.20,
) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = 1
    x, y = int(pixel_xy[0]), int(pixel_xy[1])
    cv2.putText(image_bgr, text, (x, y), font, float(font_scale), color_bgr, thickness, cv2.LINE_AA)


def annotate_candidate_labels(
    image_bgr: np.ndarray,
    camera,
    intrinsic: np.ndarray,
    active_frame: Optional[int],
    common_candidates_per_frame: Dict[int, List[CandidatePose]],
    arm_display_candidates: Dict[str, Dict[int, List[CandidatePose]]],
    selected_keyframes: List[SelectedKeyframe],
) -> None:
    if active_frame is None:
        return

    width = int(image_bgr.shape[1])
    height = int(image_bgr.shape[0])
    frame = int(active_frame)

    selected_by_frame_and_arm = {
        (int(item.source_frame), item.arm): int(item.candidate.candidate_idx) for item in selected_keyframes
    }

    common_candidates = common_candidates_per_frame.get(frame, [])
    for cand in common_candidates:
        origin = np.asarray(cand.pose_world_wxyz[:3], dtype=np.float64)
        pixel = project_world_point_to_image(camera, intrinsic, origin, width, height)
        if pixel is None:
            continue
        draw_small_candidate_label(image_bgr, str(int(cand.candidate_idx)), (pixel[0] + 2, pixel[1] - 2), (0, 150, 0), font_scale=0.15)

    arm_styles = {
        "left": ((220, 120, 20), np.array([0.0, 0.0, 0.018], dtype=np.float64)),
        "right": ((0, 140, 255), np.array([0.0, 0.0, -0.018], dtype=np.float64)),
    }
    for arm_name, (label_color, local_offset) in arm_styles.items():
        frame_candidates = arm_display_candidates.get(arm_name, {}).get(frame, [])
        selected_idx = selected_by_frame_and_arm.get((frame, arm_name))
        for cand in frame_candidates:
            pose_world = np.asarray(cand.pose_world_matrix, dtype=np.float64)
            label_world = pose_world[:3, 3] + pose_world[:3, :3] @ local_offset
            pixel = project_world_point_to_image(camera, intrinsic, label_world, width, height)
            if pixel is None:
                continue
            color = (0, 0, 255) if selected_idx is not None and int(cand.candidate_idx) == selected_idx else label_color
            draw_small_candidate_label(image_bgr, str(int(cand.candidate_idx)), (pixel[0] + 2, pixel[1] - 2), color)


def selected_keyframes_for_active_frame(
    selected_keyframes: Sequence[SelectedKeyframe],
    active_frame: Optional[int],
) -> List[SelectedKeyframe]:
    if active_frame is None:
        return []
    frame = int(active_frame)
    return [item for item in selected_keyframes if int(item.source_frame) == frame]


def selected_keyframes_for_active_frames_by_arm(
    selected_keyframes: Sequence[SelectedKeyframe],
    active_frame: Optional[int],
    active_frame_by_arm: Optional[Dict[str, int]],
) -> List[SelectedKeyframe]:
    if active_frame_by_arm:
        result: List[SelectedKeyframe] = []
        for item in selected_keyframes:
            arm_frame = active_frame_by_arm.get(str(item.arm))
            if arm_frame is not None and int(item.source_frame) == int(arm_frame):
                result.append(item)
        return result
    return selected_keyframes_for_active_frame(selected_keyframes, active_frame)


def build_display_candidates_per_frame(
    args: argparse.Namespace,
    keyframes: Sequence[int],
    ranked_candidates_per_frame: Dict[int, List[CandidatePose]],
    all_candidates_per_frame: Dict[int, List[CandidatePose]],
) -> Dict[int, List[CandidatePose]]:
    display: Dict[int, List[CandidatePose]] = {}
    common_top_k = int(args.debug_common_candidate_top_k)
    for frame in keyframes:
        frame = int(frame)
        if bool(args.debug_show_all_candidates):
            ranked_common = sorted(all_candidates_per_frame.get(frame, []), key=lambda item: float(item.score), reverse=True)
            display[frame] = ranked_common if common_top_k < 0 else ranked_common[: max(common_top_k, 0)]
        else:
            display[frame] = []
    return display


def create_debug_visual_bundle(
    renderer: ReplayRenderer,
    args: argparse.Namespace,
    keyframes: Sequence[int],
    common_candidates_per_frame: Dict[int, List[CandidatePose]],
    arm_display_candidates: Dict[str, Dict[int, List[CandidatePose]]],
    selected_keyframes: List[SelectedKeyframe],
) -> DebugVisualBundle:
    axis_colors = {
        int(keyframes[0]): ([1.0, 0.35, 0.10], [1.0, 0.65, 0.15], [1.0, 0.90, 0.20]),
        int(keyframes[1]): ([0.15, 0.85, 1.0], [0.15, 0.45, 1.0], [0.55, 0.25, 1.0]),
    }
    keyframe_axis_actors = {
        (int(frame), arm_name): create_colored_axis_actor(
            renderer.scene,
            f"debug_axis_keyframe_{int(frame)}_{arm_name}",
            axis_length=float(args.debug_target_axis_length),
            thickness=float(args.debug_target_axis_thickness),
            colors=axis_colors[int(frame)],
        )
        for frame in keyframes
        for arm_name in ("left", "right")
    }
    selected_by_frame_and_arm = {
        (int(item.source_frame), item.arm): item.candidate.candidate_idx for item in selected_keyframes
    }
    common_candidate_actors = {
        int(frame): [
            create_gripper_candidate_actor(
                renderer.scene,
                f"debug_common_candidate_{int(frame)}_{rank}",
                color=[0.1, 0.85, 0.2],
                marker_side="none",
                scale=0.82,
                opening_width_m=float(_cand.width_m),
                forward_axis=str(args.debug_gripper_actor_forward_axis),
            )
            for rank, _cand in enumerate(common_candidates_per_frame.get(int(frame), []))
        ]
        for frame in keyframes
    }
    arm_candidate_actors: Dict[str, Dict[int, List[sapien.Entity]]] = {}
    arm_candidate_axis_actors: Dict[str, Dict[int, List[sapien.Entity]]] = {}
    standard_axis_colors = ([1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.4, 1.0])
    for arm_name, frame_candidates in arm_display_candidates.items():
        arm_candidate_actors[arm_name] = {}
        arm_candidate_axis_actors[arm_name] = {}
        for frame in keyframes:
            frame = int(frame)
            selected_idx = selected_by_frame_and_arm.get((frame, arm_name))
            actors = []
            axis_actors = []
            for rank, cand in enumerate(frame_candidates.get(frame, [])):
                actors.append(
                    create_gripper_candidate_actor(
                        renderer.scene,
                        f"debug_{arm_name}_candidate_{frame}_{rank}",
                        color=[1.0, 0.1, 0.1] if cand.candidate_idx == selected_idx else ([0.1, 0.45, 1.0] if arm_name == "left" else [1.0, 0.55, 0.0]),
                        marker_side="none",
                        scale=1.65 if cand.candidate_idx == selected_idx else 1.0,
                        opening_width_m=float(cand.width_m),
                        forward_axis=str(args.debug_gripper_actor_forward_axis),
                    )
                )
                axis_actors.append(
                    create_colored_axis_actor(
                        renderer.scene,
                        f"debug_{arm_name}_candidate_axis_{frame}_{rank}",
                        axis_length=float(args.debug_target_axis_length) * 0.72,
                        thickness=float(args.debug_target_axis_thickness) * 0.72,
                        colors=standard_axis_colors,
                    )
                )
            arm_candidate_actors[arm_name][frame] = actors
            arm_candidate_axis_actors[arm_name][frame] = axis_actors
    return DebugVisualBundle(
        keyframe_axis_actors=keyframe_axis_actors,
        common_candidate_actors=common_candidate_actors,
        arm_candidate_actors=arm_candidate_actors,
        arm_candidate_axis_actors=arm_candidate_axis_actors,
    )


def set_single_arm_target_visual(renderer: ReplayRenderer, arm: str, pose_world_wxyz: Optional[np.ndarray]) -> None:
    if pose_world_wxyz is None:
        renderer.update_target_axis_visuals(None, None)
        return
    if arm == "left":
        renderer.update_target_axis_visuals(pose_world_wxyz, None)
    elif arm == "right":
        renderer.update_target_axis_visuals(None, pose_world_wxyz)
    else:
        renderer.update_target_axis_visuals(None, None)


def set_dual_arm_target_visuals(
    renderer: ReplayRenderer,
    left_pose_world_wxyz: Optional[np.ndarray],
    right_pose_world_wxyz: Optional[np.ndarray],
) -> None:
    renderer.update_target_axis_visuals(left_pose_world_wxyz, right_pose_world_wxyz)


def apply_target_visuals_from_state(renderer: ReplayRenderer, debug_execution_state: Optional[DebugExecutionState]) -> None:
    if debug_execution_state is None or not debug_execution_state.target_pose_by_arm:
        renderer.update_target_axis_visuals(None, None)
        return
    left_pose = debug_execution_state.target_pose_by_arm.get("left")
    right_pose = debug_execution_state.target_pose_by_arm.get("right")
    renderer.update_target_axis_visuals(left_pose, right_pose)


def make_skipped_stage_result(reason: str) -> Dict[str, object]:
    return {
        "status": f"skipped:{reason}",
        "attempts": 0,
        "reached": False,
        "pos_err_m": float("inf"),
        "rot_err_deg": float("inf"),
        "attempt_history": [],
    }


def stage_attempt_budget(args: argparse.Namespace) -> Optional[int]:
    if bool(args.replan_until_reached):
        max_attempts = int(args.replan_until_reached_max_attempts)
        return None if max_attempts <= 0 else max(max_attempts, 1)
    return max(int(args.max_stage_replans), 1)


def stage_result_failed(stage_result: Dict[str, object]) -> bool:
    return not bool(stage_result.get("reached", False))


def execution_failed(stages_by_executed_arm: Dict[str, Dict[str, object]]) -> bool:
    for arm_stages in stages_by_executed_arm.values():
        for stage_name in ("pregrasp", "grasp", "action"):
            stage_result = arm_stages.get(stage_name)
            if isinstance(stage_result, dict) and stage_result_failed(stage_result):
                return True
    return False


def collect_failed_stage_records(stages_by_executed_arm: Dict[str, Dict[str, object]]) -> List[Dict[str, object]]:
    failed_records: List[Dict[str, object]] = []
    for arm_name, arm_stages in stages_by_executed_arm.items():
        for stage_name in ("pregrasp", "grasp", "action"):
            stage_result = arm_stages.get(stage_name)
            if not isinstance(stage_result, dict):
                continue
            if stage_result_failed(stage_result):
                failed_records.append(
                    {
                        "arm": str(arm_name),
                        "stage": str(stage_name),
                        "status": str(stage_result.get("status", "Missing")),
                        "attempts": int(stage_result.get("attempts", 0)),
                        "reached": bool(stage_result.get("reached", False)),
                        "pos_err_m": float(stage_result.get("pos_err_m", float("inf"))),
                        "rot_err_deg": float(stage_result.get("rot_err_deg", float("inf"))),
                    }
                )
    return failed_records


def record_frame(
    renderer: ReplayRenderer,
    head_writer: cv2.VideoWriter,
    third_writer: Optional[cv2.VideoWriter],
    overlay_lines: Sequence[str],
    use_overlay: bool,
    debug_visuals: Optional[DebugVisualBundle] = None,
    debug_execution_state: Optional[DebugExecutionState] = None,
    pure_scene_main: bool = False,
    use_overlay_debug: Optional[bool] = None,
) -> None:
    if use_overlay_debug is None:
        use_overlay_debug = bool(use_overlay)
    viewer_active = bool(getattr(renderer, "viewer", None) is not None and not renderer.viewer.closed)
    if debug_visuals is not None:
        for actor in debug_visuals.keyframe_axis_actors.values():
            hide_actor(actor)
        for actors in debug_visuals.common_candidate_actors.values():
            for actor in actors:
                hide_actor(actor)
        for per_frame in debug_visuals.arm_candidate_actors.values():
            for actors in per_frame.values():
                for actor in actors:
                    hide_actor(actor)
        for per_frame in debug_visuals.arm_candidate_axis_actors.values():
            for actors in per_frame.values():
                for actor in actors:
                    hide_actor(actor)
        if debug_execution_state is not None and not bool(pure_scene_main):
            if bool(debug_execution_state.show_selected_keyframe_axes):
                for item in selected_keyframes_for_active_frames_by_arm(
                    debug_execution_state.selected_keyframes,
                    debug_execution_state.active_frame,
                    debug_execution_state.active_frame_by_arm,
                ):
                    actor = debug_visuals.keyframe_axis_actors.get((int(item.source_frame), item.arm))
                    if actor is not None:
                        set_actor_pose(actor, item.candidate.pose_world_wxyz)
            update_candidate_debug_visuals(
                debug_visuals,
                debug_execution_state.active_frame,
                debug_execution_state.common_candidates_per_frame,
                debug_execution_state.arm_display_candidates,
                debug_execution_state.active_frame_by_arm,
            )
    if bool(pure_scene_main):
        renderer.update_target_axis_visuals(None, None)
    else:
        apply_target_visuals_from_state(renderer, debug_execution_state)
    renderer.update_robot_link_cameras()
    renderer.scene.update_render()
    if viewer_active:
        renderer.viewer.render()
    head_rgb, _ = renderer.capture_camera(renderer.zed_camera)
    head_bgr = base.overlay_text(head_rgb, overlay_lines) if use_overlay else cv2.cvtColor(head_rgb, cv2.COLOR_RGB2BGR)
    head_writer.write(head_bgr)
    if third_writer is not None:
        third_rgb, _ = renderer.capture_camera(renderer.third_camera)
        third_bgr = base.overlay_text(third_rgb, overlay_lines) if use_overlay else cv2.cvtColor(third_rgb, cv2.COLOR_RGB2BGR)
        third_writer.write(third_bgr)
    if debug_execution_state is not None and debug_execution_state.left_wrist_writer is not None and getattr(renderer, "_left_wrist_camera_link", None) is not None:
        left_wrist_rgb, _ = renderer.capture_camera(renderer.left_wrist_camera)
        left_wrist_rgb = rotate_wrist_rgb_for_export(left_wrist_rgb)
        left_wrist_lines = list(overlay_lines) + ["left_wrist"]
        left_wrist_bgr = base.overlay_text(left_wrist_rgb, left_wrist_lines) if use_overlay else cv2.cvtColor(left_wrist_rgb, cv2.COLOR_RGB2BGR)
        debug_execution_state.left_wrist_writer.write(left_wrist_bgr)
    if debug_execution_state is not None and debug_execution_state.right_wrist_writer is not None and getattr(renderer, "_right_wrist_camera_link", None) is not None:
        right_wrist_rgb, _ = renderer.capture_camera(renderer.right_wrist_camera)
        right_wrist_rgb = rotate_wrist_rgb_for_export(right_wrist_rgb)
        right_wrist_lines = list(overlay_lines) + ["right_wrist"]
        right_wrist_bgr = base.overlay_text(right_wrist_rgb, right_wrist_lines) if use_overlay else cv2.cvtColor(right_wrist_rgb, cv2.COLOR_RGB2BGR)
        debug_execution_state.right_wrist_writer.write(right_wrist_bgr)
    frame_metrics = None
    if debug_execution_state is not None:
        frame_metrics = build_execution_frame_metrics(renderer, debug_execution_state)
    if debug_execution_state is not None and debug_execution_state.writer is not None and debug_visuals is not None:
        apply_target_visuals_from_state(renderer, debug_execution_state)
        renderer.update_robot_link_cameras()
        renderer.scene.update_render()
        debug_overlay = list(overlay_lines)
        debug_overlay.extend(
            [
                "mode=debug_execution",
                (
                    f"active_keyframe={int(debug_execution_state.active_frame)}"
                    if debug_execution_state.active_frame is not None
                    else "active_keyframe=none"
                ),
                "color=green:raw blue:left orange:right red:selected",
                "labels=tiny candidate_idx",
            ]
        )
        debug_rgb, _ = renderer.capture_camera(renderer.zed_camera)
        debug_bgr = base.overlay_text(debug_rgb, debug_overlay) if use_overlay_debug else cv2.cvtColor(debug_rgb, cv2.COLOR_RGB2BGR)
        annotate_candidate_labels(
            debug_bgr,
            renderer.zed_camera,
            debug_execution_state.head_intrinsic,
            debug_execution_state.active_frame,
            debug_execution_state.common_candidates_per_frame,
            debug_execution_state.arm_display_candidates,
            selected_keyframes_for_active_frames_by_arm(
                debug_execution_state.selected_keyframes,
                debug_execution_state.active_frame,
                debug_execution_state.active_frame_by_arm,
            ),
        )
        if frame_metrics is not None:
            draw_execution_metric_panel(debug_bgr, frame_metrics)
        debug_execution_state.writer.write(debug_bgr)
    if debug_execution_state is not None and debug_execution_state.pose_debug_path is not None:
        current_head_pose = renderer.get_head_camera_pose()
        current_left_ee_pose = renderer.robot.get_left_ee_pose()
        current_right_ee_pose = renderer.robot.get_right_ee_pose()
        current_left_wrist_camera_pose = None
        current_right_wrist_camera_pose = None
        if getattr(renderer, "_left_wrist_camera_link", None) is not None:
            left_wrist_pose = renderer.get_wrist_camera_pose("left")
            current_left_wrist_camera_pose = (
                np.asarray(left_wrist_pose.p, dtype=np.float64).reshape(3).tolist()
                + base.normalize_quat_wxyz(np.asarray(left_wrist_pose.q, dtype=np.float64).reshape(4)).tolist()
            )
        if getattr(renderer, "_right_wrist_camera_link", None) is not None:
            right_wrist_pose = renderer.get_wrist_camera_pose("right")
            current_right_wrist_camera_pose = (
                np.asarray(right_wrist_pose.p, dtype=np.float64).reshape(3).tolist()
                + base.normalize_quat_wxyz(np.asarray(right_wrist_pose.q, dtype=np.float64).reshape(4)).tolist()
            )
        left_arm_qpos = np.asarray(renderer.robot.get_left_arm_real_jointState()[:6], dtype=np.float64).reshape(6)
        right_arm_qpos = np.asarray(renderer.robot.get_right_arm_real_jointState()[:6], dtype=np.float64).reshape(6)
        left_gripper_joint_qpos = _get_gripper_joint_positions(renderer, "left")
        right_gripper_joint_qpos = _get_gripper_joint_positions(renderer, "right")
        object_actor_poses = {}
        if debug_execution_state.object_tracks is not None:
            for name, track in debug_execution_state.object_tracks.items():
                if track.actor is None:
                    continue
                actor_pose = track.actor.get_pose()
                object_actor_poses[name] = {
                    "actor_pose_world_wxyz": (
                        np.asarray(actor_pose.p, dtype=np.float64).reshape(3).tolist()
                        + base.normalize_quat_wxyz(np.asarray(actor_pose.q, dtype=np.float64).reshape(4)).tolist()
                    )
                }
                if debug_execution_state.active_frame is not None:
                    frame_to_idx = {int(frame): idx for idx, frame in enumerate(track.frame_indices.tolist())}
                    idx = frame_to_idx.get(int(debug_execution_state.active_frame))
                    if idx is not None:
                        object_actor_poses[name]["replay_pose_world_wxyz"] = np.asarray(track.pose_world_wxyz[idx], dtype=np.float64).tolist()
        pose_debug = {
            "record_index": int(debug_execution_state.record_index),
            "active_frame": None if debug_execution_state.active_frame is None else int(debug_execution_state.active_frame),
            "active_frame_by_arm": {str(k): int(v) for k, v in debug_execution_state.active_frame_by_arm.items()},
            "stage": debug_execution_state.current_stage,
            "overlay_lines": list(overlay_lines),
            "current_head_camera_pose_world_wxyz": pose_like_to_world_wxyz(current_head_pose).tolist(),
            "current_left_wrist_camera_pose_world_wxyz": current_left_wrist_camera_pose,
            "current_right_wrist_camera_pose_world_wxyz": current_right_wrist_camera_pose,
            "current_left_tcp_pose_world_wxyz": np.asarray(renderer.get_current_tcp_pose("left"), dtype=np.float64).tolist(),
            "current_right_tcp_pose_world_wxyz": np.asarray(renderer.get_current_tcp_pose("right"), dtype=np.float64).tolist(),
            "current_left_ee_pose_world_wxyz": pose_like_to_world_wxyz(current_left_ee_pose).tolist(),
            "current_right_ee_pose_world_wxyz": pose_like_to_world_wxyz(current_right_ee_pose).tolist(),
            "current_left_arm_qpos_rad": left_arm_qpos.tolist(),
            "current_right_arm_qpos_rad": right_arm_qpos.tolist(),
            "current_left_gripper_joint_qpos_rad": left_gripper_joint_qpos.tolist(),
            "current_right_gripper_joint_qpos_rad": right_gripper_joint_qpos.tolist(),
            "current_left_gripper_command": float(renderer.robot.get_left_gripper_val()),
            "current_right_gripper_command": float(renderer.robot.get_right_gripper_val()),
            "replay_head_camera_pose_world_wxyz": None
            if debug_execution_state.replay_head_camera_pose_by_frame is None or debug_execution_state.active_frame is None
            else np.asarray(
                debug_execution_state.replay_head_camera_pose_by_frame.get(int(debug_execution_state.active_frame), np.full(7, np.nan)),
                dtype=np.float64,
            ).tolist(),
            "object_actor_poses": object_actor_poses,
            "frame_metrics": frame_metrics,
        }
        with debug_execution_state.pose_debug_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(pose_debug, ensure_ascii=False) + "\n")
    if debug_execution_state is not None and debug_execution_state.metrics_debug_path is not None and frame_metrics is not None:
        with debug_execution_state.metrics_debug_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(frame_metrics, ensure_ascii=False) + "\n")
    if debug_execution_state is not None:
        debug_execution_state.record_index += 1
    if debug_visuals is not None and not viewer_active:
        for actors in debug_visuals.common_candidate_actors.values():
            for actor in actors:
                hide_actor(actor)
        for per_frame in debug_visuals.arm_candidate_actors.values():
            for actors in per_frame.values():
                for actor in actors:
                    hide_actor(actor)
        for per_frame in debug_visuals.arm_candidate_axis_actors.values():
            for actors in per_frame.values():
                for actor in actors:
                    hide_actor(actor)
        for actor in debug_visuals.keyframe_axis_actors.values():
            hide_actor(actor)


def pose_with_offset_along_local_x(pose_world_wxyz: np.ndarray, offset_m: float) -> np.ndarray:
    return pose_with_offset_along_local_axis(pose_world_wxyz, offset_m, "local_x")


def pose_with_offset_along_local_z(pose_world_wxyz: np.ndarray, offset_m: float) -> np.ndarray:
    return pose_with_offset_along_local_axis(pose_world_wxyz, offset_m, "local_z")


def pose_with_offset_along_local_axis(pose_world_wxyz: np.ndarray, offset_m: float, axis: str) -> np.ndarray:
    # Pregrasp generation only.
    # This does NOT change the actual grasp target. It creates a separate
    # pregrasp pose by retreating backward from the grasp pose along the gripper
    # local forward axis.
    pose_world_wxyz = np.asarray(pose_world_wxyz, dtype=np.float64).reshape(7)
    pose_world_matrix = pose_wxyz_to_matrix(pose_world_wxyz)
    axis_index = 2 if axis == "local_z" else 0
    pose_world_matrix[:3, 3] -= pose_world_matrix[:3, axis_index] * float(offset_m)
    rot = pose_world_matrix[:3, :3]
    quat = base.quat_xyzw_to_wxyz(R.from_matrix(rot).as_quat())
    return np.concatenate([pose_world_matrix[:3, 3], quat]).astype(np.float64)


def vector_with_distance(vec_xyz: np.ndarray) -> Dict[str, object]:
    vec_xyz = np.asarray(vec_xyz, dtype=np.float64).reshape(3)
    return {
        "xyz_m": vec_xyz.tolist(),
        "distance_m": float(np.linalg.norm(vec_xyz)),
    }


def build_execution_frame_metrics(
    renderer: ReplayRenderer,
    debug_execution_state: DebugExecutionState,
) -> Dict[str, object]:
    current_head_pose = renderer.get_head_camera_pose()
    metrics: Dict[str, object] = {
        "active_frame": None if debug_execution_state.active_frame is None else int(debug_execution_state.active_frame),
        "active_frame_by_arm": {str(k): int(v) for k, v in debug_execution_state.active_frame_by_arm.items()},
        "stage": debug_execution_state.current_stage,
        "reach_error_pose_source": str(debug_execution_state.reach_error_pose_source),
        "current_head_camera_pose_world_wxyz": (
            np.asarray(current_head_pose.p, dtype=np.float64).reshape(3).tolist()
            + base.normalize_quat_wxyz(np.asarray(current_head_pose.q, dtype=np.float64).reshape(4)).tolist()
        ),
        "replay_head_camera_pose_world_wxyz": None,
        "arms": {},
    }
    if debug_execution_state.replay_head_camera_pose_by_frame is not None and debug_execution_state.active_frame is not None:
        replay_head_pose = debug_execution_state.replay_head_camera_pose_by_frame.get(int(debug_execution_state.active_frame))
        if replay_head_pose is not None:
            metrics["replay_head_camera_pose_world_wxyz"] = np.asarray(replay_head_pose, dtype=np.float64).tolist()

    target_pose_by_arm = debug_execution_state.target_pose_by_arm or {}
    target_object_by_arm = debug_execution_state.target_object_by_arm or {}
    goal_label_by_arm = debug_execution_state.goal_label_by_arm or {}
    object_tracks = debug_execution_state.object_tracks or {}

    for arm_name, target_pose_world in target_pose_by_arm.items():
        target_pose_world = np.asarray(target_pose_world, dtype=np.float64).reshape(7)
        current_eval_pose = get_current_pose_for_error(renderer, arm_name, debug_execution_state.reach_error_pose_source)
        target_eval_pose = target_pose_for_error(renderer, arm_name, target_pose_world, debug_execution_state.reach_error_pose_source)
        current_tcp_pose = np.asarray(renderer.get_current_tcp_pose(arm_name), dtype=np.float64).reshape(7)
        arm_metrics: Dict[str, object] = {
            "goal_label": goal_label_by_arm.get(arm_name),
            "target_object": target_object_by_arm.get(arm_name),
            "target_pose_world_wxyz": target_pose_world.tolist(),
            "target_eval_pose_world_wxyz": target_eval_pose.tolist(),
            "current_eval_pose_world_wxyz": current_eval_pose.tolist(),
            "current_tcp_pose_world_wxyz": current_tcp_pose.tolist(),
            "target_minus_current": vector_with_distance(target_eval_pose[:3] - current_eval_pose[:3]),
        }

        obj_name = target_object_by_arm.get(arm_name)
        track = None if obj_name is None else object_tracks.get(obj_name)
        actual_obj_pose = None
        replay_obj_pose = None
        if track is not None and track.actor is not None:
            actor_pose = track.actor.get_pose()
            actual_obj_pose = np.concatenate(
                [
                    np.asarray(actor_pose.p, dtype=np.float64).reshape(3),
                    base.normalize_quat_wxyz(np.asarray(actor_pose.q, dtype=np.float64).reshape(4)),
                ]
            ).astype(np.float64)
            arm_metrics["actual_object_pose_world_wxyz"] = actual_obj_pose.tolist()
            if debug_execution_state.active_frame is not None:
                frame_to_idx = {int(frame): idx for idx, frame in enumerate(track.frame_indices.tolist())}
                replay_idx = frame_to_idx.get(int(debug_execution_state.active_frame))
                if replay_idx is not None:
                    replay_obj_pose = np.asarray(track.pose_world_wxyz[replay_idx], dtype=np.float64).reshape(7)
                    arm_metrics["planned_object_pose_world_wxyz"] = replay_obj_pose.tolist()
        if actual_obj_pose is not None and replay_obj_pose is not None:
            arm_metrics["planned_minus_actual_object"] = vector_with_distance(replay_obj_pose[:3] - actual_obj_pose[:3])
        if actual_obj_pose is not None:
            arm_metrics["target_minus_actual_object"] = vector_with_distance(target_eval_pose[:3] - actual_obj_pose[:3])
            arm_metrics["actual_pose_minus_actual_object"] = vector_with_distance(current_eval_pose[:3] - actual_obj_pose[:3])
        metrics["arms"][arm_name] = arm_metrics
    return metrics


def draw_execution_metric_panel(image_bgr: np.ndarray, frame_metrics: Dict[str, object]) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.42
    thickness = 1
    line_h = 15
    left_col_x = 8
    right_col_x = max(int(image_bgr.shape[1] * 0.53), 320)
    top_y = max(int(image_bgr.shape[0] * 0.42), 150)
    colors = {
        "header": (0, 255, 255),
        "target": (255, 255, 0),
        "actual": (0, 255, 0),
        "error": (0, 0, 255),
        "object": (0, 165, 255),
    }

    def short_xyz(vec_entry: Optional[Dict[str, object]]) -> str:
        if not vec_entry:
            return "n/a"
        xyz = np.asarray(vec_entry.get("xyz_m", [np.nan, np.nan, np.nan]), dtype=np.float64).reshape(3)
        dist = float(vec_entry.get("distance_m", np.nan))
        return f"[{xyz[0]:+.3f},{xyz[1]:+.3f},{xyz[2]:+.3f}] d={dist:.3f}"

    arm_columns = [("left", left_col_x), ("right", right_col_x)]
    for arm_name, col_x in arm_columns:
        arm_metrics = frame_metrics.get("arms", {}).get(arm_name)
        if not arm_metrics:
            continue
        lines = [
            (f"{arm_name.upper()} stage={frame_metrics.get('stage', 'n/a')} obj={arm_metrics.get('target_object', 'n/a')}", colors["header"]),
            (f"goal={arm_metrics.get('goal_label', 'n/a')}", colors["header"]),
            (f"tgt-cur {short_xyz(arm_metrics.get('target_minus_current'))}", colors["target"]),
            (f"obj plan-act {short_xyz(arm_metrics.get('planned_minus_actual_object'))}", colors["error"]),
            (f"tgt-obj {short_xyz(arm_metrics.get('target_minus_actual_object'))}", colors["object"]),
            (f"cur-obj {short_xyz(arm_metrics.get('actual_pose_minus_actual_object'))}", colors["actual"]),
        ]
        y = top_y
        for text, color in lines:
            cv2.putText(image_bgr, text, (col_x, y), font, font_scale, color, thickness, cv2.LINE_AA)
            y += line_h


def load_jsonl_records(path: Path) -> List[Dict[str, object]]:
    records: List[Dict[str, object]] = []
    if not path.is_file():
        return records
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def transcode_mp4_for_vscode(path: Path) -> bool:
    path = Path(path)
    if not path.is_file():
        return False
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        print(f"[video] skip vscode transcode because ffmpeg is unavailable: {path}")
        return False
    tmp_path = path.with_name(f"{path.stem}.vscode_tmp{path.suffix}")
    cmd = [
        ffmpeg,
        "-y",
        "-v",
        "error",
        "-i",
        str(path),
        "-an",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(tmp_path),
    ]
    try:
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False)
    except Exception as exc:
        print(f"[video] failed to launch ffmpeg for {path}: {exc}")
        return False
    if result.returncode != 0 or not tmp_path.is_file() or tmp_path.stat().st_size <= 0:
        if tmp_path.exists():
            tmp_path.unlink()
        err = result.stderr.strip().splitlines()[-1] if result.stderr.strip() else "unknown ffmpeg error"
        print(f"[video] vscode transcode failed path={path} error={err}")
        return False
    tmp_path.replace(path)
    print(f"[video] vscode-compatible h264/yuv420p: {path}")
    return True


def iter_stage_spans(stage_names: Sequence[str]) -> List[Tuple[int, int, str]]:
    if not stage_names:
        return []
    spans: List[Tuple[int, int, str]] = []
    start_idx = 0
    current = str(stage_names[0])
    for idx in range(1, len(stage_names)):
        stage = str(stage_names[idx])
        if stage != current:
            spans.append((start_idx, idx - 1, current))
            start_idx = idx
            current = stage
    spans.append((start_idx, len(stage_names) - 1, current))
    return spans


def generate_execution_analysis_plots(
    metrics_path: Path,
    output_dir: Path,
    fps: float,
) -> List[str]:
    records = load_jsonl_records(metrics_path)
    if not records:
        return []

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    time_s = np.arange(len(records), dtype=np.float64) / max(float(fps), 1e-6)
    stages = [str(record.get("stage", "unknown")) for record in records]
    stage_spans = iter_stage_spans(stages)
    stage_colors = {
        "init": "#f0f0f0",
        "pregrasp": "#dceeff",
        "grasp": "#dff7df",
        "close_gripper": "#fff2cc",
        "pause_after_keyframe1": "#f5e1ff",
        "action": "#ffe4d6",
    }

    def arm_series(arm_name: str, metric_name: str) -> np.ndarray:
        values = []
        for record in records:
            arm_metrics = record.get("arms", {}).get(arm_name, {})
            metric = arm_metrics.get(metric_name)
            if not metric:
                values.append([np.nan, np.nan, np.nan])
                continue
            if isinstance(metric, dict) and "xyz_m" in metric:
                xyz = np.asarray(metric.get("xyz_m", [np.nan, np.nan, np.nan]), dtype=np.float64).reshape(3)
                values.append(xyz)
            else:
                values.append([np.nan, np.nan, np.nan])
        return np.asarray(values, dtype=np.float64)

    def arm_distance_series(arm_name: str, metric_name: str) -> np.ndarray:
        values = []
        for record in records:
            arm_metrics = record.get("arms", {}).get(arm_name, {})
            metric = arm_metrics.get(metric_name)
            values.append(float(metric.get("distance_m", np.nan)) if isinstance(metric, dict) else np.nan)
        return np.asarray(values, dtype=np.float64)

    def shade_stage_background(ax) -> None:
        for start_idx, end_idx, stage_name in stage_spans:
            color = stage_colors.get(stage_name, "#eeeeee")
            ax.axvspan(time_s[start_idx], time_s[end_idx], color=color, alpha=0.18, linewidth=0.0)

    written: List[str] = []
    for arm_name in ("left", "right"):
        arm_present = any(arm_name in record.get("arms", {}) for record in records)
        if not arm_present:
            continue

        target_current_xyz = arm_series(arm_name, "target_minus_current")
        target_current_d = arm_distance_series(arm_name, "target_minus_current")
        fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
        shade_stage_background(axes[0])
        axes[0].plot(time_s, target_current_d, color="black", linewidth=1.8, label="distance")
        axes[0].set_ylabel("meters")
        axes[0].set_title(f"{arm_name} target-current error vs time")
        axes[0].grid(True, alpha=0.3)
        axes[0].legend(loc="upper right")

        shade_stage_background(axes[1])
        axes[1].plot(time_s, target_current_xyz[:, 0], color="#d62728", linewidth=1.2, label="dx")
        axes[1].plot(time_s, target_current_xyz[:, 1], color="#2ca02c", linewidth=1.2, label="dy")
        axes[1].plot(time_s, target_current_xyz[:, 2], color="#1f77b4", linewidth=1.2, label="dz")
        axes[1].set_ylabel("meters")
        axes[1].set_xlabel("time (s)")
        axes[1].grid(True, alpha=0.3)
        axes[1].legend(loc="upper right")
        fig.tight_layout()
        out_path = output_dir / f"{arm_name}_target_current_error_vs_time.png"
        fig.savefig(out_path, dpi=180)
        plt.close(fig)
        written.append(str(out_path))

        planned_actual_d = arm_distance_series(arm_name, "planned_minus_actual_object")
        target_object_d = arm_distance_series(arm_name, "target_minus_actual_object")
        current_object_d = arm_distance_series(arm_name, "actual_pose_minus_actual_object")
        fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
        shade_stage_background(axes[0])
        axes[0].plot(time_s, planned_actual_d, color="#d62728", linewidth=1.4, label="planned-actual object")
        axes[0].plot(time_s, target_object_d, color="#ff7f0e", linewidth=1.4, label="target-object")
        axes[0].plot(time_s, current_object_d, color="#2ca02c", linewidth=1.4, label="current-object")
        axes[0].set_ylabel("meters")
        axes[0].set_title(f"{arm_name} object distance diagnostics vs time")
        axes[0].grid(True, alpha=0.3)
        axes[0].legend(loc="upper right")

        target_object_xyz = arm_series(arm_name, "target_minus_actual_object")
        shade_stage_background(axes[1])
        axes[1].plot(time_s, target_object_xyz[:, 0], color="#d62728", linewidth=1.2, label="dx")
        axes[1].plot(time_s, target_object_xyz[:, 1], color="#2ca02c", linewidth=1.2, label="dy")
        axes[1].plot(time_s, target_object_xyz[:, 2], color="#1f77b4", linewidth=1.2, label="dz")
        axes[1].set_ylabel("meters")
        axes[1].set_xlabel("time (s)")
        axes[1].grid(True, alpha=0.3)
        axes[1].legend(loc="upper right")
        fig.tight_layout()
        out_path = output_dir / f"{arm_name}_object_distance_vs_time.png"
        fig.savefig(out_path, dpi=180)
        plt.close(fig)
        written.append(str(out_path))

    return written


def sapien_pose_to_wxyz(pose: sapien.Pose) -> np.ndarray:
    return np.concatenate([np.asarray(pose.p, dtype=np.float64), np.asarray(pose.q, dtype=np.float64)]).astype(np.float64)


def pose_like_to_world_wxyz(pose_like) -> np.ndarray:
    if hasattr(pose_like, "p") and hasattr(pose_like, "q"):
        return np.concatenate(
            [
                np.asarray(pose_like.p, dtype=np.float64).reshape(3),
                base.normalize_quat_wxyz(np.asarray(pose_like.q, dtype=np.float64).reshape(4)),
            ]
        ).astype(np.float64)
    arr = np.asarray(pose_like, dtype=np.float64).reshape(7)
    arr[3:] = base.normalize_quat_wxyz(arr[3:])
    return arr


def rotate_wrist_rgb_for_export(rgb: np.ndarray) -> np.ndarray:
    # With the R1 wrist-camera local pose aligned to galaxea_sim/robots/r1.py,
    # planner exports can use the captured frame directly without post-rotation.
    return np.asarray(rgb)


def get_current_pose_for_error(renderer: ReplayRenderer, arm: str, pose_source: str) -> np.ndarray:
    if renderer.robot is None:
        raise RuntimeError("Robot is unavailable.")
    if pose_source == "tcp":
        # This must match the same planner-side TCP/gripper convention used by
        # target_pose_for_error(..., pose_source="tcp") and by renderer.plan_path(...).
        return renderer.get_current_tcp_pose(arm)
    if pose_source == "ee":
        # EE here means the wrist/endlink convention returned by robot.get_*_ee_pose().
        pose = renderer.robot.get_left_ee_pose() if arm == "left" else renderer.robot.get_right_ee_pose()
        return np.asarray(pose, dtype=np.float64)
    raise ValueError(f"Unsupported pose_source: {pose_source}")


def target_pose_for_error(renderer: ReplayRenderer, arm: str, target_tcp_pose_world_wxyz: np.ndarray, pose_source: str) -> np.ndarray:
    if renderer.robot is None:
        raise RuntimeError("Robot is unavailable.")
    target_tcp_pose_world_wxyz = np.asarray(target_tcp_pose_world_wxyz, dtype=np.float64).reshape(7)
    if pose_source == "tcp":
        # Direct comparison in planner TCP/gripper space.
        return target_tcp_pose_world_wxyz
    if pose_source == "ee":
        if hasattr(renderer, "world_pose_to_base_pose_for_arm") and hasattr(renderer, "base_pose_to_world_pose_for_arm"):
            # Piper dual robot.get_*_ee_pose() reports the same visible gripper
            # orientation convention as the preview target, with only the
            # reference point shifted to the EE position. Under the current Piper
            # config gripper_bias == 0.12, so the target position is unchanged.
            # Do not convert this to raw link6/endlink space; that would compare
            # the corrected visual target against the wrong local axes.
            return target_tcp_pose_world_wxyz
        # Convert the planner TCP/gripper target into the wrist/endlink convention before
        # comparing against robot.get_*_ee_pose(). Under the current R1 config,
        # gripper_bias == 0.12, so this conversion contributes no extra translation and
        # mainly changes the orientation convention for IK/endlink space.
        target_pose_base = renderer.world_pose_to_base_pose(target_tcp_pose_world_wxyz)
        target_pose_ee_base = renderer.robot._trans_from_gripper_to_endlink(target_pose_base.tolist(), arm_tag=arm)
        base_world = base.pose_to_matrix(renderer._base_pose)
        ee_base = base.pose_to_matrix(target_pose_ee_base)
        ee_world = base_world @ ee_base
        quat = base.quat_xyzw_to_wxyz(R.from_matrix(base.orthonormalize_rotation(ee_world[:3, :3])).as_quat())
        return np.concatenate([ee_world[:3, 3], quat]).astype(np.float64)
    raise ValueError(f"Unsupported pose_source: {pose_source}")


def build_supervision_targets(
    arm_display_candidates: Dict[str, Dict[int, List[CandidatePose]]],
    selected_keyframes: List[SelectedKeyframe],
) -> Dict[str, np.ndarray]:
    targets: Dict[str, np.ndarray] = {}
    active_frame = int(selected_keyframes[0].source_frame)
    for arm_name, frame_map in arm_display_candidates.items():
        frame_candidates = frame_map.get(active_frame, [])
        if frame_candidates:
            targets[arm_name] = np.asarray(frame_candidates[0].pose_world_wxyz, dtype=np.float64)
    for item in selected_keyframes:
        if int(item.source_frame) == active_frame:
            targets[item.arm] = np.asarray(item.candidate.pose_world_wxyz, dtype=np.float64)
    return targets


def compute_supervision_errors(
    renderer: ReplayRenderer,
    supervision_targets: Dict[str, np.ndarray],
    pose_source: str,
) -> Dict[str, Dict[str, float]]:
    errors: Dict[str, Dict[str, float]] = {}
    for arm_name, target_pose in supervision_targets.items():
        current_pose = get_current_pose_for_error(renderer, arm_name, pose_source)
        target_eval_pose = target_pose_for_error(renderer, arm_name, target_pose, pose_source)
        breakdown = pose_error_breakdown(target_eval_pose, current_pose)
        errors[arm_name] = {
            "pose_source": pose_source,
            "dx_m": float(breakdown["dx_m"]),
            "dy_m": float(breakdown["dy_m"]),
            "dz_m": float(breakdown["dz_m"]),
            "pos_err_m": float(breakdown["dist_m"]),
            "rot_err_deg": float(breakdown["rot_err_deg"]),
            "forward_axis_err_deg": float(breakdown["forward_axis_err_deg"]),
            "forward_axis_signed_err_m": float(breakdown["forward_axis_signed_err_m"]),
            "forward_axis_signed_err_cm": float(breakdown["forward_axis_signed_err_cm"]),
            "lateral_to_forward_axis_m": float(breakdown["lateral_to_forward_axis_m"]),
            "lateral_to_forward_axis_cm": float(breakdown["lateral_to_forward_axis_cm"]),
        }
    return errors


def pose_error_breakdown(
    target_pose_world_wxyz: np.ndarray,
    current_pose_world_wxyz: np.ndarray,
) -> Dict[str, float]:
    target_pose_world_wxyz = np.asarray(target_pose_world_wxyz, dtype=np.float64).reshape(7)
    current_pose_world_wxyz = np.asarray(current_pose_world_wxyz, dtype=np.float64).reshape(7)
    delta = target_pose_world_wxyz[:3] - current_pose_world_wxyz[:3]
    dist = float(np.linalg.norm(delta))
    rot_err = float(base.quat_angle_deg_wxyz(current_pose_world_wxyz[3:], target_pose_world_wxyz[3:]))
    target_pose_world_matrix = pose_wxyz_to_matrix(target_pose_world_wxyz)
    current_pose_world_matrix = pose_wxyz_to_matrix(current_pose_world_wxyz)
    # AnyGrasp candidate diagnostics use local +X as the wireframe finger-depth
    # axis. Direct hand replay uses local +Z as its gripper approach axis.
    target_forward_world = base.orthonormalize_rotation(target_pose_world_matrix[:3, :3])[:, 0]
    current_forward_world = base.orthonormalize_rotation(current_pose_world_matrix[:3, :3])[:, 0]
    forward_axis_cos = float(np.clip(np.dot(target_forward_world, current_forward_world), -1.0, 1.0))
    forward_axis_err_deg = float(np.rad2deg(np.arccos(forward_axis_cos)))
    # Positive means the actual gripper is ahead of the target along the target forward axis.
    # Negative means the actual gripper is behind the target along that same axis.
    actual_minus_target = current_pose_world_wxyz[:3] - target_pose_world_wxyz[:3]
    forward_axis_signed_err_m = float(np.dot(actual_minus_target, target_forward_world))
    lateral_to_forward_axis_m = float(
        np.linalg.norm(actual_minus_target - forward_axis_signed_err_m * target_forward_world)
    )
    return {
        "dx_m": float(delta[0]),
        "dy_m": float(delta[1]),
        "dz_m": float(delta[2]),
        "dist_m": dist,
        "rot_err_deg": rot_err,
        "forward_axis_err_deg": forward_axis_err_deg,
        "forward_axis_signed_err_m": forward_axis_signed_err_m,
        "forward_axis_signed_err_cm": float(forward_axis_signed_err_m * 100.0),
        "lateral_to_forward_axis_m": lateral_to_forward_axis_m,
        "lateral_to_forward_axis_cm": float(lateral_to_forward_axis_m * 100.0),
    }


def plan_direction_label(forward_axis_signed_err_m: float, eps_m: float = 1e-4) -> str:
    signed_err_m = float(forward_axis_signed_err_m)
    if signed_err_m > float(eps_m):
        return "move_backward_along_target_forward"
    if signed_err_m < -float(eps_m):
        return "move_forward_along_target_forward"
    return "already_aligned_on_forward_axis"


def short_direction_label(label: str) -> str:
    if label == "move_forward_along_target_forward":
        return "forward"
    if label == "move_backward_along_target_forward":
        return "backward"
    if label == "already_aligned_on_forward_axis":
        return "aligned"
    return str(label)


def colorize_forward_cm(value_cm: float) -> str:
    text = f"{float(value_cm):+.2f}"
    if not sys.stdout.isatty():
        return text
    if value_cm > 1e-6:
        return f"\033[1;31m{text}\033[0m"
    if value_cm < -1e-6:
        return f"\033[1;36m{text}\033[0m"
    return f"\033[1;33m{text}\033[0m"


def short_stage_status(stage_info: Optional[Dict[str, object]]) -> str:
    # Original fields are:
    # - reached: whether the stage satisfied the configured reach thresholds
    # - pos_err_m / rot_err_deg: final pose error after execution
    if not isinstance(stage_info, dict):
        return "na"
    reached = bool(stage_info.get("reached", False))
    pos_err = stage_info.get("pos_err_m", None)
    rot_err = stage_info.get("rot_err_deg", None)
    if pos_err is None or rot_err is None:
        return "ok" if reached else "miss"
    return f"{'ok' if reached else 'miss'}(p={float(pos_err):.3f},r={float(rot_err):.1f})"


def plan_request_diagnostics(
    current_eval_pose: np.ndarray,
    target_eval_pose: np.ndarray,
) -> Dict[str, object]:
    current_eval_pose = np.asarray(current_eval_pose, dtype=np.float64).reshape(7)
    target_eval_pose = np.asarray(target_eval_pose, dtype=np.float64).reshape(7)
    breakdown = pose_error_breakdown(target_eval_pose, current_eval_pose)
    return {
        "current_pose_world_wxyz": current_eval_pose.tolist(),
        "target_pose_world_wxyz": target_eval_pose.tolist(),
        "error": breakdown,
        "theoretical_forward_axis_motion": plan_direction_label(breakdown["forward_axis_signed_err_m"]),
    }


def _ee_base_pose_to_world_wxyz(renderer: ReplayRenderer, ee_pos_base: np.ndarray, ee_quat_wxyz_base: np.ndarray) -> np.ndarray:
    base_world = base.pose_to_matrix(renderer._base_pose)
    ee_base = np.eye(4, dtype=np.float64)
    ee_base[:3, :3] = base.orthonormalize_rotation(R.from_quat(base.quat_wxyz_to_xyzw(ee_quat_wxyz_base)).as_matrix())
    ee_base[:3, 3] = np.asarray(ee_pos_base, dtype=np.float64).reshape(3)
    ee_world = base_world @ ee_base
    quat = base.quat_xyzw_to_wxyz(R.from_matrix(base.orthonormalize_rotation(ee_world[:3, :3])).as_quat())
    return np.concatenate([ee_world[:3, 3], quat]).astype(np.float64)


def _ee_base_pose_to_world_wxyz_for_arm(
    renderer: ReplayRenderer,
    arm: str,
    ee_pos_base: np.ndarray,
    ee_quat_wxyz_base: np.ndarray,
) -> np.ndarray:
    ee_pose_base = np.concatenate(
        [
            np.asarray(ee_pos_base, dtype=np.float64).reshape(3),
            base.normalize_quat_wxyz(ee_quat_wxyz_base),
        ]
    ).astype(np.float64)
    if hasattr(renderer, "base_pose_to_world_pose_for_arm"):
        return np.asarray(renderer.base_pose_to_world_pose_for_arm(ee_pose_base, arm), dtype=np.float64).reshape(7)
    return _ee_base_pose_to_world_wxyz(renderer, ee_pose_base[:3], ee_pose_base[3:])


def planned_eval_pose_from_plan(
    renderer: ReplayRenderer,
    arm: str,
    plan: Optional[Dict],
    pose_source: str,
) -> Optional[np.ndarray]:
    if not isinstance(plan, dict):
        return None
    if str(plan.get("status", "")) not in {"Success", "Partial"}:
        return None
    if "target_joints" not in plan:
        return None
    solver = getattr(renderer, f"{arm}_ik_solver", None)
    if solver is None or not hasattr(solver, "forward_kinematics"):
        return None

    target_arm = np.asarray(plan["target_joints"], dtype=np.float64).reshape(6)
    if hasattr(renderer, "base_pose_to_world_pose_for_arm"):
        fk_joints = target_arm
    else:
        fk_joints = np.concatenate([np.asarray(renderer.torso_qpos, dtype=np.float64).reshape(4), target_arm], dtype=np.float64)
    ee_pos_base, ee_quat_wxyz_base, _ = solver.forward_kinematics(fk_joints)
    ee_world_wxyz = _ee_base_pose_to_world_wxyz_for_arm(renderer, arm, ee_pos_base, ee_quat_wxyz_base)
    robot = getattr(renderer, "robot", None)
    global_trans = None if robot is None else getattr(robot, f"{arm}_global_trans_matrix", None)
    delta_matrix = None if robot is None else getattr(robot, f"{arm}_delta_matrix", None)
    gripper_bias = 0.12 if robot is None else float(getattr(robot, f"{arm}_gripper_bias", 0.12))
    if global_trans is not None and delta_matrix is not None:
        link_world = pose_wxyz_to_matrix(ee_world_wxyz)
        report_rot = base.orthonormalize_rotation(
            link_world[:3, :3]
            @ np.asarray(global_trans, dtype=np.float64).reshape(3, 3)
            @ np.asarray(delta_matrix, dtype=np.float64).reshape(3, 3)
        )
        report_quat = base.quat_xyzw_to_wxyz(R.from_matrix(report_rot).as_quat())
        if pose_source == "ee":
            report_pos = link_world[:3, 3] + report_rot @ np.array([gripper_bias - 0.12, 0.0, 0.0], dtype=np.float64)
            return np.concatenate([report_pos, report_quat]).astype(np.float64)
        if pose_source == "tcp":
            report_pos = link_world[:3, 3] + report_rot @ np.array([gripper_bias, 0.0, 0.0], dtype=np.float64)
            return np.concatenate([report_pos, report_quat]).astype(np.float64)
    if pose_source == "ee":
        return ee_world_wxyz
    if pose_source == "tcp":
        inv_delta = None if robot is None else getattr(robot, f"{arm}_inv_delta_matrix", None)
        if inv_delta is not None:
            # robot._trans_from_gripper_to_endlink applies E = G * T, where G is the
            # planner TCP/gripper pose and E is the wrist/endlink pose used by IK.
            # Invert that same static transform so the FK target joints are evaluated
            # in the same TCP convention as get_current_tcp_pose().
            gripper_to_ee = np.eye(4, dtype=np.float64)
            gripper_to_ee[:3, :3] = np.asarray(inv_delta, dtype=np.float64).reshape(3, 3)
            gripper_to_ee[:3, 3] = np.array([0.12 - gripper_bias, 0.0, 0.0], dtype=np.float64)
            tcp_world = pose_wxyz_to_matrix(ee_world_wxyz) @ np.linalg.inv(gripper_to_ee)
        else:
            tcp_world = pose_wxyz_to_matrix(ee_world_wxyz)
            # Fallback for older R1-style configs that do not expose the static
            # gripper/endlink transform on the robot object.
            tcp_world[:3, 3] += tcp_world[:3, 0] * 0.12
        quat = base.quat_xyzw_to_wxyz(R.from_matrix(base.orthonormalize_rotation(tcp_world[:3, :3])).as_quat())
        return np.concatenate([tcp_world[:3, 3], quat]).astype(np.float64)
    return None


def plan_is_executable(plan: Optional[Dict]) -> bool:
    if not isinstance(plan, dict):
        return False
    if str(plan.get("status", "")) not in {"Success", "Partial"}:
        return False
    if "position" not in plan or "velocity" not in plan:
        return False
    try:
        return np.asarray(plan["position"], dtype=np.float64).reshape(-1, 6).shape[0] >= 2
    except Exception:
        return False


def tcp_pose_errors(target_pose_world_wxyz: np.ndarray, current_pose_world_wxyz: np.ndarray) -> Tuple[float, float]:
    target_pose_world_wxyz = np.asarray(target_pose_world_wxyz, dtype=np.float64).reshape(7)
    current_pose_world_wxyz = np.asarray(current_pose_world_wxyz, dtype=np.float64).reshape(7)
    pos_err = float(np.linalg.norm(target_pose_world_wxyz[:3] - current_pose_world_wxyz[:3]))
    rot_err = float(base.quat_angle_deg_wxyz(current_pose_world_wxyz[3:], target_pose_world_wxyz[3:]))
    return pos_err, rot_err


def poses_are_effectively_same(
    pose_a_world_wxyz: np.ndarray,
    pose_b_world_wxyz: np.ndarray,
    pos_tol_m: float = 1e-5,
    rot_tol_deg: float = 1e-3,
) -> bool:
    pos_err, rot_err = tcp_pose_errors(
        np.asarray(pose_a_world_wxyz, dtype=np.float64).reshape(7),
        np.asarray(pose_b_world_wxyz, dtype=np.float64).reshape(7),
    )
    return pos_err <= float(pos_tol_m) and rot_err <= float(rot_tol_deg)


def interpolate_joint_trajectory(position: np.ndarray, velocity: np.ndarray, num_steps: int) -> Tuple[np.ndarray, np.ndarray]:
    position = np.asarray(position, dtype=np.float64)
    velocity = np.asarray(velocity, dtype=np.float64)
    if position.shape[0] < 2 or int(num_steps) <= position.shape[0]:
        return position, velocity

    out_pos = []
    dof = position.shape[1]
    zero_vel = np.zeros(dof, dtype=np.float64)
    for idx in range(position.shape[0] - 1):
        start = position[idx]
        end = position[idx + 1]
        seg = np.linspace(start, end, num=max(int(num_steps / max(position.shape[0] - 1, 1)), 2), endpoint=False)
        out_pos.extend(seg.tolist())
    out_pos.append(position[-1].tolist())
    out_pos_arr = np.asarray(out_pos, dtype=np.float64)
    out_vel_arr = np.tile(zero_vel[None, :], (out_pos_arr.shape[0], 1))
    return out_pos_arr, out_vel_arr


def apply_robot_init_pose(renderer: ReplayRenderer, open_gripper: float, settle_steps: int) -> Dict[str, object]:
    left_init = getattr(renderer, "init_left_arm_joints", None)
    right_init = getattr(renderer, "init_right_arm_joints", None)
    init_gripper_open = getattr(renderer, "init_gripper_open", None)
    if init_gripper_open is None:
        init_gripper_open = float(open_gripper)

    vel = np.zeros(6, dtype=np.float64)
    left_applied = False
    right_applied = False
    if left_init is not None:
        renderer.robot.set_arm_joints(np.asarray(left_init, dtype=np.float64).reshape(6), vel, "left")
        left_applied = True
    if right_init is not None:
        renderer.robot.set_arm_joints(np.asarray(right_init, dtype=np.float64).reshape(6), vel, "right")
        right_applied = True

    if left_applied or right_applied:
        renderer.step_scene(steps=max(int(settle_steps), 1))
    renderer.set_grippers(float(init_gripper_open), float(init_gripper_open))
    renderer.step_scene(steps=1)

    return {
        "left_applied": bool(left_applied),
        "right_applied": bool(right_applied),
        "left_joints": None if left_init is None else np.asarray(left_init, dtype=np.float64).reshape(6),
        "right_joints": None if right_init is None else np.asarray(right_init, dtype=np.float64).reshape(6),
        "gripper_open": float(init_gripper_open),
    }


def get_current_arm_joint_vector(renderer: ReplayRenderer, arm: str) -> np.ndarray:
    if renderer.robot is None:
        raise RuntimeError("Robot is unavailable.")
    if arm == "left":
        return np.asarray(renderer.robot.get_left_arm_real_jointState()[:6], dtype=np.float64)
    if arm == "right":
        return np.asarray(renderer.robot.get_right_arm_real_jointState()[:6], dtype=np.float64)
    raise ValueError(f"Unsupported arm: {arm}")


def joint_error_metrics(current_joints: np.ndarray, target_joints: np.ndarray) -> Dict[str, float]:
    current_joints = np.asarray(current_joints, dtype=np.float64).reshape(-1)
    target_joints = np.asarray(target_joints, dtype=np.float64).reshape(-1)
    delta = target_joints - current_joints
    return {
        "max_abs_err_rad": float(np.max(np.abs(delta))) if delta.size > 0 else 0.0,
        "l2_err_rad": float(np.linalg.norm(delta)),
    }


def settle_arms_to_targets(
    renderer: ReplayRenderer,
    target_joints_by_arm: Dict[str, np.ndarray],
    max_wait_steps: int,
    tol_rad: float,
    attached_actor_by_arm: Optional[Dict[str, Optional[sapien.Entity]]] = None,
    tcp_to_object_by_arm: Optional[Dict[str, Optional[np.ndarray]]] = None,
    object_replay: Optional[ExecutionObjectReplayConfig] = None,
) -> Dict[str, Dict[str, float]]:
    arms = [arm for arm in ("left", "right") if arm in target_joints_by_arm]
    targets = {
        arm: np.asarray(target_joints_by_arm[arm], dtype=np.float64).reshape(6)
        for arm in arms
    }
    max_wait_steps = max(int(max_wait_steps), 0)
    tol_rad = float(tol_rad)

    def _collect() -> Dict[str, Dict[str, float]]:
        metrics: Dict[str, Dict[str, float]] = {}
        for arm_name in arms:
            metrics[arm_name] = joint_error_metrics(get_current_arm_joint_vector(renderer, arm_name), targets[arm_name])
        return metrics

    zero_vel = np.zeros(6, dtype=np.float64)
    for arm_name in arms:
        renderer.robot.set_arm_joints(targets[arm_name], zero_vel, arm_name)

    final_metrics = _collect()
    if max_wait_steps <= 0:
        return final_metrics

    for _ in range(max_wait_steps):
        if all(metrics["max_abs_err_rad"] <= tol_rad for metrics in final_metrics.values()):
            break
        for arm_name in arms:
            renderer.robot.set_arm_joints(targets[arm_name], zero_vel, arm_name)
        if object_replay is not None:
            update_execution_object_replay(object_replay, 1.0)
        else:
            for arm_name in arms:
                actor = None if attached_actor_by_arm is None else attached_actor_by_arm.get(arm_name)
                tcp_to_object = None if tcp_to_object_by_arm is None else tcp_to_object_by_arm.get(arm_name)
                if actor is None or tcp_to_object is None:
                    continue
                tcp_pose = renderer.get_current_tcp_pose(arm_name)
                object_world = pose_wxyz_to_matrix(tcp_pose) @ tcp_to_object
                quat = base.quat_xyzw_to_wxyz(R.from_matrix(object_world[:3, :3]).as_quat())
                set_actor_pose(actor, np.concatenate([object_world[:3, 3], quat]))
        renderer.step_scene(steps=1)
        final_metrics = _collect()

    return final_metrics


def emit_init_prefix_frames(
    renderer: ReplayRenderer,
    head_writer: cv2.VideoWriter,
    third_writer: Optional[cv2.VideoWriter],
    use_overlay: bool,
    init_info: Dict[str, object],
    fixed_frames: int,
    arm_label: str,
    debug_visuals: Optional[DebugVisualBundle] = None,
    debug_execution_state: Optional[DebugExecutionState] = None,
    pure_scene_main: bool = False,
    use_overlay_debug: Optional[bool] = None,
) -> int:
    total = max(int(fixed_frames), 0)
    if total <= 0:
        return 0
    for idx in range(total):
        record_frame(
            renderer,
            head_writer,
            third_writer,
            [
                "stage=init_prefix",
                f"arm={arm_label}",
                f"fixed_frame={idx + 1}/{total}",
                f"gripper_open={float(init_info['gripper_open']):.3f}",
            ],
            use_overlay,
            debug_visuals,
            debug_execution_state,
            pure_scene_main=pure_scene_main,
            use_overlay_debug=use_overlay_debug,
        )
    return total


def execute_single_arm_plan(
    renderer: ReplayRenderer,
    arm: str,
    plan: Optional[Dict],
    label: str,
    head_writer: cv2.VideoWriter,
    third_writer: Optional[cv2.VideoWriter],
    use_overlay: bool,
    attached_actor: Optional[sapien.Entity] = None,
    tcp_to_object: Optional[np.ndarray] = None,
    execute_interp_steps: int = 24,
    settle_steps: int = 4,
    hold_frames_after_stage: int = 0,
    target_visual_pose: Optional[np.ndarray] = None,
    target_visual_label: Optional[str] = None,
    debug_visuals: Optional[DebugVisualBundle] = None,
    debug_execution_state: Optional[DebugExecutionState] = None,
    object_replay: Optional[ExecutionObjectReplayConfig] = None,
    pure_scene_main: bool = False,
    use_overlay_debug: Optional[bool] = None,
    joint_command_scene_steps: int = 2,
    joint_target_wait_steps: int = 60,
    joint_target_wait_tol_rad: float = 0.01,
) -> str:
    status = renderer._plan_status(plan)
    overlay_lines = [f"stage={label}", f"arm={arm}", f"status={status}"]
    if target_visual_label:
        overlay_lines.append(f"goal={target_visual_label}")
    set_single_arm_target_visual(renderer, arm, target_visual_pose)
    if not plan_is_executable(plan):
        record_frame(renderer, head_writer, third_writer, overlay_lines, use_overlay, debug_visuals, debug_execution_state, pure_scene_main=pure_scene_main, use_overlay_debug=use_overlay_debug)
        return status

    position = np.asarray(plan["position"], dtype=np.float64)
    velocity = np.asarray(plan["velocity"], dtype=np.float64)
    position, velocity = interpolate_joint_trajectory(position, velocity, execute_interp_steps)
    scene_steps_per_waypoint = max(int(joint_command_scene_steps), 1)
    for idx in range(position.shape[0]):
        renderer.robot.set_arm_joints(position[idx], velocity[idx], arm)
        if object_replay is not None:
            update_execution_object_replay(object_replay, 0.0 if position.shape[0] <= 1 else float(idx) / float(position.shape[0] - 1))
        elif attached_actor is not None and tcp_to_object is not None:
            tcp_pose = renderer.get_current_tcp_pose(arm)
            object_world = pose_wxyz_to_matrix(tcp_pose) @ tcp_to_object
            quat = base.quat_xyzw_to_wxyz(R.from_matrix(object_world[:3, :3]).as_quat())
            set_actor_pose(attached_actor, np.concatenate([object_world[:3, 3], quat]))
        renderer.step_scene(steps=scene_steps_per_waypoint)
        record_frame(
            renderer,
            head_writer,
            third_writer,
            overlay_lines + [f"plan_step={idx + 1}/{position.shape[0]}"],
            use_overlay,
            debug_visuals,
            debug_execution_state,
            pure_scene_main=pure_scene_main,
            use_overlay_debug=use_overlay_debug,
        )
    renderer.step_scene(steps=max(int(settle_steps), 0))
    final_joint_metrics = settle_arms_to_targets(
        renderer,
        {arm: np.asarray(position[-1], dtype=np.float64).reshape(6)},
        max_wait_steps=joint_target_wait_steps,
        tol_rad=joint_target_wait_tol_rad,
        attached_actor_by_arm={arm: attached_actor} if attached_actor is not None else None,
        tcp_to_object_by_arm={arm: tcp_to_object} if tcp_to_object is not None else None,
        object_replay=object_replay,
    )
    if object_replay is not None:
        update_execution_object_replay(object_replay, 1.0)
    final_joint_max_err = float(final_joint_metrics.get(arm, {}).get("max_abs_err_rad", 0.0))
    record_frame(
        renderer,
        head_writer,
        third_writer,
        overlay_lines + ["plan_step=done", f"joint_max_err={final_joint_max_err:.4f}rad"],
        use_overlay,
        debug_visuals,
        debug_execution_state,
        pure_scene_main=pure_scene_main,
        use_overlay_debug=use_overlay_debug,
    )
    for hold_idx in range(max(int(hold_frames_after_stage), 0)):
        record_frame(
            renderer,
            head_writer,
            third_writer,
            overlay_lines + [f"hold={hold_idx + 1}/{hold_frames_after_stage}"],
            use_overlay,
            debug_visuals,
            debug_execution_state,
            pure_scene_main=pure_scene_main,
            use_overlay_debug=use_overlay_debug,
        )
    return status


def execute_stage_until_reached(
    renderer: ReplayRenderer,
    arm: str,
    target_pose_world_wxyz: np.ndarray,
    label: str,
    head_writer: cv2.VideoWriter,
    third_writer: Optional[cv2.VideoWriter],
    use_overlay: bool,
    args: argparse.Namespace,
    attached_actor: Optional[sapien.Entity] = None,
    tcp_to_object: Optional[np.ndarray] = None,
    target_visual_pose: Optional[np.ndarray] = None,
    target_visual_label: Optional[str] = None,
    debug_visuals: Optional[DebugVisualBundle] = None,
    debug_execution_state: Optional[DebugExecutionState] = None,
    supervision_targets: Optional[Dict[str, np.ndarray]] = None,
    object_replay: Optional[ExecutionObjectReplayConfig] = None,
    pure_scene_main: bool = False,
    use_overlay_debug: Optional[bool] = None,
) -> Dict[str, object]:
    last_status = "Missing"
    last_pos_err = float("inf")
    last_rot_err = float("inf")
    attempts = 0
    attempt_history = []
    max_attempts = stage_attempt_budget(args)
    attempt = 0
    while True:
        attempt += 1
        attempts = attempt
        current_eval_pose_before_plan = get_current_pose_for_error(renderer, arm, args.reach_error_pose_source)
        target_eval_pose = target_pose_for_error(renderer, arm, target_pose_world_wxyz, args.reach_error_pose_source)
        pre_plan_diag = plan_request_diagnostics(current_eval_pose_before_plan, target_eval_pose)
        print(
            f"[plan-request] stage={label} arm={arm} try={attempt} "
            f"dx={float(pre_plan_diag['error']['dx_m']):.4f} dy={float(pre_plan_diag['error']['dy_m']):.4f} dz={float(pre_plan_diag['error']['dz_m']):.4f} "
            f"dist={float(pre_plan_diag['error']['dist_m']):.4f} rot={float(pre_plan_diag['error']['rot_err_deg']):.2f} "
            f"fwd_rot={float(pre_plan_diag['error']['forward_axis_err_deg']):.2f} "
            f"fwd_cm={colorize_forward_cm(float(pre_plan_diag['error']['forward_axis_signed_err_cm']))} "
            f"lat_cm={float(pre_plan_diag['error']['lateral_to_forward_axis_cm']):.2f} "
            f"theory={short_direction_label(str(pre_plan_diag['theoretical_forward_axis_motion']))}"
        )
        plan = renderer.plan_path(arm, target_pose_world_wxyz)
        update_ik_waypoint_visuals(renderer, args, {arm: plan})
        planned_eval_pose = planned_eval_pose_from_plan(renderer, arm, plan, args.reach_error_pose_source)
        if planned_eval_pose is not None:
            plan_vs_target = plan_request_diagnostics(planned_eval_pose, target_eval_pose)
            plan_vs_current = plan_request_diagnostics(current_eval_pose_before_plan, planned_eval_pose)
            print(
                f"[plan-solution] stage={label} arm={arm} try={attempt} "
                f"plan_vs_target_fwd_cm={colorize_forward_cm(float(plan_vs_target['error']['forward_axis_signed_err_cm']))} "
                f"plan_vs_target_lat_cm={float(plan_vs_target['error']['lateral_to_forward_axis_cm']):.2f} "
                f"plan_vs_target_dist={float(plan_vs_target['error']['dist_m']):.4f} "
                f"plan_vs_target_rot={float(plan_vs_target['error']['rot_err_deg']):.2f} "
                f"plan_vs_current_fwd_cm={colorize_forward_cm(float(plan_vs_current['error']['forward_axis_signed_err_cm']))} "
                f"plan_vs_current_lat_cm={float(plan_vs_current['error']['lateral_to_forward_axis_cm']):.2f} "
                f"plan_vs_current_dist={float(plan_vs_current['error']['dist_m']):.4f} "
                f"plan_vs_current_rot={float(plan_vs_current['error']['rot_err_deg']):.2f} "
                f"theory={short_direction_label(str(plan_vs_current['theoretical_forward_axis_motion']))}"
            )
        last_status = execute_single_arm_plan(
            renderer=renderer,
            arm=arm,
            plan=plan,
            label=f"{label}_try{attempt}",
            head_writer=head_writer,
            third_writer=third_writer,
            use_overlay=use_overlay,
            attached_actor=attached_actor,
            tcp_to_object=tcp_to_object,
            execute_interp_steps=args.execute_interp_steps,
            settle_steps=args.settle_steps,
            hold_frames_after_stage=args.hold_frames_after_stage,
            target_visual_pose=target_visual_pose,
            target_visual_label=target_visual_label,
            debug_visuals=debug_visuals,
            debug_execution_state=debug_execution_state,
            object_replay=object_replay,
            pure_scene_main=pure_scene_main,
            use_overlay_debug=use_overlay_debug,
            joint_command_scene_steps=args.joint_command_scene_steps,
            joint_target_wait_steps=args.joint_target_wait_steps,
            joint_target_wait_tol_rad=args.joint_target_wait_tol_rad,
        )
        current_eval_pose = get_current_pose_for_error(renderer, arm, args.reach_error_pose_source)
        stage_error = pose_error_breakdown(target_eval_pose, current_eval_pose)
        last_pos_err = float(stage_error["dist_m"])
        last_rot_err = float(stage_error["rot_err_deg"])
        reached = (
            last_status == "Success"
            and last_pos_err <= float(args.reach_pos_tol_m)
            and last_rot_err <= float(args.reach_rot_tol_deg)
        )
        supervision_errors = compute_supervision_errors(renderer, supervision_targets or {}, args.reach_error_pose_source) if supervision_targets else {}
        attempt_history.append(
            {
                "attempt": attempt,
                "pre_plan": pre_plan_diag,
                "status": last_status,
                "target_error": stage_error,
                "pos_err_m": last_pos_err,
                "rot_err_deg": last_rot_err,
                "reached": bool(reached),
                "supervision_errors": supervision_errors,
            }
        )
        print(
            f"[attempt] stage={label} arm={arm} try={attempt} status={last_status} "
            f"dx={stage_error['dx_m']:.4f} dy={stage_error['dy_m']:.4f} dz={stage_error['dz_m']:.4f} "
            f"dist={stage_error['dist_m']:.4f} rot={stage_error['rot_err_deg']:.2f} "
            f"fwd_rot={stage_error['forward_axis_err_deg']:.2f} "
            f"fwd_cm={colorize_forward_cm(float(stage_error['forward_axis_signed_err_cm']))} "
            f"lat_cm={float(stage_error['lateral_to_forward_axis_cm']):.2f} "
            f"reached={int(reached)}"
        )
        for supervision_arm, supervision_error in supervision_errors.items():
            print(
                f"[attempt-supervision] stage={label} exec_arm={arm} supervised_arm={supervision_arm} try={attempt} "
                f"dx={float(supervision_error.get('dx_m', 0.0)):.4f} dy={float(supervision_error.get('dy_m', 0.0)):.4f} "
                f"dz={float(supervision_error.get('dz_m', 0.0)):.4f} dist={float(supervision_error.get('pos_err_m', 0.0)):.4f} "
                f"rot={float(supervision_error.get('rot_err_deg', 0.0)):.2f} "
                f"fwd_rot={float(supervision_error.get('forward_axis_err_deg', 0.0)):.2f} "
                f"fwd_cm={colorize_forward_cm(float(supervision_error.get('forward_axis_signed_err_cm', 0.0)))} "
                f"lat_cm={float(supervision_error.get('lateral_to_forward_axis_cm', 0.0)):.2f}"
            )
        record_frame(
            renderer,
            head_writer,
            third_writer,
            [
                f"stage={label}",
                f"arm={arm}",
                f"attempt={attempt}/{max_attempts if max_attempts is not None else 'until_reached'}",
                f"status={last_status}",
                f"pos_err={last_pos_err:.4f}m",
                f"rot_err={last_rot_err:.2f}deg",
                f"reached={int(reached)}",
                *([f"right_sup_pos={supervision_errors['right']['pos_err_m']:.4f}m", f"right_sup_rot={supervision_errors['right']['rot_err_deg']:.2f}deg"] if "right" in supervision_errors and arm != "right" else []),
            ],
            use_overlay,
            debug_visuals,
            debug_execution_state,
            pure_scene_main=pure_scene_main,
            use_overlay_debug=use_overlay_debug,
        )
        if reached:
            clear_ik_waypoint_visuals(renderer)
            return {
                "status": last_status,
                "attempts": attempts,
                "reached": True,
                "pos_err_m": last_pos_err,
                "rot_err_deg": last_rot_err,
                "attempt_history": attempt_history,
            }
        if max_attempts is not None and attempt >= max_attempts:
            break

    clear_ik_waypoint_visuals(renderer)
    return {
        "status": last_status,
        "attempts": attempts,
        "reached": False,
        "pos_err_m": last_pos_err,
        "rot_err_deg": last_rot_err,
        "attempt_history": attempt_history,
    }


def skipped_action_result_for_arm(
    renderer: ReplayRenderer,
    arm: str,
    action_pose_world_wxyz: np.ndarray,
    args: argparse.Namespace,
    reason: str,
) -> Dict[str, object]:
    current_eval_pose = get_current_pose_for_error(renderer, arm, args.reach_error_pose_source)
    target_eval_pose = target_pose_for_error(renderer, arm, action_pose_world_wxyz, args.reach_error_pose_source)
    error_breakdown = pose_error_breakdown(target_eval_pose, current_eval_pose)
    return {
        "status": "Skipped",
        "skip_reason": str(reason),
        "attempts": 0,
        "reached": False,
        "target_error": error_breakdown,
        "pos_err_m": float(error_breakdown["dist_m"]),
        "rot_err_deg": float(error_breakdown["rot_err_deg"]),
        "attempt_history": [],
    }


def skipped_dual_action_result(
    renderer: ReplayRenderer,
    action_targets: Dict[str, np.ndarray],
    args: argparse.Namespace,
    reason: str,
) -> Dict[str, object]:
    return {
        "attempts": 0,
        "reached": False,
        "skip_reason": str(reason),
        "arms": {
            arm: skipped_action_result_for_arm(renderer, arm, target, args, reason)
            for arm, target in action_targets.items()
        },
        "attempt_history": [],
    }


def execute_dual_arm_plan(
    renderer: ReplayRenderer,
    plans_by_arm: Dict[str, Optional[Dict]],
    label: str,
    head_writer: cv2.VideoWriter,
    third_writer: Optional[cv2.VideoWriter],
    use_overlay: bool,
    execute_interp_steps: int = 24,
    settle_steps: int = 4,
    hold_frames_after_stage: int = 0,
    debug_visuals: Optional[DebugVisualBundle] = None,
    debug_execution_state: Optional[DebugExecutionState] = None,
    attached_actor_by_arm: Optional[Dict[str, Optional[sapien.Entity]]] = None,
    tcp_to_object_by_arm: Optional[Dict[str, Optional[np.ndarray]]] = None,
    object_replay: Optional[ExecutionObjectReplayConfig] = None,
    pure_scene_main: bool = False,
    use_overlay_debug: Optional[bool] = None,
    joint_command_scene_steps: int = 2,
    joint_target_wait_steps: int = 60,
    joint_target_wait_tol_rad: float = 0.01,
    require_all_plans: bool = True,
    print_pose_every: int = 0,
) -> Dict[str, str]:
    arms = [arm for arm in ("left", "right") if arm in plans_by_arm]
    statuses: Dict[str, str] = {arm: renderer._plan_status(plans_by_arm.get(arm)) for arm in arms}
    overlay_lines = [
        f"stage={label}",
        f"left_status={statuses.get('left', 'NA')}",
        f"right_status={statuses.get('right', 'NA')}",
    ]

    if bool(require_all_plans) and len(arms) > 1 and any(not plan_is_executable(plans_by_arm.get(arm)) for arm in arms):
        print(
            "[dual-plan] skip stage execution because not all arm plans are executable: "
            + " ".join(f"{arm}={statuses.get(arm, 'Missing')}" for arm in arms)
        )
        record_frame(
            renderer,
            head_writer,
            third_writer,
            overlay_lines + ["dual_skip=not_all_plans_success"],
            use_overlay,
            debug_visuals,
            debug_execution_state,
            pure_scene_main=pure_scene_main,
            use_overlay_debug=use_overlay_debug,
        )
        return statuses

    trajectories: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    max_steps = 0
    for arm in arms:
        if not plan_is_executable(plans_by_arm.get(arm)):
            continue
        plan = plans_by_arm[arm]
        position = np.asarray(plan["position"], dtype=np.float64)
        velocity = np.asarray(plan["velocity"], dtype=np.float64)
        position, velocity = interpolate_joint_trajectory(position, velocity, execute_interp_steps)
        trajectories[arm] = (position, velocity)
        max_steps = max(max_steps, int(position.shape[0]))

    if max_steps <= 0:
        record_frame(renderer, head_writer, third_writer, overlay_lines, use_overlay, debug_visuals, debug_execution_state, pure_scene_main=pure_scene_main, use_overlay_debug=use_overlay_debug)
        return statuses

    hold_joints_by_arm: Dict[str, np.ndarray] = {}
    zero_hold_vel = np.zeros(6, dtype=np.float64)
    for arm in arms:
        if arm not in trajectories:
            hold_joints_by_arm[arm] = get_current_arm_joint_vector(renderer, arm)

    scene_steps_per_waypoint = max(int(joint_command_scene_steps), 1)
    for step_idx in range(max_steps):
        for arm in arms:
            if arm not in trajectories:
                continue
            position, velocity = trajectories[arm]
            local_idx = min(step_idx, int(position.shape[0]) - 1)
            renderer.robot.set_arm_joints(position[local_idx], velocity[local_idx], arm)
        for arm, hold_joints in hold_joints_by_arm.items():
            renderer.robot.set_arm_joints(hold_joints, zero_hold_vel, arm)
        for arm in arms:
            actor = None if attached_actor_by_arm is None else attached_actor_by_arm.get(arm)
            tcp_to_object = None if tcp_to_object_by_arm is None else tcp_to_object_by_arm.get(arm)
            if object_replay is not None:
                continue
            if actor is not None and tcp_to_object is not None:
                tcp_pose = renderer.get_current_tcp_pose(arm)
                object_world = pose_wxyz_to_matrix(tcp_pose) @ tcp_to_object
                quat = base.quat_xyzw_to_wxyz(R.from_matrix(object_world[:3, :3]).as_quat())
                set_actor_pose(actor, np.concatenate([object_world[:3, 3], quat]))
        if object_replay is not None:
            update_execution_object_replay(object_replay, 0.0 if max_steps <= 1 else float(step_idx) / float(max_steps - 1))
        renderer.step_scene(steps=scene_steps_per_waypoint)
        if int(print_pose_every) > 0 and (step_idx == 0 or (step_idx + 1) % int(print_pose_every) == 0 or step_idx == max_steps - 1):
            pose_parts = []
            for arm in arms:
                tcp = np.asarray(renderer.get_current_tcp_pose(arm), dtype=np.float64).reshape(7)
                ee_pose = renderer.robot.get_left_ee_pose() if arm == "left" else renderer.robot.get_right_ee_pose()
                ee = pose_like_to_world_wxyz(ee_pose)
                pose_parts.append(
                    f"{arm}:tcp=({tcp[0]:+.4f},{tcp[1]:+.4f},{tcp[2]:+.4f}) "
                    f"ee=({ee[0]:+.4f},{ee[1]:+.4f},{ee[2]:+.4f})"
                )
            print(f"[exec-pose] stage={label} step={step_idx + 1}/{max_steps} " + " ".join(pose_parts), flush=True)
        record_frame(
            renderer,
            head_writer,
            third_writer,
            overlay_lines + [f"plan_step={step_idx + 1}/{max_steps}"],
            use_overlay,
            debug_visuals,
            debug_execution_state,
            pure_scene_main=pure_scene_main,
            use_overlay_debug=use_overlay_debug,
        )

    renderer.step_scene(steps=max(int(settle_steps), 0))
    final_joint_targets = {
        arm: np.asarray(trajectories[arm][0][-1], dtype=np.float64).reshape(6)
        for arm in arms
        if arm in trajectories
    }
    final_joint_targets.update(hold_joints_by_arm)
    final_joint_metrics = settle_arms_to_targets(
        renderer,
        final_joint_targets,
        max_wait_steps=joint_target_wait_steps,
        tol_rad=joint_target_wait_tol_rad,
        attached_actor_by_arm=attached_actor_by_arm,
        tcp_to_object_by_arm=tcp_to_object_by_arm,
        object_replay=object_replay,
    )
    if int(print_pose_every) > 0:
        pose_parts = []
        for arm in arms:
            tcp = np.asarray(renderer.get_current_tcp_pose(arm), dtype=np.float64).reshape(7)
            ee_pose = renderer.robot.get_left_ee_pose() if arm == "left" else renderer.robot.get_right_ee_pose()
            ee = pose_like_to_world_wxyz(ee_pose)
            joint_err = float(final_joint_metrics.get(arm, {}).get("max_abs_err_rad", 0.0))
            pose_parts.append(
                f"{arm}:tcp=({tcp[0]:+.4f},{tcp[1]:+.4f},{tcp[2]:+.4f}) "
                f"ee=({ee[0]:+.4f},{ee[1]:+.4f},{ee[2]:+.4f}) joint_err={joint_err:.4f}"
            )
        print(f"[exec-pose] stage={label} settled " + " ".join(pose_parts), flush=True)
    if object_replay is not None:
        update_execution_object_replay(object_replay, 1.0)
    record_frame(
        renderer,
        head_writer,
        third_writer,
        overlay_lines
        + ["plan_step=done"]
        + [
            f"{arm}_joint_max_err={float(final_joint_metrics.get(arm, {}).get('max_abs_err_rad', 0.0)):.4f}rad"
            for arm in arms
            if arm in final_joint_metrics
        ],
        use_overlay,
        debug_visuals,
        debug_execution_state,
        pure_scene_main=pure_scene_main,
        use_overlay_debug=use_overlay_debug,
    )
    for hold_idx in range(max(int(hold_frames_after_stage), 0)):
        record_frame(
            renderer,
            head_writer,
            third_writer,
            overlay_lines + [f"hold={hold_idx + 1}/{hold_frames_after_stage}"],
            use_overlay,
            debug_visuals,
            debug_execution_state,
            pure_scene_main=pure_scene_main,
            use_overlay_debug=use_overlay_debug,
        )
    return statuses


def execute_dual_stage_until_reached(
    renderer: ReplayRenderer,
    target_pose_world_wxyz_by_arm: Dict[str, np.ndarray],
    label: str,
    head_writer: cv2.VideoWriter,
    third_writer: Optional[cv2.VideoWriter],
    use_overlay: bool,
    args: argparse.Namespace,
    debug_visuals: Optional[DebugVisualBundle] = None,
    debug_execution_state: Optional[DebugExecutionState] = None,
    attached_actor_by_arm: Optional[Dict[str, Optional[sapien.Entity]]] = None,
    tcp_to_object_by_arm: Optional[Dict[str, Optional[np.ndarray]]] = None,
    object_replay: Optional[ExecutionObjectReplayConfig] = None,
    pure_scene_main: bool = False,
    use_overlay_debug: Optional[bool] = None,
) -> Dict[str, object]:
    arms = [arm for arm in ("left", "right") if arm in target_pose_world_wxyz_by_arm]
    max_attempts = stage_attempt_budget(args)

    attempt_history: List[Dict[str, object]] = []
    last_arm_metrics: Dict[str, Dict[str, object]] = {}
    attempt = 0
    while True:
        attempt += 1
        pre_plan_by_arm: Dict[str, Dict[str, object]] = {}
        for arm in arms:
            current_eval_pose_before_plan = get_current_pose_for_error(renderer, arm, args.reach_error_pose_source)
            target_eval_pose = target_pose_for_error(renderer, arm, target_pose_world_wxyz_by_arm[arm], args.reach_error_pose_source)
            pre_plan_diag = plan_request_diagnostics(current_eval_pose_before_plan, target_eval_pose)
            pre_plan_by_arm[arm] = pre_plan_diag
        print(f"[plan-request] stage={label} try={attempt}")
        for arm in arms:
            print(
                f"  {arm}: "
                f"dx={float(pre_plan_by_arm[arm]['error']['dx_m']):.4f} "
                f"dy={float(pre_plan_by_arm[arm]['error']['dy_m']):.4f} "
                f"dz={float(pre_plan_by_arm[arm]['error']['dz_m']):.4f} "
                f"dist={float(pre_plan_by_arm[arm]['error']['dist_m']):.4f} "
                f"rot={float(pre_plan_by_arm[arm]['error']['rot_err_deg']):.2f} "
                f"fwd_rot={float(pre_plan_by_arm[arm]['error']['forward_axis_err_deg']):.2f} "
                f"fwd_cm={colorize_forward_cm(float(pre_plan_by_arm[arm]['error']['forward_axis_signed_err_cm']))} "
                f"lat_cm={float(pre_plan_by_arm[arm]['error']['lateral_to_forward_axis_cm']):.2f} "
                f"theory={short_direction_label(str(pre_plan_by_arm[arm]['theoretical_forward_axis_motion']))}"
            )
        plans_by_arm: Dict[str, Optional[Dict]] = {
            arm: renderer.plan_path(arm, target_pose_world_wxyz_by_arm[arm]) for arm in arms
        }
        for arm in arms:
            plan = plans_by_arm.get(arm)
            if renderer._plan_status(plan) not in {"Success", "Partial"} and isinstance(plan, dict):
                reason = plan.get("reason", "unknown")
                waypoint_index = plan.get("failed_waypoint_index")
                waypoint_count = plan.get("waypoint_count")
                waypoint_text = ""
                if waypoint_index is not None and waypoint_count is not None:
                    waypoint_text = f" waypoint={int(waypoint_index)}/{int(waypoint_count)}"
                print(f"[plan-fail] stage={label} try={attempt} arm={arm} reason={reason}{waypoint_text}")
            elif renderer._plan_status(plan) == "Partial" and isinstance(plan, dict):
                reason = plan.get("reason", "partial")
                waypoint_index = plan.get("failed_waypoint_index")
                waypoint_count = plan.get("waypoint_count")
                solved_count = plan.get("solved_waypoint_count")
                waypoint_text = ""
                if waypoint_index is not None and waypoint_count is not None:
                    waypoint_text = f" failed_waypoint={int(waypoint_index)}/{int(waypoint_count)}"
                solved_text = "" if solved_count is None else f" solved_prefix={int(solved_count)}"
                print(f"[plan-partial] stage={label} try={attempt} arm={arm} reason={reason}{waypoint_text}{solved_text}")
        update_ik_waypoint_visuals(renderer, args, plans_by_arm)
        plan_solution_by_arm: Dict[str, Dict[str, object]] = {}
        for arm in arms:
            planned_eval_pose = planned_eval_pose_from_plan(renderer, arm, plans_by_arm.get(arm), args.reach_error_pose_source)
            if planned_eval_pose is None:
                continue
            target_eval_pose = target_pose_for_error(renderer, arm, target_pose_world_wxyz_by_arm[arm], args.reach_error_pose_source)
            current_eval_pose_before_plan = np.asarray(pre_plan_by_arm[arm]["current_pose_world_wxyz"], dtype=np.float64).reshape(7)
            plan_vs_target = plan_request_diagnostics(planned_eval_pose, target_eval_pose)
            plan_vs_current = plan_request_diagnostics(current_eval_pose_before_plan, planned_eval_pose)
            plan_solution_by_arm[arm] = {
                "planned_eval_pose_world_wxyz": planned_eval_pose.tolist(),
                "plan_vs_target": plan_vs_target,
                "plan_vs_current": plan_vs_current,
            }
        if plan_solution_by_arm:
            print(f"[plan-solution] stage={label} try={attempt}")
            for arm in arms:
                if arm not in plan_solution_by_arm:
                    continue
                print(
                    f"  {arm}: "
                    f"plan_vs_target_fwd_cm={colorize_forward_cm(float(plan_solution_by_arm[arm]['plan_vs_target']['error']['forward_axis_signed_err_cm']))} "
                    f"plan_vs_target_lat_cm={float(plan_solution_by_arm[arm]['plan_vs_target']['error']['lateral_to_forward_axis_cm']):.2f} "
                    f"plan_vs_target_dist={float(plan_solution_by_arm[arm]['plan_vs_target']['error']['dist_m']):.4f} "
                    f"plan_vs_target_rot={float(plan_solution_by_arm[arm]['plan_vs_target']['error']['rot_err_deg']):.2f} "
                    f"plan_vs_current_fwd_cm={colorize_forward_cm(float(plan_solution_by_arm[arm]['plan_vs_current']['error']['forward_axis_signed_err_cm']))} "
                    f"plan_vs_current_lat_cm={float(plan_solution_by_arm[arm]['plan_vs_current']['error']['lateral_to_forward_axis_cm']):.2f} "
                    f"plan_vs_current_dist={float(plan_solution_by_arm[arm]['plan_vs_current']['error']['dist_m']):.4f} "
                    f"plan_vs_current_rot={float(plan_solution_by_arm[arm]['plan_vs_current']['error']['rot_err_deg']):.2f} "
                    f"theory={short_direction_label(str(plan_solution_by_arm[arm]['plan_vs_current']['theoretical_forward_axis_motion']))}"
                )
        statuses = execute_dual_arm_plan(
            renderer=renderer,
            plans_by_arm=plans_by_arm,
            label=f"{label}_try{attempt}",
            head_writer=head_writer,
            third_writer=third_writer,
            use_overlay=use_overlay,
            execute_interp_steps=args.execute_interp_steps,
            settle_steps=args.settle_steps,
            hold_frames_after_stage=args.hold_frames_after_stage,
            debug_visuals=debug_visuals,
            debug_execution_state=debug_execution_state,
            attached_actor_by_arm=attached_actor_by_arm,
            tcp_to_object_by_arm=tcp_to_object_by_arm,
            object_replay=object_replay,
            pure_scene_main=pure_scene_main,
            use_overlay_debug=use_overlay_debug,
            joint_command_scene_steps=args.joint_command_scene_steps,
            joint_target_wait_steps=args.joint_target_wait_steps,
            joint_target_wait_tol_rad=args.joint_target_wait_tol_rad,
            require_all_plans=bool(args.dual_stage_require_all_plans),
            print_pose_every=int(args.print_execution_pose_every),
        )

        arm_metrics: Dict[str, Dict[str, object]] = {}
        for arm in arms:
            current_eval_pose = get_current_pose_for_error(renderer, arm, args.reach_error_pose_source)
            target_eval_pose = target_pose_for_error(renderer, arm, target_pose_world_wxyz_by_arm[arm], args.reach_error_pose_source)
            error_breakdown = pose_error_breakdown(target_eval_pose, current_eval_pose)
            pos_err = float(error_breakdown["dist_m"])
            rot_err = float(error_breakdown["rot_err_deg"])
            reached = (
                statuses.get(arm, "Missing") == "Success"
                and pos_err <= float(args.reach_pos_tol_m)
                and rot_err <= float(args.reach_rot_tol_deg)
            )
            arm_metrics[arm] = {
                "status": statuses.get(arm, "Missing"),
                "target_error": error_breakdown,
                "pos_err_m": float(pos_err),
                "rot_err_deg": float(rot_err),
                "reached": bool(reached),
            }

        stage_reached = all(bool(arm_metrics[arm]["reached"]) for arm in arms)
        attempt_history.append(
            {
                "attempt": attempt,
                "pre_plan_by_arm": pre_plan_by_arm,
                "plan_solution_by_arm": plan_solution_by_arm,
                "arms": arm_metrics,
                "reached": bool(stage_reached),
            }
        )
        print(f"[attempt] stage={label} try={attempt}")
        for arm in arms:
            print(
                f"  {arm}: "
                f"dx={float(arm_metrics[arm]['target_error']['dx_m']):.4f} "
                f"dy={float(arm_metrics[arm]['target_error']['dy_m']):.4f} "
                f"dz={float(arm_metrics[arm]['target_error']['dz_m']):.4f} "
                f"dist={float(arm_metrics[arm]['target_error']['dist_m']):.4f} "
                f"rot={float(arm_metrics[arm]['target_error']['rot_err_deg']):.2f} "
                f"fwd_rot={float(arm_metrics[arm]['target_error']['forward_axis_err_deg']):.2f} "
                f"fwd_cm={colorize_forward_cm(float(arm_metrics[arm]['target_error']['forward_axis_signed_err_cm']))} "
                f"lat_cm={float(arm_metrics[arm]['target_error']['lateral_to_forward_axis_cm']):.2f} "
                f"reached={int(bool(arm_metrics[arm]['reached']))}"
            )

        overlay_lines = [
            f"stage={label}",
            f"attempt={attempt}/{max_attempts if max_attempts is not None else 'until_reached'}",
            f"left_reached={int(arm_metrics.get('left', {}).get('reached', False))}",
            f"right_reached={int(arm_metrics.get('right', {}).get('reached', False))}",
            f"both_reached={int(stage_reached)}",
        ]
        record_frame(renderer, head_writer, third_writer, overlay_lines, use_overlay, debug_visuals, debug_execution_state, pure_scene_main=pure_scene_main, use_overlay_debug=use_overlay_debug)
        last_arm_metrics = arm_metrics
        if stage_reached:
            clear_ik_waypoint_visuals(renderer)
            return {
                "attempts": attempt,
                "reached": True,
                "arms": arm_metrics,
                "attempt_history": attempt_history,
            }
        if max_attempts is not None and attempt >= max_attempts:
            break

    clear_ik_waypoint_visuals(renderer)
    return {
        "attempts": attempt,
        "reached": False,
        "arms": last_arm_metrics,
        "attempt_history": attempt_history,
    }


def pause_after_keyframe1(
    renderer: ReplayRenderer,
    head_writer: cv2.VideoWriter,
    third_writer: Optional[cv2.VideoWriter],
    use_overlay: bool,
    args: argparse.Namespace,
    arm_label: str,
    goal_frame: int,
    debug_visuals: Optional[DebugVisualBundle] = None,
    debug_execution_state: Optional[DebugExecutionState] = None,
    pure_scene_main: bool = False,
    use_overlay_debug: Optional[bool] = None,
) -> int:
    seconds = max(float(args.pause_after_keyframe1_seconds), 0.0)
    if seconds <= 0.0:
        return 0
    num_frames = max(int(round(seconds * float(args.fps))), 1)
    viewer_sleep = max(float(args.viewer_frame_delay), 0.0)
    print(f"[stage] reached keyframe_{int(goal_frame)} arm={arm_label}; pausing for {seconds:.2f}s before next target")
    if debug_execution_state is not None:
        debug_execution_state.current_stage = "pause_after_keyframe1"
    for pause_idx in range(num_frames):
        record_frame(
            renderer,
            head_writer,
            third_writer,
            [
                "stage=pause_after_keyframe1",
                f"arm={arm_label}",
                f"goal=keyframe_{int(goal_frame)}",
                f"pause_frame={pause_idx + 1}/{num_frames}",
                f"pause_seconds={seconds:.2f}",
            ],
            use_overlay,
            debug_visuals,
            debug_execution_state,
            pure_scene_main=pure_scene_main,
            use_overlay_debug=use_overlay_debug,
        )
        if viewer_sleep > 0.0:
            time.sleep(viewer_sleep)
    return num_frames


def update_candidate_debug_visuals(
    debug_visuals: DebugVisualBundle,
    active_frame: Optional[int],
    common_candidates_per_frame: Dict[int, List[CandidatePose]],
    arm_display_candidates: Dict[str, Dict[int, List[CandidatePose]]],
    active_frame_by_arm: Optional[Dict[str, int]] = None,
) -> None:
    active_frame_set = set()
    if active_frame is not None:
        active_frame_set.add(int(active_frame))
    if active_frame_by_arm:
        active_frame_set.update(int(v) for v in active_frame_by_arm.values())
    for frame, actors in debug_visuals.common_candidate_actors.items():
        candidates = common_candidates_per_frame.get(int(frame), [])
        for idx, actor in enumerate(actors):
            if int(frame) in active_frame_set and idx < len(candidates):
                set_actor_pose(actor, candidates[idx].pose_world_wxyz)
            else:
                hide_actor(actor)
    for arm_name, per_frame in debug_visuals.arm_candidate_actors.items():
        candidates_map = arm_display_candidates.get(arm_name, {})
        axis_map = debug_visuals.arm_candidate_axis_actors.get(arm_name, {})
        for frame, actors in per_frame.items():
            candidates = candidates_map.get(int(frame), [])
            axis_actors = axis_map.get(frame, [])
            arm_active_frame = active_frame_by_arm.get(arm_name) if active_frame_by_arm else active_frame
            for idx, actor in enumerate(actors):
                if arm_active_frame is not None and int(frame) == int(arm_active_frame) and idx < len(candidates):
                    set_actor_pose(actor, candidates[idx].pose_world_wxyz)
                    if idx < len(axis_actors):
                        set_actor_pose(axis_actors[idx], candidates[idx].pose_world_wxyz)
                else:
                    hide_actor(actor)
                    if idx < len(axis_actors):
                        hide_actor(axis_actors[idx])




def export_rank_preview_images(
    renderer: ReplayRenderer,
    args: argparse.Namespace,
    keyframes: Sequence[int],
    object_states_per_frame: Dict[int, Dict[str, ObjectState]],
    arm_debugs: Dict[str, ArmDebugInfo],
    selected_keyframes: List[SelectedKeyframe],
    head_intrinsic: np.ndarray,
) -> List[RankPreviewRecord]:
    if not bool(args.save_rank_preview_images):
        return []

    preview_dir = args.output_dir / "rank_previews"
    preview_dir.mkdir(parents=True, exist_ok=True)
    top_n = max(int(args.rank_preview_top_n), 0)
    if top_n <= 0:
        return []

    temp_actors = {
        "left": create_gripper_candidate_actor(
            renderer.scene,
            "rank_preview_left",
            color=[0.1, 0.45, 1.0],
            scale=1.35,
            opening_width_m=0.04,
            forward_axis=str(args.debug_gripper_actor_forward_axis),
        ),
        "right": create_gripper_candidate_actor(
            renderer.scene,
            "rank_preview_right",
            color=[1.0, 0.55, 0.0],
            scale=1.35,
            opening_width_m=0.04,
            forward_axis=str(args.debug_gripper_actor_forward_axis),
        ),
    }
    selected_map = {int(item.source_frame): (item.arm, int(item.candidate.candidate_idx)) for item in selected_keyframes}
    preview_frames = sorted(
        {
            int(frame)
            for frame in keyframes
        }
        | {
            int(item.source_frame)
            for item in selected_keyframes
        }
    )
    records: List[RankPreviewRecord] = []

    try:
        for frame in preview_frames:
            frame = int(frame)
            object_states = object_states_per_frame.get(frame)
            if object_states is None:
                continue
            for state in object_states.values():
                if state.actor is None:
                    continue
                if state.visible:
                    set_object_state_pose(state, state.pose_world_wxyz)
                else:
                    hide_object_state(state)

            left_ranked = arm_debugs.get("left", ArmDebugInfo("left", None, {}, {}, {})).ranked_candidates_per_frame.get(frame, [])
            right_ranked = arm_debugs.get("right", ArmDebugInfo("right", None, {}, {}, {})).ranked_candidates_per_frame.get(frame, [])

            for rank_idx in range(top_n):
                left_cand = left_ranked[rank_idx] if rank_idx < len(left_ranked) else None
                right_cand = right_ranked[rank_idx] if rank_idx < len(right_ranked) else None
                if left_cand is None and right_cand is None:
                    continue

                if left_cand is not None:
                    set_actor_pose(temp_actors["left"], left_cand.pose_world_wxyz)
                else:
                    hide_actor(temp_actors["left"])
                if right_cand is not None:
                    set_actor_pose(temp_actors["right"], right_cand.pose_world_wxyz)
                else:
                    hide_actor(temp_actors["right"])

                renderer.update_robot_link_cameras()
                renderer.scene.update_render()
                overlay_lines = [
                    "mode=rank_preview",
                    f"source_frame={frame}",
                    f"rank={rank_idx + 1}",
                    (
                        f"left_candidate={left_cand.candidate_idx}"
                        + (" selected" if selected_map.get(frame) == ("left", int(left_cand.candidate_idx)) else "")
                        if left_cand is not None
                        else "left_candidate=none"
                    ),
                    (
                        f"right_candidate={right_cand.candidate_idx}"
                        + (" selected" if selected_map.get(frame) == ("right", int(right_cand.candidate_idx)) else "")
                        if right_cand is not None
                        else "right_candidate=none"
                    ),
                    "color=blue:left orange:right",
                ]
                head_rgb, _ = renderer.capture_camera(renderer.zed_camera)
                image_bgr = base.overlay_text(head_rgb, overlay_lines)
                annotate_candidate_labels(
                    image_bgr,
                    renderer.zed_camera,
                    head_intrinsic,
                    frame,
                    {},
                    {
                        "left": {frame: [] if left_cand is None else [left_cand]},
                        "right": {frame: [] if right_cand is None else [right_cand]},
                    },
                    selected_keyframes,
                )
                out_path = preview_dir / f"keyframe_{frame:06d}_rank_{rank_idx + 1}.png"
                if not cv2.imwrite(str(out_path), image_bgr):
                    raise RuntimeError(f"Failed to write {out_path}")
                records.append(
                    RankPreviewRecord(
                        frame=frame,
                        rank=rank_idx + 1,
                        image_path=str(out_path),
                        left_candidate_idx=None if left_cand is None else int(left_cand.candidate_idx),
                        right_candidate_idx=None if right_cand is None else int(right_cand.candidate_idx),
                    )
                )
    finally:
        for actor in temp_actors.values():
            hide_actor(actor)
    return records


def export_source_preview_compare(
    args: argparse.Namespace,
    reused_preview_summary: Optional[Dict],
    selected_keyframes: List[SelectedKeyframe],
) -> List[SourcePreviewCompareRecord]:
    if reused_preview_summary is None or args.reuse_preview_summary_json is None:
        return []

    compare_dir = args.output_dir / "source_preview_compare"
    compare_dir.mkdir(parents=True, exist_ok=True)
    frame_entries = {
        int(item["frame"]): item
        for item in reused_preview_summary.get("frames", [])
        if isinstance(item, dict) and "frame" in item
    }
    selected_by_frame_arm = {
        (int(item.source_frame), str(item.arm)): item for item in selected_keyframes
    }
    frames = sorted({int(item.source_frame) for item in selected_keyframes})
    records: List[SourcePreviewCompareRecord] = []
    metadata: Dict[str, object] = {
        "reuse_preview_summary_json": str(args.reuse_preview_summary_json),
        "reuse_preview_candidate_group": str(args.reuse_preview_candidate_group),
        "reuse_preview_top_rank": int(args.reuse_preview_top_rank),
        "candidate_target_local_x_offset_m": float(args.candidate_target_local_x_offset_m),
        "candidate_target_local_z_offset_m": float(args.candidate_target_local_z_offset_m),
        "approach_axis": str(args.approach_axis),
        "frames": {},
    }

    def _copy_one(src: Optional[str], frame: int, kind: str) -> None:
        if not src:
            return
        src_path = Path(str(src))
        if not src_path.is_file():
            return
        dst = compare_dir / f"frame_{frame:06d}_{kind}{src_path.suffix}"
        shutil.copy2(src_path, dst)
        records.append(SourcePreviewCompareRecord(frame=int(frame), image_path=str(dst), source_path=str(src_path), source_kind=str(kind)))

    for frame in frames:
        frame_entry = frame_entries.get(int(frame), {})
        _copy_one(frame_entry.get("orientation_image"), frame, "d435_orientation_rank")
        _copy_one(frame_entry.get("fused_image"), frame, "d435_fused_rank")
        _copy_one(frame_entry.get("object_distance_debug_path"), frame, "d435_object_distance_debug")

        # Also copy the legacy wide-FOV preview with the same relative path when it exists.
        for key, kind in (
            ("orientation_image", "legacy_orientation_rank"),
            ("fused_image", "legacy_fused_rank"),
            ("object_distance_debug_path", "legacy_object_distance_debug"),
        ):
            src = frame_entry.get(key)
            if not src:
                continue
            legacy = Path(str(src).replace("/anygrasp_h2o_preview_d435/", "/anygrasp_h2o_preview/"))
            _copy_one(str(legacy), frame, kind)

        frame_meta: Dict[str, object] = {
            "source_frame": int(frame),
            "orientation_image": frame_entry.get("orientation_image"),
            "fused_image": frame_entry.get("fused_image"),
            "object_distance_debug_path": frame_entry.get("object_distance_debug_path"),
            "selected": {},
        }
        for arm in ("left", "right"):
            selected = selected_by_frame_arm.get((int(frame), arm))
            candidate_key = f"{arm}_{args.reuse_preview_candidate_group}"
            ranked_entries = list(frame_entry.get("top_candidates", {}).get(candidate_key, [])) if isinstance(frame_entry, dict) else []
            source_entry = None
            if selected is not None:
                for rank_idx, entry in enumerate(ranked_entries, start=1):
                    if int(entry.get("candidate_idx", -1)) == int(selected.candidate.candidate_idx):
                        source_entry = dict(entry)
                        source_entry["rank"] = int(rank_idx)
                        break
                frame_meta["selected"][arm] = {
                    "candidate_idx": int(selected.candidate.candidate_idx),
                    "rank": None if source_entry is None else int(source_entry["rank"]),
                    "source_entry_translation_world": None if source_entry is None else source_entry.get("translation_world"),
                    "source_entry_translation_cam": None if source_entry is None else source_entry.get("translation_cam"),
                    "source_entry_rotation_matrix": None if source_entry is None else source_entry.get("rotation_matrix"),
                    "planner_raw_pose_world_wxyz": np.asarray(selected.candidate.raw_pose_world_wxyz, dtype=np.float64).reshape(7).tolist(),
                    "planner_target_pose_world_wxyz": np.asarray(selected.candidate.pose_world_wxyz, dtype=np.float64).reshape(7).tolist(),
                    "local_x_offset_applied_m": float(args.candidate_target_local_x_offset_m),
                    "local_z_offset_applied_m": float(args.candidate_target_local_z_offset_m),
                    "nearest_object": str(selected.candidate.nearest_object),
                    "rotation_distance_deg": float(selected.candidate.rotation_distance_deg),
                }
        metadata["frames"][str(int(frame))] = frame_meta

    metadata_path = compare_dir / "selected_candidate_mapping.json"
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    records.append(
        SourcePreviewCompareRecord(
            frame=-1,
            image_path=str(metadata_path),
            source_path=str(args.reuse_preview_summary_json),
            source_kind="selected_candidate_mapping",
        )
    )
    return records


def generate_debug_preview(
    renderer: ReplayRenderer,
    args: argparse.Namespace,
    replay_frame_indices: np.ndarray,
    object_tracks: Dict[str, ObjectTrack],
    selected_keyframes: List[SelectedKeyframe],
    common_candidates_per_frame: Dict[int, List[CandidatePose]],
    arm_display_candidates: Dict[str, Dict[int, List[CandidatePose]]],
    head_intrinsic: np.ndarray,
    debug_visuals: DebugVisualBundle,
) -> Optional[Path]:
    if not bool(args.save_debug_preview) or bool(args.pure_scene_output):
        renderer.update_target_axis_visuals(None, None)
        return None

    debug_video_path = args.output_dir / "debug_selection_preview.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(debug_video_path), fourcc, int(args.debug_preview_fps), (args.image_width, args.image_height))
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open {debug_video_path}")

    selected_by_frame = {int(item.source_frame): item for item in selected_keyframes}
    selected_frames = sorted(selected_by_frame.keys())
    for item in selected_keyframes:
        actor = debug_visuals.keyframe_axis_actors.get((int(item.source_frame), item.arm))
        if actor is not None:
            set_actor_pose(actor, item.candidate.pose_world_wxyz)
    try:
        for idx, source_frame in enumerate(replay_frame_indices.tolist()):
            overlay_lines = [f"mode=debug_preview", f"source_frame={source_frame}"]
            for name, track in object_tracks.items():
                if track.actor is None:
                    continue
                is_visible = bool(track.visible[idx])
                if is_visible:
                    set_actor_pose(track.actor, track.pose_world_wxyz[idx])
                else:
                    hide_actor(track.actor)
                overlay_lines.append(f"{name}={int(is_visible)}")

            selected = None
            for keyframe in selected_frames:
                if int(source_frame) >= keyframe and int(source_frame) < keyframe + max(int(args.debug_keyframe_hold_frames), 1):
                    selected = selected_by_frame[keyframe]
                    break
            if selected is None:
                selected = selected_by_frame.get(int(source_frame))
            if selected is not None:
                update_candidate_debug_visuals(
                    debug_visuals,
                    int(selected.source_frame),
                    common_candidates_per_frame,
                    arm_display_candidates,
                )
                overlay_lines.extend(
                    [
                        f"selected_keyframe={selected.source_frame}",
                        f"selected_arm={selected.arm}",
                        f"expected_object={selected.candidate.nearest_object}",
                        f"selected_object={selected.candidate.nearest_object}",
                        f"candidate_idx={selected.candidate.candidate_idx}",
                        f"rot_err={selected.candidate.rotation_distance_deg:.2f}deg",
                        f"raw_candidates={len(common_candidates_per_frame.get(int(selected.source_frame), []))}",
                        f"per_arm_top_k={int(args.debug_candidate_top_k)}",
                        "color=green:raw blue:left orange:right red:selected",
                        "labels=tiny candidate_idx",
                    ]
                )
            else:
                update_candidate_debug_visuals(
                    debug_visuals,
                    None,
                    common_candidates_per_frame,
                    arm_display_candidates,
                )

            renderer.update_robot_link_cameras()
            renderer.scene.update_render()
            debug_rgb, _ = renderer.capture_camera(renderer.zed_camera)
            debug_bgr = base.overlay_text(debug_rgb, overlay_lines)
            annotate_candidate_labels(
                debug_bgr,
                renderer.zed_camera,
                head_intrinsic,
                int(selected.source_frame) if selected is not None else None,
                common_candidates_per_frame,
                arm_display_candidates,
                selected_keyframes,
            )
            writer.write(debug_bgr)
    finally:
        writer.release()
        update_candidate_debug_visuals(
            debug_visuals,
            None,
            common_candidates_per_frame,
            arm_display_candidates,
        )
    return debug_video_path


def main() -> None:
    args = parse_args()
    args.anygrasp_dir = args.anygrasp_dir.resolve()
    args.replay_dir = args.replay_dir.resolve()
    args.hand_npz = args.hand_npz.resolve()
    args.output_dir = args.output_dir.resolve()
    args.robot_config = args.robot_config.resolve()
    if args.reuse_plan_summary_json is not None:
        args.reuse_plan_summary_json = args.reuse_plan_summary_json.resolve()
    if args.reuse_preview_summary_json is not None:
        args.reuse_preview_summary_json = args.reuse_preview_summary_json.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if not args.anygrasp_dir.is_dir():
        raise NotADirectoryError(f"anygrasp_dir not found: {args.anygrasp_dir}")
    if not args.replay_dir.is_dir():
        raise NotADirectoryError(f"replay_dir not found: {args.replay_dir}")
    if not args.hand_npz.is_file():
        raise FileNotFoundError(f"hand_npz not found: {args.hand_npz}")
    if args.reuse_plan_summary_json is not None and not args.reuse_plan_summary_json.is_file():
        raise FileNotFoundError(f"reuse_plan_summary_json not found: {args.reuse_plan_summary_json}")
    if args.reuse_preview_summary_json is not None and not args.reuse_preview_summary_json.is_file():
        raise FileNotFoundError(f"reuse_preview_summary_json not found: {args.reuse_preview_summary_json}")
    if args.reuse_plan_summary_json is not None and args.reuse_preview_summary_json is not None:
        raise ValueError("Specify only one of --reuse_plan_summary_json or --reuse_preview_summary_json")
    save_pose_debug_effective = bool(args.save_pose_debug) or bool(args.pure_scene_output)
    if save_pose_debug_effective:
        pose_debug_path = args.output_dir / "pose_debug.jsonl"
        if pose_debug_path.exists():
            pose_debug_path.unlink()
    metrics_debug_path = args.output_dir / "debug_execution_metrics.jsonl"
    if metrics_debug_path.exists():
        metrics_debug_path.unlink()

    args.manual_candidate_overrides = parse_manual_candidate_overrides(args.manual_candidate)
    args.object_mesh_overrides = parse_object_mesh_overrides(args.object_mesh_override)
    args.execution_object_scale_overrides = parse_named_scale_overrides(
        args.execution_object_scale_override,
        "--execution_object_scale_override",
    )
    args.execution_object_visual_scale_overrides = parse_named_scale_overrides(
        args.execution_object_visual_scale_override,
        "--execution_object_visual_scale_override",
    )
    args.execution_object_collision_scale_overrides = parse_named_scale_overrides(
        args.execution_object_collision_scale_override,
        "--execution_object_collision_scale_override",
    )
    merged_visual_scale_overrides = dict(args.execution_object_scale_overrides)
    merged_visual_scale_overrides.update(args.execution_object_visual_scale_overrides)
    merged_collision_scale_overrides = dict(args.execution_object_scale_overrides)
    merged_collision_scale_overrides.update(args.execution_object_collision_scale_overrides)
    args.execution_object_visual_scale_overrides = merged_visual_scale_overrides
    args.execution_object_collision_scale_overrides = merged_collision_scale_overrides
    args.candidate_orientation_remap_label, args.candidate_orientation_remap_matrix = base.resolve_orientation_remap(args.candidate_orientation_remap_label)
    args.candidate_post_rot_xyz_deg = np.asarray(args.candidate_post_rot_xyz_deg, dtype=np.float64)
    args.candidate_post_rot_matrix = base.orthonormalize_rotation(
        R.from_euler("xyz", np.deg2rad(args.candidate_post_rot_xyz_deg)).as_matrix()
    )
    requested_keyframes = [int(v) for v in args.keyframes]
    if args.reuse_preview_summary_json is not None:
        with args.reuse_preview_summary_json.open("r", encoding="utf-8") as f:
            preview_summary_for_resolution = json.load(f)
        requested_keyframes_used, keyframes, resolved_keyframe_pairs, resolved_relative_frame = resolve_frames_from_preview_summary(
            preview_summary=preview_summary_for_resolution,
            frame_mode=str(args.reuse_preview_frame_mode),
            requested_keyframes=requested_keyframes,
            requested_relative_frame=(None if args.candidate_selection_relative_frame is None else int(args.candidate_selection_relative_frame)),
        )
        requested_keyframes = list(requested_keyframes_used)
        args.resolved_candidate_selection_relative_frame = resolved_relative_frame
    else:
        available_grasp_frames = list_available_grasp_frames(args.anygrasp_dir)
        resolved_keyframe_pairs = resolve_requested_frames(requested_keyframes, available_grasp_frames)
        keyframes = [int(resolved) for _, resolved in resolved_keyframe_pairs]
        args.resolved_candidate_selection_relative_frame = None
        if str(args.candidate_selection_mode) == "top_score_auto" and args.candidate_selection_relative_frame is not None:
            resolved_rel = resolve_requested_frames([int(args.candidate_selection_relative_frame)], available_grasp_frames)[0][1]
            args.resolved_candidate_selection_relative_frame = int(resolved_rel)
            if len(keyframes) >= 2:
                keyframes[1] = max(int(keyframes[1]), int(resolved_rel))
                args.resolved_keyframes = list(keyframes)
    args.requested_keyframes = requested_keyframes
    args.resolved_keyframes = list(keyframes)
    args.resolved_keyframe_pairs = [(int(req), int(res)) for req, res in resolved_keyframe_pairs]
    if args.resolved_keyframes != requested_keyframes:
        print(
            "[frame-resolve] "
            f"requested_keyframes={requested_keyframes} "
            f"resolved_keyframes={args.resolved_keyframes}"
        )
    if args.resolved_candidate_selection_relative_frame is not None:
        print(
            "[candidate-selection] "
            f"mode={args.candidate_selection_mode} "
            f"requested_relative_frame={int(args.candidate_selection_relative_frame)} "
            f"resolved_relative_frame={int(args.resolved_candidate_selection_relative_frame)} "
            f"final_keyframes={args.resolved_keyframes}"
        )

    renderer = build_renderer(args)
    hand_data = load_hand_data(args.hand_npz)
    replay_frame_indices, object_tracks = load_object_tracks(args.replay_dir, args.object_mesh_overrides)
    replay_head_camera_pose_by_frame = load_replay_head_camera_poses(args.replay_dir)
    object_states_per_frame = {
        frame: load_object_states(
            args.replay_dir,
            frame,
            args.object_mesh_overrides,
            args.execution_object_visual_scale_overrides,
            args.execution_object_collision_scale_overrides,
        )
        for frame in keyframes
    }
    reused_plan_summary: Optional[Dict] = None
    reused_preview_summary: Optional[Dict] = None
    if args.reuse_preview_summary_json is not None:
        selection_result, arm_debugs, execution_sequences, reused_preview_summary, requested_keyframes_used, keyframes, resolved_keyframe_pairs, resolved_relative_frame = load_reused_preview_summary(
            renderer=renderer,
            args=args,
            hand_data=hand_data,
            path=args.reuse_preview_summary_json,
            requested_arm_mode=str(args.arm),
            requested_keyframes=requested_keyframes,
            requested_relative_frame=(None if args.candidate_selection_relative_frame is None else int(args.candidate_selection_relative_frame)),
        )
        if not (bool(args.execute_both_arms) and args.arm == "auto"):
            execution_sequences = [(selection_result.arm, selection_result.selected_keyframes)]
        selected_keyframes = (
            [item for _, seq in execution_sequences for item in seq]
            if bool(args.execute_both_arms) and args.arm == "auto"
            else selection_result.selected_keyframes
        )
        args.requested_keyframes = [int(v) for v in requested_keyframes_used]
        args.resolved_keyframes = list(keyframes)
        args.resolved_keyframe_pairs = [(int(req), int(res)) for req, res in resolved_keyframe_pairs]
        args.resolved_candidate_selection_relative_frame = None if resolved_relative_frame is None else int(resolved_relative_frame)
        preview_object_frames = sorted({int(frame) for frame in keyframes} | {int(item.source_frame) for item in selected_keyframes})
        object_states_per_frame = {
            frame: load_object_states(
                args.replay_dir,
                frame,
                args.object_mesh_overrides,
                args.execution_object_visual_scale_overrides,
                args.execution_object_collision_scale_overrides,
            )
            for frame in preview_object_frames
        }
        ranked_candidates_per_frame = selection_result.ranked_candidates_per_frame
        all_candidates_per_frame = selection_result.all_candidates_per_frame
        print(
            "[reuse-preview-summary] "
            f"path={args.reuse_preview_summary_json} "
            f"frame_mode={args.reuse_preview_frame_mode} "
            f"group={args.reuse_preview_candidate_group} "
            f"rank={int(args.reuse_preview_top_rank)} "
            f"executed_arms={[arm_name for arm_name, _ in execution_sequences]} "
            f"keyframes={keyframes}"
        )
    elif args.reuse_plan_summary_json is not None:
        selection_result, arm_debugs, execution_sequences, reused_plan_summary = load_reused_plan_summary(
            path=args.reuse_plan_summary_json,
            requested_arm_mode=str(args.arm),
        )
        if not (bool(args.execute_both_arms) and args.arm == "auto"):
            execution_sequences = [(selection_result.arm, selection_result.selected_keyframes)]
        selected_keyframes = selection_result.selected_keyframes
        keyframes = [int(item.source_frame) for item in selected_keyframes]
        args.requested_keyframes = list(keyframes)
        args.resolved_keyframes = list(keyframes)
        args.resolved_keyframe_pairs = [(int(frame), int(frame)) for frame in keyframes]
        object_states_per_frame = {
            frame: load_object_states(
                args.replay_dir,
                frame,
                args.object_mesh_overrides,
                args.execution_object_visual_scale_overrides,
                args.execution_object_collision_scale_overrides,
            )
            for frame in keyframes
        }
        ranked_candidates_per_frame = selection_result.ranked_candidates_per_frame
        all_candidates_per_frame = selection_result.all_candidates_per_frame
        print(
            "[reuse-plan-summary] "
            f"path={args.reuse_plan_summary_json} "
            f"executed_arms={[arm_name for arm_name, _ in execution_sequences]} "
            f"keyframes={keyframes}"
        )
    else:
        selection_result, arm_debugs = choose_keyframes(
            renderer=renderer,
            args=args,
            anygrasp_dir=args.anygrasp_dir,
            hand_data=hand_data,
            keyframes=keyframes,
            arm_mode=args.arm,
            object_states_per_frame=object_states_per_frame,
        )
        selected_keyframes = selection_result.selected_keyframes
        ranked_candidates_per_frame = selection_result.ranked_candidates_per_frame
        all_candidates_per_frame = selection_result.all_candidates_per_frame
        arm = selected_keyframes[0].arm
        execution_sequences = [(arm, selected_keyframes)]
        if bool(args.execute_both_arms) and args.arm == "auto":
            for candidate_arm in ("left", "right"):
                if candidate_arm == arm:
                    continue
                candidate_info = arm_debugs.get(candidate_arm)
                if candidate_info is None or candidate_info.selected_keyframes is None:
                    continue
                execution_sequences.append((candidate_arm, candidate_info.selected_keyframes))
    common_candidates_per_frame = build_display_candidates_per_frame(
        args,
        keyframes,
        ranked_candidates_per_frame,
        all_candidates_per_frame,
    )
    arm_display_candidates: Dict[str, Dict[int, List[CandidatePose]]] = {}
    for arm_name, arm_info in arm_debugs.items():
        arm_display_candidates[arm_name] = {}
        for frame in keyframes:
            frame = int(frame)
            arm_display_candidates[arm_name][frame] = list(
                arm_info.ranked_candidates_per_frame.get(frame, [])[: max(int(args.debug_candidate_top_k), 0)]
            )

    arm = selected_keyframes[0].arm
    primary_object_name = selected_keyframes[0].candidate.nearest_object
    if bool(args.execute_both_arms) and len(execution_sequences) < 2:
        print("[exec-mode] execute_both_arms=1 but only one arm has valid selected keyframes; falling back to single-arm execution")
    debug_visuals = create_debug_visual_bundle(
        renderer,
        args,
        keyframes,
        common_candidates_per_frame,
        arm_display_candidates,
        selected_keyframes,
    )

    object_states = object_states_per_frame[keyframes[0]]
    selected_object_names = {seq[0].candidate.nearest_object for _, seq in execution_sequences if seq}
    use_grasp_action_object_collision = bool(args.enable_grasp_action_object_collision)
    collision_start_stage = str(args.grasp_action_object_collision_start_stage)
    for state in object_states.values():
        state.actor, state.collision_debug_info = create_execution_object_actor(
            renderer.scene,
            state.mesh_file,
            f"planned_object_{state.name}",
            ignore_collision=(bool(args.replay_objects_ignore_collision) and not (use_grasp_action_object_collision and state.name in selected_object_names)),
            collision_mode=str(args.execution_object_collision_mode),
            visual_scale=state.visual_scale,
            collision_scale=state.collision_scale,
        )
        state.collision_mode = str(args.execution_object_collision_mode)
        if bool(args.debug_visualize_object_collision_bbox):
            info = state.collision_debug_info or {}
            if info.get("effective_mode") == "solid_bbox" and "center" in info and "half_size" in info:
                state.collision_bbox_actor = create_debug_collision_bbox_actor(
                    renderer.scene,
                    f"planned_object_{state.name}_collision_bbox",
                    center=info["center"],
                    half_size=info["half_size"],
                )
        if use_grasp_action_object_collision and state.name in selected_object_names and collision_start_stage != "pregrasp":
            state.collision_groups_cache = set_actor_collision_enabled(state.actor, enabled=False, cached_groups=None)
        set_object_state_pose(state, state.pose_world_wxyz)
        if state.name in object_tracks:
            object_tracks[state.name].actor = state.actor

    for item in selected_keyframes:
        actor = debug_visuals.keyframe_axis_actors.get((int(item.source_frame), item.arm))
        if actor is not None:
            set_actor_pose(actor, item.candidate.pose_world_wxyz)

    renderer.update_robot_link_cameras()
    renderer.step_scene(steps=1)
    renderer.set_grippers(args.open_gripper if arm == "left" else None, args.open_gripper if arm == "right" else None)
    head_intrinsic = get_camera_intrinsic_matrix(
        renderer.zed_camera,
        image_width=args.image_width,
        image_height=args.image_height,
        fovy_deg=args.fovy_deg,
    )

    debug_video_path = generate_debug_preview(
        renderer,
        args,
        replay_frame_indices,
        object_tracks,
        selected_keyframes,
        common_candidates_per_frame,
        arm_display_candidates,
        head_intrinsic,
        debug_visuals,
    )
    rank_preview_records = export_rank_preview_images(
        renderer=renderer,
        args=args,
        keyframes=keyframes,
        object_states_per_frame=object_states_per_frame,
        arm_debugs=arm_debugs,
        selected_keyframes=selected_keyframes,
        head_intrinsic=head_intrinsic,
    )
    source_preview_compare_records = export_source_preview_compare(
        args=args,
        reused_preview_summary=reused_preview_summary,
        selected_keyframes=selected_keyframes,
    )
    for state in object_states.values():
        if state.actor is not None:
            set_object_state_pose(state, state.pose_world_wxyz)
    update_candidate_debug_visuals(debug_visuals, None, common_candidates_per_frame, arm_display_candidates)
    renderer.step_scene(steps=1)

    head_video_path = args.output_dir / "head_cam_plan.mp4"
    third_video_path = args.output_dir / "third_cam_plan.mp4"
    left_wrist_video_path = args.output_dir / "left_wrist_cam_plan.mp4"
    right_wrist_video_path = args.output_dir / "right_wrist_cam_plan.mp4"
    debug_execution_video_path = args.output_dir / "debug_execution_preview.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    wrist_export_size = (int(args.image_width), int(args.image_height))
    head_writer = cv2.VideoWriter(str(head_video_path), fourcc, args.fps, (args.image_width, args.image_height))
    if not head_writer.isOpened():
        raise RuntimeError(f"Failed to open {head_video_path}")
    third_writer = None
    if bool(args.third_person_view) and not bool(args.head_only):
        third_writer = cv2.VideoWriter(str(third_video_path), fourcc, args.fps, (args.image_width, args.image_height))
        if not third_writer.isOpened():
            raise RuntimeError(f"Failed to open {third_video_path}")
    left_wrist_writer = None
    if getattr(renderer, "_left_wrist_camera_link", None) is not None:
        left_wrist_writer = cv2.VideoWriter(str(left_wrist_video_path), fourcc, args.fps, wrist_export_size)
        if not left_wrist_writer.isOpened():
            raise RuntimeError(f"Failed to open {left_wrist_video_path}")
    right_wrist_writer = None
    if getattr(renderer, "_right_wrist_camera_link", None) is not None:
        right_wrist_writer = cv2.VideoWriter(str(right_wrist_video_path), fourcc, args.fps, wrist_export_size)
        if not right_wrist_writer.isOpened():
            raise RuntimeError(f"Failed to open {right_wrist_video_path}")
    debug_execution_writer = None
    if bool(args.save_debug_execution_preview):
        debug_execution_writer = cv2.VideoWriter(str(debug_execution_video_path), fourcc, int(args.debug_execution_fps), (args.image_width, args.image_height))
        if not debug_execution_writer.isOpened():
            raise RuntimeError(f"Failed to open {debug_execution_video_path}")
    debug_execution_state = DebugExecutionState(
        writer=debug_execution_writer,
        left_wrist_writer=left_wrist_writer,
        right_wrist_writer=right_wrist_writer,
        selected_keyframes=selected_keyframes,
        common_candidates_per_frame=common_candidates_per_frame,
        arm_display_candidates=arm_display_candidates,
        head_intrinsic=head_intrinsic,
        active_frame=None,
        pose_debug_path=(args.output_dir / "pose_debug.jsonl") if save_pose_debug_effective else None,
        metrics_debug_path=metrics_debug_path,
        replay_head_camera_pose_by_frame=replay_head_camera_pose_by_frame,
        object_tracks=object_tracks,
        current_stage="init",
        target_pose_by_arm={},
        target_object_by_arm={},
        goal_label_by_arm={},
        reach_error_pose_source=str(args.reach_error_pose_source),
        show_selected_keyframe_axes=bool(args.debug_visualize_selected_keyframe_axes),
    )
    use_overlay = bool(args.overlay_text) and not bool(args.pure_scene_output)
    use_overlay_debug = bool(args.overlay_text)
    pure_scene_main = bool(args.pure_scene_output)
    dual_sync_mode = bool(args.execute_both_arms) and args.arm == "auto" and len(execution_sequences) >= 2
    stages_by_executed_arm: Dict[str, Dict[str, object]] = {}
    supervision_targets_by_executed_arm: Dict[str, Dict[str, np.ndarray]] = {}
    supervision_only_arms_by_executed_arm: Dict[str, List[str]] = {}
    selected_candidates_by_executed_arm: Dict[str, List[SelectedKeyframe]] = {}
    selected_objects_by_executed_arm: Dict[str, str] = {}
    init_pose_info_by_executed_arm: Dict[str, Dict[str, object]] = {}
    init_prefix_frames_written_by_executed_arm: Dict[str, int] = {}

    try:
        if dual_sync_mode:
            exec_selected_by_arm = {arm_name: seq for arm_name, seq in execution_sequences[:2]}
            dual_arms = [arm_name for arm_name in ("left", "right") if arm_name in exec_selected_by_arm]
            for arm_name in dual_arms:
                seq = exec_selected_by_arm[arm_name]
                selected_objects_by_executed_arm[arm_name] = seq[0].candidate.nearest_object
                selected_candidates_by_executed_arm[arm_name] = list(seq)
                supervision_targets_by_executed_arm[arm_name] = {
                    "left": np.asarray(exec_selected_by_arm["left"][0].candidate.pose_world_wxyz, dtype=np.float64),
                    "right": np.asarray(exec_selected_by_arm["right"][0].candidate.pose_world_wxyz, dtype=np.float64),
                }
                supervision_only_arms_by_executed_arm[arm_name] = []

            dual_selected_keyframes = [item for arm_name in dual_arms for item in exec_selected_by_arm[arm_name]]
            debug_execution_state.selected_keyframes = dual_selected_keyframes
            print("[exec-mode] selected_arm=both supervision_only_arms=none")

            init_info = apply_robot_init_pose(renderer, open_gripper=args.open_gripper, settle_steps=args.settle_steps)
            init_pose_info_by_executed_arm["left"] = init_info
            init_pose_info_by_executed_arm["right"] = init_info
            init_prefix_written = emit_init_prefix_frames(
                renderer=renderer,
                head_writer=head_writer,
                third_writer=third_writer,
                use_overlay=use_overlay,
                init_info=init_info,
                fixed_frames=args.init_prefix_frames,
                arm_label="both",
                debug_visuals=debug_visuals,
                debug_execution_state=debug_execution_state,
                pure_scene_main=pure_scene_main,
                use_overlay_debug=use_overlay_debug,
            )
            init_prefix_frames_written_by_executed_arm["left"] = int(init_prefix_written)
            init_prefix_frames_written_by_executed_arm["right"] = int(init_prefix_written)
            renderer.set_grippers(args.open_gripper, args.open_gripper)

            record_frame(
                renderer,
                head_writer,
                third_writer,
                [
                    "stage=init",
                    "arm=both",
                    f"left_object={selected_objects_by_executed_arm['left']}",
                    f"right_object={selected_objects_by_executed_arm['right']}",
                    f"left_goal_frame={exec_selected_by_arm['left'][0].source_frame}",
                    f"right_goal_frame={exec_selected_by_arm['right'][0].source_frame}",
                ],
                use_overlay,
                debug_visuals,
                debug_execution_state,
                pure_scene_main=pure_scene_main,
                use_overlay_debug=use_overlay_debug,
            )

            pregrasp_targets = {
                arm_name: pose_with_offset_along_local_axis(seq[0].candidate.pose_world_wxyz, args.approach_offset_m, args.approach_axis)
                for arm_name, seq in exec_selected_by_arm.items()
            }
            grasp_targets = {arm_name: seq[0].candidate.pose_world_wxyz for arm_name, seq in exec_selected_by_arm.items()}
            skip_grasp_dual = all(
                poses_are_effectively_same(pregrasp_targets[arm_name], grasp_targets[arm_name])
                for arm_name in dual_arms
            )
            action_targets = {arm_name: seq[1].candidate.pose_world_wxyz for arm_name, seq in exec_selected_by_arm.items()}
            action_object_replay = None
            if bool(args.replay_objects_during_action):
                action_object_replay = ExecutionObjectReplayConfig(
                    replay_frame_indices=np.asarray(replay_frame_indices, dtype=np.int32),
                    object_tracks=object_tracks,
                    start_frame=int(exec_selected_by_arm["left"][0].source_frame),
                    end_frame=int(exec_selected_by_arm["left"][1].source_frame),
                )

            debug_execution_state.active_frame = int(exec_selected_by_arm["left"][0].source_frame)
            debug_execution_state.active_frame_by_arm = {
                arm_name: int(seq[0].source_frame) for arm_name, seq in exec_selected_by_arm.items()
            }
            debug_execution_state.current_stage = "pregrasp"
            debug_execution_state.target_pose_by_arm = {k: np.asarray(v, dtype=np.float64) for k, v in pregrasp_targets.items()}
            debug_execution_state.target_object_by_arm = dict(selected_objects_by_executed_arm)
            debug_execution_state.goal_label_by_arm = {
                arm_name: f"keyframe_{int(seq[0].source_frame)}_pregrasp" for arm_name, seq in exec_selected_by_arm.items()
            }
            set_dual_arm_target_visuals(
                renderer,
                pregrasp_targets.get("left"),
                pregrasp_targets.get("right"),
            )
            pregrasp_dual = execute_dual_stage_until_reached(
                renderer=renderer,
                target_pose_world_wxyz_by_arm=pregrasp_targets,
                label="pregrasp",
                head_writer=head_writer,
                third_writer=third_writer,
                use_overlay=use_overlay,
                args=args,
                debug_visuals=debug_visuals,
                debug_execution_state=debug_execution_state,
                pure_scene_main=pure_scene_main,
                use_overlay_debug=use_overlay_debug,
            )

            if use_grasp_action_object_collision and collision_start_stage == "grasp":
                set_object_collision_for_names(
                    object_states,
                    [selected_objects_by_executed_arm[arm_name] for arm_name in dual_arms],
                    enabled=True,
                )

            if skip_grasp_dual:
                debug_execution_state.active_frame = int(exec_selected_by_arm["left"][0].source_frame)
                debug_execution_state.active_frame_by_arm = {
                    arm_name: int(seq[0].source_frame) for arm_name, seq in exec_selected_by_arm.items()
                }
                debug_execution_state.current_stage = "grasp_skipped_same_target"
                debug_execution_state.target_pose_by_arm = {k: np.asarray(v, dtype=np.float64) for k, v in grasp_targets.items()}
                debug_execution_state.goal_label_by_arm = {
                    arm_name: f"keyframe_{int(seq[0].source_frame)}_grasp_skipped" for arm_name, seq in exec_selected_by_arm.items()
                }
                set_dual_arm_target_visuals(
                    renderer,
                    grasp_targets.get("left"),
                    grasp_targets.get("right"),
                )
                record_frame(
                    renderer,
                    head_writer,
                    third_writer,
                    ["stage=grasp_skipped_same_target", "arm=both"],
                    use_overlay,
                    debug_visuals,
                    debug_execution_state,
                    pure_scene_main=pure_scene_main,
                    use_overlay_debug=use_overlay_debug,
                )
                grasp_dual = {
                    "attempts": int(pregrasp_dual.get("attempts", 0)),
                    "reached": bool(pregrasp_dual.get("reached", False)),
                    "arms": pregrasp_dual.get("arms", {}),
                    "attempt_history": list(pregrasp_dual.get("attempt_history", [])),
                    "skipped_same_target": True,
                }
                print("[stage] grasp skipped because pregrasp target equals grasp target for both arms")
            else:
                debug_execution_state.active_frame = int(exec_selected_by_arm["left"][0].source_frame)
                debug_execution_state.active_frame_by_arm = {
                    arm_name: int(seq[0].source_frame) for arm_name, seq in exec_selected_by_arm.items()
                }
                debug_execution_state.current_stage = "grasp"
                debug_execution_state.target_pose_by_arm = {k: np.asarray(v, dtype=np.float64) for k, v in grasp_targets.items()}
                debug_execution_state.goal_label_by_arm = {
                    arm_name: f"keyframe_{int(seq[0].source_frame)}_grasp" for arm_name, seq in exec_selected_by_arm.items()
                }
                set_dual_arm_target_visuals(
                    renderer,
                    grasp_targets.get("left"),
                    grasp_targets.get("right"),
                )
                grasp_dual = execute_dual_stage_until_reached(
                    renderer=renderer,
                    target_pose_world_wxyz_by_arm=grasp_targets,
                    label="grasp",
                    head_writer=head_writer,
                    third_writer=third_writer,
                    use_overlay=use_overlay,
                    args=args,
                    debug_visuals=debug_visuals,
                    debug_execution_state=debug_execution_state,
                    pure_scene_main=pure_scene_main,
                    use_overlay_debug=use_overlay_debug,
                )
            print(
                "[stage] keyframe_1 grasp finished "
                f"left_reached={int(bool(grasp_dual['arms'].get('left', {}).get('reached', False)))} "
                f"right_reached={int(bool(grasp_dual['arms'].get('right', {}).get('reached', False)))}"
            )
            if not all(bool(grasp_dual["arms"].get(arm_name, {}).get("reached", False)) for arm_name in dual_arms):
                print(
                    "[warn] grasp_not_reached_before_close "
                    + " ".join(
                        f"{arm_name}:pos={float(grasp_dual['arms'][arm_name]['pos_err_m']):.4f},"
                        f"rot={float(grasp_dual['arms'][arm_name]['rot_err_deg']):.2f}"
                        for arm_name in dual_arms
                    )
                )

            keyframe1_reached_dual = all(bool(grasp_dual["arms"].get(arm_name, {}).get("reached", False)) for arm_name in dual_arms)
            stop_before_close_reason = None
            if bool(args.debug_stop_after_keyframe1):
                stop_before_close_reason = "debug_stop_after_keyframe1"
            elif bool(args.require_keyframe1_reached_before_close) and not keyframe1_reached_dual:
                stop_before_close_reason = "keyframe1_grasp_not_reached_before_close"

            if stop_before_close_reason is not None:
                action_dual = skipped_dual_action_result(
                    renderer=renderer,
                    action_targets=action_targets,
                    args=args,
                    reason=stop_before_close_reason,
                )
                print(f"[stage] stopping before close_gripper reason={stop_before_close_reason}")
                record_frame(
                    renderer,
                    head_writer,
                    third_writer,
                    ["stage=close_skipped", "arm=both", f"reason={stop_before_close_reason}"],
                    use_overlay,
                    debug_visuals,
                    debug_execution_state,
                    pure_scene_main=pure_scene_main,
                    use_overlay_debug=use_overlay_debug,
                )
                for arm_name in dual_arms:
                    stages_by_executed_arm[arm_name] = {
                        "pregrasp": {
                            "status": str(pregrasp_dual["arms"][arm_name]["status"]),
                            "attempts": int(pregrasp_dual["attempts"]),
                            "reached": bool(pregrasp_dual["arms"][arm_name]["reached"]),
                            "pos_err_m": float(pregrasp_dual["arms"][arm_name]["pos_err_m"]),
                            "rot_err_deg": float(pregrasp_dual["arms"][arm_name]["rot_err_deg"]),
                            "attempt_history": pregrasp_dual["attempt_history"],
                        },
                        "grasp": {
                            "status": str(grasp_dual["arms"][arm_name]["status"]),
                            "attempts": int(grasp_dual["attempts"]),
                            "reached": bool(grasp_dual["arms"][arm_name]["reached"]),
                            "pos_err_m": float(grasp_dual["arms"][arm_name]["pos_err_m"]),
                            "rot_err_deg": float(grasp_dual["arms"][arm_name]["rot_err_deg"]),
                            "attempt_history": grasp_dual["attempt_history"],
                        },
                        "action": {
                            "status": str(action_dual["arms"][arm_name]["status"]),
                            "attempts": int(action_dual["attempts"]),
                            "reached": bool(action_dual["arms"][arm_name]["reached"]),
                            "pos_err_m": float(action_dual["arms"][arm_name]["pos_err_m"]),
                            "rot_err_deg": float(action_dual["arms"][arm_name]["rot_err_deg"]),
                            "attempt_history": action_dual["attempt_history"],
                        },
                    }
                continue_execution_after_keyframe1 = False
            else:
                continue_execution_after_keyframe1 = True

            if not continue_execution_after_keyframe1:
                pass
            else:
                if bool(args.enable_grasp_action_object_collision):
                    export_close_stage_snapshot(
                        args.output_dir,
                        tag="dual_before_close",
                        renderer=renderer,
                        object_states=object_states,
                        selected_objects_by_arm=selected_objects_by_executed_arm,
                        arms=dual_arms,
                    )
                    set_object_collision_for_names(
                        object_states,
                        [selected_objects_by_executed_arm[arm_name] for arm_name in dual_arms],
                        enabled=True,
                    )
                    close_summary = close_grippers_progressively_with_collision_stop(
                        renderer,
                        args.close_gripper,
                        args.close_gripper,
                        {arm_name: object_states[selected_objects_by_executed_arm[arm_name]].actor for arm_name in dual_arms},
                        object_debug_by_arm={
                            arm_name: {
                                "object_name": selected_objects_by_executed_arm[arm_name],
                                "collision_mode": object_states[selected_objects_by_executed_arm[arm_name]].collision_mode,
                                "collision_debug_info": object_states[selected_objects_by_executed_arm[arm_name]].collision_debug_info,
                            }
                            for arm_name in dual_arms
                        },
                        debug_collision_report=bool(args.debug_collision_report),
                        gripper_contact_monitor_mode=str(args.gripper_contact_monitor_mode),
                    )
                    print(
                        "[gripper-close] "
                        + " ".join(
                            f"{arm_name}:monitor={close_summary.get(arm_name, {}).get('monitor_mode', 'n/a')},"
                            f"reason={close_summary.get(arm_name, {}).get('reason', 'n/a')},"
                            f"cmd={float(close_summary.get(arm_name, {}).get('final_cmd', args.close_gripper)):.3f},"
                            f"contact={int(bool(close_summary.get(arm_name, {}).get('had_contact', False)))},"
                            f"base_contact={int(bool(close_summary.get(arm_name, {}).get('had_base_contact', False)))},"
                            f"raw_target_contact={int(bool(close_summary.get(arm_name, {}).get('had_raw_target_contact', False)))}"
                            for arm_name in dual_arms
                        )
                    )
                else:
                    renderer.set_grippers(args.close_gripper, args.close_gripper)
                debug_execution_state.current_stage = "close_gripper"
                record_frame(
                    renderer,
                    head_writer,
                    third_writer,
                    ["stage=close_gripper", "arm=both"],
                    use_overlay,
                    debug_visuals,
                    debug_execution_state,
                    pure_scene_main=pure_scene_main,
                    use_overlay_debug=use_overlay_debug,
                )
                pause_after_keyframe1(
                    renderer=renderer,
                    head_writer=head_writer,
                    third_writer=third_writer,
                    use_overlay=use_overlay,
                    args=args,
                    arm_label="both",
                    goal_frame=int(exec_selected_by_arm["left"][0].source_frame),
                    debug_visuals=debug_visuals,
                    debug_execution_state=debug_execution_state,
                    pure_scene_main=pure_scene_main,
                    use_overlay_debug=use_overlay_debug,
                )

                attached_actor_by_arm: Dict[str, Optional[sapien.Entity]] = {}
                tcp_to_object_by_arm: Dict[str, Optional[np.ndarray]] = {}
                if not bool(args.replay_objects_during_action):
                    for arm_name in dual_arms:
                        obj_name = selected_objects_by_executed_arm[arm_name]
                        attached_actor_by_arm[arm_name] = object_states[obj_name].actor
                        tcp_pose = renderer.get_current_tcp_pose(arm_name)
                        tcp_to_object_by_arm[arm_name] = np.linalg.inv(pose_wxyz_to_matrix(tcp_pose)) @ object_states[obj_name].pose_world_matrix

                debug_execution_state.active_frame = int(exec_selected_by_arm["left"][1].source_frame)
                debug_execution_state.active_frame_by_arm = {
                    arm_name: int(seq[1].source_frame) for arm_name, seq in exec_selected_by_arm.items()
                }
                debug_execution_state.current_stage = "action"
                debug_execution_state.target_pose_by_arm = {k: np.asarray(v, dtype=np.float64) for k, v in action_targets.items()}
                debug_execution_state.goal_label_by_arm = {
                    arm_name: f"keyframe_{int(seq[1].source_frame)}_action" for arm_name, seq in exec_selected_by_arm.items()
                }
                set_dual_arm_target_visuals(
                    renderer,
                    action_targets.get("left"),
                    action_targets.get("right"),
                )
                action_dual = execute_dual_stage_until_reached(
                    renderer=renderer,
                    target_pose_world_wxyz_by_arm=action_targets,
                    label="action",
                    head_writer=head_writer,
                    third_writer=third_writer,
                    use_overlay=use_overlay,
                    args=args,
                    debug_visuals=debug_visuals,
                    debug_execution_state=debug_execution_state,
                    attached_actor_by_arm=attached_actor_by_arm,
                    tcp_to_object_by_arm=tcp_to_object_by_arm,
                    object_replay=action_object_replay,
                    pure_scene_main=pure_scene_main,
                    use_overlay_debug=use_overlay_debug,
                ) if (
                    not bool(args.require_keyframe1_reached_before_action)
                    or all(bool(grasp_dual["arms"].get(arm_name, {}).get("reached", False)) for arm_name in dual_arms)
                ) else skipped_dual_action_result(
                    renderer=renderer,
                    action_targets=action_targets,
                    args=args,
                    reason="keyframe1_grasp_not_reached",
                )
                if bool(action_dual.get("skip_reason")):
                    print(f"[stage] action skipped reason={action_dual.get('skip_reason')}")

                for arm_name in dual_arms:
                    stages_by_executed_arm[arm_name] = {
                        "pregrasp": {
                            "status": str(pregrasp_dual["arms"][arm_name]["status"]),
                            "attempts": int(pregrasp_dual["attempts"]),
                            "reached": bool(pregrasp_dual["arms"][arm_name]["reached"]),
                            "pos_err_m": float(pregrasp_dual["arms"][arm_name]["pos_err_m"]),
                            "rot_err_deg": float(pregrasp_dual["arms"][arm_name]["rot_err_deg"]),
                            "attempt_history": pregrasp_dual["attempt_history"],
                        },
                        "grasp": {
                            "status": str(grasp_dual["arms"][arm_name]["status"]),
                            "attempts": int(grasp_dual["attempts"]),
                            "reached": bool(grasp_dual["arms"][arm_name]["reached"]),
                            "pos_err_m": float(grasp_dual["arms"][arm_name]["pos_err_m"]),
                            "rot_err_deg": float(grasp_dual["arms"][arm_name]["rot_err_deg"]),
                            "attempt_history": grasp_dual["attempt_history"],
                        },
                        "action": {
                            "status": str(action_dual["arms"][arm_name]["status"]),
                            "attempts": int(action_dual["attempts"]),
                            "reached": bool(action_dual["arms"][arm_name]["reached"]),
                            "pos_err_m": float(action_dual["arms"][arm_name]["pos_err_m"]),
                            "rot_err_deg": float(action_dual["arms"][arm_name]["rot_err_deg"]),
                            "attempt_history": action_dual["attempt_history"],
                        },
                    }
        else:
            for exec_arm, exec_selected_keyframes in execution_sequences:
                exec_object_name = exec_selected_keyframes[0].candidate.nearest_object
                selected_objects_by_executed_arm[exec_arm] = exec_object_name
                selected_candidates_by_executed_arm[exec_arm] = list(exec_selected_keyframes)
                debug_execution_state.selected_keyframes = exec_selected_keyframes

                init_info = apply_robot_init_pose(renderer, open_gripper=args.open_gripper, settle_steps=args.settle_steps)
                init_pose_info_by_executed_arm[exec_arm] = init_info
                init_prefix_frames_written_by_executed_arm[exec_arm] = int(
                    emit_init_prefix_frames(
                        renderer=renderer,
                        head_writer=head_writer,
                        third_writer=third_writer,
                        use_overlay=use_overlay,
                        init_info=init_info,
                        fixed_frames=args.init_prefix_frames,
                        arm_label=exec_arm,
                        debug_visuals=debug_visuals,
                        debug_execution_state=debug_execution_state,
                        pure_scene_main=pure_scene_main,
                        use_overlay_debug=use_overlay_debug,
                    )
                )

                supervision_targets = build_supervision_targets(arm_display_candidates, exec_selected_keyframes)
                supervision_targets_by_executed_arm[exec_arm] = supervision_targets
                supervision_only_arms = sorted([name for name in supervision_targets.keys() if name != exec_arm])
                supervision_only_arms_by_executed_arm[exec_arm] = supervision_only_arms
                print(
                    f"[exec-mode] selected_arm={exec_arm} "
                    f"supervision_only_arms={supervision_only_arms if supervision_only_arms else 'none'}"
                )

                renderer.set_grippers(
                    args.open_gripper if exec_arm == "left" else None,
                    args.open_gripper if exec_arm == "right" else None,
                )

                grasp_pose = exec_selected_keyframes[0].candidate.pose_world_wxyz
                pregrasp_pose = pose_with_offset_along_local_axis(grasp_pose, args.approach_offset_m, args.approach_axis)
                action_pose = exec_selected_keyframes[1].candidate.pose_world_wxyz
                action_object_replay = None
                if bool(args.replay_objects_during_action):
                    action_object_replay = ExecutionObjectReplayConfig(
                        replay_frame_indices=np.asarray(replay_frame_indices, dtype=np.int32),
                        object_tracks=object_tracks,
                        start_frame=int(exec_selected_keyframes[0].source_frame),
                        end_frame=int(exec_selected_keyframes[1].source_frame),
                    )
                debug_execution_state.active_frame = int(exec_selected_keyframes[0].source_frame)
                debug_execution_state.active_frame_by_arm = {exec_arm: int(exec_selected_keyframes[0].source_frame)}
                debug_execution_state.current_stage = "init"
                debug_execution_state.target_object_by_arm = {exec_arm: exec_object_name}
                debug_execution_state.goal_label_by_arm = {exec_arm: f"keyframe_{int(exec_selected_keyframes[0].source_frame)}_grasp"}
                debug_execution_state.target_pose_by_arm = {exec_arm: np.asarray(grasp_pose, dtype=np.float64)}
                set_single_arm_target_visual(renderer, exec_arm, grasp_pose)
                record_frame(
                    renderer,
                    head_writer,
                    third_writer,
                    [
                        "stage=init",
                        f"arm={exec_arm}",
                        f"object={exec_object_name}",
                        f"goal=keyframe_{exec_selected_keyframes[0].source_frame}",
                        f"selected_f1_candidate={exec_selected_keyframes[0].candidate.candidate_idx}",
                        f"selected_f22_candidate={exec_selected_keyframes[1].candidate.candidate_idx}",
                    ],
                    use_overlay,
                    debug_visuals,
                    debug_execution_state,
                    pure_scene_main=pure_scene_main,
                    use_overlay_debug=use_overlay_debug,
                )

                debug_execution_state.current_stage = "pregrasp"
                debug_execution_state.active_frame_by_arm = {exec_arm: int(exec_selected_keyframes[0].source_frame)}
                debug_execution_state.target_pose_by_arm = {exec_arm: np.asarray(pregrasp_pose, dtype=np.float64)}
                debug_execution_state.goal_label_by_arm = {exec_arm: f"keyframe_{int(exec_selected_keyframes[0].source_frame)}_pregrasp"}
                pregrasp_result = execute_stage_until_reached(
                    renderer=renderer,
                    arm=exec_arm,
                    target_pose_world_wxyz=pregrasp_pose,
                    label="pregrasp",
                    head_writer=head_writer,
                    third_writer=third_writer,
                    use_overlay=use_overlay,
                    args=args,
                    target_visual_pose=grasp_pose,
                    target_visual_label=f"keyframe_{exec_selected_keyframes[0].source_frame}",
                    debug_visuals=debug_visuals,
                    debug_execution_state=debug_execution_state,
                    supervision_targets=supervision_targets,
                    pure_scene_main=pure_scene_main,
                    use_overlay_debug=use_overlay_debug,
                )

                if use_grasp_action_object_collision and collision_start_stage == "grasp":
                    set_object_collision_for_names(object_states, [exec_object_name], enabled=True)

                if poses_are_effectively_same(pregrasp_pose, grasp_pose):
                    debug_execution_state.current_stage = "grasp_skipped_same_target"
                    debug_execution_state.active_frame_by_arm = {exec_arm: int(exec_selected_keyframes[0].source_frame)}
                    debug_execution_state.target_pose_by_arm = {exec_arm: np.asarray(grasp_pose, dtype=np.float64)}
                    debug_execution_state.goal_label_by_arm = {exec_arm: f"keyframe_{int(exec_selected_keyframes[0].source_frame)}_grasp_skipped"}
                    set_single_arm_target_visual(renderer, exec_arm, grasp_pose)
                    record_frame(
                        renderer,
                        head_writer,
                        third_writer,
                        ["stage=grasp_skipped_same_target", f"arm={exec_arm}"],
                        use_overlay,
                        debug_visuals,
                        debug_execution_state,
                        pure_scene_main=pure_scene_main,
                        use_overlay_debug=use_overlay_debug,
                    )
                    grasp_result = dict(pregrasp_result)
                    grasp_result["skipped_same_target"] = True
                    print(f"[stage] grasp skipped because pregrasp target equals grasp target arm={exec_arm}")
                else:
                    debug_execution_state.current_stage = "grasp"
                    debug_execution_state.active_frame_by_arm = {exec_arm: int(exec_selected_keyframes[0].source_frame)}
                    debug_execution_state.target_pose_by_arm = {exec_arm: np.asarray(grasp_pose, dtype=np.float64)}
                    debug_execution_state.goal_label_by_arm = {exec_arm: f"keyframe_{int(exec_selected_keyframes[0].source_frame)}_grasp"}
                    grasp_result = execute_stage_until_reached(
                        renderer=renderer,
                        arm=exec_arm,
                        target_pose_world_wxyz=grasp_pose,
                        label="grasp",
                        head_writer=head_writer,
                        third_writer=third_writer,
                        use_overlay=use_overlay,
                        args=args,
                        target_visual_pose=grasp_pose,
                        target_visual_label=f"keyframe_{exec_selected_keyframes[0].source_frame}",
                        debug_visuals=debug_visuals,
                        debug_execution_state=debug_execution_state,
                        supervision_targets=supervision_targets,
                        pure_scene_main=pure_scene_main,
                        use_overlay_debug=use_overlay_debug,
                    )
                print(
                    "[stage] keyframe_1 grasp finished "
                    f"arm={exec_arm} reached={int(bool(grasp_result.get('reached', False)))}"
                )
                if not bool(grasp_result.get("reached", False)):
                    print(
                        "[warn] grasp_not_reached_before_close "
                        f"{exec_arm}:pos={float(grasp_result.get('pos_err_m', 0.0)):.4f},"
                        f"rot={float(grasp_result.get('rot_err_deg', 0.0)):.2f}"
                    )

                stop_before_close_reason = None
                if bool(args.debug_stop_after_keyframe1):
                    stop_before_close_reason = "debug_stop_after_keyframe1"
                elif bool(args.require_keyframe1_reached_before_close) and not bool(grasp_result.get("reached", False)):
                    stop_before_close_reason = "keyframe1_grasp_not_reached_before_close"

                if stop_before_close_reason is not None:
                    action_result = skipped_action_result_for_arm(
                        renderer=renderer,
                        arm=exec_arm,
                        action_pose_world_wxyz=action_pose,
                        args=args,
                        reason=stop_before_close_reason,
                    )
                    print(f"[stage] stopping before close_gripper arm={exec_arm} reason={stop_before_close_reason}")
                    record_frame(
                        renderer,
                        head_writer,
                        third_writer,
                        ["stage=close_skipped", f"arm={exec_arm}", f"reason={stop_before_close_reason}"],
                        use_overlay,
                        debug_visuals,
                        debug_execution_state,
                        pure_scene_main=pure_scene_main,
                        use_overlay_debug=use_overlay_debug,
                    )
                    stages_by_executed_arm[exec_arm] = {
                        "pregrasp": pregrasp_result,
                        "grasp": grasp_result,
                        "action": action_result,
                    }
                    continue

                set_single_arm_target_visual(renderer, exec_arm, grasp_pose)
                if bool(args.enable_grasp_action_object_collision):
                    export_close_stage_snapshot(
                        args.output_dir,
                        tag=f"{exec_arm}_before_close",
                        renderer=renderer,
                        object_states=object_states,
                        selected_objects_by_arm={exec_arm: exec_object_name},
                        arms=[exec_arm],
                    )
                    set_object_collision_for_names(object_states, [exec_object_name], enabled=True)
                    close_summary = close_grippers_progressively_with_collision_stop(
                        renderer,
                        args.close_gripper if exec_arm == "left" else None,
                        args.close_gripper if exec_arm == "right" else None,
                        {exec_arm: object_states[exec_object_name].actor},
                        object_debug_by_arm={
                            exec_arm: {
                                "object_name": exec_object_name,
                                "collision_mode": object_states[exec_object_name].collision_mode,
                                "collision_debug_info": object_states[exec_object_name].collision_debug_info,
                            }
                        },
                        debug_collision_report=bool(args.debug_collision_report),
                        gripper_contact_monitor_mode=str(args.gripper_contact_monitor_mode),
                    )
                    print(
                        "[gripper-close] "
                        f"{exec_arm}:monitor={close_summary.get(exec_arm, {}).get('monitor_mode', 'n/a')},"
                        f"reason={close_summary.get(exec_arm, {}).get('reason', 'n/a')},"
                        f"cmd={float(close_summary.get(exec_arm, {}).get('final_cmd', args.close_gripper)):.3f},"
                        f"contact={int(bool(close_summary.get(exec_arm, {}).get('had_contact', False)))},"
                        f"base_contact={int(bool(close_summary.get(exec_arm, {}).get('had_base_contact', False)))},"
                        f"raw_target_contact={int(bool(close_summary.get(exec_arm, {}).get('had_raw_target_contact', False)))}"
                    )
                else:
                    renderer.set_grippers(args.close_gripper if exec_arm == "left" else None, args.close_gripper if exec_arm == "right" else None)
                debug_execution_state.current_stage = "close_gripper"
                record_frame(
                    renderer,
                    head_writer,
                    third_writer,
                    ["stage=close_gripper", f"arm={exec_arm}", f"goal=keyframe_{exec_selected_keyframes[0].source_frame}"],
                    use_overlay,
                    debug_visuals,
                    debug_execution_state,
                    pure_scene_main=pure_scene_main,
                    use_overlay_debug=use_overlay_debug,
                )
                pause_after_keyframe1(
                    renderer=renderer,
                    head_writer=head_writer,
                    third_writer=third_writer,
                    use_overlay=use_overlay,
                    args=args,
                    arm_label=exec_arm,
                    goal_frame=int(exec_selected_keyframes[0].source_frame),
                    debug_visuals=debug_visuals,
                    debug_execution_state=debug_execution_state,
                    pure_scene_main=pure_scene_main,
                    use_overlay_debug=use_overlay_debug,
                )

                attached_actor = None
                tcp_to_object = None
                if not bool(args.replay_objects_during_action):
                    attached_actor = object_states[exec_object_name].actor
                    tcp_pose = renderer.get_current_tcp_pose(exec_arm)
                    tcp_to_object = np.linalg.inv(pose_wxyz_to_matrix(tcp_pose)) @ object_states[exec_object_name].pose_world_matrix

                debug_execution_state.active_frame = int(exec_selected_keyframes[1].source_frame)
                debug_execution_state.active_frame_by_arm = {exec_arm: int(exec_selected_keyframes[1].source_frame)}
                debug_execution_state.current_stage = "action"
                debug_execution_state.target_pose_by_arm = {exec_arm: np.asarray(action_pose, dtype=np.float64)}
                debug_execution_state.goal_label_by_arm = {exec_arm: f"keyframe_{int(exec_selected_keyframes[1].source_frame)}_action"}
                if bool(args.require_keyframe1_reached_before_action) and not bool(grasp_result.get("reached", False)):
                    action_result = skipped_action_result_for_arm(
                        renderer=renderer,
                        arm=exec_arm,
                        action_pose_world_wxyz=action_pose,
                        args=args,
                        reason="keyframe1_grasp_not_reached",
                    )
                    print(f"[stage] action skipped arm={exec_arm} reason={action_result.get('skip_reason')}")
                    record_frame(
                        renderer,
                        head_writer,
                        third_writer,
                        ["stage=action_skipped", f"arm={exec_arm}", f"reason={action_result.get('skip_reason')}"],
                        use_overlay,
                        debug_visuals,
                        debug_execution_state,
                        pure_scene_main=pure_scene_main,
                        use_overlay_debug=use_overlay_debug,
                    )
                else:
                    action_result = execute_stage_until_reached(
                        renderer=renderer,
                        arm=exec_arm,
                        target_pose_world_wxyz=action_pose,
                        label="action",
                        head_writer=head_writer,
                        third_writer=third_writer,
                        use_overlay=use_overlay,
                        args=args,
                        attached_actor=attached_actor,
                        tcp_to_object=tcp_to_object,
                        target_visual_pose=action_pose,
                        target_visual_label=f"keyframe_{exec_selected_keyframes[1].source_frame}",
                        debug_visuals=debug_visuals,
                        debug_execution_state=debug_execution_state,
                        object_replay=action_object_replay,
                        pure_scene_main=pure_scene_main,
                        use_overlay_debug=use_overlay_debug,
                    )
                stages_by_executed_arm[exec_arm] = {
                    "pregrasp": pregrasp_result,
                    "grasp": grasp_result,
                    "action": action_result,
                }
    finally:
        if bool(args.enable_grasp_action_object_collision):
            set_object_collision_for_names(object_states, selected_objects_by_executed_arm.values(), enabled=False)
        head_writer.release()
        if third_writer is not None:
            third_writer.release()
        if left_wrist_writer is not None:
            left_wrist_writer.release()
        if right_wrist_writer is not None:
            right_wrist_writer.release()
        if debug_execution_writer is not None:
            debug_execution_writer.release()
        renderer.update_target_axis_visuals(None, None)
        if bool(args.vscode_compatible_video):
            transcode_mp4_for_vscode(head_video_path)
            if third_writer is not None:
                transcode_mp4_for_vscode(third_video_path)

    primary_exec_arm = execution_sequences[0][0]
    primary_exec_selected_keyframes = execution_sequences[0][1]
    primary_object_name = selected_objects_by_executed_arm[primary_exec_arm]
    primary_stages = stages_by_executed_arm[primary_exec_arm]
    primary_supervision_targets = supervision_targets_by_executed_arm[primary_exec_arm]
    primary_supervision_only_arms = supervision_only_arms_by_executed_arm[primary_exec_arm]
    analysis_plot_paths = generate_execution_analysis_plots(
        metrics_path=metrics_debug_path,
        output_dir=args.output_dir / "analysis_plots",
        fps=float(args.debug_execution_fps),
    )

    summary = {
        "anygrasp_dir": str(args.anygrasp_dir),
        "replay_dir": str(args.replay_dir),
        "hand_npz": str(args.hand_npz),
        "reuse_plan_summary_json": None if args.reuse_plan_summary_json is None else str(args.reuse_plan_summary_json),
        "reuse_preview_summary_json": None if args.reuse_preview_summary_json is None else str(args.reuse_preview_summary_json),
        "reuse_preview_frame_mode": str(args.reuse_preview_frame_mode),
        "reuse_preview_candidate_group": str(args.reuse_preview_candidate_group),
        "reuse_preview_top_rank": int(args.reuse_preview_top_rank),
        "selection_source": (
            "reused_preview_summary_json"
            if args.reuse_preview_summary_json is not None
            else ("reused_plan_summary_json" if args.reuse_plan_summary_json is not None else "recomputed_from_anygrasp")
        ),
        "requested_keyframes": [int(v) for v in args.requested_keyframes],
        "resolved_keyframes": [int(v) for v in args.resolved_keyframes],
        "resolved_keyframe_pairs": [{"requested": int(req), "resolved": int(res)} for req, res in args.resolved_keyframe_pairs],
        "reuse_preview_frame_selection": None if reused_preview_summary is None else reused_preview_summary.get("frame_selection"),
        "keyframes": keyframes,
        "selected_arm": primary_exec_arm,
        "expected_object_for_selected_arm": selection_result.expected_object,
        "selection_diagnostics": selection_result.diagnostics,
        "candidate_orientation_remap_label": args.candidate_orientation_remap_label,
        "candidate_post_rot_xyz_deg": args.candidate_post_rot_xyz_deg.tolist(),
        "planner_backend": str(args.planner_backend),
        "candidate_selection_mode": str(args.candidate_selection_mode),
        "candidate_selection_relative_frame": None if args.candidate_selection_relative_frame is None else int(args.candidate_selection_relative_frame),
        "resolved_candidate_selection_relative_frame": None if args.resolved_candidate_selection_relative_frame is None else int(args.resolved_candidate_selection_relative_frame),
        "candidate_max_rotation_distance_deg": float(args.candidate_max_rotation_distance_deg),
        "urdfik_trajectory_mode": str(args.urdfik_trajectory_mode),
        "urdfik_joint_interp_waypoints": int(args.urdfik_joint_interp_waypoints),
        "urdfik_cartesian_interp_steps": int(args.urdfik_cartesian_interp_steps),
        "urdfik_cartesian_interp_auto_step_m": float(args.urdfik_cartesian_interp_auto_step_m),
        "urdfik_position_threshold_m": float(args.urdfik_position_threshold_m),
        "urdfik_rotation_threshold_rad": float(args.urdfik_rotation_threshold_rad),
        "urdfik_max_position_threshold_m": None if args.urdfik_max_position_threshold_m is None else float(args.urdfik_max_position_threshold_m),
        "urdfik_max_rotation_threshold_rad": None if args.urdfik_max_rotation_threshold_rad is None else float(args.urdfik_max_rotation_threshold_rad),
        "urdfik_num_seeds": int(args.urdfik_num_seeds),
        "execute_partial_cartesian_plan": int(args.execute_partial_cartesian_plan),
        "piper_urdfik_apply_global_trans_to_ik": int(args.piper_urdfik_apply_global_trans_to_ik),
        "execute_interp_steps": int(args.execute_interp_steps),
        "settle_steps": int(args.settle_steps),
        "joint_command_scene_steps": int(args.joint_command_scene_steps),
        "joint_target_wait_steps": int(args.joint_target_wait_steps),
        "joint_target_wait_tol_rad": float(args.joint_target_wait_tol_rad),
        "print_execution_pose_every": int(args.print_execution_pose_every),
        "reach_pos_tol_m": float(args.reach_pos_tol_m),
        "reach_rot_tol_deg": float(args.reach_rot_tol_deg),
        "candidate_keep_camera_up": int(args.candidate_keep_camera_up),
        "candidate_camera_top_axis": str(args.candidate_camera_top_axis),
        "candidate_target_local_x_offset_m": float(args.candidate_target_local_x_offset_m),
        "candidate_target_local_z_offset_m": float(args.candidate_target_local_z_offset_m),
        "approach_axis": str(args.approach_axis),
        "debug_gripper_actor_forward_axis": str(args.debug_gripper_actor_forward_axis),
        "enable_grasp_action_object_collision": int(args.enable_grasp_action_object_collision),
        "grasp_action_object_collision_start_stage": str(args.grasp_action_object_collision_start_stage),
        "disable_table": int(args.disable_table),
        "base_occluder_enable": int(args.base_occluder_enable),
        "pure_scene_output": int(args.pure_scene_output),
        "save_pose_debug_effective": int(save_pose_debug_effective),
        "reach_error_pose_source": args.reach_error_pose_source,
        "init_prefix_frames": int(args.init_prefix_frames),
        "execute_both_arms": int(args.execute_both_arms),
        "dual_stage_require_all_plans": int(args.dual_stage_require_all_plans),
        "require_keyframe1_reached_before_close": int(args.require_keyframe1_reached_before_close),
        "require_keyframe1_reached_before_action": int(args.require_keyframe1_reached_before_action),
        "debug_stop_after_keyframe1": int(args.debug_stop_after_keyframe1),
        "execution_mode": "dual_sync" if dual_sync_mode else "single_or_sequential",
        "manual_candidate_overrides": {arm: {str(frame): idx for frame, idx in frame_map.items()} for arm, frame_map in args.manual_candidate_overrides.items() if frame_map},
        "arm_debugs": {
            arm_name: {
                "expected_object": arm_info.expected_object,
                "diagnostics": arm_info.diagnostics,
                "selected_keyframes": (
                    [
                        {
                            "source_frame": item.source_frame,
                            "candidate_idx": item.candidate.candidate_idx,
                            "nearest_object": item.candidate.nearest_object,
                            "rotation_distance_deg": item.candidate.rotation_distance_deg,
                            "top_axis_up_dot": item.candidate.top_axis_up_dot,
                            "original_top_axis_up_dot": item.candidate.original_top_axis_up_dot,
                            "camera_up_flip_applied": item.candidate.camera_up_flip_applied,
                            "forward_axis_change_deg": item.candidate.forward_axis_change_deg,
                        }
                        for item in arm_info.selected_keyframes
                    ]
                    if arm_info.selected_keyframes is not None
                    else None
                ),
            }
            for arm_name, arm_info in arm_debugs.items()
        },
        "selected_object": primary_object_name,
        "selected_objects_by_executed_arm": selected_objects_by_executed_arm,
        "executed_arms": [item[0] for item in execution_sequences],
        "supervision_only_arms": primary_supervision_only_arms,
        "supervision_only_arms_by_executed_arm": supervision_only_arms_by_executed_arm,
        "supervision_targets": {arm_name: target.tolist() for arm_name, target in primary_supervision_targets.items()},
        "supervision_targets_by_executed_arm": {
            arm_name: {target_arm: target.tolist() for target_arm, target in targets.items()}
            for arm_name, targets in supervision_targets_by_executed_arm.items()
        },
        "init_pose_by_executed_arm": {
            arm_name: {
                "left_applied": bool(info["left_applied"]),
                "right_applied": bool(info["right_applied"]),
                "left_joints": None if info["left_joints"] is None else np.asarray(info["left_joints"], dtype=np.float64).reshape(6).tolist(),
                "right_joints": None if info["right_joints"] is None else np.asarray(info["right_joints"], dtype=np.float64).reshape(6).tolist(),
                "gripper_open": float(info["gripper_open"]),
            }
            for arm_name, info in init_pose_info_by_executed_arm.items()
        },
        "init_prefix_frames_written_by_executed_arm": init_prefix_frames_written_by_executed_arm,
        "stages": primary_stages,
        "stages_by_executed_arm": stages_by_executed_arm,
        "selected_candidates": [
            {
                "source_frame": item.source_frame,
                "arm": item.arm,
                "candidate_idx": item.candidate.candidate_idx,
                "score": item.candidate.score,
                "rotation_distance_deg": item.candidate.rotation_distance_deg,
                "nearest_object": item.candidate.nearest_object,
                "nearest_object_distance_m": item.candidate.nearest_object_distance_m,
                "width_m": item.candidate.width_m,
                "depth_m": item.candidate.depth_m,
                "top_axis_up_dot": item.candidate.top_axis_up_dot,
                "original_top_axis_up_dot": item.candidate.original_top_axis_up_dot,
                "camera_up_flip_applied": item.candidate.camera_up_flip_applied,
                "forward_axis_change_deg": item.candidate.forward_axis_change_deg,
                "camera_up_selection_mode": item.candidate.camera_up_selection_mode,
                "raw_pose_world_wxyz": item.candidate.raw_pose_world_wxyz.tolist(),
                "pose_world_wxyz": item.candidate.pose_world_wxyz.tolist(),
                "translation_cam": item.candidate.translation_cam.tolist(),
                "rotation_cam": item.candidate.rotation_cam.tolist(),
                "hand_rotation_cam": item.hand_rotation_cam.tolist(),
            }
            for item in primary_exec_selected_keyframes
        ],
        "selected_candidates_by_executed_arm": {
            arm_name: [
                {
                    "source_frame": item.source_frame,
                    "arm": item.arm,
                    "candidate_idx": item.candidate.candidate_idx,
                    "score": item.candidate.score,
                    "rotation_distance_deg": item.candidate.rotation_distance_deg,
                    "nearest_object": item.candidate.nearest_object,
                    "nearest_object_distance_m": item.candidate.nearest_object_distance_m,
                    "width_m": item.candidate.width_m,
                    "depth_m": item.candidate.depth_m,
                    "top_axis_up_dot": item.candidate.top_axis_up_dot,
                    "original_top_axis_up_dot": item.candidate.original_top_axis_up_dot,
                    "camera_up_flip_applied": item.candidate.camera_up_flip_applied,
                    "forward_axis_change_deg": item.candidate.forward_axis_change_deg,
                    "camera_up_selection_mode": item.candidate.camera_up_selection_mode,
                    "raw_pose_world_wxyz": item.candidate.raw_pose_world_wxyz.tolist(),
                    "pose_world_wxyz": item.candidate.pose_world_wxyz.tolist(),
                    "translation_cam": item.candidate.translation_cam.tolist(),
                    "rotation_cam": item.candidate.rotation_cam.tolist(),
                    "hand_rotation_cam": item.hand_rotation_cam.tolist(),
                }
                for item in candidates
            ]
            for arm_name, candidates in selected_candidates_by_executed_arm.items()
        },
        "top_candidates_per_keyframe": {
            str(frame): [
                {
                    "candidate_idx": cand.candidate_idx,
                    "score": cand.score,
                    "nearest_object": cand.nearest_object,
                    "nearest_object_distance_m": cand.nearest_object_distance_m,
                    "rotation_distance_deg": cand.rotation_distance_deg,
                    "width_m": cand.width_m,
                    "depth_m": cand.depth_m,
                    "top_axis_up_dot": cand.top_axis_up_dot,
                    "original_top_axis_up_dot": cand.original_top_axis_up_dot,
                    "camera_up_flip_applied": cand.camera_up_flip_applied,
                    "forward_axis_change_deg": cand.forward_axis_change_deg,
                    "camera_up_selection_mode": cand.camera_up_selection_mode,
                    "raw_pose_world_wxyz": cand.raw_pose_world_wxyz.tolist(),
                    "pose_world_wxyz": cand.pose_world_wxyz.tolist(),
                }
                for cand in ranked_candidates_per_frame.get(int(frame), [])[: max(int(args.debug_candidate_top_k), 0)]
            ]
            for frame in keyframes
        },
        "all_candidates_per_keyframe": {
            str(frame): [
                {
                    "candidate_idx": cand.candidate_idx,
                    "score": cand.score,
                    "nearest_object": cand.nearest_object,
                    "nearest_object_distance_m": cand.nearest_object_distance_m,
                    "rotation_distance_deg": cand.rotation_distance_deg,
                    "width_m": cand.width_m,
                    "depth_m": cand.depth_m,
                    "top_axis_up_dot": cand.top_axis_up_dot,
                    "original_top_axis_up_dot": cand.original_top_axis_up_dot,
                    "camera_up_flip_applied": cand.camera_up_flip_applied,
                    "forward_axis_change_deg": cand.forward_axis_change_deg,
                    "camera_up_selection_mode": cand.camera_up_selection_mode,
                    "raw_pose_world_wxyz": cand.raw_pose_world_wxyz.tolist(),
                    "pose_world_wxyz": cand.pose_world_wxyz.tolist(),
                }
                for cand in all_candidates_per_frame.get(int(frame), [])
            ]
            for frame in keyframes
        },
        "debug_preview_video": str(debug_video_path) if debug_video_path is not None else None,
        "rank_preview_images": [
            {
                "frame": item.frame,
                "rank": item.rank,
                "image_path": item.image_path,
                "left_candidate_idx": item.left_candidate_idx,
                "right_candidate_idx": item.right_candidate_idx,
            }
            for item in rank_preview_records
        ],
        "source_preview_compare": [
            {
                "frame": item.frame,
                "source_kind": item.source_kind,
                "image_path": item.image_path,
                "source_path": item.source_path,
            }
            for item in source_preview_compare_records
        ],
        "debug_execution_video": str(debug_execution_video_path) if debug_execution_writer is not None else None,
        "debug_execution_metrics": str(metrics_debug_path),
        "pose_debug": str(args.output_dir / "pose_debug.jsonl") if save_pose_debug_effective else None,
        "analysis_plots": analysis_plot_paths,
        "head_video": str(head_video_path),
        "third_video": str(third_video_path) if third_writer is not None else None,
        "vscode_compatible_video": int(args.vscode_compatible_video),
        "left_wrist_video": str(left_wrist_video_path) if left_wrist_writer is not None else None,
        "right_wrist_video": str(right_wrist_video_path) if right_wrist_writer is not None else None,
    }
    summary["failed_stage_records"] = collect_failed_stage_records(stages_by_executed_arm)
    summary["execution_failed"] = bool(summary["failed_stage_records"])
    summary["execution_success"] = not bool(summary["execution_failed"])
    with (args.output_dir / "plan_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    status_prefix = "[warn]" if bool(summary["execution_failed"]) else "[done]"
    sel0 = primary_exec_selected_keyframes[0]
    sel1 = primary_exec_selected_keyframes[1]
    print(
        f"{status_prefix} "
        f"arms={','.join(summary['executed_arms'])} arm={primary_exec_arm} obj={primary_object_name} "
        f"f{sel0.source_frame}=c{sel0.candidate.candidate_idx} "
        f"f{sel1.source_frame}=c{sel1.candidate.candidate_idx} "
        f"pre={short_stage_status(primary_stages.get('pregrasp'))} "
        f"gr={short_stage_status(primary_stages.get('grasp'))} "
        f"act={short_stage_status(primary_stages.get('action'))} "
        f"video={head_video_path}"
    )
    renderer.hold_viewer()


if __name__ == "__main__":
    main()
