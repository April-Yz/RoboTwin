"""O.1: FoundationPose positions and source OBJ meshes with Piper Cartesian IK."""

import json
import pickle
import re
from copy import deepcopy
from pathlib import Path

import numpy as np
import sapien.core as sapien
import transforms3d as t3d
import trimesh

from .pick_diverse_bottles_piper_ik import pick_diverse_bottles_piper_ik
from .utils import Action
from .utils.actor_utils import Actor


class pick_diverse_bottles_piper_ik_foundation(pick_diverse_bottles_piper_ik):
    """Use FoundationPose positions and source visual meshes instead of random assets."""

    FOUNDATION_OBJECT_KEYS = {
        "left": "left_bottle",
        "right": "right_bottle",
    }
    FOUNDATION_ACTOR_IDS = {
        "left": "foundation_left_bottle",
        "right": "foundation_right_bottle",
    }
    FOUNDATION_DEFAULT_ANNOTATION_JSON = (
        "code_painting/h2o_manual_review/pick_diverse_bottles/hand_keyframes_all.json"
    )
    FOUNDATION_DEFAULT_HAND_TARGETS_ROOT = "code_painting/human_replay/pick_diverse_bottles"
    FOUNDATION_DEFAULT_HAND_TARGETS_PATTERN = "id_{episode}/world_targets_and_status.npz"
    FOUNDATION_DEFAULT_LEFT_DESCRIPTION = "left cola bottle"
    FOUNDATION_DEFAULT_RIGHT_DESCRIPTION = "right bottle"
    FOUNDATION_DEFAULT_OPEN_AFTER_ACTION = False
    FOUNDATION_DEFAULT_ACTION_TARGET_SOURCE = "hand_ee"
    FOUNDATION_DEFAULT_PREGRASP_CLEARANCE = 0.0

    def setup_demo(self, **kwargs):
        self._foundation_mode = str(kwargs.get("foundation_mode", "o1")).lower()
        if self._foundation_mode not in {"o1", "o1.1", "o1.2"}:
            raise ValueError(f"unsupported foundation_mode={self._foundation_mode!r}")
        self._foundation_object_keys = dict(type(self).FOUNDATION_OBJECT_KEYS)
        self._foundation_actor_ids = dict(type(self).FOUNDATION_ACTOR_IDS)
        self._foundation_input_dir = kwargs.get("foundation_input_dir")
        self._foundation_frame = int(kwargs.get("foundation_frame", 0))
        self._foundation_annotation_json = Path(
            kwargs.get(
                "foundation_annotation_json",
                type(self).FOUNDATION_DEFAULT_ANNOTATION_JSON,
            )
        ).expanduser().resolve()
        self._foundation_hand_targets_root = Path(
            kwargs.get(
                "foundation_hand_targets_root",
                type(self).FOUNDATION_DEFAULT_HAND_TARGETS_ROOT,
            )
        ).expanduser().resolve()
        self._foundation_hand_targets_pattern = str(
            kwargs.get(
                "foundation_hand_targets_pattern",
                type(self).FOUNDATION_DEFAULT_HAND_TARGETS_PATTERN,
            )
        )
        self._foundation_episode_id = kwargs.get("foundation_episode_id")
        self._foundation_keyframes = None
        self._foundation_action_positions = None
        self._foundation_action_object_centers = None
        self._foundation_use_orientation = bool(kwargs.get("foundation_use_orientation", False))
        self._foundation_table_clearance = float(kwargs.get("foundation_table_clearance", 0.002))
        self._foundation_mesh_mass = float(kwargs.get("foundation_mesh_mass", 0.05))
        self._foundation_mesh_friction = float(kwargs.get("foundation_mesh_friction", 1.5))
        self._foundation_grasp_standoff = float(kwargs.get("foundation_grasp_standoff", 0.105))
        self._foundation_grasp_lateral_offset = float(
            kwargs.get("foundation_grasp_lateral_offset", 0.0)
        )
        self._foundation_collision_radius_padding = float(
            kwargs.get("foundation_collision_radius_padding", 0.0)
        )
        self._foundation_collision_mode = kwargs.get("foundation_collision_mode", "support_proxy")
        self._foundation_grasp_assist = bool(kwargs.get("foundation_grasp_assist", True))
        self._foundation_grasp_assist_max_distance = float(
            kwargs.get("foundation_grasp_assist_max_distance", 0.14)
        )
        self._foundation_pregrasp_distance = float(
            kwargs.get("foundation_pregrasp_distance", 0.12)
        )
        self._foundation_grasp_max_displacement = float(
            kwargs.get("foundation_grasp_max_displacement", 0.025)
        )
        self._foundation_grasp_max_rotation_deg = float(
            kwargs.get("foundation_grasp_max_rotation_deg", 15.0)
        )
        self._foundation_capture_radial_tolerance = float(
            kwargs.get("foundation_capture_radial_tolerance", 0.065)
        )
        self._foundation_capture_segment_margin = float(
            kwargs.get("foundation_capture_segment_margin", 0.15)
        )
        self._foundation_grasp_require_contact = bool(
            kwargs.get("foundation_grasp_require_contact", False)
        )
        self._foundation_action_target_source = str(
            kwargs.get(
                "foundation_action_target_source",
                type(self).FOUNDATION_DEFAULT_ACTION_TARGET_SOURCE,
            )
        ).lower()
        if self._foundation_action_target_source not in {"hand_ee", "object_keyframe"}:
            raise ValueError(
                "foundation_action_target_source must be 'hand_ee' or 'object_keyframe'"
            )
        self._foundation_pregrasp_clearance = float(
            kwargs.get(
                "foundation_pregrasp_clearance",
                type(self).FOUNDATION_DEFAULT_PREGRASP_CLEARANCE,
            )
        )
        if self._foundation_pregrasp_clearance < 0:
            raise ValueError("foundation_pregrasp_clearance must be non-negative")
        if np.hypot(
            self._foundation_grasp_standoff,
            self._foundation_grasp_lateral_offset,
        ) <= 0:
            raise ValueError("Foundation grasp offset must be nonzero")
        self._foundation_left_description = kwargs.get(
            "foundation_left_description", type(self).FOUNDATION_DEFAULT_LEFT_DESCRIPTION
        )
        self._foundation_right_description = kwargs.get(
            "foundation_right_description", type(self).FOUNDATION_DEFAULT_RIGHT_DESCRIPTION
        )
        self._foundation_open_after_action = bool(
            kwargs.get(
                "foundation_open_after_action",
                type(self).FOUNDATION_DEFAULT_OPEN_AFTER_ACTION,
            )
        )
        self._foundation_grasp_drives = []
        self._foundation_data = None
        self._foundation_geometry = {}
        self._foundation_settled_poses = {}
        self._foundation_settled_centers = {}
        self._foundation_grasp_diagnostics = {}
        self._foundation_grasp_valid = False

        if not self._foundation_input_dir:
            raise ValueError(
                "O.1 requires foundation_input_dir in the task config; it must contain "
                "multi_object_world_poses.npz"
            )
        npz_path = Path(self._foundation_input_dir) / "multi_object_world_poses.npz"
        if not npz_path.is_file():
            raise FileNotFoundError(f"Foundation NPZ not found: {npz_path}")

        if self._foundation_episode_id is None:
            match = re.search(r"foundation_input_(\d+)$", Path(self._foundation_input_dir).name)
            if match is None:
                raise ValueError(
                    "foundation_episode_id is required when foundation_input_dir does not end "
                    "with foundation_input_<ID>"
                )
            self._foundation_episode_id = int(match.group(1))
        else:
            self._foundation_episode_id = int(self._foundation_episode_id)

        self._foundation_data = dict(np.load(npz_path, allow_pickle=True))
        required = tuple(
            f"{object_key}__{suffix}"
            for object_key in self._foundation_object_keys.values()
            for suffix in ("pose_world_wxyz", "mesh_file")
        )
        missing = [key for key in required if key not in self._foundation_data]
        if missing:
            raise KeyError(f"Foundation NPZ is missing keys: {missing}")

        if self._foundation_mode in {"o1.1", "o1.2"}:
            self._load_annotated_keyframes()
            self._foundation_frame = self._foundation_keyframes[0]
        if self._foundation_mode == "o1.2":
            move_names = ["pregrasp", "grasp", "action"]
            if self._foundation_pregrasp_clearance > 0:
                move_names.insert(0, "approach_clearance")
            self.MOVE_ACTION_NAMES = tuple(move_names)
            self._load_keyframe_action_positions()
        else:
            self.MOVE_ACTION_NAMES = type(self).MOVE_ACTION_NAMES

        for side in ("left", "right"):
            object_key = self._foundation_object_keys[side]
            visible = self._foundation_data.get(f"{object_key}__visible")
            poses = self._foundation_data[f"{object_key}__pose_world_wxyz"]
            if not 0 <= self._foundation_frame < len(poses):
                raise IndexError(
                    f"foundation_frame={self._foundation_frame} is outside {side} pose count {len(poses)}"
                )
            if visible is not None and not bool(visible[self._foundation_frame]):
                raise ValueError(f"{object_key} is not visible at frame {self._foundation_frame}")

        print(
            f"[piper-ik-foundation] mode={self._foundation_mode} source={npz_path} "
            f"episode={self._foundation_episode_id} frame={self._foundation_frame} "
            f"keyframes={self._foundation_keyframes} use_orientation={self._foundation_use_orientation} "
            f"grasp_standoff={self._foundation_grasp_standoff:.3f}m "
            f"pregrasp_distance={self._foundation_pregrasp_distance:.3f}m "
            f"pregrasp_clearance={self._foundation_pregrasp_clearance:.3f}m "
            f"action_target_source={self._foundation_action_target_source} "
            f"objects={self._foundation_object_keys}"
        )
        super().setup_demo(**kwargs)

    def _load_annotated_keyframes(self):
        if not self._foundation_annotation_json.is_file():
            raise FileNotFoundError(
                f"Foundation keyframe annotation not found: {self._foundation_annotation_json}"
            )
        with self._foundation_annotation_json.open(encoding="utf-8") as file:
            annotation = json.load(file)
        video_key = f"hand_vis_{self._foundation_episode_id}.mp4"
        record = annotation.get("videos", {}).get(video_key)
        if not isinstance(record, dict):
            raise KeyError(f"keyframe annotation is missing {video_key}")
        if str(record.get("status", "done")).lower() in {"reject", "discard", "bad"}:
            raise ValueError(f"{video_key} is marked {record.get('status')}")
        keyframes = [int(value) for value in record.get("keyframes", [])]
        if len(keyframes) < 2:
            raise ValueError(f"{video_key} requires at least two global keyframes, got {keyframes}")
        self._foundation_keyframes = (keyframes[0], keyframes[1])

    def _load_keyframe_action_positions(self):
        action_frame = self._foundation_keyframes[1]
        if self._foundation_action_target_source == "object_keyframe":
            centers = {}
            for side, object_key in self._foundation_object_keys.items():
                poses = self._foundation_data[f"{object_key}__pose_world_wxyz"]
                if not 0 <= action_frame < len(poses):
                    raise IndexError(
                        f"O.1.2 action frame={action_frame} is outside {object_key} pose count {len(poses)}"
                    )
                visible = self._foundation_data.get(f"{object_key}__visible")
                if visible is not None and not bool(visible[action_frame]):
                    raise ValueError(f"{object_key} is not visible at action frame {action_frame}")
                mesh_path = self._scalar_path(self._foundation_data[f"{object_key}__mesh_file"])
                mesh = trimesh.load(mesh_path, force="mesh", process=False)
                if mesh.is_empty:
                    raise ValueError(f"Foundation {side} mesh is empty: {mesh_path}")
                bounds = np.asarray(mesh.bounds, dtype=np.float64)
                pose_values = np.asarray(poses[action_frame], dtype=np.float64)
                position = pose_values[:3].copy()
                foundation_quat = pose_values[3:7].copy()
                foundation_quat /= np.linalg.norm(foundation_quat)
                if self._foundation_use_orientation:
                    quat = foundation_quat
                else:
                    quat = np.asarray([0.66, 0.66, -0.25, -0.25], dtype=np.float64)
                    quat /= np.linalg.norm(quat)
                rotation = t3d.quaternions.quat2mat(quat)
                center = position + rotation @ bounds.mean(axis=0)
                centers[side] = center
            self._foundation_action_positions = centers
            self._foundation_action_object_centers = centers
            self._foundation_action_source = (
                f"foundation_npz_object_keyframe:{Path(self._foundation_input_dir).resolve()}"
            )
            print(
                f"[piper-ik-foundation] O.1.2 object action frame={action_frame} "
                f"left_center={np.round(centers['left'], 4).tolist()} "
                f"right_center={np.round(centers['right'], 4).tolist()}"
            )
            return

        relative_path = self._foundation_hand_targets_pattern.format(
            episode=self._foundation_episode_id
        )
        target_path = self._foundation_hand_targets_root / relative_path
        if not target_path.is_file():
            raise FileNotFoundError(f"O.1.2 hand target NPZ not found: {target_path}")
        data = np.load(target_path, allow_pickle=True)
        required = ("selected_indices", "left_world_targets", "right_world_targets")
        missing = [key for key in required if key not in data]
        if missing:
            raise KeyError(f"O.1.2 hand target NPZ is missing keys: {missing}")
        selected_indices = np.asarray(data["selected_indices"], dtype=np.int64).reshape(-1)
        matches = np.flatnonzero(selected_indices == action_frame)
        if matches.size != 1:
            raise ValueError(
                f"O.1.2 action frame {action_frame} must map to exactly one target row, "
                f"got {matches.size}"
            )
        row = int(matches[0])
        positions = {
            "left": np.asarray(data["left_world_targets"][row, :3], dtype=np.float64),
            "right": np.asarray(data["right_world_targets"][row, :3], dtype=np.float64),
        }
        if not all(np.isfinite(value).all() for value in positions.values()):
            raise ValueError(f"O.1.2 action frame {action_frame} contains invalid EE positions")
        self._foundation_action_positions = positions
        self._foundation_action_source = str(target_path.resolve())
        print(
            f"[piper-ik-foundation] O.1.2 action frame={action_frame} "
            f"left={np.round(positions['left'], 4).tolist()} "
            f"right={np.round(positions['right'], 4).tolist()}"
        )

    @staticmethod
    def _scalar_path(value):
        if isinstance(value, np.ndarray):
            value = value.item()
        return Path(str(value)).expanduser().resolve()

    @staticmethod
    def _mesh_actor_data(bounds):
        bounds = np.asarray(bounds, dtype=np.float64)
        center = bounds.mean(axis=0)
        extents = bounds[1] - bounds[0]

        def point_matrix(point):
            matrix = np.eye(4, dtype=np.float64)
            matrix[:3, 3] = point
            return matrix.tolist()

        center_matrix = point_matrix(center)
        return {
            "center": center.tolist(),
            "extents": extents.tolist(),
            "scale": [1.0, 1.0, 1.0],
            "target_pose": [deepcopy(center_matrix)],
            "contact_points_pose": [deepcopy(center_matrix)],
            "functional_matrix": [deepcopy(center_matrix)],
            "orientation_point": deepcopy(center_matrix),
            "stable": True,
        }

    def _create_foundation_mesh_actor(self, side, mesh_path, pose_values):
        if not mesh_path.is_file():
            raise FileNotFoundError(f"Foundation {side} mesh not found: {mesh_path}")
        mesh = trimesh.load(mesh_path, force="mesh", process=False)
        if mesh.is_empty:
            raise ValueError(f"Foundation {side} mesh is empty: {mesh_path}")
        bounds = np.asarray(mesh.bounds, dtype=np.float64)
        if bounds.shape != (2, 3) or not np.isfinite(bounds).all():
            raise ValueError(f"Foundation {side} mesh has invalid bounds: {bounds}")

        position = np.asarray(pose_values[:3], dtype=np.float64).copy()
        foundation_quat = np.asarray(pose_values[3:7], dtype=np.float64)
        quaternion_norm = np.linalg.norm(foundation_quat)
        if not np.isfinite(position).all() or not np.isfinite(quaternion_norm) or quaternion_norm <= 0:
            raise ValueError(f"Foundation {side} pose is invalid: {pose_values}")
        foundation_quat /= quaternion_norm
        if self._foundation_use_orientation:
            quat = foundation_quat
            orientation_source = "FoundationPose"
        else:
            quat = np.asarray([0.66, 0.66, -0.25, -0.25], dtype=np.float64)
            quat /= np.linalg.norm(quat)
            orientation_source = "upright"

        rotation = t3d.quaternions.quat2mat(quat)
        corners = np.array([
            [x, y, z]
            for x in (bounds[0, 0], bounds[1, 0])
            for y in (bounds[0, 1], bounds[1, 1])
            for z in (bounds[0, 2], bounds[1, 2])
        ])
        table_top = 0.74 + self.table_z_bias + self._foundation_table_clearance
        world_min_z = np.min((rotation @ corners.T).T[:, 2] + position[2])
        floor_shift = max(0.0, table_top - world_min_z)
        position[2] += floor_shift

        builder = self.scene.create_actor_builder()
        builder.set_physx_body_type("dynamic")
        if self._foundation_collision_mode == "exact_convex":
            builder.add_multiple_convex_collisions_from_file(
                filename=str(mesh_path), scale=[1, 1, 1]
            )
        elif self._foundation_collision_mode in {"cylinder_proxy", "support_proxy"}:
            extents = bounds[1] - bounds[0]
            radius = (
                0.5 * max(extents[0], extents[2])
                + self._foundation_collision_radius_padding
            )
            if self._foundation_collision_mode == "support_proxy":
                half_length = min(0.006, 0.05 * extents[1])
                center_local = bounds.mean(axis=0)
                center_local[1] = bounds[0, 1] + half_length
            else:
                half_length = 0.5 * extents[1]
                center_local = bounds.mean(axis=0)
            # SAPIEN cylinders use local X as their long axis; Foundation bottle
            # meshes use local Y as vertical, so rotate the collision X axis to Y.
            cylinder_pose = sapien.Pose(
                center_local,
                t3d.euler.euler2quat(0.0, 0.0, np.pi / 2),
            )
            high_friction_material = self.scene.create_physical_material(
                static_friction=self._foundation_mesh_friction,
                dynamic_friction=self._foundation_mesh_friction,
                restitution=0.0,
            )
            builder.add_cylinder_collision(
                pose=cylinder_pose,
                radius=radius,
                half_length=half_length,
                material=high_friction_material,
            )
        else:
            raise ValueError(
                f"unsupported foundation_collision_mode={self._foundation_collision_mode!r}"
            )
        builder.add_visual_from_file(filename=str(mesh_path), scale=[1, 1, 1])
        entity = builder.build(name=self._foundation_actor_ids.get(side, f"foundation_{side}_object"))
        entity.set_pose(sapien.Pose(position, quat))
        actor = Actor(entity, self._mesh_actor_data(bounds), mass=self._foundation_mesh_mass)

        center_local = bounds.mean(axis=0)
        center_world = position + rotation @ center_local
        self._foundation_geometry[side] = {
            "mesh_path": str(mesh_path),
            "bounds": bounds.tolist(),
            "center_local": center_local.tolist(),
            "source_pose": np.asarray(pose_values, dtype=np.float64).tolist(),
            "spawn_pose": position.tolist() + quat.tolist(),
        }
        print(
            f"[piper-ik-foundation] {side}: mesh={mesh_path} orientation={orientation_source} "
            f"collision={self._foundation_collision_mode} "
            f"extents={tuple(round(v, 4) for v in bounds[1] - bounds[0])} "
            f"floor_shift={floor_shift:.4f} center_world="
            f"({center_world[0]:.3f},{center_world[1]:.3f},{center_world[2]:.3f})"
        )
        return actor

    def load_actors(self):
        frame = self._foundation_frame
        data = self._foundation_data
        left_key = self._foundation_object_keys["left"]
        right_key = self._foundation_object_keys["right"]
        left_pose = np.asarray(data[f"{left_key}__pose_world_wxyz"][frame], dtype=np.float64)
        right_pose = np.asarray(data[f"{right_key}__pose_world_wxyz"][frame], dtype=np.float64)
        left_mesh = self._scalar_path(data[f"{left_key}__mesh_file"])
        right_mesh = self._scalar_path(data[f"{right_key}__mesh_file"])

        self.bottle1_id = self._foundation_actor_ids["left"]
        self.bottle2_id = self._foundation_actor_ids["right"]
        self.bottle1 = self._create_foundation_mesh_actor("left", left_mesh, left_pose)
        self.bottle2 = self._create_foundation_mesh_actor("right", right_mesh, right_pose)

        self.delay(4)
        self.add_prohibit_area(self.bottle1, padding=0.08)
        self.add_prohibit_area(self.bottle2, padding=0.08)
        self.prohibited_area.append([-0.20, -0.24, 0.28, -0.04])
        self.left_target_pose = [-0.28, -0.15, 1.0, 0, 1, 0, 0]
        self.right_target_pose = [0.52, -0.15, 1.0, 0, 1, 0, 0]

        left_center = self.bottle1.get_functional_point(0)
        right_center = self.bottle2.get_functional_point(0)
        self._foundation_settled_poses = {
            "left": deepcopy(self.bottle1.get_pose()),
            "right": deepcopy(self.bottle2.get_pose()),
        }
        self._foundation_settled_centers = {
            "left": np.asarray(left_center[:3], dtype=np.float64),
            "right": np.asarray(right_center[:3], dtype=np.float64),
        }
        print(
            "[piper-ik-foundation] actors settled: "
            f"left_center=({left_center[0]:.3f},{left_center[1]:.3f},{left_center[2]:.3f}) "
            f"right_center=({right_center[0]:.3f},{right_center[1]:.3f},{right_center[2]:.3f})"
        )

    def _foundation_grasp_poses(self, actor, arm_tag, pre_grasp_dis):
        pre_grasp_dis = self._foundation_pregrasp_distance
        center = np.asarray(actor.get_functional_point(0)[:3], dtype=np.float64)
        side_sign = -1.0 if arm_tag == "left" else 1.0
        grasp_offset = np.asarray([
            side_sign * self._foundation_grasp_lateral_offset,
            -self._foundation_grasp_standoff,
            0.0,
        ])
        grasp_pos = center + grasp_offset
        approach = -grasp_offset / np.linalg.norm(grasp_offset)
        pregrasp_pos = grasp_pos - approach * pre_grasp_dis
        ee = (self.robot.left_ee if arm_tag == "left" else self.robot.right_ee).child_link.get_pose()
        quat = np.asarray(ee.q, dtype=np.float64).tolist()
        return center, pregrasp_pos, grasp_pos, quat

    def _cartesian_grasp_actor(self, actor, arm_tag, pre_grasp_dis=0.08, gripper_pos=0.0):
        if not self.plan_success:
            return None, []
        center, pregrasp_pos, grasp_pos, quat = self._foundation_grasp_poses(
            actor, str(arm_tag), pre_grasp_dis
        )
        print(
            f"[piper-ik-foundation] grasp {arm_tag}: center="
            f"({center[0]:.3f},{center[1]:.3f},{center[2]:.3f}) "
            f"pregrasp=({pregrasp_pos[0]:.3f},{pregrasp_pos[1]:.3f},{pregrasp_pos[2]:.3f}) "
            f"grasp=({grasp_pos[0]:.3f},{grasp_pos[1]:.3f},{grasp_pos[2]:.3f})"
        )
        return arm_tag, [
            Action(arm_tag, "move", target_pose=pregrasp_pos.tolist() + quat),
            Action(
                arm_tag,
                "move",
                target_pose=grasp_pos.tolist() + quat,
                constraint_pose=[1, 1, 1, 0, 0, 0],
            ),
            Action(arm_tag, "close", target_gripper_pos=gripper_pos),
        ]

    def get_debug_axis_poses(self):
        axes = []
        for side, actor in (("left", self.bottle1), ("right", self.bottle2)):
            ee = (self.robot.left_ee if side == "left" else self.robot.right_ee).child_link.get_pose()
            axes.append((f"ee_current_{side}", sapien.Pose(ee.p, ee.q), 0.09, (0.0, 1.0, 1.0)))
            _, pregrasp, grasp, quat = self._foundation_grasp_poses(actor, side, 0.08)
            axes.append((f"plan_pregrasp_{side}", sapien.Pose(pregrasp, quat), 0.07, (0.3, 0.5, 1.0)))
            axes.append((f"plan_grasp_{side}", sapien.Pose(grasp, quat), 0.07, (0.3, 1.0, 0.5)))
        return axes

    def _build_action_sequence(self):
        sequence = super()._build_action_sequence()
        if self._foundation_mode != "o1.2":
            return sequence
        pregrasp, grasp, close = sequence[:3]
        left_grasp_pose = list(grasp[1].target_pose)
        right_grasp_pose = list(grasp[2].target_pose)
        if self._foundation_action_target_source == "object_keyframe":
            left_action_position = (
                self._foundation_action_positions["left"]
                + np.asarray(left_grasp_pose[:3], dtype=np.float64)
                - self._foundation_settled_centers["left"]
            )
            right_action_position = (
                self._foundation_action_positions["right"]
                + np.asarray(right_grasp_pose[:3], dtype=np.float64)
                - self._foundation_settled_centers["right"]
            )
        else:
            left_action_position = self._foundation_action_positions["left"]
            right_action_position = self._foundation_action_positions["right"]
        left_action_pose = left_action_position.tolist() + left_grasp_pose[3:]
        right_action_pose = right_action_position.tolist() + right_grasp_pose[3:]
        action = (
            "action",
            Action(grasp[1].arm_tag, "move", target_pose=left_action_pose),
            Action(grasp[2].arm_tag, "move", target_pose=right_action_pose),
        )
        sequence = [pregrasp, grasp, close, action]
        if self._foundation_pregrasp_clearance > 0:
            left_clearance_pose = list(pregrasp[1].target_pose)
            right_clearance_pose = list(pregrasp[2].target_pose)
            left_clearance_pose[2] += self._foundation_pregrasp_clearance
            right_clearance_pose[2] += self._foundation_pregrasp_clearance
            approach_clearance = (
                "approach_clearance",
                Action(pregrasp[1].arm_tag, "move", target_pose=left_clearance_pose),
                Action(pregrasp[2].arm_tag, "move", target_pose=right_clearance_pose),
            )
            sequence.insert(0, approach_clearance)
        if self._foundation_open_after_action:
            sequence.append(
                (
                    "open_gripper",
                    Action(grasp[1].arm_tag, "open", target_gripper_pos=1.0),
                    Action(grasp[2].arm_tag, "open", target_gripper_pos=1.0),
                )
            )
        self._trajectory_targets = {
            "left": {
                name: list(left.target_pose)
                for name, left, _ in sequence
                if left.action == "move"
            },
            "right": {
                name: list(right.target_pose)
                for name, _, right in sequence
                if right.action == "move"
            },
        }
        print(
            "[piper-ik-foundation] O.1.2 replaces lift/place with action: "
            f"left={np.round(left_action_pose[:3], 4).tolist()} "
            f"right={np.round(right_action_pose[:3], 4).tolist()} "
            f"source={self._foundation_action_target_source} "
            f"pregrasp_clearance={self._foundation_pregrasp_clearance:.3f}m "
            f"open_after_action={self._foundation_open_after_action}"
        )
        return sequence

    def _sequence_from_loaded_trajectory(self):
        if self._foundation_mode != "o1.2":
            return super()._sequence_from_loaded_trajectory()
        metadata = self._loaded_trajectory_metadata
        if metadata is None:
            raise ValueError("trajectory metadata was not loaded before replay")
        targets = metadata["targets"]
        from .utils import ArmTag

        left_arm = ArmTag("left")
        right_arm = ArmTag("right")
        sequence = []
        if "approach_clearance" in targets["left"]:
            sequence.append(
                (
                    "approach_clearance",
                    Action(left_arm, "move", target_pose=targets["left"]["approach_clearance"]),
                    Action(right_arm, "move", target_pose=targets["right"]["approach_clearance"]),
                )
            )
        sequence.extend([
            (
                "pregrasp",
                Action(left_arm, "move", target_pose=targets["left"]["pregrasp"]),
                Action(right_arm, "move", target_pose=targets["right"]["pregrasp"]),
            ),
            (
                "grasp",
                Action(
                    left_arm,
                    "move",
                    target_pose=targets["left"]["grasp"],
                    constraint_pose=[1, 1, 1, 0, 0, 0],
                ),
                Action(
                    right_arm,
                    "move",
                    target_pose=targets["right"]["grasp"],
                    constraint_pose=[1, 1, 1, 0, 0, 0],
                ),
            ),
            (
                "close_gripper",
                Action(left_arm, "close", target_gripper_pos=0.0),
                Action(right_arm, "close", target_gripper_pos=0.0),
            ),
            (
                "action",
                Action(left_arm, "move", target_pose=targets["left"]["action"]),
                Action(right_arm, "move", target_pose=targets["right"]["action"]),
            ),
        ])
        if self._foundation_open_after_action:
            sequence.append(
                (
                    "open_gripper",
                    Action(left_arm, "open", target_gripper_pos=1.0),
                    Action(right_arm, "open", target_gripper_pos=1.0),
                )
            )
        self._trajectory_targets = deepcopy(targets)
        return sequence

    def _update_place_targets_from_closed_grasp(self, sequence):
        if self._foundation_mode == "o1.2":
            return
        super()._update_place_targets_from_closed_grasp(sequence)

    @staticmethod
    def _quat_distance_deg(first, second):
        first = np.asarray(first, dtype=np.float64)
        second = np.asarray(second, dtype=np.float64)
        first /= np.linalg.norm(first)
        second /= np.linalg.norm(second)
        dot = np.clip(abs(float(np.dot(first, second))), 0.0, 1.0)
        return float(np.degrees(2.0 * np.arccos(dot)))

    def _finger_contact_indices(self, actor, finger_links):
        indices = set()
        for contact in self.scene.get_contacts():
            if hasattr(contact, "bodies") and contact.bodies:
                entities = [getattr(body, "entity", None) for body in contact.bodies]
            else:
                entities = [getattr(contact, "actor0", None), getattr(contact, "actor1", None)]
            if actor.actor not in entities:
                continue
            for index, finger_link in enumerate(finger_links):
                if finger_link in entities:
                    indices.add(index)
        return sorted(indices)

    def _validate_foundation_grasp(self, side, actor, ee_joint, gripper_joints):
        current_pose = actor.get_pose()
        settled_pose = self._foundation_settled_poses[side]
        current_center = np.asarray(actor.get_functional_point(0)[:3], dtype=np.float64)
        settled_center = self._foundation_settled_centers[side]
        displacement = float(np.linalg.norm(current_center - settled_center))
        rotation_deg = self._quat_distance_deg(current_pose.q, settled_pose.q)
        ee_position = np.asarray(ee_joint.child_link.get_pose().p, dtype=np.float64)
        ee_distance = float(np.linalg.norm(current_center - ee_position))

        finger_links = [
            item[0].child_link
            for item in gripper_joints
            if getattr(item[0], "child_link", None) is not None
        ]
        if len(finger_links) != 2:
            raise RuntimeError(f"{side} grasp requires exactly two finger links")
        finger_positions = np.asarray(
            [link.get_pose().p for link in finger_links], dtype=np.float64
        )
        segment = finger_positions[1] - finger_positions[0]
        segment_norm_sq = float(np.dot(segment, segment))
        if segment_norm_sq <= 1e-8:
            raise RuntimeError(f"{side} finger segment is degenerate")
        projection = float(
            np.dot(current_center - finger_positions[0], segment) / segment_norm_sq
        )
        closest = finger_positions[0] + np.clip(projection, 0.0, 1.0) * segment
        radial_distance = float(np.linalg.norm(current_center - closest))
        contact_indices = self._finger_contact_indices(actor, finger_links)
        within_segment = (
            -self._foundation_capture_segment_margin
            <= projection
            <= 1.0 + self._foundation_capture_segment_margin
        )
        valid = (
            displacement <= self._foundation_grasp_max_displacement
            and rotation_deg <= self._foundation_grasp_max_rotation_deg
            and ee_distance <= self._foundation_grasp_assist_max_distance
            and within_segment
            and radial_distance <= self._foundation_capture_radial_tolerance
            and (
                not self._foundation_grasp_require_contact
                or len(contact_indices) == len(finger_links)
            )
        )
        diagnostics = {
            "displacement_m": displacement,
            "rotation_deg": rotation_deg,
            "ee_distance_m": ee_distance,
            "finger_projection": projection,
            "finger_radial_distance_m": radial_distance,
            "finger_contacts": contact_indices,
            "valid": valid,
        }
        self._foundation_grasp_diagnostics[side] = diagnostics
        print(
            f"[piper-ik-foundation] grasp-state {side}: valid={valid} "
            f"displacement={displacement:.4f}m rotation={rotation_deg:.1f}deg "
            f"ee_distance={ee_distance:.3f}m projection={projection:.3f} "
            f"radial={radial_distance:.3f}m contacts={contact_indices}"
        )
        if not valid:
            raise RuntimeError(f"{side} grasp-state validation failed: {diagnostics}")
        return current_pose, ee_joint.child_link.get_pose()

    def _attach_foundation_grasp_drives(self):
        if self._foundation_grasp_drives:
            return
        validated = []
        for side, actor, ee_joint, gripper_joints in (
            ("left", self.bottle1, self.robot.left_ee, self.robot.left_gripper),
            ("right", self.bottle2, self.robot.right_ee, self.robot.right_gripper),
        ):
            actor_pose, ee_pose = self._validate_foundation_grasp(
                side, actor, ee_joint, gripper_joints
            )
            validated.append((side, actor, ee_joint.child_link, actor_pose, ee_pose))
        self._foundation_grasp_valid = True
        if not self._foundation_grasp_assist:
            print("[piper-ik-foundation] grasp-assist disabled; validation only")
            return
        for side, actor, ee_link, actor_pose, ee_pose in validated:
            object_anchor = actor_pose.inv() * ee_pose
            drive = self.scene.create_drive(
                ee_link,
                sapien.Pose(),
                actor.actor,
                object_anchor,
            )
            for axis in ("x", "y", "z"):
                getattr(drive, f"set_limit_{axis}")(0.0, 0.0, 1e5, 1e4)
                getattr(drive, f"set_drive_property_{axis}")(1e5, 1e4)
            drive.set_limit_twist(0.0, 0.0, 1e5, 1e4)
            drive.set_limit_cone(0.0, 0.0)
            drive.set_drive_property_slerp(1e5, 1e4)
            self._foundation_grasp_drives.append(drive)
            print(
                f"[piper-ik-foundation] grasp-assist attached {side} at current object pose"
            )

    def _release_foundation_grasp_drives(self):
        for drive in self._foundation_grasp_drives:
            drive.entity.remove_from_scene()
        if self._foundation_grasp_drives:
            print("[piper-ik-foundation] grasp-assist released")
        self._foundation_grasp_drives = []

    def close_env(self, clear_cache=False):
        self._release_foundation_grasp_drives()
        super().close_env(clear_cache=clear_cache)

    def _execute_gripper_pair(self, action_name, left_action, right_action):
        if action_name == "open_gripper":
            self._release_foundation_grasp_drives()
        super()._execute_gripper_pair(action_name, left_action, right_action)
        if action_name == "close_gripper":
            self._attach_foundation_grasp_drives()

    def play_once(self):
        info = super().play_once()
        if self.plan_success:
            info["info"] = {
                "{A}": self._foundation_left_description,
                "{B}": self._foundation_right_description,
            }
        return info

    def check_success(self):
        if self._foundation_mode != "o1.2":
            return super().check_success()
        targets = self._trajectory_targets
        left_ee = np.asarray(self.robot.left_ee.child_link.get_pose().p, dtype=np.float64)
        right_ee = np.asarray(self.robot.right_ee.child_link.get_pose().p, dtype=np.float64)
        left_target = np.asarray(targets["left"]["action"][:3], dtype=np.float64)
        right_target = np.asarray(targets["right"]["action"][:3], dtype=np.float64)
        left_error = float(np.linalg.norm(left_ee - left_target))
        right_error = float(np.linalg.norm(right_ee - right_target))
        left_center = np.asarray(self.bottle1.get_functional_point(0)[:3], dtype=np.float64)
        right_center = np.asarray(self.bottle2.get_functional_point(0)[:3], dtype=np.float64)
        left_motion = float(
            np.linalg.norm(left_center - self._foundation_settled_centers["left"])
        )
        right_motion = float(
            np.linalg.norm(right_center - self._foundation_settled_centers["right"])
        )
        success = bool(
            self.plan_success
            and self._foundation_grasp_valid
            and left_error < 0.12
            and right_error < 0.12
            and left_motion > 0.04
            and right_motion > 0.04
        )
        self._last_success = success
        print(
            f"[piper-ik-foundation][O.1.2-check] success={success} "
            f"ee_error=({left_error:.3f},{right_error:.3f})m "
            f"object_motion=({left_motion:.3f},{right_motion:.3f})m "
            f"grasp_valid={self._foundation_grasp_valid}"
        )
        if self._foundation_action_object_centers is not None:
            left_object_target = self._foundation_action_object_centers["left"]
            right_object_target = self._foundation_action_object_centers["right"]
            left_object_error = float(np.linalg.norm(left_center - left_object_target))
            right_object_error = float(np.linalg.norm(right_center - right_object_target))
            print(
                "[piper-ik-foundation][O.1.2-object-check] "
                f"object_error=({left_object_error:.3f},{right_object_error:.3f})m "
                f"target=({np.round(left_object_target, 4).tolist()}, "
                f"{np.round(right_object_target, 4).tolist()})"
            )
        if self.save_all_episodes:
            return True
        return success

    def save_traj_data(self, idx):
        super().save_traj_data(idx)
        path = Path(self.save_dir) / "_traj_data" / f"episode{idx}.pkl"
        with path.open("rb") as file:
            trajectory = pickle.load(file)
        trajectory["foundation_source"] = self._foundation_context()
        with path.open("wb") as file:
            pickle.dump(trajectory, file)

    def load_tran_data(self, idx):
        trajectory = super().load_tran_data(idx)
        expected = self._foundation_context()
        if trajectory.get("foundation_source") != expected:
            raise ValueError(
                "Foundation trajectory source does not match the current input/frame/meshes; "
                "run Phase 1 in a fresh task_config output directory"
            )
        return trajectory

    def _foundation_context(self):
        return {
            "mode": self._foundation_mode,
            "input_dir": str(Path(self._foundation_input_dir).resolve()),
            "frame": self._foundation_frame,
            "episode_id": self._foundation_episode_id,
            "annotated_keyframes": self._foundation_keyframes,
            "annotation_json": (
                str(self._foundation_annotation_json)
                if self._foundation_mode in {"o1.1", "o1.2"}
                else None
            ),
            "action_source": getattr(self, "_foundation_action_source", None),
            "action_target_source": self._foundation_action_target_source,
            "action_positions": (
                {
                    side: value.tolist()
                    for side, value in self._foundation_action_positions.items()
                }
                if self._foundation_action_positions is not None
                else None
            ),
            "use_orientation": self._foundation_use_orientation,
            "collision_mode": self._foundation_collision_mode,
            "collision_radius_padding": self._foundation_collision_radius_padding,
            "pregrasp_distance": self._foundation_pregrasp_distance,
            "pregrasp_clearance": self._foundation_pregrasp_clearance,
            "grasp_assist": self._foundation_grasp_assist,
            "open_after_action": self._foundation_open_after_action,
            "object_keys": deepcopy(self._foundation_object_keys),
            "hand_targets_pattern": self._foundation_hand_targets_pattern,
            "grasp_state_limits": {
                "max_displacement": self._foundation_grasp_max_displacement,
                "max_rotation_deg": self._foundation_grasp_max_rotation_deg,
                "capture_radial_tolerance": self._foundation_capture_radial_tolerance,
                "capture_segment_margin": self._foundation_capture_segment_margin,
                "require_contact": self._foundation_grasp_require_contact,
            },
            "geometry": deepcopy(self._foundation_geometry),
        }
