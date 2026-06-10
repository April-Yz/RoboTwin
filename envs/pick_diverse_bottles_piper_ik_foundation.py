"""O.1: FoundationPose positions and source OBJ meshes with Piper Cartesian IK."""

import pickle
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

    def setup_demo(self, **kwargs):
        self._foundation_input_dir = kwargs.get("foundation_input_dir")
        self._foundation_frame = int(kwargs.get("foundation_frame", 0))
        self._foundation_use_orientation = bool(kwargs.get("foundation_use_orientation", True))
        self._foundation_table_clearance = float(kwargs.get("foundation_table_clearance", 0.002))
        self._foundation_mesh_mass = float(kwargs.get("foundation_mesh_mass", 0.05))
        self._foundation_mesh_friction = float(kwargs.get("foundation_mesh_friction", 1.5))
        self._foundation_grasp_standoff = float(kwargs.get("foundation_grasp_standoff", 0.085))
        self._foundation_grasp_lateral_offset = float(
            kwargs.get("foundation_grasp_lateral_offset", 0.0)
        )
        self._foundation_collision_radius_padding = float(
            kwargs.get("foundation_collision_radius_padding", 0.0)
        )
        self._foundation_collision_mode = kwargs.get("foundation_collision_mode", "cylinder_proxy")
        self._foundation_grasp_assist = bool(kwargs.get("foundation_grasp_assist", True))
        self._foundation_grasp_assist_max_distance = float(
            kwargs.get("foundation_grasp_assist_max_distance", 0.14)
        )
        if np.hypot(
            self._foundation_grasp_standoff,
            self._foundation_grasp_lateral_offset,
        ) <= 0:
            raise ValueError("Foundation grasp offset must be nonzero")
        self._foundation_left_description = kwargs.get(
            "foundation_left_description", "left cola bottle"
        )
        self._foundation_right_description = kwargs.get(
            "foundation_right_description", "right bottle"
        )
        self._foundation_grasp_drives = []
        self._foundation_data = None
        self._foundation_geometry = {}
        self._foundation_settled_poses = {}

        if not self._foundation_input_dir:
            raise ValueError(
                "O.1 requires foundation_input_dir in the task config; it must contain "
                "multi_object_world_poses.npz"
            )
        npz_path = Path(self._foundation_input_dir) / "multi_object_world_poses.npz"
        if not npz_path.is_file():
            raise FileNotFoundError(f"Foundation NPZ not found: {npz_path}")

        self._foundation_data = dict(np.load(npz_path, allow_pickle=True))
        required = (
            "left_bottle__pose_world_wxyz",
            "right_bottle__pose_world_wxyz",
            "left_bottle__mesh_file",
            "right_bottle__mesh_file",
        )
        missing = [key for key in required if key not in self._foundation_data]
        if missing:
            raise KeyError(f"Foundation NPZ is missing keys: {missing}")
        for side in ("left", "right"):
            visible = self._foundation_data.get(f"{side}_bottle__visible")
            poses = self._foundation_data[f"{side}_bottle__pose_world_wxyz"]
            if not 0 <= self._foundation_frame < len(poses):
                raise IndexError(
                    f"foundation_frame={self._foundation_frame} is outside {side} pose count {len(poses)}"
                )
            if visible is not None and not bool(visible[self._foundation_frame]):
                raise ValueError(f"{side}_bottle is not visible at frame {self._foundation_frame}")

        print(
            f"[piper-ik-foundation] source={npz_path} frame={self._foundation_frame} "
            f"use_orientation={self._foundation_use_orientation}"
        )
        super().setup_demo(**kwargs)

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
        elif self._foundation_collision_mode == "cylinder_proxy":
            extents = bounds[1] - bounds[0]
            radius = (
                0.5 * max(extents[0], extents[2])
                + self._foundation_collision_radius_padding
            )
            half_length = 0.5 * extents[1]
            # SAPIEN cylinders use local X as their long axis; Foundation bottle
            # meshes use local Y as vertical, so rotate the collision X axis to Y.
            cylinder_pose = sapien.Pose(
                bounds.mean(axis=0),
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
        entity = builder.build(name=f"foundation_{side}_bottle")
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
        left_pose = np.asarray(data["left_bottle__pose_world_wxyz"][frame], dtype=np.float64)
        right_pose = np.asarray(data["right_bottle__pose_world_wxyz"][frame], dtype=np.float64)
        left_mesh = self._scalar_path(data["left_bottle__mesh_file"])
        right_mesh = self._scalar_path(data["right_bottle__mesh_file"])

        self.bottle1_id = "foundation_left_bottle"
        self.bottle2_id = "foundation_right_bottle"
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
        print(
            "[piper-ik-foundation] actors settled: "
            f"left_center=({left_center[0]:.3f},{left_center[1]:.3f},{left_center[2]:.3f}) "
            f"right_center=({right_center[0]:.3f},{right_center[1]:.3f},{right_center[2]:.3f})"
        )

    def _foundation_grasp_poses(self, actor, arm_tag, pre_grasp_dis):
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

    def _attach_foundation_grasp_drives(self):
        if not self._foundation_grasp_assist or self._foundation_grasp_drives:
            return
        for side, actor, ee_joint in (
            ("left", self.bottle1, self.robot.left_ee),
            ("right", self.bottle2, self.robot.right_ee),
        ):
            settled_pose = self._foundation_settled_poses[side]
            actor.actor.set_pose(settled_pose)
            for component in actor.actor.get_components():
                if hasattr(component, "set_linear_velocity"):
                    component.set_linear_velocity([0.0, 0.0, 0.0])
                    component.set_angular_velocity([0.0, 0.0, 0.0])
            ee_link = ee_joint.child_link
            ee_pose = ee_link.get_pose()
            actor_pose = actor.get_pose()
            center = np.asarray(actor.get_functional_point(0)[:3], dtype=np.float64)
            distance = float(np.linalg.norm(center - np.asarray(ee_pose.p)))
            if distance > self._foundation_grasp_assist_max_distance:
                raise RuntimeError(
                    f"{side} grasp-assist refused: object/EE distance {distance:.3f}m exceeds "
                    f"{self._foundation_grasp_assist_max_distance:.3f}m"
                )
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
                f"[piper-ik-foundation] grasp-assist attached {side}: "
                f"object/EE distance={distance:.3f}m"
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
        if action_name == "close_gripper":
            self._attach_foundation_grasp_drives()
        elif action_name == "open_gripper":
            self._release_foundation_grasp_drives()
        super()._execute_gripper_pair(action_name, left_action, right_action)

    def play_once(self):
        info = super().play_once()
        if self.plan_success:
            info["info"] = {
                "{A}": self._foundation_left_description,
                "{B}": self._foundation_right_description,
            }
        return info

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
            "input_dir": str(Path(self._foundation_input_dir).resolve()),
            "frame": self._foundation_frame,
            "use_orientation": self._foundation_use_orientation,
            "collision_mode": self._foundation_collision_mode,
            "collision_radius_padding": self._foundation_collision_radius_padding,
            "grasp_assist": self._foundation_grasp_assist,
            "geometry": deepcopy(self._foundation_geometry),
        }
