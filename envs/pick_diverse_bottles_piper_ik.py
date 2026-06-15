"""
O.0 对照实验 — Piper IK 版本，遵循 Cartesian 抓取逻辑。

支持 4 种 IK 策略 (v1-v4)，通过 config 中的 ik_version 选择。

与原始 pick_diverse_bottles 的区别：
- 底层 planner 替换为 PiperIKPlanner
- play_once 使用简化的 Cartesian pregrasp/grasp/lift/place 流程
- 不经过 choose_best_pose 的旋转候选搜索（Piper 上全部失败的问题）

用法:
  python collect_data.py pick_diverse_bottles_piper_ik demo_piper_ik_v1 0
  python view_pick_diverse_bottles_piper_scene.py --task_name pick_diverse_bottles_piper_ik --task_config demo_piper_ik_v1 --seed 0
"""

import os
import pickle
from copy import deepcopy

import numpy as np
import sapien.core as sapien
import transforms3d as t3d

from .pick_diverse_bottles import pick_diverse_bottles
from ._GLOBAL_CONFIGS import ROOT_PATH
from .utils import ArmTag, Action


class pick_diverse_bottles_piper_ik(pick_diverse_bottles):
    """Piper IK task — Cartesian 抓取 + 多种 IK 后端。"""

    TRAJECTORY_SCHEMA = "piper_ik_cartesian"
    TRAJECTORY_VERSION = 2
    MOVE_ACTION_NAMES = ("pregrasp", "grasp", "lift", "place")

    def setup_demo(self, **kwargs):
        super().setup_demo(**kwargs)

        # 确保 folder_path 存在（Phase 2 replay 时需要）
        if not hasattr(self, 'folder_path') or self.folder_path is None:
            self.folder_path = {
                "cache": f"{self.save_dir}/.cache/episode{self.ep_num}/"
            }

        ik_version = kwargs.get("ik_version", "v1")
        interp_steps = kwargs.get("ik_interp_steps", 50)
        self.ik_version = ik_version
        self.lift_height = float(kwargs.get("lift_height", 0.12))
        self.move_settle_steps = int(kwargs.get("move_settle_steps", 80))
        self.gripper_settle_steps = int(kwargs.get("gripper_settle_steps", 120))
        self.save_all_episodes = kwargs.get("save_all_episodes", False)
        self._trajectory_targets = {"left": {}, "right": {}}
        self._loaded_trajectory_metadata = None
        self._last_success = False
        print(
            f"[piper-ik-task] IK version={ik_version} interp_steps={interp_steps} "
            f"lift_height={self.lift_height:.3f} move_settle={self.move_settle_steps} "
            f"save_all={self.save_all_episodes}"
        )

        if not self.need_plan:
            print("[piper-ik-task] replay mode: planner initialization skipped")
            return

        from envs.robot.piper_ik import create_piper_ik_planner

        left_yml = os.path.join(ROOT_PATH, self.robot.left_curobo_yml_path)
        right_yml = os.path.join(ROOT_PATH, self.robot.right_curobo_yml_path)

        left_all_joints = [j.get_name() for j in self.robot.left_entity.get_active_joints()]
        right_all_joints = [j.get_name() for j in self.robot.right_entity.get_active_joints()]

        self.robot.left_planner = create_piper_ik_planner(
            version=ik_version,
            robot_origin_pose=self.robot.left_entity_origion_pose,
            active_joints_name=self.robot.left_arm_joints_name,
            all_joints=left_all_joints,
            yml_path=left_yml,
            interp_steps=interp_steps,
        )
        self.robot.right_planner = create_piper_ik_planner(
            version=ik_version,
            robot_origin_pose=self.robot.right_entity_origion_pose,
            active_joints_name=self.robot.right_arm_joints_name,
            all_joints=right_all_joints,
            yml_path=right_yml,
            interp_steps=interp_steps,
        )
        print("[piper-ik-task] planners replaced successfully")

    # ------------------------------------------------------------------
    # Piper 专用瓶子范围 & 放置目标（覆盖父类 ALOHA 范围）
    # ------------------------------------------------------------------

    def load_actors(self):
        """Piper 专用：瓶子 x 范围比 ALOHA 更宽（基座间距更大），目标位置适配 Piper workspace。"""
        import numpy as np
        from .utils import rand_create_actor

        self.id_list = [i for i in range(20)]
        self.bottle1_id = np.random.choice(self.id_list)
        self.bottle2_id = np.random.choice(self.id_list)

        # Piper 左基座 x≈-0.30，右基座 x≈0.556
        # 瓶子 x 范围：左瓶偏左、右瓶偏右，避免右臂过度跨身体够取
        self.bottle1 = rand_create_actor(
            self,
            xlim=[-0.35, -0.18],
            ylim=[0.03, 0.23],
            modelname="001_bottle",
            rotate_rand=True,
            rotate_lim=[0, 1, 0],
            qpos=[0.66, 0.66, -0.25, -0.25],
            convex=True,
            model_id=self.bottle1_id,
        )
        self.bottle2 = rand_create_actor(
            self,
            xlim=[0.38, 0.52],
            ylim=[0.03, 0.23],
            modelname="001_bottle",
            rotate_rand=True,
            rotate_lim=[0, 1, 0],
            qpos=[0.65, 0.65, 0.27, 0.27],
            convex=True,
            model_id=self.bottle2_id,
        )

        self.delay(4)
        self.add_prohibit_area(self.bottle1, padding=0.08)
        self.add_prohibit_area(self.bottle2, padding=0.08)
        self.prohibited_area.append([-0.20, -0.24, 0.28, -0.04])

        # 放置目标：适配 Piper workspace (y 比 ALOHA 的 -0.105 更前)
        self.left_target_pose = [-0.28, -0.15, 1.0, 0, 1, 0, 0]
        self.right_target_pose = [0.52, -0.15, 1.0, 0, 1, 0, 0]

        print(
            "[piper-ik][setup] bottle ranges "
            "left=x[-0.35,-0.18],y[0.03,0.23] "
            "right=x[0.38,0.52],y[0.03,0.23]"
        )
        print(
            "[piper-ik][setup] place targets "
            f"left={self.left_target_pose[:3]} "
            f"right={self.right_target_pose[:3]}"
        )

    # ------------------------------------------------------------------
    # Debug 坐标轴（与 pick_diverse_bottles_piper_motion 兼容）
    # ------------------------------------------------------------------

    @staticmethod
    def _forward_offset_pose(pose, distance=0.10):
        """将位姿沿局部 +Z（夹爪前进方向）向前偏移。"""
        import sapien.core as sapien
        R = pose.to_transformation_matrix()[:3, :3]
        forward = R[:, 2]
        return sapien.Pose(pose.p + forward * distance, pose.q)

    def get_debug_axis_poses(self):
        """返回当前 EE 尖端 + 规划目标位姿的坐标轴。

        viewer 通过 origin_color 区分：
        - ee_current (亮青色): 当前夹爪尖端
        - plan_pregrasp (浅蓝色): pregrasp 规划目标
        - plan_grasp (浅绿色): grasp 规划目标
        """
        import numpy as np
        import sapien.core as sapien

        axes = []

        # 当前 EE 尖端（左右）
        for arm_tag, side in [("left", "left"), ("right", "right")]:
            if arm_tag == "left":
                wrist = sapien.Pose(
                    self.robot.left_ee.child_link.get_pose().p,
                    self.robot.left_ee.child_link.get_pose().q,
                )
            else:
                wrist = sapien.Pose(
                    self.robot.right_ee.child_link.get_pose().p,
                    self.robot.right_ee.child_link.get_pose().q,
                )
            tip = self._forward_offset_pose(wrist, 0.10)
            axes.append((f"ee_current_{side}", tip, 0.09, (0.0, 1.0, 1.0)))  # 亮青色

        # 规划目标位姿（与 _cartesian_grasp_actor 一致的后方→前方逻辑）
        if hasattr(self, "bottle1") and hasattr(self, "bottle2"):
            for bottle, bottle_side in [(self.bottle1, "left"), (self.bottle2, "right")]:
                contact_matrix = bottle.get_contact_point(0, "matrix")
                if contact_matrix is None:
                    continue
                global_contact_pose_matrix = contact_matrix @ np.array(
                    [[0, 0, 1, 0], [-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 0, 1]]
                )
                grasp_rot = global_contact_pose_matrix[:3, :3]
                contact_pos = global_contact_pose_matrix[:3, 3]
                grasp_quat = t3d.quaternions.mat2quat(grasp_rot)
                approach_dir = -grasp_rot[:, 0]  # 取反：从机器人侧向瓶子

                pre_offset = -0.10  # Piper 短臂展
                pregrasp_pos = contact_pos + approach_dir * pre_offset
                grasp_pos = pregrasp_pos + approach_dir * 0.08

                pregrasp_pose = sapien.Pose(pregrasp_pos, grasp_quat)
                grasp_pose = sapien.Pose(grasp_pos, grasp_quat)
                axes.append((f"plan_pregrasp_{bottle_side}", pregrasp_pose, 0.07, (0.3, 0.5, 1.0)))
                axes.append((f"plan_grasp_{bottle_side}", grasp_pose, 0.07, (0.3, 1.0, 0.5)))

                print(
                    f"[piper-ik][debug-axis] {bottle_side}: "
                    f"contact=({contact_pos[0]:.3f},{contact_pos[1]:.3f},{contact_pos[2]:.3f}) "
                    f"approach=({approach_dir[0]:.2f},{approach_dir[1]:.2f},{approach_dir[2]:.2f}) "
                    f"pregrasp=({pregrasp_pos[0]:.3f},{pregrasp_pos[1]:.3f},{pregrasp_pos[2]:.3f})"
                )

        return axes

    # ------------------------------------------------------------------
    # 简化的 Cartesian 抓取（不经过 choose_best_pose 旋转搜索）
    # ------------------------------------------------------------------

    def _cartesian_grasp_actor(self, actor, arm_tag, pre_grasp_dis=0.08, gripper_pos=0.0):
        """Piper Cartesian 抓取 — 沿瓶子前进轴后方→前方（与原始逻辑一致）。

        pregrasp: contact_point - 0.20 * approach_dir（瓶子后方 20cm，沿前进轴）
        grasp:    pregrasp + 0.08 * approach_dir（向瓶子前进 8cm）
        """
        if not self.plan_success:
            return None, []

        # 获取瓶子 contact point 的变换矩阵
        contact_matrix = actor.get_contact_point(0, "matrix")
        if contact_matrix is None:
            print(f"[piper-ik] no contact matrix for {arm_tag}")
            return None, []

        # 与原始 get_grasp_pose 一致的坐标系变换
        global_contact_pose_matrix = contact_matrix @ np.array(
            [[0, 0, 1, 0], [-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 0, 1]]
        )
        grasp_rot = global_contact_pose_matrix[:3, :3]         # 抓取坐标系旋转矩阵
        contact_pos = global_contact_pose_matrix[:3, 3]        # contact point 世界坐标
        grasp_quat = t3d.quaternions.mat2quat(grasp_rot)       # 抓取坐标系四元数

        # 前进方向 = 抓取坐标系局部 +X（从瓶子指向机器人方向）
        # pregrasp 在机器人与瓶子之间，approach_dir 沿 +Y（从机器人→瓶子）时为负偏移
        approach_dir = grasp_rot[:, 0]

        # 使用当前 EE 朝向（可达），而非接触点朝向（可能不可达）
        if arm_tag == "left":
            ee = self.robot.left_ee.child_link.get_pose()
        else:
            ee = self.robot.right_ee.child_link.get_pose()
        ee_quat = [float(ee.q[0]), float(ee.q[1]), float(ee.q[2]), float(ee.q[3])]

        # pregrasp: 机器人与瓶子之间
        # approach_dir 指向机器人侧时用正偏移，反则用负偏移
        # 左臂 base y=-0.25，右臂 base y=-0.27；base 都在 -Y 方向
        if approach_dir[1] < 0:  # 指向 base 方向 (-Y)
            pre_offset = 0.10
        else:                     # 背离 base 方向 (+Y)
            pre_offset = -0.10
        pregrasp_pos = contact_pos + approach_dir * pre_offset
        pregrasp_pose = list(pregrasp_pos) + ee_quat

        # grasp: 从 pregrasp 向瓶子前进（方向取决于 approach_dir 朝向）
        if approach_dir[1] < 0:  # 指向 base → grasp 反向（朝瓶子）
            grasp_pos = pregrasp_pos - approach_dir * pre_grasp_dis
        else:                     # 背离 base → grasp 正向（朝瓶子）
            grasp_pos = pregrasp_pos + approach_dir * pre_grasp_dis
        grasp_pose = list(grasp_pos) + ee_quat

        actor_pos = actor.get_pose().p
        print(f"[piper-ik] grasp_actor {arm_tag}: "
              f"bottle=({actor_pos[0]:.3f},{actor_pos[1]:.3f},{actor_pos[2]:.3f}) "
              f"contact=({contact_pos[0]:.3f},{contact_pos[1]:.3f},{contact_pos[2]:.3f})")
        print(f"[piper-ik]   approach_dir=({approach_dir[0]:.2f},{approach_dir[1]:.2f},{approach_dir[2]:.2f})")
        print(f"[piper-ik]   pregrasp=({pregrasp_pos[0]:.3f},{pregrasp_pos[1]:.3f},{pregrasp_pos[2]:.3f}) "
              f"grasp=({grasp_pos[0]:.3f},{grasp_pos[1]:.3f},{grasp_pos[2]:.3f})")

        return arm_tag, [
            Action(arm_tag, "move", target_pose=pregrasp_pose),
            Action(arm_tag, "move", target_pose=grasp_pose, constraint_pose=[1, 1, 1, 0, 0, 0]),
            Action(arm_tag, "close", target_gripper_pos=gripper_pos),
        ]

    def _build_action_sequence(self):
        left_arm = ArmTag("left")
        right_arm = ArmTag("right")
        left_grasp = self._cartesian_grasp_actor(self.bottle1, left_arm, pre_grasp_dis=0.08)[1]
        right_grasp = self._cartesian_grasp_actor(self.bottle2, right_arm, pre_grasp_dis=0.08)[1]
        if len(left_grasp) != 3 or len(right_grasp) != 3:
            raise RuntimeError("failed to construct pregrasp/grasp actions")

        left_grasp_pose = list(left_grasp[1].target_pose)
        right_grasp_pose = list(right_grasp[1].target_pose)
        left_lift_pose = left_grasp_pose.copy()
        right_lift_pose = right_grasp_pose.copy()
        left_lift_pose[2] += self.lift_height
        right_lift_pose[2] += self.lift_height

        # The task targets describe the bottle functional points, not the gripper.
        # Preserve the bottle-to-gripper XY offset measured at grasp construction.
        left_functional_point = np.asarray(self.bottle1.get_functional_point(0), dtype=np.float64)
        right_functional_point = np.asarray(self.bottle2.get_functional_point(0), dtype=np.float64)
        left_place_xy = (
            np.asarray(left_grasp_pose[:2])
            + np.asarray(self.left_target_pose[:2])
            - left_functional_point[:2]
        )
        right_place_xy = (
            np.asarray(right_grasp_pose[:2])
            + np.asarray(self.right_target_pose[:2])
            - right_functional_point[:2]
        )
        left_place_pose = list(left_place_xy) + [left_lift_pose[2]] + left_grasp_pose[3:]
        right_place_pose = list(right_place_xy) + [right_lift_pose[2]] + right_grasp_pose[3:]
        print(
            "[piper-ik] place gripper targets from bottle offsets: "
            f"left=({left_place_pose[0]:.3f},{left_place_pose[1]:.3f},{left_place_pose[2]:.3f}) "
            f"right=({right_place_pose[0]:.3f},{right_place_pose[1]:.3f},{right_place_pose[2]:.3f})"
        )

        sequence = [
            ("pregrasp", left_grasp[0], right_grasp[0]),
            ("grasp", left_grasp[1], right_grasp[1]),
            ("close_gripper", left_grasp[2], right_grasp[2]),
            ("lift", Action(left_arm, "move", target_pose=left_lift_pose),
             Action(right_arm, "move", target_pose=right_lift_pose)),
            ("place", Action(left_arm, "move", target_pose=left_place_pose),
             Action(right_arm, "move", target_pose=right_place_pose)),
            ("open_gripper", Action(left_arm, "open", target_gripper_pos=1.0),
             Action(right_arm, "open", target_gripper_pos=1.0)),
        ]
        self._trajectory_targets = {
            "left": {name: list(left.target_pose) for name, left, _ in sequence if left.action == "move"},
            "right": {name: list(right.target_pose) for name, _, right in sequence if right.action == "move"},
        }
        return sequence

    def _sequence_from_loaded_trajectory(self):
        metadata = self._loaded_trajectory_metadata
        if metadata is None:
            raise ValueError("trajectory metadata was not loaded before replay")
        targets = metadata["targets"]
        left_arm = ArmTag("left")
        right_arm = ArmTag("right")
        sequence = []
        for name in ("pregrasp", "grasp"):
            constraint = [1, 1, 1, 0, 0, 0] if name == "grasp" else None
            kwargs = {"constraint_pose": constraint} if constraint is not None else {}
            sequence.append((
                name,
                Action(left_arm, "move", target_pose=targets["left"][name], **kwargs),
                Action(right_arm, "move", target_pose=targets["right"][name], **kwargs),
            ))
        sequence.extend([
            ("close_gripper", Action(left_arm, "close", target_gripper_pos=0.0),
             Action(right_arm, "close", target_gripper_pos=0.0)),
            ("lift", Action(left_arm, "move", target_pose=targets["left"]["lift"]),
             Action(right_arm, "move", target_pose=targets["right"]["lift"])),
            ("place", Action(left_arm, "move", target_pose=targets["left"]["place"]),
             Action(right_arm, "move", target_pose=targets["right"]["place"])),
            ("open_gripper", Action(left_arm, "open", target_gripper_pos=1.0),
             Action(right_arm, "open", target_gripper_pos=1.0)),
        ])
        self._trajectory_targets = deepcopy(targets)
        return sequence

    @staticmethod
    def _validate_plan_result(result, side, action_name):
        if not isinstance(result, dict) or result.get("status") != "Success":
            status = result.get("status") if isinstance(result, dict) else type(result).__name__
            raise ValueError(f"{side} {action_name} trajectory is invalid: status={status}")
        position = np.asarray(result.get("position"))
        velocity = np.asarray(result.get("velocity"))
        if position.ndim != 2 or position.shape[0] < 2 or position.shape[1] != 6:
            raise ValueError(f"{side} {action_name} position shape is invalid: {position.shape}")
        if velocity.shape != position.shape:
            raise ValueError(
                f"{side} {action_name} velocity shape {velocity.shape} != position shape {position.shape}"
            )
        if not np.isfinite(position).all() or not np.isfinite(velocity).all():
            raise ValueError(f"{side} {action_name} trajectory contains non-finite values")

    def _full_qpos_with_plan_endpoint(self, side, result):
        entity = self.robot.left_entity if side == "left" else self.robot.right_entity
        arm_joints = self.robot.left_arm_joints if side == "left" else self.robot.right_arm_joints
        active_joints = list(entity.get_active_joints())
        qpos = entity.get_qpos().copy()
        for idx, joint in enumerate(arm_joints):
            qpos[active_joints.index(joint)] = result["position"][-1][idx]
        return qpos

    def _execute_move_pair(self, action_name, left_action, right_action, left_result, right_result):
        self._validate_plan_result(left_result, "left", action_name)
        self._validate_plan_result(right_result, "right", action_name)
        left_steps = left_result["position"].shape[0]
        right_steps = right_result["position"].shape[0]
        max_steps = max(left_steps, right_steps)
        for step in range(max_steps):
            left_idx = min(step, left_steps - 1)
            right_idx = min(step, right_steps - 1)
            self.robot.set_arm_joints(
                left_result["position"][left_idx], left_result["velocity"][left_idx], "left")
            self.robot.set_arm_joints(
                right_result["position"][right_idx], right_result["velocity"][right_idx], "right")
            self.scene.step()
            if self.save_data and step % self.save_freq == 0:
                self._take_picture()
            self._render_execution_step(step)

        # Keep commanding the final IK endpoint so the PD controller converges
        # before the next contact-sensitive stage starts.
        for step in range(self.move_settle_steps):
            self.robot.set_arm_joints(
                left_result["position"][-1], np.zeros(6, dtype=np.float64), "left")
            self.robot.set_arm_joints(
                right_result["position"][-1], np.zeros(6, dtype=np.float64), "right")
            self.scene.step()
            if self.save_data and step % self.save_freq == 0:
                self._take_picture()
            self._render_execution_step(step)

        left_ee = self.robot.left_ee.child_link.get_pose()
        right_ee = self.robot.right_ee.child_link.get_pose()
        left_err = np.linalg.norm(np.asarray(left_ee.p) - np.asarray(left_action.target_pose[:3]))
        right_err = np.linalg.norm(np.asarray(right_ee.p) - np.asarray(right_action.target_pose[:3]))
        print(
            f"[piper-ik][fk-check] {action_name}: "
            f"L=({left_ee.p[0]:.3f},{left_ee.p[1]:.3f},{left_ee.p[2]:.3f}) err={left_err:.3f}m | "
            f"R=({right_ee.p[0]:.3f},{right_ee.p[1]:.3f},{right_ee.p[2]:.3f}) err={right_err:.3f}m"
        )

    def _execute_gripper_pair(self, action_name, left_action, right_action):
        self.robot.set_gripper(left_action.target_gripper_pos, "left")
        self.robot.set_gripper(right_action.target_gripper_pos, "right")
        for step in range(self.gripper_settle_steps):
            self.scene.step()
            if self.save_data and step % self.save_freq == 0:
                self._take_picture()
            self._render_execution_step(step)
        print(f"[piper-ik] {action_name}: grippers settled for {self.gripper_settle_steps} steps")

    def _render_execution_step(self, step):
        """Refresh observation cameras and the interactive viewer during execution."""
        self._update_render()
        if self.render_freq and step % self.render_freq == 0:
            viewer = getattr(self, "viewer", None)
            if viewer is None:
                raise RuntimeError("render_freq is enabled but the SAPIEN viewer is missing")
            viewer.render()
            self._piper_ik_live_render_count = getattr(
                self, "_piper_ik_live_render_count", 0
            ) + 1

    def _update_place_targets_from_closed_grasp(self, sequence):
        """Use the measured post-close bottle/EE offset to place bottle functional points."""
        place_actions = next((item for item in sequence if item[0] == "place"), None)
        if place_actions is None:
            raise RuntimeError("place action is missing")
        _, left_place, right_place = place_actions
        left_ee = np.asarray(self.robot.left_ee.child_link.get_pose().p, dtype=np.float64)
        right_ee = np.asarray(self.robot.right_ee.child_link.get_pose().p, dtype=np.float64)
        left_bottle = np.asarray(self.bottle1.get_functional_point(0), dtype=np.float64)
        right_bottle = np.asarray(self.bottle2.get_functional_point(0), dtype=np.float64)
        left_offset = left_bottle[:2] - left_ee[:2]
        right_offset = right_bottle[:2] - right_ee[:2]
        left_place.target_pose[:2] = (np.asarray(self.left_target_pose[:2]) - left_offset).tolist()
        right_place.target_pose[:2] = (np.asarray(self.right_target_pose[:2]) - right_offset).tolist()
        self._trajectory_targets["left"]["place"] = list(left_place.target_pose)
        self._trajectory_targets["right"]["place"] = list(right_place.target_pose)
        print(
            "[piper-ik] measured closed-grasp offsets: "
            f"left=({left_offset[0]:.3f},{left_offset[1]:.3f}) "
            f"right=({right_offset[0]:.3f},{right_offset[1]:.3f}); "
            f"place left=({left_place.target_pose[0]:.3f},{left_place.target_pose[1]:.3f}) "
            f"right=({right_place.target_pose[0]:.3f},{right_place.target_pose[1]:.3f})"
        )

    def _wait_step_mode(self, action_name):
        if not getattr(self, "_step_mode", False):
            return
        import select
        import sys
        print(f"[piper-ik][step-mode] '{action_name}' done. Press Enter...", flush=True)
        while True:
            self._update_render()
            viewer = getattr(self, "viewer", None)
            if viewer is not None:
                viewer.render()
            if sys.stdin in select.select([sys.stdin], [], [], 0.05)[0]:
                sys.stdin.readline()
                break

    def play_once(self):
        """Plan/replay pregrasp, grasp, lift and place as a state-continuous sequence."""
        if self.need_plan:
            print("[piper-ik] Phase 1: sequential planning + execution")
            sequence = self._build_action_sequence()
            self.left_joint_path = []
            self.right_joint_path = []
            left_start_qpos = None
            right_start_qpos = None
            move_index = 0
            for action_name, left_action, right_action in sequence:
                print(f"[piper-ik] {action_name}: {left_action.action} / {right_action.action}")
                if left_action.action == "move":
                    left_result = self.robot.left_plan_path(
                        left_action.target_pose,
                        constraint_pose=left_action.args.get("constraint_pose"),
                        last_qpos=left_start_qpos,
                    )
                    right_result = self.robot.right_plan_path(
                        right_action.target_pose,
                        constraint_pose=right_action.args.get("constraint_pose"),
                        last_qpos=right_start_qpos,
                    )
                    try:
                        self._validate_plan_result(left_result, "left", action_name)
                        self._validate_plan_result(right_result, "right", action_name)
                    except ValueError as exc:
                        print(f"[piper-ik] plan FAILED: {exc}")
                        self.plan_success = False
                        return self.info
                    self.left_joint_path.append(deepcopy(left_result))
                    self.right_joint_path.append(deepcopy(right_result))
                    self._execute_move_pair(
                        action_name, left_action, right_action, left_result, right_result)
                    left_start_qpos = self._full_qpos_with_plan_endpoint("left", left_result)
                    right_start_qpos = self._full_qpos_with_plan_endpoint("right", right_result)
                    move_index += 1
                else:
                    self._execute_gripper_pair(action_name, left_action, right_action)
                    if action_name == "close_gripper":
                        self._update_place_targets_from_closed_grasp(sequence)
                self._wait_step_mode(action_name)
            if move_index != len(self.MOVE_ACTION_NAMES):
                raise RuntimeError(f"planned {move_index} move actions, expected {len(self.MOVE_ACTION_NAMES)}")
        else:
            print("[piper-ik] Phase 2: validated trajectory replay")
            sequence = self._sequence_from_loaded_trajectory()
            move_index = 0
            for action_name, left_action, right_action in sequence:
                print(f"[piper-ik] replay {action_name}: {left_action.action} / {right_action.action}")
                if left_action.action == "move":
                    left_result = deepcopy(self.left_joint_path[move_index])
                    right_result = deepcopy(self.right_joint_path[move_index])
                    self._execute_move_pair(
                        action_name, left_action, right_action, left_result, right_result)
                    move_index += 1
                else:
                    self._execute_gripper_pair(action_name, left_action, right_action)
                self._wait_step_mode(action_name)

        self.info["info"] = {
            "{A}": f"001_bottle/base{self.bottle1_id}",
            "{B}": f"001_bottle/base{self.bottle2_id}",
        }
        print("[piper-ik] play_once finished")
        return self.info

    def save_traj_data(self, idx):
        """Save a named, versioned trajectory that cannot be confused with legacy paths."""
        for action_name, left_result, right_result in zip(
                self.MOVE_ACTION_NAMES, self.left_joint_path, self.right_joint_path):
            self._validate_plan_result(left_result, "left", action_name)
            self._validate_plan_result(right_result, "right", action_name)
        if len(self.left_joint_path) != len(self.MOVE_ACTION_NAMES):
            raise ValueError("trajectory does not contain all required move actions")
        trajectory = {
            "schema": self.TRAJECTORY_SCHEMA,
            "version": self.TRAJECTORY_VERSION,
            "ik_version": self.ik_version,
            "move_action_names": list(self.MOVE_ACTION_NAMES),
            "targets": deepcopy(self._trajectory_targets),
            "left_joint_path": deepcopy(self.left_joint_path),
            "right_joint_path": deepcopy(self.right_joint_path),
        }
        trajectory_dir = os.path.join(self.save_dir, "_traj_data")
        os.makedirs(trajectory_dir, exist_ok=True)
        file_path = os.path.join(trajectory_dir, f"episode{idx}.pkl")
        with open(file_path, "wb") as file:
            pickle.dump(trajectory, file)
        print(f"[piper-ik] trajectory v{self.TRAJECTORY_VERSION} saved: {file_path}")

    def load_tran_data(self, idx):
        file_path = os.path.join(self.save_dir, "_traj_data", f"episode{idx}.pkl")
        with open(file_path, "rb") as file:
            trajectory = pickle.load(file)
        if trajectory.get("schema") != self.TRAJECTORY_SCHEMA or trajectory.get("version") != self.TRAJECTORY_VERSION:
            raise ValueError(
                f"legacy/incompatible trajectory '{file_path}'; expected "
                f"{self.TRAJECTORY_SCHEMA} v{self.TRAJECTORY_VERSION}. Re-run Phase 1 with a fresh output directory."
            )
        if trajectory.get("ik_version") != self.ik_version:
            raise ValueError(
                f"trajectory IK version {trajectory.get('ik_version')} != task IK version {self.ik_version}"
            )
        if tuple(trajectory.get("move_action_names", ())) != self.MOVE_ACTION_NAMES:
            raise ValueError(f"trajectory action order is invalid: {trajectory.get('move_action_names')}")
        for side in ("left", "right"):
            paths = trajectory.get(f"{side}_joint_path")
            if not isinstance(paths, list) or len(paths) != len(self.MOVE_ACTION_NAMES):
                raise ValueError(f"{side} trajectory count is invalid")
            for action_name, result in zip(self.MOVE_ACTION_NAMES, paths):
                self._validate_plan_result(result, side, action_name)
        targets = trajectory.get("targets")
        for side in ("left", "right"):
            if not isinstance(targets, dict) or set(targets.get(side, {})) != set(self.MOVE_ACTION_NAMES):
                raise ValueError(f"{side} trajectory targets are missing or invalid")
        self._loaded_trajectory_metadata = {
            "schema": trajectory["schema"],
            "version": trajectory["version"],
            "ik_version": trajectory["ik_version"],
            "move_action_names": trajectory["move_action_names"],
            "targets": deepcopy(trajectory["targets"]),
        }
        print(f"[piper-ik] validated trajectory loaded: {file_path}")
        return trajectory

    def check_success(self):
        """检查抓取是否成功，并打印调试信息。"""
        eps = 0.1
        bottle1_target = self.left_target_pose[:2]
        bottle2_target = self.right_target_pose[:2]
        b1_pose = self.bottle1.get_functional_point(0)
        b2_pose = self.bottle2.get_functional_point(0)

        b1_dist = np.linalg.norm(np.array(b1_pose[:2]) - np.array(bottle1_target))
        b2_dist = np.linalg.norm(np.array(b2_pose[:2]) - np.array(bottle2_target))
        b1_z_ok = b1_pose[2] > 0.78
        b2_z_ok = b2_pose[2] > 0.78

        success = (
            abs(b1_pose[0] - bottle1_target[0]) < eps
            and abs(b1_pose[1] - bottle1_target[1]) < eps
            and b1_pose[2] > 0.78
            and abs(b2_pose[0] - bottle2_target[0]) < eps
            and abs(b2_pose[1] - bottle2_target[1]) < eps
            and b2_pose[2] > 0.78
        )

        # 记录最近一次 success 状态（供视频重命名用）
        self._last_success = success

        print(
            f"[piper-ik][check] success={success} "
            f"b1=({b1_pose[0]:.3f},{b1_pose[1]:.3f},{b1_pose[2]:.3f}) "
            f"t1=({bottle1_target[0]:.3f},{bottle1_target[1]:.3f}) "
            f"b1_z_ok={b1_z_ok} dist1={b1_dist:.3f} | "
            f"b2=({b2_pose[0]:.3f},{b2_pose[1]:.3f},{b2_pose[2]:.3f}) "
            f"t2=({bottle2_target[0]:.3f},{bottle2_target[1]:.3f}) "
            f"b2_z_ok={b2_z_ok} dist2={b2_dist:.3f}"
        )

        # save_all_episodes: 强制保存所有 episode（包括失败的）用于调试
        if self.save_all_episodes:
            return True
        return success

    def merge_pkl_to_hdf5_video(self):
        """Merge observations into HDF5 and one video per available RGB camera."""
        import os as _os, numpy as _np
        from envs.utils.pkl2hdf5 import load_pkl_file, parse_dict_structure, append_data_to_structure, images_to_video
        import h5py as _h5py

        if not self.save_data:
            return

        cache_path = self.folder_path["cache"]
        self.check_success()
        tag = "succ" if getattr(self, "_last_success", False) else "fail"
        hdf5_path = f"{self.save_dir}/data/episode{self.ep_num}_{tag}.hdf5"
        os.makedirs(f"{self.save_dir}/data", exist_ok=True)
        os.makedirs(f"{self.save_dir}/video", exist_ok=True)

        # 收集 pkl 文件
        pkl_files = []
        for fname in _os.listdir(cache_path):
            if fname.endswith(".pkl") and fname[:-4].isdigit():
                pkl_files.append((int(fname[:-4]), _os.path.join(cache_path, fname)))
        if not pkl_files:
            print("[piper-ik] No pkl files found, falling back to parent merge")
            super().merge_pkl_to_hdf5_video()
            return

        pkl_files.sort()
        pkl_files = [f[1] for f in pkl_files]

        # 解析数据
        data_list = parse_dict_structure(load_pkl_file(pkl_files[0]))
        for pf in pkl_files:
            append_data_to_structure(data_list, load_pkl_file(pf))

        # Generate videos for every static/wrist RGB camera, including
        # third_camera (right side) and opposite_top_camera (opposite overhead).
        for cam_name, cam_data in data_list.get("observation", {}).items():
            if isinstance(cam_data, dict) and "rgb" in cam_data:
                cam_rgb = _np.array(cam_data["rgb"])
                cam_video = f"{self.save_dir}/video/episode{self.ep_num}_{tag}_{cam_name}.mp4"
                images_to_video(cam_rgb, out_path=cam_video)
                print(f"[piper-ik] {cam_name} video: {cam_video}")
        if "third_view_rgb" in data_list:
            cam_video = f"{self.save_dir}/video/episode{self.ep_num}_{tag}_third_view.mp4"
            images_to_video(_np.array(data_list["third_view_rgb"]), out_path=cam_video)
            print(f"[piper-ik] third_view video: {cam_video}")

        # hdf5
        with _h5py.File(hdf5_path, "w") as f:
            from envs.utils.pkl2hdf5 import create_hdf5_from_dict
            create_hdf5_from_dict(f, data_list)
        print(f"[piper-ik] hdf5 saved: {hdf5_path}")
