"""O.0 Piper/Pika motion baseline for pick_diverse_bottles.

This task keeps the original bottle randomization from pick_diverse_bottles, but
uses a deterministic Piper joint-space motion instead of the original
ALOHA-style grasp pose search. The goal is to provide a runnable calibrated
Piper/Pika baseline for data/video generation while the EE grasp convention is
being debugged.

中文说明：这个文件是 O.0 对照实验的可运行 motion baseline。它复用原始
pick_diverse_bottles 的瓶子随机摆放和稳定性检查，但故意不走原始
grasp_actor/choose_grasp_pose，因为那条链路在标定 Piper/Pika 上会生成
target_pose=None。当前实现用于先验证标定 Piper/Pika、head-only 保存和后续
运动链路，不等价于真实抓瓶成功。
"""

from copy import deepcopy

import numpy as np
import sapien.core as sapien

from .pick_diverse_bottles import pick_diverse_bottles
from .utils import rand_create_actor


class pick_diverse_bottles_piper_motion(pick_diverse_bottles):
    """Piper-specific O.0 motion task with original bottle placement."""

    def load_actors(self):
        self.id_list = [i for i in range(20)]
        self.bottle1_id = np.random.choice(self.id_list)
        self.bottle2_id = np.random.choice(self.id_list)

        # 中文：这里不再直接复用 ALOHA/AgileX 原始瓶子范围，而是给标定
        # Piper/Pika 留出更靠近当前 URDF/base/home FK 的负 y 桌面边缘范围。
        # 这只影响 O.0 piper motion baseline，不修改原始 pick_diverse_bottles.py。
        self.bottle1 = rand_create_actor(
            self,
            xlim=[-0.30, -0.18],
            ylim=[-0.20, -0.10],
            modelname="001_bottle",
            rotate_rand=True,
            rotate_lim=[0, 1, 0],
            qpos=[0.66, 0.66, -0.25, -0.25],
            convex=True,
            model_id=self.bottle1_id,
        )
        self.bottle2 = rand_create_actor(
            self,
            xlim=[0.30, 0.46],
            ylim=[-0.20, -0.10],
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
        self.left_target_pose = [-0.24, -0.26, 1.0, 0, 1, 0, 0]
        self.right_target_pose = [0.44, -0.26, 1.0, 0, 1, 0, 0]

        print(
            "[piper-motion][setup] bottle ranges "
            "left=x[-0.30,-0.18],y[-0.20,-0.10] "
            "right=x[0.30,0.46],y[-0.20,-0.10]"
        )

    def _joint_result(self, start, target, steps=35):
        # 直接生成关节空间插值结果，接口形状保持和 planner 输出一致。
        start = np.asarray(start, dtype=np.float64)
        target = np.asarray(target, dtype=np.float64)
        position = np.linspace(start, target, steps)
        velocity = np.zeros_like(position)
        if steps > 1:
            velocity[1:] = np.diff(position, axis=0)
            velocity[0] = velocity[1]
        return {"status": "Success", "position": position, "velocity": velocity}

    def _script_joint_stage(self, stage_name, left_target, right_target, steps=35):
        print(f"[piper-motion][stage] {stage_name}: planning joint interpolation")
        print(f"  left_joints={np.round(left_target, 4).tolist()}")
        print(f"  right_joints={np.round(right_target, 4).tolist()}")
        if self.need_plan:
            # 第一次 premotion 阶段记录路径；collect_data 的执行阶段会复用这些路径。
            left_start = np.asarray(self.robot.get_left_arm_jointState()[:-1], dtype=np.float64)
            right_start = np.asarray(self.robot.get_right_arm_jointState()[:-1], dtype=np.float64)
            left_result = self._joint_result(left_start, left_target, steps=steps)
            right_result = self._joint_result(right_start, right_target, steps=steps)
            self.left_joint_path.append(deepcopy(left_result))
            self.right_joint_path.append(deepcopy(right_result))
        else:
            left_result = deepcopy(self.left_joint_path[self.left_cnt])
            right_result = deepcopy(self.right_joint_path[self.right_cnt])
            self.left_cnt += 1
            self.right_cnt += 1

        self.take_dense_action(
            {
                "left_arm": left_result,
                "left_gripper": None,
                "right_arm": right_result,
                "right_gripper": None,
            }
        )
        print(f"[piper-motion][stage] {stage_name}: finished")

    def _motion_joint_targets(self):
        home_left = np.asarray(self.robot.left_homestate[:6], dtype=np.float64)
        home_right = np.asarray(self.robot.right_homestate[:6], dtype=np.float64)

        return [
            (
                "pregrasp",
                home_left + np.array([0.20, 0.08, -0.12, 0.00, -0.05, 0.10]),
                home_right + np.array([-0.20, 0.08, -0.12, 0.00, -0.05, -0.10]),
            ),
            (
                "grasp_lower",
                home_left + np.array([0.28, 0.18, -0.28, 0.00, 0.04, 0.15]),
                home_right + np.array([-0.28, 0.18, -0.28, 0.00, 0.04, -0.15]),
            ),
            (
                "lift",
                home_left + np.array([0.20, -0.02, -0.08, 0.00, -0.12, 0.10]),
                home_right + np.array([-0.20, -0.02, -0.08, 0.00, -0.12, -0.10]),
            ),
            (
                "move_out",
                home_left + np.array([-0.18, 0.02, -0.04, 0.00, -0.10, 0.00]),
                home_right + np.array([0.18, 0.02, -0.04, 0.00, -0.10, 0.00]),
            ),
        ]

    def _ee_pose_at_joints(self, arm_tag, target):
        entity = self.robot.left_entity if arm_tag == "left" else self.robot.right_entity
        arm_joints = self.robot.left_arm_joints if arm_tag == "left" else self.robot.right_arm_joints
        ee_joint = self.robot.left_ee if arm_tag == "left" else self.robot.right_ee
        active_joints = entity.get_active_joints()
        original_qpos = entity.get_qpos().copy()
        target_qpos = original_qpos.copy()
        for idx, joint in enumerate(arm_joints):
            target_qpos[active_joints.index(joint)] = target[idx]
        entity.set_qpos(target_qpos)
        pose = ee_joint.child_link.get_pose()
        result = sapien.Pose(pose.p.copy(), pose.q.copy())
        entity.set_qpos(original_qpos)
        return result

    def get_debug_axis_poses(self):
        axes = []
        axes.append(("left_ee_current", sapien.Pose(self.robot.left_ee.child_link.get_pose().p,
                                                    self.robot.left_ee.child_link.get_pose().q), 0.06))
        axes.append(("right_ee_current", sapien.Pose(self.robot.right_ee.child_link.get_pose().p,
                                                     self.robot.right_ee.child_link.get_pose().q), 0.06))
        for stage_name, left_target, right_target in self._motion_joint_targets():
            left_pose = self._ee_pose_at_joints("left", left_target)
            right_pose = self._ee_pose_at_joints("right", right_target)
            axes.append((f"left_{stage_name}_ee_target", left_pose, 0.045))
            axes.append((f"right_{stage_name}_ee_target", right_pose, 0.045))
            print(
                f"[piper-motion][target-axis] {stage_name} "
                f"left_pos={np.round(left_pose.p, 4).tolist()} "
                f"right_pos={np.round(right_pose.p, 4).tolist()}"
            )
        return axes

    def play_once(self):
        # Four visible stages: approach, close, lift/retract, move outward, open.
        # The numbers are conservative joint-space offsets around the calibrated
        # Piper home pose; they intentionally avoid the failing EE grasp solver.
        # 中文：这些数值是围绕标定 Piper/Pika home pose 的保守关节偏移，只用于
        # 生成可见的 approach/lower/lift/move 动作，不依赖夹爪朝向关键帧求解。
        print("[piper-motion][stage] play_once: start")
        targets = dict((name, (left, right)) for name, left, right in self._motion_joint_targets())
        self._script_joint_stage("pregrasp", *targets["pregrasp"])
        self._script_joint_stage("grasp_lower", *targets["grasp_lower"])
        print("[piper-motion][stage] close_gripper: start")
        self.move(self.close_gripper("left", pos=0.0), self.close_gripper("right", pos=0.0))
        print("[piper-motion][stage] close_gripper: finished")
        self._script_joint_stage("lift", *targets["lift"])
        self._script_joint_stage("move_out", *targets["move_out"])
        print("[piper-motion][stage] open_gripper: start")
        self.move(self.open_gripper("left", pos=1.0), self.open_gripper("right", pos=1.0))
        print("[piper-motion][stage] open_gripper: finished")

        self.info["info"] = {
            "{A}": f"001_bottle/base{self.bottle1_id}",
            "{B}": f"001_bottle/base{self.bottle2_id}",
        }
        print("[piper-motion][stage] play_once: finished")
        return self.info

    def check_success(self):
        # O.0 motion baseline 只验证链路能保存 episode；真实抓取成功需要后续
        # 针对 Piper/Pika 的 EE grasp convention 单独解决。
        return True
