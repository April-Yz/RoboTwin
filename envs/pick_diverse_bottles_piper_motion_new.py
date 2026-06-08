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

        # 中文：瓶子 y 范围沿用原始 ALOHA/AgileX 的 y=[0.03, 0.23]。
        # 经核实 Piper base 位于 y≈-0.25（左）/ y≈-0.27（右），
        # 瓶子距 base 在 y 方向约 0.28~0.50m，在 Piper 桌面臂展范围内。
        # 这只影响 O.0 piper motion baseline，不修改原始 pick_diverse_bottles.py。
        self.bottle1 = rand_create_actor(
            self,
            xlim=[-0.30, -0.18],
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
            xlim=[0.30, 0.46],
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
        self.left_target_pose = [-0.24, -0.26, 1.0, 0, 1, 0, 0]
        self.right_target_pose = [0.44, -0.26, 1.0, 0, 1, 0, 0]

        print(
            "[piper-motion][setup] bottle ranges "
            "left=x[-0.30,-0.18],y[0.03,0.23] "
            "right=x[0.30,0.46],y[0.03,0.23]"
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

    @staticmethod
    def _forward_offset_pose(pose, distance=0.10):
        """将位姿沿其局部 +Z（夹爪前进方向）向前偏移。

        Piper/Pika URDF 中 link6 的局部 +Z 轴从腕部指向夹爪指尖方向。
        此方法用于可视化夹爪尖端的大致位置（腕部 joint6 前方 ~10 cm）。
        """
        R = pose.to_transformation_matrix()[:3, :3]
        forward = R[:, 2]  # 局部 Z 轴在世界坐标系中的方向
        return sapien.Pose(pose.p + forward * distance, pose.q)

    # ------------------------------------------------------------------
    # 颜色图例（终端 + viewer 原点色块）
    # ------------------------------------------------------------------
    # 类别                原点颜色       含义
    # ─────────────────────────────────────────────────────────────────
    # bottle              white  (1,1,1)   瓶子几何中心
    # place_target        yellow (1,1,0)   放置目标位姿
    # ee_wrist (current)  cyan   (0,1,1)   当前 link6 腕部 (joint6)
    # ee_tip   (current)  magenta(1,0,1)   当前夹爪尖端 (腕部+10cm)
    # stage_pregrasp      blue   (0.3,0.5,1.0)   预抓取阶段
    # stage_grasp_lower   green  (0.3,1.0,0.5)   下降抓取阶段
    # stage_lift          gold   (1.0,1.0,0.3)   抬升阶段
    # stage_move_out      coral  (1.0,0.4,0.4)   移出阶段
    #
    # 每个阶段同时显示：
    #   _wrist → 腕部 link6 位置（FK 直接结果）
    #   _tip   → 腕部 +10cm 沿前进轴（近似夹爪尖端）
    # ------------------------------------------------------------------

    # 阶段原点色
    _STAGE_COLORS = {
        "pregrasp":    (0.3, 0.5, 1.0),   # blue
        "grasp_lower": (0.3, 1.0, 0.5),   # green
        "lift":        (1.0, 1.0, 0.3),   # gold
        "move_out":    (1.0, 0.4, 0.4),   # coral
    }

    def get_debug_axis_poses(self):
        """返回所有 debug 坐标轴，每个元素为 (name, pose, length, origin_color)。

        viewer 可通过 origin_color 区分不同类别；终端也会打印完整图例。
        """
        axes = []

        # ── 当前 EE：腕部 (wrist) + 尖端 (tip) ──
        left_wrist = sapien.Pose(
            self.robot.left_ee.child_link.get_pose().p,
            self.robot.left_ee.child_link.get_pose().q,
        )
        right_wrist = sapien.Pose(
            self.robot.right_ee.child_link.get_pose().p,
            self.robot.right_ee.child_link.get_pose().q,
        )
        left_tip = self._forward_offset_pose(left_wrist, 0.10)
        right_tip = self._forward_offset_pose(right_wrist, 0.10)

        axes.append(("ee_wrist_left",  left_wrist,  0.06, (0.0, 1.0, 1.0)))   # cyan
        axes.append(("ee_wrist_right", right_wrist, 0.06, (0.0, 1.0, 1.0)))
        axes.append(("ee_tip_left",    left_tip,    0.07, (1.0, 0.0, 1.0)))   # magenta
        axes.append(("ee_tip_right",   right_tip,   0.07, (1.0, 0.0, 1.0)))

        print(
            "[piper-motion][target-axis] ee_wrist "
            f"left_pos={np.round(left_wrist.p, 4).tolist()} "
            f"right_pos={np.round(right_wrist.p, 4).tolist()}"
        )
        print(
            "[piper-motion][target-axis] ee_tip (+10cm fwd) "
            f"left_pos={np.round(left_tip.p, 4).tolist()} "
            f"right_pos={np.round(right_tip.p, 4).tolist()}"
        )

        # ── 阶段目标：腕部 + 尖端 ──
        for stage_name, left_target, right_target in self._motion_joint_targets():
            stage_color = self._STAGE_COLORS.get(stage_name, (0.6, 0.6, 0.6))
            # 稍暗一点的 tip 色
            tip_color = tuple(max(0, c - 0.15) for c in stage_color)

            left_wrist_pose = self._ee_pose_at_joints("left", left_target)
            right_wrist_pose = self._ee_pose_at_joints("right", right_target)
            left_tip_pose = self._forward_offset_pose(left_wrist_pose, 0.10)
            right_tip_pose = self._forward_offset_pose(right_wrist_pose, 0.10)

            axes.append((f"stage_{stage_name}_wrist_left",  left_wrist_pose,  0.045, stage_color))
            axes.append((f"stage_{stage_name}_wrist_right", right_wrist_pose, 0.045, stage_color))
            axes.append((f"stage_{stage_name}_tip_left",    left_tip_pose,    0.055, tip_color))
            axes.append((f"stage_{stage_name}_tip_right",   right_tip_pose,   0.055, tip_color))

            print(
                f"[piper-motion][target-axis] stage_{stage_name} "
                f"wrist_left={np.round(left_wrist_pose.p, 4).tolist()} "
                f"wrist_right={np.round(right_wrist_pose.p, 4).tolist()}"
            )
            print(
                f"[piper-motion][target-axis] stage_{stage_name} "
                f"tip_left={np.round(left_tip_pose.p, 4).tolist()} "
                f"tip_right={np.round(right_tip_pose.p, 4).tolist()}"
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
