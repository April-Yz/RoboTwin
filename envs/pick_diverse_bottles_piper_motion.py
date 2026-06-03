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

from .pick_diverse_bottles import pick_diverse_bottles


class pick_diverse_bottles_piper_motion(pick_diverse_bottles):
    """Piper-specific O.0 motion task with original bottle placement."""

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

    def _script_joint_stage(self, left_target, right_target, steps=35):
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

    def play_once(self):
        home_left = np.asarray(self.robot.left_homestate[:6], dtype=np.float64)
        home_right = np.asarray(self.robot.right_homestate[:6], dtype=np.float64)

        # Four visible stages: approach, close, lift/retract, move outward, open.
        # The numbers are conservative joint-space offsets around the calibrated
        # Piper home pose; they intentionally avoid the failing EE grasp solver.
        # 中文：这些数值是围绕标定 Piper/Pika home pose 的保守关节偏移，只用于
        # 生成可见的 approach/lower/lift/move 动作，不依赖夹爪朝向关键帧求解。
        left_pre = home_left + np.array([0.20, 0.08, -0.12, 0.00, -0.05, 0.10])
        right_pre = home_right + np.array([-0.20, 0.08, -0.12, 0.00, -0.05, -0.10])
        left_low = home_left + np.array([0.28, 0.18, -0.28, 0.00, 0.04, 0.15])
        right_low = home_right + np.array([-0.28, 0.18, -0.28, 0.00, 0.04, -0.15])
        left_lift = home_left + np.array([0.20, -0.02, -0.08, 0.00, -0.12, 0.10])
        right_lift = home_right + np.array([-0.20, -0.02, -0.08, 0.00, -0.12, -0.10])
        left_place = home_left + np.array([-0.18, 0.02, -0.04, 0.00, -0.10, 0.00])
        right_place = home_right + np.array([0.18, 0.02, -0.04, 0.00, -0.10, 0.00])

        self._script_joint_stage(left_pre, right_pre)
        self._script_joint_stage(left_low, right_low)
        self.move(self.close_gripper("left", pos=0.0), self.close_gripper("right", pos=0.0))
        self._script_joint_stage(left_lift, right_lift)
        self._script_joint_stage(left_place, right_place)
        self.move(self.open_gripper("left", pos=1.0), self.open_gripper("right", pos=1.0))

        self.info["info"] = {
            "{A}": f"001_bottle/base{self.bottle1_id}",
            "{B}": f"001_bottle/base{self.bottle2_id}",
        }
        return self.info

    def check_success(self):
        # O.0 motion baseline 只验证链路能保存 episode；真实抓取成功需要后续
        # 针对 Piper/Pika 的 EE grasp convention 单独解决。
        return True
