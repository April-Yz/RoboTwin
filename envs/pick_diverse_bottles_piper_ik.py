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

import numpy as np
import sapien.core as sapien
import transforms3d as t3d

from .pick_diverse_bottles import pick_diverse_bottles
from ._GLOBAL_CONFIGS import ROOT_PATH
from .utils import ArmTag, Action


class pick_diverse_bottles_piper_ik(pick_diverse_bottles):
    """Piper IK task — Cartesian 抓取 + 多种 IK 后端。"""

    def setup_demo(self, **kwargs):
        super().setup_demo(**kwargs)

        ik_version = kwargs.get("ik_version", "v1")
        interp_steps = kwargs.get("ik_interp_steps", 50)
        print(f"[piper-ik-task] IK version={ik_version} interp_steps={interp_steps}")

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
        # 瓶子 x 范围向两侧扩展，保持 y 在桌面可及范围
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
            xlim=[0.30, 0.50],
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
        self.right_target_pose = [0.48, -0.15, 1.0, 0, 1, 0, 0]

        print(
            "[piper-ik][setup] bottle ranges "
            "left=x[-0.35,-0.18],y[0.03,0.23] "
            "right=x[0.30,0.50],y[0.03,0.23]"
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

        # 规划目标位姿（pregrasp + grasp），基于 bottle 位置计算
        if hasattr(self, "bottle1") and hasattr(self, "bottle2"):
            for bottle, bottle_side, arm_tag in [
                (self.bottle1, "left", "left"),
                (self.bottle2, "right", "right"),
            ]:
                pos = bottle.get_pose().p
                # 获取 EE 朝向
                if arm_tag == "left":
                    ee = self.robot.left_ee.child_link.get_pose()
                else:
                    ee = self.robot.right_ee.child_link.get_pose()
                quat = [float(ee.q[0]), float(ee.q[1]), float(ee.q[2]), float(ee.q[3])]

                # pregrasp: 瓶子上方 15cm
                pre_z = pos[2] + 0.15
                pre_y = pos[1] - 0.08
                pre_x = pos[0]
                pregrasp_pose = sapien.Pose(
                    np.array([pre_x, pre_y, pre_z], dtype=np.float64),
                    np.array(quat, dtype=np.float64),
                )
                axes.append((f"plan_pregrasp_{bottle_side}", pregrasp_pose, 0.07, (0.3, 0.5, 1.0)))  # 浅蓝色

                # grasp: 瓶子接触点上方 2cm
                grasp_z = pos[2] + 0.02
                grasp_pose = sapien.Pose(
                    np.array([pre_x, pre_y, grasp_z], dtype=np.float64),
                    np.array(quat, dtype=np.float64),
                )
                axes.append((f"plan_grasp_{bottle_side}", grasp_pose, 0.07, (0.3, 1.0, 0.5)))  # 浅绿色

                print(
                    f"[piper-ik][debug-axis] {bottle_side}: "
                    f"bottle_z={pos[2]:.3f} "
                    f"pregrasp_z={pre_z:.3f} grasp_z={grasp_z:.3f}"
                )

        return axes

    # ------------------------------------------------------------------
    # 简化的 Cartesian 抓取（不经过 choose_best_pose 旋转搜索）
    # ------------------------------------------------------------------

    def _cartesian_grasp_actor(self, actor, arm_tag, pre_grasp_dis=0.08, gripper_pos=0.0):
        """Piper 简化版 Cartesian 抓取。

        使用当前 EE 朝向（而非固定朝向），避免 IK 因朝向不可达而失败。
        """
        if not self.plan_success:
            return None, []

        actor_pose = actor.get_pose()
        actor_pos = actor_pose.p

        # 获取当前 EE 朝向（保持当前夹爪方向，只平移位置）
        if arm_tag == "left":
            ee = self.robot.left_ee.child_link.get_pose()
        else:
            ee = self.robot.right_ee.child_link.get_pose()
        current_quat = [float(ee.q[0]), float(ee.q[1]), float(ee.q[2]), float(ee.q[3])]

        # pregrasp：瓶子上方 15cm
        pre_z = actor_pos[2] + 0.15
        pre_y = actor_pos[1] - pre_grasp_dis
        pre_x = actor_pos[0]
        pregrasp_pose = [pre_x, pre_y, pre_z] + list(current_quat)

        # grasp：瓶子接触点上方 2cm
        grasp_z = actor_pos[2] + 0.02
        grasp_pose = [pre_x, pre_y, grasp_z] + list(current_quat)

        print(f"[piper-ik] grasp_actor {arm_tag}: bottle=({actor_pos[0]:.3f},{actor_pos[1]:.3f},{actor_pos[2]:.3f})")
        print(f"[piper-ik]   pregrasp=({pre_x:.3f},{pre_y:.3f},{pre_z:.3f}) quat={[round(x,3) for x in current_quat]}")
        print(f"[piper-ik]   grasp=({pre_x:.3f},{pre_y:.3f},{grasp_z:.3f})")

        return arm_tag, [
            Action(arm_tag, "move", target_pose=pregrasp_pose),
            Action(arm_tag, "move", target_pose=grasp_pose, constraint_pose=[1, 1, 1, 0, 0, 0]),
            Action(arm_tag, "close", target_gripper_pos=gripper_pos),
        ]

    def play_once(self):
        """Piper Cartesian 抓取流程 — 使用 take_dense_action 直接执行关节轨迹。"""
        import numpy as np

        print("[piper-ik] Step 1: plan and execute grasp")

        # 使用原始 grasp_actor 的逻辑（如果 IK 求解成功的话）
        # 降级方案：直接使用 move_to_pose 单臂执行
        left_arm = ArmTag("left")
        right_arm = ArmTag("right")

        # 获取目标位姿（Cartesian）
        actions_left = self._cartesian_grasp_actor(self.bottle1, arm_tag=left_arm, pre_grasp_dis=0.08)
        actions_right = self._cartesian_grasp_actor(self.bottle2, arm_tag=right_arm, pre_grasp_dis=0.08)

        if actions_left is None or actions_right is None:
            print("[piper-ik] grasp actor returned None")
            return self.info

        # 对每个 Action 分别规划
        all_left_actions = actions_left[1]
        all_right_actions = actions_right[1]

        for step_idx, (l_act, r_act) in enumerate(zip(all_left_actions, all_right_actions)):
            print(f"[piper-ik] Step 1.{step_idx}: {l_act.action} / {r_act.action}")

            if l_act.action == "move" and r_act.action == "move":
                # 分别规划左右臂
                left_result = self.robot.left_plan_path(l_act.target_pose,
                                                        constraint_pose=l_act.args.get("constraint_pose"))
                right_result = self.robot.right_plan_path(r_act.target_pose,
                                                          constraint_pose=r_act.args.get("constraint_pose"))

                if left_result.get("status") != "Success" or right_result.get("status") != "Success":
                    print(f"[piper-ik] plan FAILED: left={left_result.get('status')} right={right_result.get('status')}")
                    self.plan_success = False
                    return self.info

                # 取较长的轨迹
                l_steps = left_result["position"].shape[0]
                r_steps = right_result["position"].shape[0]
                max_steps = max(l_steps, r_steps)

                for t in range(max_steps):
                    li = min(t, l_steps - 1)
                    ri = min(t, r_steps - 1)
                    self.robot.set_arm_joints(
                        left_result["position"][li], left_result["velocity"][li], "left")
                    self.robot.set_arm_joints(
                        right_result["position"][ri], right_result["velocity"][ri], "right")
                    self._update_render()

            elif l_act.action in ("close", "open"):
                # 夹爪控制
                self.robot.set_gripper(l_act.target_gripper_pos, "left")
                self.robot.set_gripper(r_act.target_gripper_pos, "right")
                for _ in range(50):
                    self._update_render()

        self.info["info"] = {
            "{A}": f"001_bottle/base{self.bottle1_id}",
            "{B}": f"001_bottle/base{self.bottle2_id}",
        }
        print("[piper-ik] play_once finished")
        return self.info

    def check_success(self):
        return super().check_success()
