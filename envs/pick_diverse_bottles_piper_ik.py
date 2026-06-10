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

        # 确保 folder_path 存在（Phase 2 replay 时需要）
        if not hasattr(self, 'folder_path') or self.folder_path is None:
            self.folder_path = {
                "cache": f"{self.save_dir}/.cache/episode{self.ep_num}/"
            }

        ik_version = kwargs.get("ik_version", "v1")
        interp_steps = kwargs.get("ik_interp_steps", 50)
        self.save_all_episodes = kwargs.get("save_all_episodes", False)
        print(f"[piper-ik-task] IK version={ik_version} interp_steps={interp_steps} save_all={self.save_all_episodes}")

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

    def play_once(self):
        """Piper Cartesian 抓取流程 — Phase 1/2 兼容。

        Phase 1 (need_plan=True):  IK 求解 → 存入 left_joint_path → 执行
        Phase 2 (need_plan=False): 从 left_joint_path 回放 → 执行 → hdf5

        步骤：pregrasp → grasp → gripper_close → lift → place
        """
        import numpy as np
        from copy import deepcopy

        left_arm = ArmTag("left")
        right_arm = ArmTag("right")

        if self.need_plan:
            print("[piper-ik] Phase 1: planning + executing")
            actions_left = self._cartesian_grasp_actor(self.bottle1, arm_tag=left_arm, pre_grasp_dis=0.08)
            actions_right = self._cartesian_grasp_actor(self.bottle2, arm_tag=right_arm, pre_grasp_dis=0.08)

            if actions_left is None or actions_right is None:
                print("[piper-ik] grasp actor returned None")
                return self.info

            # 追加 lift + place 步骤
            all_left = actions_left[1]   # [pregrasp_move, grasp_move, close_gripper]
            all_right = actions_right[1]

            # lift: 用瓶子位置 + 固定 z 偏移（避免 move_by_displacement 的相对误差）
            l_bottle_z = self.bottle1.get_pose().p[2]
            r_bottle_z = self.bottle2.get_pose().p[2]
            l_ee = self.robot.left_ee.child_link.get_pose()
            r_ee = self.robot.right_ee.child_link.get_pose()
            lift_z = max(l_bottle_z, r_bottle_z) + 0.25  # 瓶子上方 25cm

            lift_left_pose = [l_ee.p[0], l_ee.p[1], lift_z,
                              float(l_ee.q[0]), float(l_ee.q[1]), float(l_ee.q[2]), float(l_ee.q[3])]
            lift_right_pose = [r_ee.p[0], r_ee.p[1], lift_z,
                               float(r_ee.q[0]), float(r_ee.q[1]), float(r_ee.q[2]), float(r_ee.q[3])]
            all_left.append(Action(left_arm, "move", target_pose=lift_left_pose))
            all_right.append(Action(right_arm, "move", target_pose=lift_right_pose))

            # place: 移动到放置目标
            all_left.append(Action(left_arm, "move", target_pose=self.left_target_pose))
            all_right.append(Action(right_arm, "move", target_pose=self.right_target_pose))

            # open gripper
            all_left.append(self.open_gripper(arm_tag=left_arm, pos=1.0)[1][0])
            all_right.append(self.open_gripper(arm_tag=right_arm, pos=1.0)[1][0])

            all_left_actions = all_left
            all_right_actions = all_right

            for step_idx, (l_act, r_act) in enumerate(zip(all_left_actions, all_right_actions)):
                print(f"[piper-ik] Step 1.{step_idx}: {l_act.action} / {r_act.action}")

                if l_act.action == "move" and r_act.action == "move":
                    left_result = self.robot.left_plan_path(
                        l_act.target_pose, constraint_pose=l_act.args.get("constraint_pose"))
                    right_result = self.robot.right_plan_path(
                        r_act.target_pose, constraint_pose=r_act.args.get("constraint_pose"))

                    if left_result.get("status") != "Success" or right_result.get("status") != "Success":
                        print(f"[piper-ik] plan FAILED: L={left_result.get('status')} R={right_result.get('status')}")
                        self.plan_success = False
                        return self.info

                    self.left_joint_path.append(deepcopy(left_result))
                    self.right_joint_path.append(deepcopy(right_result))
                # gripper step (no joint path needed)
        else:
            print("[piper-ik] Phase 2: replaying from recorded paths")
            # Phase 2 also needs to reconstruct the action list for execution
            actions_left = self._cartesian_grasp_actor(self.bottle1, arm_tag=left_arm, pre_grasp_dis=0.08)
            actions_right = self._cartesian_grasp_actor(self.bottle2, arm_tag=right_arm, pre_grasp_dis=0.08)
            all_left = actions_left[1]
            all_right = actions_right[1]
            l_bz = self.bottle1.get_pose().p[2]; r_bz = self.bottle2.get_pose().p[2]
            l_ee = self.robot.left_ee.child_link.get_pose(); r_ee = self.robot.right_ee.child_link.get_pose()
            lz = max(l_bz, r_bz) + 0.25
            all_left.append(Action(left_arm, "move", target_pose=[
                l_ee.p[0], l_ee.p[1], lz, float(l_ee.q[0]), float(l_ee.q[1]), float(l_ee.q[2]), float(l_ee.q[3])]))
            all_right.append(Action(right_arm, "move", target_pose=[
                r_ee.p[0], r_ee.p[1], lz, float(r_ee.q[0]), float(r_ee.q[1]), float(r_ee.q[2]), float(r_ee.q[3])]))
            all_left.append(Action(left_arm, "move", target_pose=self.left_target_pose))
            all_right.append(Action(right_arm, "move", target_pose=self.right_target_pose))
            all_left.append(self.open_gripper(arm_tag=left_arm, pos=1.0)[1][0])
            all_right.append(self.open_gripper(arm_tag=right_arm, pos=1.0)[1][0])
            all_left_actions = all_left
            all_right_actions = all_right

        # ═══════════════ 执行 ═══════════════
        self._move_cnt = 0
        for step_idx, (l_act, r_act) in enumerate(zip(all_left_actions, all_right_actions)):
            step_names = ["pregrasp", "grasp", "close_gripper", "lift", "place", "open_gripper"]
            step_name = step_names[step_idx] if step_idx < len(step_names) else f"step{step_idx}"
            print(f"[piper-ik] Step 1.{step_idx} ({step_name}): {l_act.action} / {r_act.action}")

            if l_act.action == "move" and r_act.action == "move":
                if self.need_plan:
                    left_result = self.left_joint_path[self._move_cnt]
                    right_result = self.right_joint_path[self._move_cnt]
                    self._move_cnt += 1
                else:
                    left_result = deepcopy(self.left_joint_path[self.left_cnt])
                    right_result = deepcopy(self.right_joint_path[self.right_cnt])
                    self.left_cnt += 1
                    self.right_cnt += 1

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
                    self.scene.step()
                    if self.save_data and t % self.save_freq == 0:
                        self._take_picture()
                    self._update_render()

                # FK 验证：直接设置 qpos 测 IK 精度（绕过 PD 控制器）
                if self.need_plan and step_idx <= 1:
                    for side, arm_tag in [("left", "left"), ("right", "right")]:
                        entity = self.robot.left_entity if arm_tag == "left" else self.robot.right_entity
                        arm_joints = self.robot.left_arm_joints if arm_tag == "left" else self.robot.right_arm_joints
                        ee_joint = self.robot.left_ee if arm_tag == "left" else self.robot.right_ee
                        result = self.left_joint_path[step_idx] if arm_tag == "left" else self.right_joint_path[step_idx]
                        sol_q = result["position"][-1]  # final joint angles from IK
                        # Direct set qpos (bypass PD)
                        qpos = entity.get_qpos().copy()
                        for j_idx, joint in enumerate(arm_joints):
                            active_idx = list(entity.get_active_joints()).index(joint)
                            qpos[active_idx] = sol_q[j_idx]
                        entity.set_qpos(qpos)
                        ee_pose = ee_joint.child_link.get_pose()
                        target = l_act.target_pose if arm_tag == "left" else r_act.target_pose
                        t_pos = np.array(target[:3]) if isinstance(target, (list, np.ndarray)) else target.p
                        direct_err = np.linalg.norm(np.array(ee_pose.p) - t_pos)
                        print(f"[piper-ik][fk-direct] {step_name} {side}: "
                              f"EE=({ee_pose.p[0]:.3f},{ee_pose.p[1]:.3f},{ee_pose.p[2]:.3f}) "
                              f"err={direct_err:.3f}m")

                # FK 验证：每次 move 后读取实际 EE 与目标对比
                if self.need_plan:
                    l_ee = self.robot.left_ee.child_link.get_pose()
                    r_ee = self.robot.right_ee.child_link.get_pose()
                    l_target = l_act.target_pose
                    r_target = r_act.target_pose
                    if isinstance(l_target, (list, np.ndarray)):
                        l_err = np.linalg.norm(np.array(l_ee.p) - np.array(l_target[:3]))
                        r_err = np.linalg.norm(np.array(r_ee.p) - np.array(r_target[:3]))
                        print(f"[piper-ik][fk-check] {step_name}: "
                              f"L_ee=({l_ee.p[0]:.3f},{l_ee.p[1]:.3f},{l_ee.p[2]:.3f}) err={l_err:.3f}m | "
                              f"R_ee=({r_ee.p[0]:.3f},{r_ee.p[1]:.3f},{r_ee.p[2]:.3f}) err={r_err:.3f}m")

            elif l_act.action in ("close", "open"):
                gripper_pos = l_act.target_gripper_pos
                print(f"[piper-ik] {l_act.action} gripper to {gripper_pos}")
                self.robot.set_gripper(gripper_pos, "left")
                self.robot.set_gripper(gripper_pos, "right")
                for _ in range(80):
                    self.scene.step()
                    self._update_render()

            # step_mode 等待
            if getattr(self, "_step_mode", False):
                import sys as _sys, select as _sel
                print(f"[piper-ik][step-mode] '{step_name}' done. Press Enter...", flush=True)
                while True:
                    self._update_render()
                    viewer = getattr(self, "viewer", None)
                    if viewer is not None:
                        viewer.render()
                    if _sys.stdin in _sel.select([_sys.stdin], [], [], 0.05)[0]:
                        _sys.stdin.readline()
                        break

        self.info["info"] = {
            "{A}": f"001_bottle/base{self.bottle1_id}",
            "{B}": f"001_bottle/base{self.bottle2_id}",
        }
        print("[piper-ik] play_once finished")
        return self.info

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
            and b1_pose[2] > 0.89
            and abs(b2_pose[0] - bottle2_target[0]) < eps
            and abs(b2_pose[1] - bottle2_target[1]) < eps
            and b2_pose[2] > 0.89
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
        """合并 pkl 为 hdf5+mp4，包含 head + third_view 双路视频，标注 success/fail。"""
        import os as _os, numpy as _np
        from envs.utils.pkl2hdf5 import load_pkl_file, parse_dict_structure, append_data_to_structure, images_to_video
        import h5py as _h5py

        if not self.save_data:
            return

        cache_path = self.folder_path["cache"]
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

        # 为所有 camera 生成 mp4（包括 Piper config 中新增的 third_camera）
        for cam_name in ["head_camera", "front_camera", "side_camera", "third_camera",
                         "third_view", "observer", "third_view_rgb"]:
            if cam_name in data_list.get("observation", {}):
                cam_rgb = _np.array(data_list["observation"][cam_name]["rgb"])
                cam_video = f"{self.save_dir}/video/episode{self.ep_num}_{tag}_{cam_name}.mp4"
                images_to_video(cam_rgb, out_path=cam_video)
                print(f"[piper-ik] {cam_name} video: {cam_video}")

        # hdf5
        with _h5py.File(hdf5_path, "w") as f:
            from envs.utils.pkl2hdf5 import create_hdf5_from_dict
            create_hdf5_from_dict(f, data_list)
        print(f"[piper-ik] hdf5 saved: {hdf5_path}")
