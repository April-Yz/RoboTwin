"""
Piper IK Planner — 多个求解器变体，可作为 CuroboPlanner.plan_path 的替代。

提供 4 个版本，在速度/平滑度/成功率之间不同权衡：

  V1: 纯 IK + 线性关节插值   — 最快，最可靠，无碰撞检测
  V2: IK + 三次样条插值      — 更平滑的速度曲线
  V3: MotionGen + IK 种子    — 碰撞感知，轨迹优化
  V4: 多种子 IK + 择优       — 最高成功率

所有版本实现统一的 plan_path() 接口：
  plan_path(curr_joint_pos, target_gripper_pose, constraint_pose, arms_tag)
  → {"status": "Success"/"Fail", "position": np.array, "velocity": np.array}

用法：
  from envs.robot.piper_ik import PiperIKPlannerV1
  planner = PiperIKPlannerV1(robot_origin_pose, active_joints_name, all_joints, yml_path)
  result = planner.plan_path(current_qpos, target_pose, arms_tag="left")
"""

import os
import time
from copy import deepcopy

import numpy as np
import torch
import transforms3d as t3d
import yaml

from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose as CuroboPose
from curobo.types.robot import JointState, RobotConfig
from curobo.wrap.reacher.ik_solver import IKSolver, IKSolverConfig
from curobo.wrap.reacher.motion_gen import (
    MotionGen,
    MotionGenConfig,
    MotionGenPlanConfig,
)

# ---------------------------------------------------------------------------
# 通用工具
# ---------------------------------------------------------------------------

def _ensure_cuda_sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _linear_interpolate(start, target, steps=50):
    """线性关节空间插值。"""
    start = np.asarray(start, dtype=np.float64).squeeze()
    target = np.asarray(target, dtype=np.float64).squeeze()
    if start.ndim != 1 or target.ndim != 1:
        raise ValueError(f"start/target must be 1D, got {start.shape} / {target.shape}")
    if steps < 2:
        steps = 2
    position = np.linspace(start, target, steps)
    velocity = np.zeros_like(position)
    velocity[1:] = np.diff(position, axis=0)
    velocity[0] = velocity[1] if steps > 1 else 0.0
    return {"status": "Success", "position": position, "velocity": velocity}


def _cubic_interpolate(start, target, steps=50):
    """三次样条关节空间插值（零初速/零终速）。"""
    start = np.asarray(start, dtype=np.float64).squeeze()
    target = np.asarray(target, dtype=np.float64).squeeze()
    if start.ndim != 1 or target.ndim != 1:
        raise ValueError(f"start/target must be 1D, got {start.shape} / {target.shape}")
    n_dof = len(start)
    if steps < 3:
        return _linear_interpolate(start, target, steps)

    t = np.linspace(0, 1, steps)
    position = np.zeros((steps, n_dof))
    velocity = np.zeros((steps, n_dof))

    for j in range(n_dof):
        a0 = start[j]
        a1 = 0.0
        a2 = 3.0 * (target[j] - start[j])
        a3 = -2.0 * (target[j] - start[j])
        position[:, j] = a0 + a1 * t + a2 * t**2 + a3 * t**3
        velocity[:, j] = a1 + 2.0 * a2 * t + 3.0 * a3 * t**2

    return {"status": "Success", "position": position, "velocity": velocity}


# ---------------------------------------------------------------------------
# 基础类：共享 IK 基础设施
# ---------------------------------------------------------------------------

class PiperIKBase:
    """所有 Piper IK Planner 变体的基类。

    处理 URDF 加载、坐标系变换、关节索引映射。
    子类只需实现 _solve_ik_impl() 方法。
    """

    def __init__(self, robot_origin_pose, active_joints_name, all_joints, yml_path=None):
        self.robot_origin_pose = robot_origin_pose
        self.active_joints_name = list(active_joints_name)
        self.all_joints = list(all_joints)

        # 加载 curobo 配置（仅用于 frame_bias 等参数）
        if yml_path and os.path.exists(yml_path):
            with open(yml_path, "r") as f:
                yml_data = yaml.safe_load(f)
            self.frame_bias = yml_data.get("planner", {}).get("frame_bias", [0.0, 0.0, 0.0])
            self.base_link = yml_data["robot_cfg"]["kinematics"]["base_link"]
            self.ee_link = yml_data["robot_cfg"]["kinematics"]["ee_link"]
        else:
            self.frame_bias = [0.0, 0.0, 0.0]
            self.base_link = "base_link"
            self.ee_link = "link6"

        # 使用与 SAPIEN 仿真一致的 URDF（piper_pika_agx），
        # 而非 curobo.yml 中引用的 piper/piper.urdf（关节原点不同！）
        # piper_pika_agx 的 link6 以上只有 6 个 arm joint，不包含夹爪 joint
        yml_dir = os.path.dirname(os.path.abspath(yml_path)) if yml_path else ""
        pika_urdf = os.path.join(yml_dir, "piper_pika_agx.urdf")
        if os.path.exists(pika_urdf):
            self.urdf_path = pika_urdf
        else:
            # 降级：使用 curobo.yml 中的 URDF
            self.urdf_path = yml_data["robot_cfg"]["kinematics"]["urdf_path"]
        print(f"[piper-ik] URDF: {self.urdf_path}")

        self.tensor_args = TensorDeviceType()

        # 构建 IKSolver（纯 IK）
        robot_cfg = RobotConfig.from_basic(
            self.urdf_path, self.base_link, self.ee_link, self.tensor_args
        )
        self.ik_config = IKSolverConfig.load_from_robot_config(
            robot_cfg,
            None,
            rotation_threshold=0.02,
            position_threshold=0.005,  # 更严格的位置精度
            num_seeds=2,               # 多种子提高成功率
            self_collision_check=False,
            self_collision_opt=False,
            tensor_args=self.tensor_args,
            use_cuda_graph=True,
        )
        self.ik_solver = IKSolver(self.ik_config)
        self._robot_cfg = robot_cfg  # 供 V3 使用

        # 关节索引映射
        self._joint_indices = [
            self.all_joints.index(name)
            for name in self.active_joints_name
            if name in self.all_joints
        ]

        self._motion_gen = None  # lazy init for V3
        self._motion_gen_config = None

        print(f"[piper-ik] loaded URDF: {self.urdf_path}")
        print(f"[piper-ik] base={self.base_link} ee={self.ee_link}")
        print(f"[piper-ik] frame_bias={self.frame_bias}")

    # ------------------------------------------------------------------
    # 坐标变换（与 CuroboPlanner 保持一致）
    # ------------------------------------------------------------------

    def _trans_from_world_to_base(self, base_pose, target_pose):
        """将 target_pose 从世界坐标系变换到机械臂 base 坐标系。"""
        base_p, base_q = base_pose[0:3], base_pose[3:]
        target_p, target_q = target_pose[0:3], target_pose[3:]
        rel_p = target_p - base_p
        wRb = t3d.quaternions.quat2mat(base_q)
        wRt = t3d.quaternions.quat2mat(target_q)
        result_p = wRb.T @ rel_p
        result_q = t3d.quaternions.mat2quat(wRb.T @ wRt)
        return result_p, result_q

    def _extract_joint_angles(self, curr_joint_pos):
        """从完整 qpos 中提取 arm joint 角度。"""
        return np.array([curr_joint_pos[idx] for idx in self._joint_indices], dtype=np.float64)

    # ------------------------------------------------------------------
    # plan_path 接口（与 CuroboPlanner 兼容）
    # ------------------------------------------------------------------

    def plan_path(self, curr_joint_pos, target_gripper_pose, constraint_pose=None, arms_tag=None):
        """主接口：输入世界坐标系 target，返回轨迹。

        子类重写 _solve_ik_impl() 来实现不同的 IK 策略。
        本方法负责坐标系变换和结果格式化。
        """
        # 坐标变换：world → base
        world_base_pose = np.concatenate([
            np.array(self.robot_origin_pose.p),
            np.array(self.robot_origin_pose.q),
        ])
        world_target_pose = np.concatenate([
            np.array(target_gripper_pose.p),
            np.array(target_gripper_pose.q),
        ])
        target_p, target_q = self._trans_from_world_to_base(world_base_pose, world_target_pose)

        # frame_bias（Piper 为 [0,0,0]）
        target_p[0] += self.frame_bias[0]
        target_p[1] += self.frame_bias[1]
        target_p[2] += self.frame_bias[2]

        # 提取当前关节角（仅 arm joints）
        current_arm_joints = self._extract_joint_angles(curr_joint_pos)

        # 调用子类的 IK 实现
        ik_result = self._solve_ik_impl(
            target_pos=target_p,
            target_quat=target_q,
            current_joints=current_arm_joints,
            constraint_pose=constraint_pose,
            arms_tag=arms_tag,
        )

        if ik_result is None or not ik_result.get("success", False):
            return {"status": "Fail"}

        # 子类可能已经返回完整轨迹（如 V3），也可能只返回 IK 解
        if "position" in ik_result:
            ik_result["status"] = "Success"
            return ik_result

        # 默认：线性插值连接当前 → IK 解
        solution = ik_result["solution"]
        return _linear_interpolate(current_arm_joints, solution, steps=50)

    # ------------------------------------------------------------------
    # plan_batch 接口（choose_best_pose 需要）
    # ------------------------------------------------------------------

    def plan_batch(self, curr_joint_pos, target_gripper_pose_list,
                   constraint_pose=None, arms_tag=None):
        """批量规划：对多个目标位姿分别求解 IK。

        与 CuroboPlanner.plan_batch 接口兼容。
        返回 {"status": [...], "position": np.array, "velocity": np.array}
        """
        n_poses = len(target_gripper_pose_list)
        statuses = []
        positions = []
        velocities = []

        for target_pose in target_gripper_pose_list:
            result = self.plan_path(
                curr_joint_pos, target_pose,
                constraint_pose=constraint_pose, arms_tag=arms_tag,
            )
            if result.get("status") == "Success":
                statuses.append("Success")
                positions.append(result["position"])
                velocities.append(result["velocity"])
            else:
                statuses.append("Failure")
                # 占位
                positions.append(np.zeros((2, len(self.active_joints_name))))
                velocities.append(np.zeros((2, len(self.active_joints_name))))

        return {
            "status": np.array(statuses, dtype=object),
            "position": np.array(positions, dtype=object),
            "velocity": np.array(velocities, dtype=object),
        }

    def plan_grippers(self, now_val, target_val):
        """夹爪控制（兼容 CuroboPlanner 接口）。"""
        num_step = 200
        dis_val = target_val - now_val
        step = dis_val / num_step
        vals = np.linspace(now_val, target_val, num_step)
        return {"num_step": num_step, "per_step": step, "result": vals}

    # ------------------------------------------------------------------
    # 子类重写此方法
    # ------------------------------------------------------------------

    def _solve_ik_impl(self, target_pos, target_quat, current_joints,
                       constraint_pose=None, arms_tag=None):
        """在 base 坐标系中求解 IK。

        参数:
          target_pos:  (3,)  base 坐标系中的目标位置
          target_quat: (4,)  base 坐标系中的目标四元数 (w,x,y,z)
          current_joints: (6,) 当前关节角 (rad)

        返回:
          None → 失败
          {"success": True, "solution": np.array(6,)} → IK 解
          {"success": True, "position": ..., "velocity": ...} → 完整轨迹
        """
        raise NotImplementedError


# ---------------------------------------------------------------------------
# V1: 纯 IK + 线性插值（默认 50 步）
# ---------------------------------------------------------------------------

class PiperIKPlannerV1(PiperIKBase):
    """最快、最可靠。用 IKSolver 求解单点 IK，线性插值连接。

    特点:
      - 无碰撞检测，适合无障碍物桌面场景
      - seed 来自当前关节角 → 解在当前位置附近
      - 带阈值松弛重试
    """

    def __init__(self, robot_origin_pose, active_joints_name, all_joints, yml_path=None,
                 interp_steps=50):
        super().__init__(robot_origin_pose, active_joints_name, all_joints, yml_path)
        self.interp_steps = interp_steps

    def _solve_ik_impl(self, target_pos, target_quat, current_joints,
                       constraint_pose=None, arms_tag=None):
        target_pos_t = torch.tensor(target_pos, device=self.tensor_args.device, dtype=torch.float32)
        target_quat_t = torch.tensor(target_quat, device=self.tensor_args.device, dtype=torch.float32)
        goal = CuroboPose(target_pos_t, target_quat_t)

        seed_t = torch.tensor(current_joints, device=self.tensor_args.device, dtype=torch.float32).view(1, -1)

        # 第一次尝试
        result = self.ik_solver.solve_batch(goal, seed_config=seed_t)
        _ensure_cuda_sync()

        # 松弛重试
        orig_pos_thresh = self.ik_solver.position_threshold
        orig_rot_thresh = self.ik_solver.rotation_threshold
        attempt = 0

        while not result.success.cpu().numpy().all():
            attempt += 1
            if attempt > 5:
                break
            self.ik_solver.position_threshold *= 3
            self.ik_solver.rotation_threshold *= 2
            result = self.ik_solver.solve_batch(goal, seed_config=seed_t)
            _ensure_cuda_sync()
            if self.ik_solver.position_threshold > 0.5:
                break

        self.ik_solver.position_threshold = orig_pos_thresh
        self.ik_solver.rotation_threshold = orig_rot_thresh

        if not result.success.cpu().numpy().all():
            return None

        solution = result.solution.cpu().numpy().squeeze()
        return {
            "success": True,
            "solution": solution,
            "position": _linear_interpolate(current_joints, solution, self.interp_steps)["position"],
            "velocity": _linear_interpolate(current_joints, solution, self.interp_steps)["velocity"],
        }


# ---------------------------------------------------------------------------
# V2: IK + 三次样条插值（更平滑的速度曲线）
# ---------------------------------------------------------------------------

class PiperIKPlannerV2(PiperIKBase):
    """与 V1 相同的 IK 策略，但使用三次样条插值生成更平滑的轨迹。

    特点:
      - 零初速/零终速 → 平滑启停
      - 适合需要 smooth velocity profile 的场景
    """

    def __init__(self, robot_origin_pose, active_joints_name, all_joints, yml_path=None,
                 interp_steps=50):
        super().__init__(robot_origin_pose, active_joints_name, all_joints, yml_path)
        self.interp_steps = interp_steps

    def _solve_ik_impl(self, target_pos, target_quat, current_joints,
                       constraint_pose=None, arms_tag=None):
        target_pos_t = torch.tensor(target_pos, device=self.tensor_args.device, dtype=torch.float32)
        target_quat_t = torch.tensor(target_quat, device=self.tensor_args.device, dtype=torch.float32)
        goal = CuroboPose(target_pos_t, target_quat_t)

        seed_t = torch.tensor(current_joints, device=self.tensor_args.device, dtype=torch.float32).view(1, -1)

        result = self.ik_solver.solve_batch(goal, seed_config=seed_t)
        _ensure_cuda_sync()

        orig_pos_thresh = self.ik_solver.position_threshold
        orig_rot_thresh = self.ik_solver.rotation_threshold
        attempt = 0

        while not result.success.cpu().numpy().all():
            attempt += 1
            if attempt > 5:
                break
            self.ik_solver.position_threshold *= 3
            self.ik_solver.rotation_threshold *= 2
            result = self.ik_solver.solve_batch(goal, seed_config=seed_t)
            _ensure_cuda_sync()
            if self.ik_solver.position_threshold > 0.5:
                break

        self.ik_solver.position_threshold = orig_pos_thresh
        self.ik_solver.rotation_threshold = orig_rot_thresh

        if not result.success.cpu().numpy().all():
            return None

        solution = result.solution.cpu().numpy().squeeze()
        traj = _cubic_interpolate(current_joints, solution, self.interp_steps)
        return {
            "success": True,
            "solution": solution,
            "position": traj["position"],
            "velocity": traj["velocity"],
        }


# ---------------------------------------------------------------------------
# V3: MotionGen + IK 种子（碰撞感知 + 轨迹优化）
# ---------------------------------------------------------------------------

class PiperIKPlannerV3(PiperIKBase):
    """先用 IKSolver 找到目标关节角，再用 MotionGen 做轨迹优化。

    特点:
      - 碰撞感知（table + self-collision）
      - 生成平滑、无碰撞的轨迹
      - 比直接用 MotionGen（随机种子）更可靠，因为种子已接近目标
    """

    def __init__(self, robot_origin_pose, active_joints_name, all_joints, yml_path=None, **kwargs):
        super().__init__(robot_origin_pose, active_joints_name, all_joints, yml_path)
        self._init_motion_gen(yml_path)

    def _init_motion_gen(self, yml_path):
        if yml_path is None:
            self._motion_gen = None
            return

        try:
            with open(yml_path, "r") as f:
                yml_data = yaml.safe_load(f)

            world_config = {
                "cuboid": {
                    "table": {
                        "dims": [0.7, 2, 0.04],
                        "pose": [
                            self.robot_origin_pose.p[1],
                            0.0,
                            0.74 - self.robot_origin_pose.p[2],
                            1, 0, 0, 0.0,
                        ],
                    },
                }
            }

            motion_gen_config = MotionGenConfig.load_from_robot_config(
                yml_path,
                world_config,
                interpolation_dt=1 / 250,
                num_trajopt_seeds=1,
            )
            self._motion_gen = MotionGen(motion_gen_config)
            self._motion_gen.warmup()
            self._motion_gen_config = motion_gen_config
            print("[piper-ik V3] MotionGen initialized (collision-aware)")
        except Exception as exc:
            print(f"[piper-ik V3] MotionGen init FAILED: {exc}")
            print("[piper-ik V3] falling back to IK + cubic interpolation (no collision)")
            self._motion_gen = None

    def _solve_ik_impl(self, target_pos, target_quat, current_joints,
                       constraint_pose=None, arms_tag=None):
        # Step 1: 先用 IKSolver 找目标 IK 解
        target_pos_t = torch.tensor(target_pos, device=self.tensor_args.device, dtype=torch.float32)
        target_quat_t = torch.tensor(target_quat, device=self.tensor_args.device, dtype=torch.float32)
        goal = CuroboPose(target_pos_t, target_quat_t)

        seed_t = torch.tensor(current_joints, device=self.tensor_args.device, dtype=torch.float32).view(1, -1)
        ik_result = self.ik_solver.solve_batch(goal, seed_config=seed_t)
        _ensure_cuda_sync()

        if not ik_result.success.cpu().numpy().all():
            return None

        # Step 2: 用 MotionGen 生成平滑轨迹（从当前关节角出发）
        if self._motion_gen is None:
            # 降级为线性插值
            solution = ik_result.solution.cpu().numpy()[0]
            return {
                "success": True,
                "solution": solution,
                "position": _linear_interpolate(current_joints, solution, 80)["position"],
                "velocity": _linear_interpolate(current_joints, solution, 80)["velocity"],
            }

        goal_pose = CuroboPose.from_list(list(target_pos) + list(target_quat))
        start_joint = JointState.from_position(
            torch.tensor(current_joints, dtype=torch.float32).cuda().reshape(1, -1),
            joint_names=self.active_joints_name,
        )

        plan_config = MotionGenPlanConfig(max_attempts=10)
        if constraint_pose is not None:
            from curobo.wrap.reacher.motion_gen import PoseCostMetric
            pose_cost_metric = PoseCostMetric(
                hold_partial_pose=True,
                hold_vec_weight=self._motion_gen.tensor_args.to_device(constraint_pose),
            )
            plan_config.pose_cost_metric = pose_cost_metric

        mg_result = self._motion_gen.plan_single(start_joint, goal_pose, plan_config)
        _ensure_cuda_sync()

        if not mg_result.success.item():
            # MotionGen 失败 → 降级为线性插值
            solution = ik_result.solution.cpu().numpy()[0]
            return {
                "success": True,
                "solution": solution,
                "position": _linear_interpolate(current_joints, solution, 80)["position"],
                "velocity": _linear_interpolate(current_joints, solution, 80)["velocity"],
            }

        return {
            "success": True,
            "status": "Success",
            "position": np.array(mg_result.interpolated_plan.position.cpu()),
            "velocity": np.array(mg_result.interpolated_plan.velocity.cpu()),
        }


# ---------------------------------------------------------------------------
# V4: 多种子 IK + 择优
# ---------------------------------------------------------------------------

class PiperIKPlannerV4(PiperIKBase):
    """从当前关节角出发，尝试多个附近种子，选择最佳 IK 解。

    特点:
      - 最高成功率 — 当前关节角 ± 小扰动作为多个种子
      - 选择离当前关节角最近的解（最小 joint space distance）
      - 适合目标位姿处于奇异位形附近的情况
    """

    def __init__(self, robot_origin_pose, active_joints_name, all_joints, yml_path=None,
                 interp_steps=50, num_perturbations=5, perturbation_scale=0.05):
        super().__init__(robot_origin_pose, active_joints_name, all_joints, yml_path)
        self.interp_steps = interp_steps
        self.num_perturbations = num_perturbations
        self.perturbation_scale = perturbation_scale

    def _solve_ik_impl(self, target_pos, target_quat, current_joints,
                       constraint_pose=None, arms_tag=None):
        target_pos_t = torch.tensor(target_pos, device=self.tensor_args.device, dtype=torch.float32)
        target_quat_t = torch.tensor(target_quat, device=self.tensor_args.device, dtype=torch.float32)
        goal = CuroboPose(target_pos_t, target_quat_t)

        best_solution = None
        best_distance = float("inf")

        # 种子列表：当前关节角 + 微小扰动
        seeds = [np.asarray(current_joints, dtype=np.float64)]
        rng = np.random.RandomState(42)
        for _ in range(self.num_perturbations):
            noise = rng.normal(0, self.perturbation_scale, size=len(current_joints))
            seeds.append(np.clip(current_joints + noise, -3.14, 3.14))

        for seed in seeds:
            seed_t = torch.tensor(seed, device=self.tensor_args.device, dtype=torch.float32).view(1, -1)
            result = self.ik_solver.solve_batch(goal, seed_config=seed_t)
            _ensure_cuda_sync()

            if result.success.cpu().numpy().all():
                sol = result.solution.cpu().numpy()[0]
                dist = np.linalg.norm(sol - current_joints)
                if dist < best_distance:
                    best_distance = dist
                    best_solution = sol

        if best_solution is None:
            # 所有种子都失败 → 松弛重试（仅用当前关节角种子）
            seed_t = torch.tensor(current_joints, device=self.tensor_args.device, dtype=torch.float32).view(1, -1)
            orig_pos = self.ik_solver.position_threshold
            orig_rot = self.ik_solver.rotation_threshold
            for _ in range(5):
                self.ik_solver.position_threshold *= 3
                self.ik_solver.rotation_threshold *= 2
                result = self.ik_solver.solve_batch(goal, seed_config=seed_t)
                _ensure_cuda_sync()
                if result.success.cpu().numpy().all():
                    best_solution = result.solution.cpu().numpy()[0]
                    break
                if self.ik_solver.position_threshold > 0.5:
                    break
            self.ik_solver.position_threshold = orig_pos
            self.ik_solver.rotation_threshold = orig_rot

        if best_solution is None:
            return None

        traj = _cubic_interpolate(current_joints, best_solution, self.interp_steps)
        return {
            "success": True,
            "solution": best_solution,
            "position": traj["position"],
            "velocity": traj["velocity"],
        }


# ---------------------------------------------------------------------------
# 便捷工厂函数
# ---------------------------------------------------------------------------

_PLANNER_REGISTRY = {
    "v1": PiperIKPlannerV1,
    "v2": PiperIKPlannerV2,
    "v3": PiperIKPlannerV3,
    "v4": PiperIKPlannerV4,
}


def create_piper_ik_planner(version="v1", robot_origin_pose=None,
                            active_joints_name=None, all_joints=None,
                            yml_path=None, **kwargs):
    """工厂函数：按版本号创建 Piper IK Planner。"""
    cls = _PLANNER_REGISTRY.get(version)
    if cls is None:
        raise ValueError(f"Unknown IK version '{version}'. Options: {list(_PLANNER_REGISTRY.keys())}")
    return cls(robot_origin_pose, active_joints_name, all_joints, yml_path, **kwargs)
