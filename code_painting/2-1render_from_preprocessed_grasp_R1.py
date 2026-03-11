#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简化版机器人渲染器 - 使用hdf5_aloha.py预处理的JSON数据
直接使用预计算的手腕位姿和机器人基座位置，无需重新计算
可视化egodex的预处理结果(wxyz)

说明：
- 该版本在初始化和后续仿真步进中持续固定 R1 的 torso 关节角；
- 目标是让视频重点展示双手动作，同时腰部保持固定姿态。
"""

import os
import numpy as np
import sapien.core as sapien
from sapien.render import clear_cache as sapien_clear_cache
from typing import Dict, Any
import cv2
from envs.robot import Robot
import json
from scipy.spatial.transform import Rotation as Rotation1
import argparse
import trimesh
import time

R1_URDF_PATH = "/projects/zaijia001/R1/galaxea_sim/assets/r1_pro/robot.urdf"
R1_CONFIG_PATH = "/data1/zjyang/program/third/RoboTwin/robot_config_R1_pro.json"
# 视频展示时固定腰部 4 个关节角（torso pos）
R1_TORSO_JOINT_NAMES = ("torso_joint1", "torso_joint2", "torso_joint3", "torso_joint4")
R1_FIXED_TORSO_POS = np.array([0.25, -0.4, -0.85, 0.0], dtype=float)


class SimplifiedRobotRenderer:
    """简化版机器人渲染器 - 直接使用预处理数据"""
    
    def __init__(self, 
                 image_width=640, 
                 image_height=360,
                 enable_viewer=False,
                 fovy_deg=90.0,
                 ground_height=0.0,
                 world_z_offset=0.0,
                 camera_x_offset=0.0,
                 arms_z_offset=0.0,
                 model_path=None,
                 poses_path=None,
                 model_path2=None,
                 poses_path2=None,
                 depth_scale=1.0,
                 depth_offset=0.0,
                 grasp_json_path=None,
                 link_cam_debug_enable=False,
                 link_cam_debug_rot_xyz_deg=(0.0, 0.0, 0.0),
                 link_cam_debug_forward=0.0,
                 link_cam_debug_right=0.0,
                 link_cam_debug_up=0.0,
                 link_cam_axis_mode="none",
                 link_cam_debug_apply_to="all",
                 third_cam_show_link_cams=True,
                 third_person_view=True):
        """初始化机器人渲染器
        
        Args:
            depth_scale: 深度缩放因子（调整深度估计误差）
            depth_offset: 深度偏移量（米）
        """
        self.image_width = image_width
        self.image_height = image_height
        self.enable_viewer = enable_viewer
        self.fovy_deg = float(np.clip(fovy_deg, 30.0, 120.0))
        self.ground_height = float(ground_height)
        self.world_z_offset = float(world_z_offset)
        self.camera_x_offset = float(camera_x_offset)
        self.arms_z_offset = float(arms_z_offset)
        self.third_person_view = third_person_view  # 是否启用第三人称视角
        self.link_cam_debug_enable = bool(link_cam_debug_enable)
        self.link_cam_debug_rot_xyz_deg = np.array(link_cam_debug_rot_xyz_deg, dtype=float).reshape(3)
        self.link_cam_debug_forward = float(link_cam_debug_forward)
        self.link_cam_debug_right = float(link_cam_debug_right)
        self.link_cam_debug_up = float(link_cam_debug_up)
        self.link_cam_axis_mode = str(link_cam_axis_mode).strip().lower()
        apply_to_tokens = [s.strip().lower() for s in str(link_cam_debug_apply_to).split(",") if s.strip()]
        if not apply_to_tokens:
            apply_to_tokens = ["all"]
        self.link_cam_debug_apply_to = set(apply_to_tokens)
        if "all" in self.link_cam_debug_apply_to:
            self.link_cam_debug_apply_to = {"head", "left", "right"}
        self.third_cam_show_link_cams = bool(third_cam_show_link_cams)
        
        # 模型相关参数（第一个模型）
        self.model_path = model_path
        self.poses_path = poses_path
        self.model_actor = None
        self.model_poses = None
        self.depth_scale = depth_scale
        self.depth_offset = depth_offset
        self.fixed_model_pose = None  # 固定的模型位置
        
        # 模型相关参数（第二个模型）
        self.model_path2 = model_path2
        self.poses_path2 = poses_path2
        self.model_actor2 = None
        self.model_poses2 = None
        
        # 存储机器人基座朝向，用于计算third_camera位置
        self.robot_base_quat_wxyz = None
        
        # 抓取相关参数
        self.grasp_json_path = grasp_json_path
        self.best_grasp = None

        # 机器人相关运行状态
        self.robot = None
        self.need_topp = True
        # 固定 torso 的目标角度：用于突出双手动作表现，避免腰部跟随规划漂移。
        self.fixed_torso_pos = R1_FIXED_TORSO_POS.copy()
        self._torso_warning_printed = False
        # 相机link缓存（仅本脚本维护，避免影响其他URDF流程）
        self._head_camera_link = None
        self._left_wrist_camera_link = None
        self._right_wrist_camera_link = None
        self._head_link_warning_printed = False
        self._left_link_warning_printed = False
        self._right_link_warning_printed = False
        self._camera_debug_summary_printed = False
        
        # 初始化SAPIEN环境
        self._setup_sapien_scene()
        self._load_robot()
        self._setup_camera()
        self._create_table()
        
        # 如果提供了模型路径，加载模型
        if self.model_path and self.poses_path:
            self._load_model(depth_scale=self.depth_scale)
        
        # 如果提供了第二个模型路径，加载第二个模型
        if self.model_path2 and self.poses_path2:
            self._load_model2()
        
        # 如果提供了抓取JSON路径，加载抓取数据
        if self.grasp_json_path:
            self._load_grasp_data()
        
        print("简化版机器人渲染器初始化成功！")
    
    def _setup_sapien_scene(self):
        """设置SAPIEN场景"""
        # 创建引擎和渲染器
        self.engine = sapien.Engine()
        
        from sapien.render import set_global_config
        set_global_config(max_num_materials=50000, max_num_textures=50000)
        
        self.renderer = sapien.SapienRenderer()
        self.engine.set_renderer(self.renderer)
        
        # 设置光线追踪参数
        sapien.render.set_camera_shader_dir("rt")
        sapien.render.set_ray_tracing_samples_per_pixel(32)
        sapien.render.set_ray_tracing_path_depth(8)
        sapien.render.set_ray_tracing_denoiser("oidn")
        
        # 创建场景
        scene_config = sapien.SceneConfig()
        self.scene = self.engine.create_scene(scene_config)
        self.scene.set_timestep(1/250)
        
        # 添加地面
        self.scene.add_ground(self.ground_height)
        
        # 设置物理材料
        self.scene.default_physical_material = self.scene.create_physical_material(
            static_friction=0.5,
            dynamic_friction=0.5,
            restitution=0
        )
        
        # 设置环境光
        self.scene.set_ambient_light([0.5, 0.5, 0.5])
        
        # 添加方向光
        self.scene.add_directional_light([0, 0.5, -1], [0.5, 0.5, 0.5], shadow=True)
        
        # 添加点光源
        self.scene.add_point_light([1, 0, 1.8], [1, 1, 1], shadow=True)
        self.scene.add_point_light([-1, 0, 1.8], [1, 1, 1], shadow=True)
        
        # 如果启用viewer
        if self.enable_viewer:
            from sapien.utils.viewer import Viewer
            self.viewer = Viewer(self.renderer)
            self.viewer.set_scene(self.scene)
            self.viewer.set_camera_xyz(x=0.4, y=0.22, z=1.5)
            self.viewer.set_camera_rpy(r=0, p=-0.8, y=2.45)
    
    def _load_robot(self):
        """加载R1机器人并完成初始姿态设置。"""
        if not os.path.exists(R1_URDF_PATH):
            print(f"!!! URDF文件未找到: {R1_URDF_PATH}")
            return
        if not os.path.exists(R1_CONFIG_PATH):
            print(f"!!! 机器人配置文件未找到: {R1_CONFIG_PATH}")
            return

        try:
            with open(R1_CONFIG_PATH, "r") as f:
                robot_cfg = json.load(f)

            self.robot = Robot(self.scene, self.need_topp, **robot_cfg)
            self.robot.init_joints()
            print("机器人加载成功")
            self._init_joint_states()
            self._refresh_robot_camera_links(verbose=True)
        except Exception as e:
            self.robot = None
            print(f"加载机器人URDF失败: {e}")
    
    def _init_joint_states(self):
        """初始化双臂、夹爪，并固定R1腰部关节。"""
        if self.robot is None:
            return

        try:
            self.robot.move_to_homestate()
            self._set_fixed_torso_joint_positions(verbose=True)
            self.robot.left_gripper_val = 0.8
            self.robot.right_gripper_val = 0.8
            self.scene.step()
        except Exception as e:
            print(f"初始化关节状态失败: {e}")

    def _set_fixed_torso_joint_positions(self, verbose: bool = False):
        """将腰部关节固定到视频展示用姿态。"""
        if self.robot is None:
            return

        applied = self._apply_torso_targets_to_entity(self.robot.left_entity, verbose=verbose)
        if self.robot.right_entity is not self.robot.left_entity:
            applied = self._apply_torso_targets_to_entity(self.robot.right_entity, verbose=verbose) or applied

        if applied and verbose:
            print(f"固定腰部关节角度: {self.fixed_torso_pos.tolist()}")

    def _apply_torso_targets_to_entity(self, entity, verbose: bool = False) -> bool:
        """对一个articulation应用torso目标角。"""
        active_joints = entity.get_active_joints()
        active_joint_map = {joint.get_name(): (idx, joint) for idx, joint in enumerate(active_joints)}
        qpos = entity.get_qpos()

        applied_count = 0
        missing_joints = []

        for joint_name, target in zip(R1_TORSO_JOINT_NAMES, self.fixed_torso_pos):
            joint_data = active_joint_map.get(joint_name)
            if joint_data is None:
                missing_joints.append(joint_name)
                continue

            idx, joint = joint_data
            target = float(target)
            qpos[idx] = target
            joint.set_drive_target(target)
            joint.set_drive_velocity_target(0.0)
            applied_count += 1

        if applied_count > 0:
            entity.set_qpos(qpos)

        # 腰部关节缺失时最多打印一次，避免逐帧刷屏
        if missing_joints and (verbose or not self._torso_warning_printed):
            print(f"警告: 未找到腰部关节 {missing_joints}")
            self._torso_warning_printed = True

        return applied_count > 0

    def _enforce_fixed_torso(self):
        """在关键步进前后调用，持续保持 torso 固定姿态。"""
        self._set_fixed_torso_joint_positions(verbose=False)

    def _find_robot_link_by_names(self, candidate_names):
        """在当前机器人实体中按候选名称查找link。"""
        if self.robot is None:
            return None

        entities = []
        if hasattr(self.robot, "left_entity") and self.robot.left_entity is not None:
            entities.append(self.robot.left_entity)
        if (
            hasattr(self.robot, "right_entity")
            and self.robot.right_entity is not None
            and self.robot.right_entity is not self.robot.left_entity
        ):
            entities.append(self.robot.right_entity)

        for entity in entities:
            for link_name in candidate_names:
                link = entity.find_link_by_name(link_name)
                if link is not None:
                    return link
        return None

    def _refresh_robot_camera_links(self, verbose=False):
        """刷新R1相机link绑定。"""
        self._head_camera_link = self._find_robot_link_by_names(
            ["zed_link", "head_camera", "head", "camera_link"]
        )
        self._left_wrist_camera_link = self._find_robot_link_by_names(
            ["left_realsense_link", "left_camera", "left_wrist_camera"]
        )
        self._right_wrist_camera_link = self._find_robot_link_by_names(
            ["right_realsense_link", "right_camera", "right_wrist_camera"]
        )

        if self._head_camera_link is not None:
            self._head_link_warning_printed = False
        if self._left_wrist_camera_link is not None:
            self._left_link_warning_printed = False
        if self._right_wrist_camera_link is not None:
            self._right_link_warning_printed = False

        if verbose:
            head_name = self._head_camera_link.get_name() if self._head_camera_link is not None else "None"
            left_name = self._left_wrist_camera_link.get_name() if self._left_wrist_camera_link is not None else "None"
            right_name = self._right_wrist_camera_link.get_name() if self._right_wrist_camera_link is not None else "None"
            print(f"[camera-links] head={head_name}, left={left_name}, right={right_name}")

        if self.link_cam_debug_enable and (verbose or not self._camera_debug_summary_printed):
            print(self.get_link_camera_debug_summary())
            self._camera_debug_summary_printed = True

    def _should_apply_link_cam_debug(self, cam_tag: str) -> bool:
        return self.link_cam_debug_enable and cam_tag in self.link_cam_debug_apply_to

    def _get_axis_mode_rotation(self, mode: str, warn_attr: str = "_axis_mode_warned") -> np.ndarray:
        """
        返回用于相机朝向调试的基准旋转矩阵。
        这些模式用于排查“轴定义不一致”问题。
        """
        mode = str(mode).strip().lower()
        presets = {
            "none": np.eye(3, dtype=np.float64),
            "yaw_p90": Rotation1.from_euler("z", np.deg2rad(90.0)).as_matrix(),
            "yaw_n90": Rotation1.from_euler("z", np.deg2rad(-90.0)).as_matrix(),
            "pitch_p90": Rotation1.from_euler("y", np.deg2rad(90.0)).as_matrix(),
            "pitch_n90": Rotation1.from_euler("y", np.deg2rad(-90.0)).as_matrix(),
            "roll_p90": Rotation1.from_euler("x", np.deg2rad(90.0)).as_matrix(),
            "roll_n90": Rotation1.from_euler("x", np.deg2rad(-90.0)).as_matrix(),
            # 轴交换（保持右手系）
            "swap_xy": np.array([[0, 1, 0], [1, 0, 0], [0, 0, -1]], dtype=np.float64),
            "swap_xz": np.array([[0, 0, 1], [0, -1, 0], [1, 0, 0]], dtype=np.float64),
            "swap_yz": np.array([[-1, 0, 0], [0, 0, 1], [0, 1, 0]], dtype=np.float64),
        }
        if mode not in presets:
            if not hasattr(self, warn_attr):
                print(f"警告: 未知 axis_mode='{mode}'，回退为 none")
                setattr(self, warn_attr, True)
            return presets["none"]
        return presets[mode]

    def _apply_link_camera_debug_transform(self, pose: sapien.Pose, cam_tag: str, verbose: bool = False) -> sapien.Pose:
        """
        对相机link位姿施加调试变换：
        1) 先应用轴模式修正（axis mode）
        2) 再在相机局部坐标系施加旋转(RPY, 度)
        3) 再沿“原始相机朝向”平移（forward/right/up, 米）
        """
        if not self._should_apply_link_cam_debug(cam_tag):
            return pose

        quat_xyzw = np.array([pose.q[1], pose.q[2], pose.q[3], pose.q[0]], dtype=float)
        rot_base = Rotation1.from_quat(quat_xyzw).as_matrix()
        rot_axis_mode = self._get_axis_mode_rotation(self.link_cam_axis_mode, "_link_cam_axis_mode_warned")
        rot_delta = Rotation1.from_euler("xyz", np.deg2rad(self.link_cam_debug_rot_xyz_deg)).as_matrix()
        rot_new = rot_base @ rot_axis_mode @ rot_delta

        # SAPIEN相机局部坐标：X右, Y上, Z后；“前进”对应局部 -Z
        shift_local = np.array(
            [self.link_cam_debug_right, self.link_cam_debug_up, -self.link_cam_debug_forward],
            dtype=float,
        )
        pos_new = pose.p + rot_base @ shift_local

        quat_new_xyzw = Rotation1.from_matrix(rot_new).as_quat()
        quat_new_wxyz = np.array(
            [quat_new_xyzw[3], quat_new_xyzw[0], quat_new_xyzw[1], quat_new_xyzw[2]],
            dtype=np.float32,
        )

        if verbose:
            print(
                f"[camera-debug:{cam_tag}] axis_mode={self.link_cam_axis_mode}, "
                f"rot_xyz_deg={self.link_cam_debug_rot_xyz_deg.tolist()}, "
                f"shift(f,r,u)=({self.link_cam_debug_forward:+.3f}, {self.link_cam_debug_right:+.3f}, {self.link_cam_debug_up:+.3f})"
            )

        return sapien.Pose(pos_new, quat_new_wxyz)

    def get_link_camera_debug_summary(self) -> str:
        if not self.link_cam_debug_enable:
            return "[camera-debug] disabled"

        apply_to = ",".join(sorted(self.link_cam_debug_apply_to))
        rx, ry, rz = self.link_cam_debug_rot_xyz_deg.tolist()
        return (
            "[camera-debug] enabled "
            f"apply_to={apply_to} "
            f"axis_mode={self.link_cam_axis_mode} "
            f"rot_xyz_deg=({rx:+.2f},{ry:+.2f},{rz:+.2f}) "
            f"shift_m(forward,right,up)=({self.link_cam_debug_forward:+.3f},"
            f"{self.link_cam_debug_right:+.3f},{self.link_cam_debug_up:+.3f})"
        )
    
    def _setup_camera(self):
        """设置相机：包括ego视角、observer视角和三个robot link相机"""
        # ========== 第一类：固定视角相机 ==========
        # ego_camera: 原来的第一视角相机（从预处理JSON设置位置）
        self.ego_camera = self.scene.add_camera(
            name="ego_camera",
            width=self.image_width,
            height=self.image_height,
            fovy=np.deg2rad(self.fovy_deg),
            near=0.01,
            far=100.0
        )
        self.ego_camera.set_entity_pose(sapien.Pose([0, 0, 1.6], [0, 0, 0, 1]))
        
        # observer_camera: 原来的第三视角相机（观察者视角）
        self.observer_camera = self.scene.add_camera(
            name="observer_camera",
            width=self.image_width,
            height=self.image_height,
            fovy=np.deg2rad(self.fovy_deg),
            near=0.01,
            far=100.0
        )
        self.observer_camera.set_entity_pose(sapien.Pose([0, 0, 1.6], [0, 0, 0, 1]))
        
        # ========== 第二类：基于robot link的相机 ==========
        # zed_camera: 头部相机（跟随zed_link）
        self.zed_camera = self.scene.add_camera(
            name="zed_camera",
            width=self.image_width,
            height=self.image_height,
            fovy=np.deg2rad(self.fovy_deg),
            near=0.01,
            far=100.0
        )
        self.zed_camera.set_entity_pose(sapien.Pose([0, 0, 1.6], [0, 0, 0, 1]))
        
        # left_wrist_camera: 左手腕相机（跟随left_realsense_link）
        self.left_wrist_camera = self.scene.add_camera(
            name="left_wrist_camera",
            width=self.image_width,
            height=self.image_height,
            fovy=np.deg2rad(self.fovy_deg),
            near=0.01,
            far=100.0
        )
        self.left_wrist_camera.set_entity_pose(sapien.Pose([0.3, 0.2, 1.0], [0, 0, 0, 1]))
        
        # right_wrist_camera: 右手腕相机（跟随right_realsense_link）
        self.right_wrist_camera = self.scene.add_camera(
            name="right_wrist_camera",
            width=self.image_width,
            height=self.image_height,
            fovy=np.deg2rad(self.fovy_deg),
            near=0.01,
            far=100.0
        )
        self.right_wrist_camera.set_entity_pose(sapien.Pose([-0.3, 0.2, 1.0], [0, 0, 0, 1]))
        
        # 为了兼容性，保留原来的camera和third_camera引用
        self.camera = self.ego_camera
        self.third_camera = self.observer_camera
    
    def _create_table(self, height=0.725):
        """创建桌子"""
        builder = self.scene.create_actor_builder()
        builder.add_box_collision(half_size=[0.2, 0.4, height]) # [机器人朝向方向, 机器人左右方向, 桌子高度]
        # builder.add_box_visual(half_size=[0.4, 0.4, 0.725], material=[0.8, 0.8, 0.8])
        builder.add_box_visual(half_size=[0.2, 0.4, height], material=[0.6, 0.4, 0.2])
        self.table = builder.build_kinematic(name="table")
        # 初始位置会在设置机器人基座后更新
        self.table.set_pose(sapien.Pose([0, 0, -0.025]))
        print("桌子创建成功")

    def _load_grasp_data(self):
        """加载anygrasp抓取数据，当前使用frame 0的第2个候选抓取。"""
        try:
            with open(self.grasp_json_path, 'r') as f:
                grasp_data = json.load(f)
            
            # 查找frame 0的数据
            frame_0_data = None
            for frame_data in grasp_data:
                if frame_data['frame'] == 0:
                    frame_0_data = frame_data
                    break
            
            if frame_0_data is None:
                print("警告: 未找到frame 0的抓取数据")
                return
            
            # 当前固定使用frame 0中的第2个抓取候选（索引1）
            grasps = frame_0_data['grasps']
            if not grasps:
                print("警告: frame 0没有抓取数据")
                return
            
            best_grasp = grasps[1] #  max(grasps, key=lambda g: g['score'])
            self.best_grasp = best_grasp
            
            print(f"加载抓取数据成功！")
            print(f"  位置: {best_grasp['translation']}")
            print(f"  宽度: {best_grasp['width']}")
            print(f"  深度: {best_grasp['depth']}")
            print(f"  得分: {best_grasp['score']}")
            
        except Exception as e:
            print(f"加载抓取数据失败: {e}")
            import traceback
            traceback.print_exc()
    
    def set_robot_base_pose_direct(self, robot_base_pos, robot_base_quat_wxyz):
        """
        直接设置机器人基座位置和朝向（使用预处理数据）
        
        Args:
            robot_base_pos: 机器人基座位置 [x, y, z]
            robot_base_quat_wxyz: SAPIEN格式四元数 [w, x, y, z]
        """
        if self.robot is not None:
            robot_base_pos = np.array(robot_base_pos, dtype=float)
            robot_base_quat_wxyz = np.array(robot_base_quat_wxyz, dtype=float)

            # 计算机器人朝向的前方方向（X轴正方向），然后后退0.1m
            quat_xyzw = np.array([robot_base_quat_wxyz[1], robot_base_quat_wxyz[2], robot_base_quat_wxyz[3], robot_base_quat_wxyz[0]])
            rotation_matrix = Rotation1.from_quat(quat_xyzw).as_matrix()
            forward_direction = rotation_matrix @ np.array([1, 0, 0])
            # 沿前方负方向后退0.4m(让相机拍不到)
            robot_base_pos = robot_base_pos  - 0.2 * forward_direction

            base_pose = sapien.Pose(robot_base_pos, robot_base_quat_wxyz)
            self.robot.left_entity.set_root_pose(base_pose)
            self.robot.right_entity.set_root_pose(base_pose)

            self.robot.left_entity_origion_pose = base_pose
            self.robot.right_entity_origion_pose = base_pose

            # 存储机器人基座朝向
            self.robot_base_quat_wxyz = robot_base_quat_wxyz

            # 每次更新基座后，重新施加torso目标角，确保腰部在整段视频中保持固定
            self._enforce_fixed_torso()

            # 设置桌子位置（在机器人前方0.5m处）
            self._update_table_pose(base_pose)

            print(f"机器人基座位置设置为: {robot_base_pos}")
            print(f"机器人朝向四元数(wxyz): {robot_base_quat_wxyz}")
    
    def _update_table_pose(self, robot_base_pose):
        """
        根据机器人基座位置更新桌子位置（放在机器人前方0.5m处）
        
        Args:
            robot_base_pose: 机器人基座的Pose对象
        """
        if not hasattr(self, 'table') or self.table is None:
            return
        
        # 获取机器人基座的位置和朝向
        base_pos = robot_base_pose.p
        base_quat = robot_base_pose.q  # wxyz格式
        
        # 将四元数转换为旋转矩阵
        # SAPIEN使用wxyz格式，scipy使用xyzw格式
        quat_xyzw = np.array([base_quat[1], base_quat[2], base_quat[3], base_quat[0]])
        rotation_matrix = Rotation1.from_quat(quat_xyzw).as_matrix()
        
        # 机器人前方0.5m的方向（通常机器人的前方是X轴正方向）
        forward_direction = rotation_matrix @ np.array([1, 0, 0])
        
        # 计算桌子位置：机器人基座位置 + 前方0.65m
        table_position = base_pos + 0.65 * forward_direction
        
        # 桌子高度：桌面顶部在z=0，所以桌子中心在z=-0.025
        table_position[2] = 0.0
        
        # 桌子使用与机器人相同的朝向
        table_pose = sapien.Pose(table_position, base_quat)
        self.table.set_pose(table_pose)
        
        print(f"桌子位置设置为: {table_position}")

    def set_camera_pose_direct(self, camera_position, camera_rotation_matrix):
        """
        直接设置相机位姿（使用预处理数据）
        
        Args:
            camera_position: 相机位置 [x, y, z]
            camera_rotation_matrix: 相机旋转矩阵 (3x3)
        """
        camera_position = np.array(camera_position, dtype=float)
        camera_rotation_matrix = np.array(camera_rotation_matrix, dtype=float)
        
        # 应用世界Z偏移和相机X偏移
        cam_pos_shifted = camera_position.copy()
        cam_pos_shifted[2] += self.world_z_offset
        cam_pos_shifted[0] += self.camera_x_offset
        
        # sapien相机坐标系转换：
        # JSON中的旋转矩阵需要转换到SAPIEN相机坐标系
        # SAPIEN相机: X右, Y上, Z后
        # 需要应用Y轴270度和X轴90度旋转
        R_y_270 = np.array([[0, 0, -1],
                            [0, 1, 0],
                            [1, 0, 0]], dtype=np.float64)
        R_x_90 = np.array([[1, 0, 0],
                           [0, 0, -1],
                           [0, 1, 0]], dtype=np.float64)
        
        rotation_matrix = camera_rotation_matrix @ R_y_270 @ R_x_90
        
        # 转换为四元数
        quat_xyzw = Rotation1.from_matrix(rotation_matrix).as_quat()
        sapien_quat_wxyz = np.array([quat_xyzw[3], *quat_xyzw[:3]], dtype=np.float32)
        
        # 设置ego_camera位姿
        camera_pose = sapien.Pose(cam_pos_shifted, sapien_quat_wxyz)
        self.ego_camera.set_pose(camera_pose)
        
        # 更新observer_camera的位置和朝向
        self._update_observer_camera_pose(camera_pose)
        
        # 为了兼容性
        self.camera.set_pose(camera_pose)
        
        print(f"Ego相机位置设置为: {cam_pos_shifted}")
    
    def _update_observer_camera_pose(self, ego_camera_pose):
        """
        根据ego_camera位置更新observer_camera位置和朝向
        
        Args:
            ego_camera_pose: ego_camera的Pose对象
        """
        if not hasattr(self, 'observer_camera') or self.observer_camera is None:
            return
        
        # 获取ego_camera的位置
        ego_cam_pos = ego_camera_pose.p
        ego_cam_quat = ego_camera_pose.q  # wxyz格式
        
        # 计算机器人前方方向
        if self.robot_base_quat_wxyz is not None:
            # 将机器人基座四元数转换为旋转矩阵
            robot_quat_xyzw = np.array([self.robot_base_quat_wxyz[1], 
                                        self.robot_base_quat_wxyz[2], 
                                        self.robot_base_quat_wxyz[3], 
                                        self.robot_base_quat_wxyz[0]])
            robot_rotation_matrix = Rotation1.from_quat(robot_quat_xyzw).as_matrix()
            
            # 机器人前方方向（X轴正方向）
            forward_direction = robot_rotation_matrix @ np.array([1, 0, 0])
            
            # observer_camera位置：ego_camera位置 + 沿机器人朝向前方1m
            observer_cam_pos = ego_cam_pos + 1.0 * forward_direction
        else:
            # 如果没有机器人基座朝向，使用ego_camera的朝向的前方方向
            ego_quat_xyzw = np.array([ego_cam_quat[1], ego_cam_quat[2], ego_cam_quat[3], ego_cam_quat[0]])
            ego_rotation_matrix = Rotation1.from_quat(ego_quat_xyzw).as_matrix()
            forward_direction = ego_rotation_matrix @ np.array([1, 0, 0])
            observer_cam_pos = ego_cam_pos + 1.0 * forward_direction
        
        # 计算observer_camera的朝向：ego_camera绕世界坐标系z轴转180度
        # 世界坐标系z轴转180度的旋转矩阵
        R_z_180 = np.array([
            [-1, 0, 0],
            [0, -1, 0],
            [0, 0, 1]
        ], dtype=np.float64)
        
        # 获取ego_camera的旋转矩阵
        ego_quat_xyzw = np.array([ego_cam_quat[1], ego_cam_quat[2], ego_cam_quat[3], ego_cam_quat[0]])
        ego_rotation_matrix = Rotation1.from_quat(ego_quat_xyzw).as_matrix()
        
        # 绕z轴转180度：先旋转ego_camera的旋转矩阵，然后应用z轴180度旋转
        # 注意：绕世界坐标系z轴旋转，应该左乘R_z_180
        observer_cam_rotation_matrix = R_z_180 @ ego_rotation_matrix
        
        # 转换为四元数
        observer_quat_xyzw = Rotation1.from_matrix(observer_cam_rotation_matrix).as_quat()
        observer_quat_wxyz = np.array([observer_quat_xyzw[3], observer_quat_xyzw[0], observer_quat_xyzw[1], observer_quat_xyzw[2]], dtype=np.float32)
        
        # 设置observer_camera位姿
        observer_cam_pose = sapien.Pose(observer_cam_pos, observer_quat_wxyz)
        self.observer_camera.set_pose(observer_cam_pose)
        
        # 为了兼容性，也更新third_camera
        self.third_camera.set_pose(observer_cam_pose)
        
        print(f"Observer相机位置设置为: {observer_cam_pos}")
    
    def update_robot_link_cameras(self, verbose=False):
        """
        更新基于robot link的相机位姿（zed_camera, left_wrist_camera, right_wrist_camera）
        这些相机跟随机器人的特定link移动
        """
        if self.robot is None:
            return
        
        try:
            # 缓存为空时尝试重新绑定，避免依赖Robot通用实现中的相机字段
            if (
                self._head_camera_link is None
                or self._left_wrist_camera_link is None
                or self._right_wrist_camera_link is None
            ):
                self._refresh_robot_camera_links(verbose=verbose)

            # 更新zed_camera（头部相机）
            if self._head_camera_link is not None:
                head_pose = self._head_camera_link.get_pose()
                head_pose = self._apply_link_camera_debug_transform(head_pose, "head", verbose=verbose)
                if verbose:
                    print(f"\033[92m  [update] head cam {head_pose} \033[0m  ")
                self.zed_camera.set_pose(head_pose)
            elif not self._head_link_warning_printed:
                print("警告: 未找到头部相机link（期望: zed_link）")
                self._head_link_warning_printed = True
            
            # 更新left_wrist_camera
            if self._left_wrist_camera_link is not None:
                left_camera_link_pose = self._left_wrist_camera_link.get_pose()
                left_camera_link_pose = self._apply_link_camera_debug_transform(left_camera_link_pose, "left", verbose=verbose)
                if verbose:
                    print(f"\033[92m  [update] left wrist cam {left_camera_link_pose} \033[0m  ")
                self.left_wrist_camera.set_pose(left_camera_link_pose)
            elif not self._left_link_warning_printed:
                print("警告: 未找到左腕相机link（期望: left_realsense_link）")
                self._left_link_warning_printed = True
            
            # 更新right_wrist_camera
            if self._right_wrist_camera_link is not None:
                right_camera_link_pose = self._right_wrist_camera_link.get_pose()
                right_camera_link_pose = self._apply_link_camera_debug_transform(right_camera_link_pose, "right", verbose=verbose)
                if verbose:
                    print(f"\033[92m  [update] right wrist cam {right_camera_link_pose} \033[0m  ")
                self.right_wrist_camera.set_pose(right_camera_link_pose)
            elif not self._right_link_warning_printed:
                print("警告: 未找到右腕相机link（期望: right_realsense_link）")
                self._right_link_warning_printed = True
                
        except Exception as e:
            print(f"更新robot link相机位姿失败: {e}")
            import traceback
            traceback.print_exc()

    def _load_model(self, depth_scale=1.0):
        """
        加载三维模型并添加碰撞形状用于抓取检测
        
        Args:
            depth_scale: 深度缩放因子，调整深度时物体尺寸也应相应缩放
                        根据相机投影原理：depth * scale = constant
                        所以 size_scale = depth_scale
        """
        try:
            if not os.path.exists(self.model_path):
                print(f"模型文件不存在: {self.model_path}")
                return
            
            if not os.path.exists(self.poses_path):
                print(f"位姿文件不存在: {self.poses_path}")
                return
            
            # 加载位姿数据
            with open(self.poses_path, 'r') as f:
                self.model_poses = json.load(f)
            
            print(f"加载了 {len(self.model_poses)} 个模型位姿")
            
            # 使用trimesh加载OBJ模型获取信息
            mesh = trimesh.load(self.model_path)
            
            # 应用深度缩放：深度变为k倍，尺寸也变为k倍（保持2D投影大小不变）
            if depth_scale != 1.0:
                mesh.apply_scale(depth_scale)
                print(f"应用深度缩放因子: {depth_scale}")
            
            print(f"模型信息: 顶点数={len(mesh.vertices)}, 面数={len(mesh.faces)}")
            
            # 创建SAPIEN actor - 使用简单的方法
            builder = self.scene.create_actor_builder()
            
            # 添加真实的3D模型作为视觉形状
            try:
                # 尝试直接加载OBJ文件作为视觉形状
                builder.add_visual_from_file(self.model_path)
                print(f"成功加载3D模型: {self.model_path}")
            except Exception as e:
                print(f"无法直接加载OBJ文件，使用简化几何体: {e}")
                # 如果无法直接加载，使用边界框作为替代
                if hasattr(mesh, 'bounds') and mesh.bounds is not None:
                    bounds = mesh.bounds
                    size = bounds[1] - bounds[0]
                    # 限制最大尺寸
                    max_size = 0.5
                    size = np.minimum(size, max_size)
                    print(f"使用边界框替代，尺寸: {size}")
                    builder.add_box_visual()
                else:
                    builder.add_box_visual()
            
            # *** 添加碰撞形状用于抓取检测 ***
            try:
                # 尝试使用凸包作为碰撞形状
                builder.add_convex_collision_from_file(self.model_path)
                print("成功添加凸包碰撞形状用于抓取检测")
            except Exception as e:
                print(f"无法使用凸包碰撞，使用边界框替代: {e}")
                # 使用边界框作为碰撞形状
                if hasattr(mesh, 'bounds') and mesh.bounds is not None:
                    bounds = mesh.bounds
                    center = (bounds[0] + bounds[1]) / 2
                    size = (bounds[1] - bounds[0]) / 2
                    # 限制最大尺寸
                    max_size = 0.25
                    size = np.minimum(size, max_size)
                    builder.add_box_collision(half_size=size, pose=sapien.Pose(center))
                    print(f"使用边界框作为碰撞形状，半尺寸: {size}, 中心: {center}")
                else:
                    # 使用默认尺寸的盒子
                    builder.add_box_collision(half_size=[0.05, 0.05, 0.05])
                    print("使用默认盒子作为碰撞形状")
            
            # 创建为kinematic actor（不受重力影响，但可以检测碰撞）
            self.model_actor = builder.build_kinematic(name="target_object")
            
            print(f"模型加载成功（带碰撞形状用于抓取检测）: {self.model_path}")
            
        except Exception as e:
            print(f"加载模型时出错: {e}")
            import traceback
            traceback.print_exc()

    def _load_model2(self):
        """加载第二个三维模型并设置忽略碰撞"""
        try:
            if not os.path.exists(self.model_path2):
                print(f"第二个模型文件不存在: {self.model_path2}")
                return
            
            if not os.path.exists(self.poses_path2):
                print(f"第二个位姿文件不存在: {self.poses_path2}")
                return
            
            # 加载位姿数据
            with open(self.poses_path2, 'r') as f:
                self.model_poses2 = json.load(f)
            
            print(f"加载了 {len(self.model_poses2)} 个第二个模型位姿")
            
            # 使用trimesh加载OBJ模型获取信息
            mesh = trimesh.load(self.model_path2)
            print(f"第二个模型信息: 顶点数={len(mesh.vertices)}, 面数={len(mesh.faces)}")
            
            # 创建SAPIEN actor - 使用简单的方法
            builder = self.scene.create_actor_builder()
            
            # 添加真实的3D模型作为视觉形状
            try:
                # 尝试直接加载OBJ文件作为视觉形状
                builder.add_visual_from_file(self.model_path2)
                print(f"成功加载第二个3D模型: {self.model_path2}")
            except Exception as e:
                print(f"无法直接加载OBJ文件，使用简化几何体: {e}")
                # 如果无法直接加载，使用边界框作为替代
                if hasattr(mesh, 'bounds') and mesh.bounds is not None:
                    bounds = mesh.bounds
                    size = bounds[1] - bounds[0]
                    # 限制最大尺寸
                    max_size = 0.5
                    size = np.minimum(size, max_size)
                    print(f"使用边界框替代，尺寸: {size}")
                    builder.add_box_visual()
                else:
                    builder.add_box_visual()
            
            # 不添加碰撞形状（忽略碰撞）
            # 注意：这里不添加任何碰撞形状，所以模型不会参与物理碰撞
            
            # 创建actor
            self.model_actor2 = builder.build_kinematic()
            
            print(f"第二个模型加载成功: {self.model_path2}")
            
        except Exception as e:
            print(f"加载第二个模型时出错: {e}")
            import traceback
            traceback.print_exc()

    def initialize_fixed_model_pose(self, frame_idx=0, table_height=0.0):
        """
        初始化固定的模型位置（只使用第一帧，并放在桌面上）
        
        Args:
            frame_idx: 使用哪一帧的位姿（默认第一帧）
            table_height: 桌面高度
        """
        if self.model_actor is None or self.model_poses is None:
            print("模型或位姿数据未加载")
            return
        
        if str(frame_idx) not in self.model_poses:
            print(f"帧 {frame_idx} 的位姿数据不存在")
            return
        
        try:
            # 获取相机坐标系下的位姿矩阵
            cam_T_obj = np.array(self.model_poses[str(frame_idx)])
            
            # 应用深度调整
            # 深度在相机坐标系的Z轴方向
            cam_T_obj[:3, 3] = cam_T_obj[:3, 3] * self.depth_scale + np.array([0, 0, self.depth_offset])
            
            # 相机坐标系到世界坐标系的转换
            camera_pose = self.camera.get_entity_pose()
            camera_position = camera_pose.p
            camera_quat = camera_pose.q
            
            # 构建相机在世界坐标系下的变换矩阵
            camera_rotation_matrix = Rotation1.from_quat([camera_quat[1], camera_quat[2], camera_quat[3], camera_quat[0]]).as_matrix()
            R_offset_inv = np.array([
                [0, 0, 1],
                [-1, 0, 0],
                [0, -1, 0]
            ])
            camera_rotation_matrix = camera_rotation_matrix @ R_offset_inv

            world_T_cam = np.eye(4)
            world_T_cam[:3, :3] = camera_rotation_matrix
            world_T_cam[:3, 3] = camera_position

            # 转换到世界坐标系
            world_T_obj = world_T_cam @ cam_T_obj

            # 提取位置和旋转
            position = world_T_obj[:3, 3]
            rotation_matrix = world_T_obj[:3, :3]
            
            # 将物体放在桌面上：调整Z坐标
            # 假设物体中心到底部的距离
            if hasattr(self, 'table') and self.table is not None:
                table_pose = self.table.get_pose()
                table_top_z = table_pose.p[2] + 0.05  # 桌面顶部
                # 简单处理：将物体底部放在桌面上
                position[2] = max(position[2], table_top_z + 0.02)  # 至少高于桌面2cm
            
            # 转换为四元数
            quat_xyzw = Rotation1.from_matrix(rotation_matrix).as_quat()
            sapien_quat_wxyz = np.array([quat_xyzw[3], *quat_xyzw[:3]], dtype=np.float32)
            
            # 保存固定位姿
            self.fixed_model_pose = sapien.Pose(position, sapien_quat_wxyz)
            
            # 设置模型位姿
            self.model_actor.set_pose(self.fixed_model_pose)
            
            print(f"初始化固定模型位姿: 世界位置={position}, depth_scale={self.depth_scale}, depth_offset={self.depth_offset}")
            
        except Exception as e:
            print(f"初始化固定模型位姿时出错: {e}")
            import traceback
            traceback.print_exc()
    
    def set_model_pose_from_camera_coords(self, frame_idx):
        """
        根据相机坐标系下的位姿设置模型位置（相机坐标系到世界坐标系转换）
        按照JSON数据每帧更新model1的位置
        
        Args:
            frame_idx: 帧索引
        """
        # 注释掉固定位姿的限制，允许模型每帧按JSON数据更新
        # if self.fixed_model_pose is not None:
        #     self.model_actor.set_pose(self.fixed_model_pose)
        #     return
        
        if self.model_actor is None or self.model_poses is None:
            print("模型或位姿数据未加载")
            return
        
        if str(frame_idx) not in self.model_poses:
            print(f"帧 {frame_idx} 的位姿数据不存在")
            return
        
        try:
            # 获取相机坐标系下的位姿矩阵 (camera->object变换)
            cam_T_obj = np.array(self.model_poses[str(frame_idx)])
            print(f"cam_T_obj: {cam_T_obj}")
            
            # 相机坐标系到世界坐标系的转换
            # 需要获取当前相机的世界位姿
            camera_pose = self.camera.get_entity_pose() # 得到的是wxyz但是X_new = Z; y_new=-X ; z_new=-Y
            print(f"camera_pose: {camera_pose}")
            camera_position = camera_pose.p
            camera_quat = camera_pose.q  # wxyz格式
            
            # 构建相机在世界坐标系下的变换矩阵
            camera_rotation_matrix = Rotation1.from_quat([camera_quat[1], camera_quat[2], camera_quat[3], camera_quat[0]]).as_matrix()
            print(f"camera_rotation_matrix: {camera_rotation_matrix}")
            R_offset_inv = np.array([
                [0, 0, 1],
                [-1, 0, 0],
                [0, -1, 0]
            ])
            # 3. 右乘修正矩阵，得到"正确"的矩阵
            camera_rotation_matrix = camera_rotation_matrix @ R_offset_inv
            print(f"camera_rotation_matrix: {camera_rotation_matrix}")

            world_T_cam = np.eye(4)
            world_T_cam[:3, :3] = camera_rotation_matrix
            world_T_cam[:3, 3] = camera_position
            print(f"world_T_cam: {world_T_cam}")

            # 转换公式: world_T_obj = world_T_cam @ cam_T_obj
            world_T_obj = world_T_cam @ cam_T_obj
            print(f"world_T_obj: {world_T_obj}")

            # 提取位置和旋转
            position = world_T_obj[:3, 3]
            rotation_matrix = world_T_obj[:3, :3]
            
            # 转换为四元数
            quat_xyzw = Rotation1.from_matrix(rotation_matrix).as_quat()
            sapien_quat_wxyz = np.array([quat_xyzw[3], *quat_xyzw[:3]], dtype=np.float32)
            
            # 设置模型位姿
            print(f"model_pose: {position}")
            # position = np.array([0.4341698884963989, -0.06660800985991955, 1.0013048648834229], dtype=float)
            model_pose = sapien.Pose(position, sapien_quat_wxyz)
            self.model_actor.set_pose(model_pose)
            
            print(f"设置模型位姿 (帧 {frame_idx}): 世界位置={position}")
            
        except Exception as e:
            print(f"设置模型位姿时出错: {e}")
            import traceback
            traceback.print_exc()

    def set_model_pose_from_camera_coords2(self, frame_idx):
        """
        根据相机坐标系下的位姿设置第二个模型位置（相机坐标系到世界坐标系转换）
        
        Args:
            frame_idx: 帧索引
        """
        if self.model_actor2 is None or self.model_poses2 is None:
            print("第二个模型或位姿数据未加载")
            return
        
        if str(frame_idx) not in self.model_poses2:
            print(f"帧 {frame_idx} 的第二个模型位姿数据不存在")
            return
        
        try:
            # 获取相机坐标系下的位姿矩阵 (camera->object变换)
            cam_T_obj = np.array(self.model_poses2[str(frame_idx)])
            print(f"cam_T_obj (model2): {cam_T_obj}")
            
            # 相机坐标系到世界坐标系的转换
            # 需要获取当前相机的世界位姿
            camera_pose = self.camera.get_entity_pose() # 得到的是wxyz但是X_new = Z; y_new=-X ; z_new=-Y
            camera_position = camera_pose.p
            camera_quat = camera_pose.q  # wxyz格式
            
            # 构建相机在世界坐标系下的变换矩阵
            camera_rotation_matrix = Rotation1.from_quat([camera_quat[1], camera_quat[2], camera_quat[3], camera_quat[0]]).as_matrix()
            R_offset_inv = np.array([
                [0, 0, 1],
                [-1, 0, 0],
                [0, -1, 0]
            ])
            # 3. 右乘修正矩阵，得到"正确"的矩阵
            camera_rotation_matrix = camera_rotation_matrix @ R_offset_inv

            world_T_cam = np.eye(4)
            world_T_cam[:3, :3] = camera_rotation_matrix
            world_T_cam[:3, 3] = camera_position

            # 转换公式: world_T_obj = world_T_cam @ cam_T_obj
            world_T_obj = world_T_cam @ cam_T_obj

            # 提取位置和旋转
            position = world_T_obj[:3, 3]
            rotation_matrix = world_T_obj[:3, :3]
            
            # 转换为四元数
            quat_xyzw = Rotation1.from_matrix(rotation_matrix).as_quat()
            sapien_quat_wxyz = np.array([quat_xyzw[3], *quat_xyzw[:3]], dtype=np.float32)
            
            # 设置模型位姿
            model_pose = sapien.Pose(position, sapien_quat_wxyz)
            self.model_actor2.set_pose(model_pose)
            
            print(f"设置第二个模型位姿 (帧 {frame_idx}): 世界位置={position}")
            
        except Exception as e:
            print(f"设置第二个模型位姿时出错: {e}")
            import traceback
            traceback.print_exc()

    def _test_single_orientation(self, left_wrist_pos, left_wrist_rot_matrix,
                                 right_wrist_pos, right_wrist_rot_matrix,
                                 euler_angles, test_idx, test_frame_idx,
                                 test_output_dir, test_results):
        """
        测试单个朝向，尝试规划路径并渲染图片
        
        Args:
            left_wrist_pos: 左手腕位置
            left_wrist_rot_matrix: 左手腕旋转矩阵
            right_wrist_pos: 右手腕位置
            right_wrist_rot_matrix: 右手腕旋转矩阵
            euler_angles: 当前测试的欧拉角 [roll, pitch, yaw]
            test_idx: 测试索引
            test_frame_idx: 帧索引
            test_output_dir: 输出目录
            test_results: 结果列表（用于记录）
        """
        euler_degrees = np.degrees(euler_angles)
        left_status = "Unknown"
        right_status = "Unknown"
        is_success = False
        error_msg = ""
        
        # ========================================================================
        # 关键修复：保存机器人当前关节状态，测试后恢复
        # ========================================================================
        # 保存左臂关节状态
        left_qpos = self.robot.left_entity.get_qpos()
        left_qvel = self.robot.left_entity.get_qvel()
        # 保存右臂关节状态
        right_qpos = self.robot.right_entity.get_qpos()
        right_qvel = self.robot.right_entity.get_qvel()
        
        print(f"  [保存初始状态] 左臂qpos: {left_qpos[:3]}..., 右臂qpos: {right_qpos[:3]}...")
        
        try:
            left_pos = np.array(left_wrist_pos, dtype=float).flatten()[:3]
            right_pos = np.array(right_wrist_pos, dtype=float).flatten()[:3]
            
            left_rot_matrix = np.array(left_wrist_rot_matrix, dtype=float).reshape(3, 3)
            right_rot_matrix = np.array(right_wrist_rot_matrix, dtype=float).reshape(3, 3)
            
            # 转换为四元数
            left_quat_xyzw = Rotation1.from_matrix(left_rot_matrix).as_quat()
            right_quat_xyzw = Rotation1.from_matrix(right_rot_matrix).as_quat()
            
            # 转换为wxyz格式
            left_quat_wxyz = np.array([left_quat_xyzw[3], left_quat_xyzw[0], left_quat_xyzw[1], left_quat_xyzw[2]])
            right_quat_wxyz = np.array([right_quat_xyzw[3], right_quat_xyzw[0], right_quat_xyzw[1], right_quat_xyzw[2]])
            
            left_target_pose = np.concatenate([left_pos, left_quat_wxyz])
            right_target_pose = np.concatenate([right_pos, right_quat_wxyz])
            
            print(f"  左臂目标: {left_target_pose}")
            print(f"  右臂目标: {right_target_pose}")
            
            # 尝试路径规划
            left_plan_result = self.robot.left_plan_path(
                target_pose=left_target_pose.tolist(),
                constraint_pose=None,
                use_point_cloud=False,
                use_attach=False
            )
            
            right_plan_result = self.robot.right_plan_path(
                target_pose=right_target_pose.tolist(),
                constraint_pose=None,
                use_point_cloud=False,
                use_attach=False
            )
            
            left_status = left_plan_result.get('status', 'Unknown')
            right_status = right_plan_result.get('status', 'Unknown')
            
            print(f"  左臂规划: {left_status}")
            print(f"  右臂规划: {right_status}")
            
            # 执行规划 - 只要有一侧成功就执行
            left_moved = False
            right_moved = False
            
            if left_status == "Success":
                left_path = left_plan_result.get('position', [])
                if len(left_path) > 0:
                    left_target_joints = left_path[-1]
                    self.robot.set_arm_joints(
                        target_position=left_target_joints,
                        target_velocity=[0.0] * len(left_target_joints),
                        arm_tag="left"
                    )
                    left_moved = True
                    print(f"  ✓ 左臂移动命令已发送")
            else:
                print(f"  ✗ 左臂未移动（规划失败）")
            
            if right_status == "Success":
                right_path = right_plan_result.get('position', [])
                if len(right_path) > 0:
                    right_target_joints = right_path[-1]
                    self.robot.set_arm_joints(
                        target_position=right_target_joints,
                        target_velocity=[0.0] * len(right_target_joints),
                        arm_tag="right"
                    )
                    right_moved = True
                    print(f"  ✓ 右臂移动命令已发送")
            else:
                print(f"  ✗ 右臂未移动（规划失败）")
            
            # 推进仿真 - 让移动完成
            if left_moved or right_moved:
                print(f"  推进仿真让机器人移动...")
                self._enforce_fixed_torso()
                for _ in range(1000):
                    self._enforce_fixed_torso()
                    self.scene.step()
                self._enforce_fixed_torso()
                print(f"  仿真完成")
            
            # 检查是否两侧都成功
            if left_status != "Success" or right_status != "Success":
                error_msg = f"路径规划失败 - 左臂: {left_status}, 右臂: {right_status}"
                is_success = False
                raise Exception(error_msg)
            
            is_success = True
            print(f"\n✓ 规划成功！机器人已移动到目标姿态")
            
        except Exception as e:
            error_msg = str(e)
            print(f"\n✗ 规划失败: {error_msg}")
        
        # 无论成功或失败，都渲染并保存当前状态
        try:
            # 更新robot link相机的位置（先更新再渲染）
            self.update_robot_link_cameras()
            
            # 渲染第一视角（ego camera）
            rgb_image, depth_image = self.render_frame(test_frame_idx)
            
            # 保存第一视角图片
            status_label = "SUCCESS" if is_success else "FAIL"
            filename = f"test_{test_idx:02d}_{status_label}_rx{euler_degrees[0]:+07.1f}_ry{euler_degrees[1]:+07.1f}_rz{euler_degrees[2]:+07.1f}_ego.png"
            filepath = os.path.join(test_output_dir, filename)
            
            bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
            
            # 在图片上添加信息
            font = cv2.FONT_HERSHEY_SIMPLEX
            # 第一行：视角标签
            text0 = "View: Ego Camera (First Person)"
            cv2.putText(bgr_image, text0, (10, 25), font, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(bgr_image, text0, (10, 25), font, 0.5, (255, 200, 0), 1, cv2.LINE_AA)
            
            # 第二行：欧拉角
            text1 = f"Euler(xyz): [{euler_degrees[0]:+.1f}, {euler_degrees[1]:+.1f}, {euler_degrees[2]:+.1f}]"
            cv2.putText(bgr_image, text1, (10, 50), font, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(bgr_image, text1, (10, 50), font, 0.5, (0, 255, 0) if is_success else (0, 0, 255), 1, cv2.LINE_AA)
            
            # 第三行：规划状态
            text2 = f"Left: {left_status} | Right: {right_status}"
            cv2.putText(bgr_image, text2, (10, 75), font, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(bgr_image, text2, (10, 75), font, 0.5, (0, 255, 0) if is_success else (0, 0, 255), 1, cv2.LINE_AA)
            
            # 第四行：总体状态
            text3 = f"Status: {status_label}"
            color = (0, 255, 0) if is_success else (0, 0, 255)
            cv2.putText(bgr_image, text3, (10, 100), font, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(bgr_image, text3, (10, 100), font, 0.6, color, 1, cv2.LINE_AA)
            
            cv2.imwrite(filepath, bgr_image)
            print(f"  第一视角图片已保存: {filename}")
            
            # 渲染第三视角（observer camera）
            if hasattr(self, 'third_person_view') and self.third_person_view and hasattr(self, 'observer_camera') and self.observer_camera is not None:
                observer_rgb_image, observer_depth_image = self.render_third_camera_frame(test_frame_idx)
                
                # 保存第三视角图片
                observer_filename = f"test_{test_idx:02d}_{status_label}_rx{euler_degrees[0]:+07.1f}_ry{euler_degrees[1]:+07.1f}_rz{euler_degrees[2]:+07.1f}_observer.png"
                observer_filepath = os.path.join(test_output_dir, observer_filename)
                
                observer_bgr_image = cv2.cvtColor(observer_rgb_image, cv2.COLOR_RGB2BGR)
                
                # 在第三视角图片上添加信息
                text0_observer = "View: Observer Camera (Third Person)"
                cv2.putText(observer_bgr_image, text0_observer, (10, 25), font, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(observer_bgr_image, text0_observer, (10, 25), font, 0.5, (255, 200, 0), 1, cv2.LINE_AA)
                
                cv2.putText(observer_bgr_image, text1, (10, 50), font, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(observer_bgr_image, text1, (10, 50), font, 0.5, (0, 255, 0) if is_success else (0, 0, 255), 1, cv2.LINE_AA)
                
                cv2.putText(observer_bgr_image, text2, (10, 75), font, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(observer_bgr_image, text2, (10, 75), font, 0.5, (0, 255, 0) if is_success else (0, 0, 255), 1, cv2.LINE_AA)
                
                cv2.putText(observer_bgr_image, text3, (10, 100), font, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(observer_bgr_image, text3, (10, 100), font, 0.6, color, 1, cv2.LINE_AA)
                
                cv2.imwrite(observer_filepath, observer_bgr_image)
                print(f"  第三视角图片已保存: {observer_filename}")
            else:
                observer_filename = None
            
            # 渲染zed camera（头部相机）
            if hasattr(self, 'zed_camera') and self.zed_camera is not None:
                zed_rgb_image, zed_depth_image = self.render_zed_camera_frame(test_frame_idx)
                
                # 保存zed camera图片
                zed_filename = f"test_{test_idx:02d}_{status_label}_rx{euler_degrees[0]:+07.1f}_ry{euler_degrees[1]:+07.1f}_rz{euler_degrees[2]:+07.1f}_zed.png"
                zed_filepath = os.path.join(test_output_dir, zed_filename)
                
                zed_bgr_image = cv2.cvtColor(zed_rgb_image, cv2.COLOR_RGB2BGR)
                
                # 在zed camera图片上添加信息
                text0_zed = "View: ZED Camera (Head Link)"
                cv2.putText(zed_bgr_image, text0_zed, (10, 25), font, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(zed_bgr_image, text0_zed, (10, 25), font, 0.5, (255, 200, 0), 1, cv2.LINE_AA)
                
                cv2.putText(zed_bgr_image, text1, (10, 50), font, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(zed_bgr_image, text1, (10, 50), font, 0.5, (0, 255, 0) if is_success else (0, 0, 255), 1, cv2.LINE_AA)
                
                cv2.putText(zed_bgr_image, text2, (10, 75), font, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(zed_bgr_image, text2, (10, 75), font, 0.5, (0, 255, 0) if is_success else (0, 0, 255), 1, cv2.LINE_AA)
                
                cv2.putText(zed_bgr_image, text3, (10, 100), font, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(zed_bgr_image, text3, (10, 100), font, 0.6, color, 1, cv2.LINE_AA)
                
                cv2.imwrite(zed_filepath, zed_bgr_image)
                print(f"  ZED相机图片已保存: {zed_filename}")
            else:
                zed_filename = None
            
            # 渲染left_wrist_camera（左手腕相机）
            if hasattr(self, 'left_wrist_camera') and self.left_wrist_camera is not None:
                left_wrist_rgb_image, left_wrist_depth_image = self.render_left_wrist_camera_frame(test_frame_idx)
                
                # 保存left_wrist_camera图片
                left_wrist_filename = f"test_{test_idx:02d}_{status_label}_rx{euler_degrees[0]:+07.1f}_ry{euler_degrees[1]:+07.1f}_rz{euler_degrees[2]:+07.1f}_left_wrist.png"
                left_wrist_filepath = os.path.join(test_output_dir, left_wrist_filename)
                
                left_wrist_bgr_image = cv2.cvtColor(left_wrist_rgb_image, cv2.COLOR_RGB2BGR)
                
                # 在left_wrist_camera图片上添加信息
                text0_left_wrist = "View: Left Wrist Camera (left_realsense_link)"
                cv2.putText(left_wrist_bgr_image, text0_left_wrist, (10, 25), font, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(left_wrist_bgr_image, text0_left_wrist, (10, 25), font, 0.5, (255, 200, 0), 1, cv2.LINE_AA)
                
                cv2.putText(left_wrist_bgr_image, text1, (10, 50), font, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(left_wrist_bgr_image, text1, (10, 50), font, 0.5, (0, 255, 0) if is_success else (0, 0, 255), 1, cv2.LINE_AA)
                
                cv2.putText(left_wrist_bgr_image, text2, (10, 75), font, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(left_wrist_bgr_image, text2, (10, 75), font, 0.5, (0, 255, 0) if is_success else (0, 0, 255), 1, cv2.LINE_AA)
                
                cv2.putText(left_wrist_bgr_image, text3, (10, 100), font, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(left_wrist_bgr_image, text3, (10, 100), font, 0.6, color, 1, cv2.LINE_AA)
                
                cv2.imwrite(left_wrist_filepath, left_wrist_bgr_image)
                print(f"  左手腕相机图片已保存: {left_wrist_filename}")
            else:
                left_wrist_filename = None
            
            # 渲染right_wrist_camera（右手腕相机）
            if hasattr(self, 'right_wrist_camera') and self.right_wrist_camera is not None:
                right_wrist_rgb_image, right_wrist_depth_image = self.render_right_wrist_camera_frame(test_frame_idx)
                
                # 保存right_wrist_camera图片
                right_wrist_filename = f"test_{test_idx:02d}_{status_label}_rx{euler_degrees[0]:+07.1f}_ry{euler_degrees[1]:+07.1f}_rz{euler_degrees[2]:+07.1f}_right_wrist.png"
                right_wrist_filepath = os.path.join(test_output_dir, right_wrist_filename)
                
                right_wrist_bgr_image = cv2.cvtColor(right_wrist_rgb_image, cv2.COLOR_RGB2BGR)
                
                # 在right_wrist_camera图片上添加信息
                text0_right_wrist = "View: Right Wrist Camera (right_realsense_link)"
                cv2.putText(right_wrist_bgr_image, text0_right_wrist, (10, 25), font, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(right_wrist_bgr_image, text0_right_wrist, (10, 25), font, 0.5, (255, 200, 0), 1, cv2.LINE_AA)
                
                cv2.putText(right_wrist_bgr_image, text1, (10, 50), font, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(right_wrist_bgr_image, text1, (10, 50), font, 0.5, (0, 255, 0) if is_success else (0, 0, 255), 1, cv2.LINE_AA)
                
                cv2.putText(right_wrist_bgr_image, text2, (10, 75), font, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(right_wrist_bgr_image, text2, (10, 75), font, 0.5, (0, 255, 0) if is_success else (0, 0, 255), 1, cv2.LINE_AA)
                
                cv2.putText(right_wrist_bgr_image, text3, (10, 100), font, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(right_wrist_bgr_image, text3, (10, 100), font, 0.6, color, 1, cv2.LINE_AA)
                
                cv2.imwrite(right_wrist_filepath, right_wrist_bgr_image)
                print(f"  右手腕相机图片已保存: {right_wrist_filename}")
            else:
                right_wrist_filename = None
            
        except Exception as render_error:
            print(f"  ⚠ 渲染图片失败: {render_error}")
            import traceback
            traceback.print_exc()
            filename = f"test_{test_idx:02d}_RENDER_ERROR.png"
            observer_filename = None
            zed_filename = None
            left_wrist_filename = None
            right_wrist_filename = None
        
        # ========================================================================
        # 关键修复：恢复机器人到初始状态，确保下次测试从相同姿态开始
        # ========================================================================
        try:
            # 恢复左臂状态
            self.robot.left_entity.set_qpos(left_qpos)
            self.robot.left_entity.set_qvel(left_qvel)
            # 恢复右臂状态
            self.robot.right_entity.set_qpos(right_qpos)
            self.robot.right_entity.set_qvel(right_qvel)
            # 推进几步让状态稳定
            for _ in range(10):
                self._enforce_fixed_torso()
                self.scene.step()
            print(f"  [恢复初始状态] 机器人已重置到测试前状态")
        except Exception as reset_error:
            print(f"  ⚠ 恢复机器人状态失败: {reset_error}")
        
        # 记录测试结果
        euler_angles_list = euler_angles.tolist() if hasattr(euler_angles, 'tolist') else list(euler_angles)
        euler_degrees_list = euler_degrees.tolist() if hasattr(euler_degrees, 'tolist') else list(euler_degrees)
        
        test_results.append({
            'test_idx': test_idx,
            'euler_angles': euler_angles_list,
            'euler_degrees': euler_degrees_list,
            'status': 'SUCCESS' if is_success else 'FAILED',
            'left_status': left_status,
            'right_status': right_status,
            'error': error_msg if not is_success else None,
            'image_filename_ego': filename,
            'image_filename_observer': observer_filename if 'observer_filename' in locals() else None,
            'image_filename_zed': zed_filename if 'zed_filename' in locals() else None,
            'image_filename_left_wrist': left_wrist_filename if 'left_wrist_filename' in locals() else None,
            'image_filename_right_wrist': right_wrist_filename if 'right_wrist_filename' in locals() else None
        })

    def set_arm_poses_direct(self, 
                            left_wrist_pos,
                            left_wrist_rot_matrix,
                            right_wrist_pos, 
                            right_wrist_rot_matrix,
                            orientation_test=False,
                            test_frame_idx=0,
                            test_output_dir=None):
        """
        直接设置双臂末端位置和旋转（使用预处理数据或抓取数据）
        如果有抓取数据，则使用抓取位姿替代原始位姿
        
        Args:
            left_wrist_pos: 左手腕位置 [x, y, z]
            left_wrist_rot_matrix: 左手腕旋转矩阵 (3x3)
            right_wrist_pos: 右手腕位置 [x, y, z]
            right_wrist_rot_matrix: 右手腕旋转矩阵 (3x3)
            orientation_test: 是否进行朝向测试（测试多个欧拉角组合）
            test_frame_idx: 测试帧索引（用于文件命名）
            test_output_dir: 测试图片输出目录
        """
        if self.robot is None:
            print("机器人未加载，无法设置手臂位姿")
            return
        
        # 如果有抓取数据，将抓取位姿从相机坐标系转换到世界坐标系
        if self.best_grasp is not None:
            print("\n" + "="*80)
            print("【抓取位姿坐标系转换 - 详细步骤】")
            print("="*80)
            # ========================================================================
            # 步骤1: 获取anygrasp输出的抓取位姿（OpenCV相机坐标系）
            # ========================================================================
            # anygrasp输出的坐标系定义：
            # - OpenCV相机坐标系：X右, Y下, Z前（指向场景）
            grasp_pos_cam_opencv = np.array(self.best_grasp['translation'], dtype=np.float64)
            grasp_rot_cam_opencv = np.array(self.best_grasp['rotation_matrix'], dtype=np.float64)
            
            print(f"\n步骤1: anygrasp输出 (OpenCV相机坐标系: X右, Y下, Z前)")
            print(f"   位置: {grasp_pos_cam_opencv}")
            print(f"  旋转矩阵:\n{grasp_rot_cam_opencv}")
            print(f"   旋转矩阵的Z轴(抓取方向): {grasp_rot_cam_opencv[:, 2]}")
            
            # ========================================================================
            # 步骤2: 获取当前SAPIEN相机在世界坐标系中的位姿
            # ========================================================================
            cam_pose = self.camera.get_entity_pose()
            cam_pos_world = cam_pose.p  # 相机在世界坐标系的位置
            cam_quat_wxyz = cam_pose.q  # 相机在世界坐标系的朝向(wxyz格式)
            
            # # 转换四元数为旋转矩阵
            # cam_quat_xyzw = np.array([cam_quat_wxyz[1], cam_quat_wxyz[2], cam_quat_wxyz[3], cam_quat_wxyz[0]])
            # R_sapien_cam_to_world = Rotation1.from_quat(cam_quat_xyzw).as_matrix()
            camera_rotation_matrix = Rotation1.from_quat([cam_quat_wxyz[1], cam_quat_wxyz[2], cam_quat_wxyz[3], cam_quat_wxyz[0]]).as_matrix()
            
            # print(f"\n步骤2: SAPIEN相机在世界坐标系中的位姿")
            # print(f"   相机位置(世界): {cam_pos_world}")
            # print(f"   相机旋转矩阵(SAPIEN相机→世界)的形状: {R_sapien_cam_to_world.shape}")
            # print(F"   相机朝向{R_sapien_cam_to_world}")
            # print(f"   相机Z轴方向(世界坐标): {R_sapien_cam_to_world[:, 2]}")
            
            # # ========================================================================
            # # 步骤3: 构建齐次变换矩阵 - OpenCV相机坐标系 → SAPIEN相机坐标系
            # # ========================================================================
            # 构建相机在世界坐标系下的变换矩阵
            R_offset_inv = np.array([[0, 0, 1],[-1, 0, 0],[0, -1, 0]])
            # 3. 右乘修正矩阵，得到"正确"的矩阵
            camera_rotation_matrix = camera_rotation_matrix @ R_offset_inv

            world_T_cam = np.eye(4)
            world_T_cam[:3, :3] = camera_rotation_matrix
            world_T_cam[:3, 3] = cam_pos_world

            T_grasp_cam = np.eye(4)
            T_grasp_cam[:3, 3] = grasp_pos_cam_opencv
            T_grasp_cam[:3, :3] = grasp_rot_cam_opencv

            # 转换公式: world_T_obj = world_T_cam @ cam_T_obj
            world_T_grasp = world_T_cam @ T_grasp_cam

            # 提取位置和旋转
            grasp_pos_world = world_T_grasp[:3, 3]
            grasp_rot_world = world_T_grasp[:3, :3]
            
            # # 转换为四元数
            # quat_xyzw = Rotation1.from_matrix(rotation_matrix).as_quat()
            # sapien_quat_wxyz = np.array([quat_xyzw[3], *quat_xyzw[:3]], dtype=np.float32)
            
            # # 设置模型位姿
            # model_pose = sapien.Pose(position, sapien_quat_wxyz)
            # self.model_actor2.set_pose(model_pose)
            # # 坐标系对比：
            # # - OpenCV相机: X右, Y下, Z前
            # # - SAPIEN相机: X右, Y上, Z后
            # # 使用单位矩阵（不进行坐标系转换）
            # R_offset_inv = np.array([
            #     [0,  0,  1],
            #     [-1, 0,  0],
            #     [0,  -1, 0]
            # ], dtype=np.float64)
            
            # # 构建齐次变换矩阵
            # T_opencv_to_sapien = np.eye(4)
            # T_opencv_to_sapien[:3, :3] = R_offset_inv
            
            # # 构建抓取点在OpenCV坐标系下的齐次变换矩阵
            # T_grasp_opencv = np.eye(4)
            # T_grasp_opencv[:3, 3] = grasp_pos_cam_opencv
            
            # print(f"\n步骤3: 构建齐次变换矩阵")
            # print(f"   OpenCV→SAPIEN转换矩阵:\n{R_offset_inv}")
            # print(f"   抓取点位置(OpenCV): {grasp_pos_cam_opencv}")
            
            # # ========================================================================
            # # 步骤4: 链式转换 - 世界坐标系 ← 相机 ← SAPIEN_Cam ← OpenCV_Cam ← 抓取点
            # # ========================================================================
            # # 构建世界到相机的齐次变换矩阵
            # T_world_cam = np.eye(4)
            # T_world_cam[:3, :3] = R_sapien_cam_to_world
            # T_world_cam[:3, 3] = cam_pos_world
            
            # # 链式转换: T_world_grasp = T_world_cam @ T_opencv_to_sapien @ T_grasp_opencv
            # T_world_grasp = T_world_cam @ T_opencv_to_sapien @ T_grasp_opencv
            
            # # 提取世界坐标系下的位置和旋转
            # grasp_pos_world = T_world_grasp[:3, 3]
            # grasp_rot_world = T_world_grasp[:3, :3] @ grasp_rot_cam_opencv
            
            # R_z_90 = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=np.float64)
            # R_z_180 = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]], dtype=np.float64)
            # R_x_90 = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float64)
            # R_z_270 = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]], dtype=np.float64)
            # R_y_180 = np.array([[-1, 0, 0],[0, 1, 0],[0, 0, -1]], dtype=np.float64)
            # R_x_180 = np.array([[1, 0, 0],[0, -1, 0],[0, 0, -1]], dtype=np.float64)
            # r_rot_y_90 = np.array([[0., 0., 1.],[0., 1., 0.],[-1., 0., 0.]])
            # r_rot_y_270 = np.array([[ 0., 0., -1.],[ 0., 1., 0.],[1., 0., 0.]])            
            # r_rot_y_180 = np.array([[-1., 0., 0.],[0., 1., 0.],[0., 0., -1.]])
            # grasp_rot_world = grasp_rot_world # @ R_x_180 #  @ r_rot_y_270
            
            # # # 逆变换：从变换后的旋转矩阵恢复变换前的z轴
            # grasp_rot_world = R_z_90 @ grasp_rot_world @ R_x_90.T @ R_z_270.T

            # grasp_rot_world = grasp_rot_world @ R_x_90 #  @ r_rot_y_270

            print(f"\n步骤4: SAPIEN相机坐标系 → 世界坐标系")
            print(f"   相机位置贡献: {cam_pos_world}")
            # print(f"   旋转后的相对位置: {R_sapien_cam_to_world @ grasp_pos_cam_sapien}")
            print(f"   最终位置(世界): {grasp_pos_world}")
            print(f"   最终旋转Z轴(世界): {grasp_rot_world[:, 2]}")
            print(f"\033[92m   set时的右手朝向{grasp_rot_world} \033[0m")
            
            # ========================================================================
            # 步骤5: 验证转换链的一致性
            # ========================================================================
            print(f"\n步骤5: 转换验证")
            print(f"   OpenCV → SAPIEN → 世界: 完成")
            print(f"   位置变化: {np.linalg.norm(grasp_pos_cam_opencv)} m (相机) → {np.linalg.norm(grasp_pos_world)} m (世界)")
            
        # ========================================================================
        # 朝向测试模式：循环测试不同的欧拉角组合
        # ========================================================================
        if orientation_test and test_output_dir is not None:
            print("\n" + "="*80)
            print("【朝向测试模式】开始测试多个欧拉角组合")
            print("="*80)
            
            # 定义要测试的欧拉角组合 (单位: 弧度)
            # 格式: [roll, pitch, yaw] 对应 'xyz' 顺序
            test_orientations = [
                # 基础朝向
                [0, 0, 0],
                [np.pi/2, 0, 0],
                [np.pi, 0, 0],
                [3*np.pi/2, 0, 0],
                
                # Y轴旋转
                [0, np.pi/2, 0],
                [0, np.pi, 0],
                [0, 3*np.pi/2, 0],
                
                # Z轴旋转
                [0, 0, np.pi/2],
                [0, 0, np.pi],
                [0, 0, 3*np.pi/2],
                
                # 组合旋转
                [np.pi/2, np.pi/2, 0],
                [np.pi/2, 0, np.pi/2],
                [0, np.pi/2, np.pi/2],
                [np.pi/4, np.pi/4, 0],
                [np.pi/4, 0, np.pi/4],
                
                # 常用抓取朝向
                [np.pi, np.pi/4, 0],
                [np.pi, 0, np.pi/4],
                [3*np.pi/4, 0, 0],
                [np.pi/4, np.pi/2, 0],
            ]
            
            test_results = []
            
            for test_idx, euler_angles in enumerate(test_orientations):
                print(f"\n{'='*80}")
                print(f"测试 {test_idx+1}/{len(test_orientations)}: 欧拉角(xyz) = [{euler_angles[0]:.4f}, {euler_angles[1]:.4f}, {euler_angles[2]:.4f}] rad")
                print(f"                          = [{np.degrees(euler_angles[0]):.1f}°, {np.degrees(euler_angles[1]):.1f}°, {np.degrees(euler_angles[2]):.1f}°]")
                print(f"{'='*80}")
                
                # 生成该朝向的旋转矩阵
                test_orientation = Rotation1.from_euler('xyz', euler_angles).as_matrix()
                
                # 使用测试朝向
                test_left_rot = test_orientation
                test_right_rot = test_orientation
                
                # 测试该朝向（内部已处理所有异常）
                self._test_single_orientation(
                    left_wrist_pos=left_wrist_pos,
                    left_wrist_rot_matrix=test_left_rot,
                    right_wrist_pos=right_wrist_pos,
                    right_wrist_rot_matrix=test_right_rot,
                    euler_angles=euler_angles,
                    test_idx=test_idx,
                    test_frame_idx=test_frame_idx,
                    test_output_dir=test_output_dir,
                    test_results=test_results
                )
            
            # 输出测试总结
            print("\n" + "="*80)
            print("【朝向测试总结】")
            print("="*80)
            success_count = sum(1 for r in test_results if r['status'] == 'SUCCESS')
            fail_count = len(test_results) - success_count
            print(f"总测试数: {len(test_results)}")
            print(f"成功: {success_count}")
            print(f"失败: {fail_count}")
            print(f"\n成功的朝向:")
            for r in test_results:
                if r['status'] == 'SUCCESS':
                    head_img = r.get('image_filename_head', r.get('image_filename', 'N/A'))
                    third_img = r.get('image_filename_third', 'N/A')
                    print(f"  [{r['euler_degrees'][0]:7.1f}°, {r['euler_degrees'][1]:7.1f}°, {r['euler_degrees'][2]:7.1f}°]")
                    print(f"    -> 第一视角: {head_img}")
                    if third_img != 'N/A' and third_img is not None:
                        print(f"    -> 第三视角: {third_img}")
            print(f"\n失败的朝向:")
            for r in test_results:
                if r['status'] == 'FAILED':
                    head_img = r.get('image_filename_head', r.get('image_filename', 'N/A'))
                    third_img = r.get('image_filename_third', 'N/A')
                    error_msg = r.get('error', 'Unknown error')
                    print(f"  [{r['euler_degrees'][0]:7.1f}°, {r['euler_degrees'][1]:7.1f}°, {r['euler_degrees'][2]:7.1f}°]")
                    print(f"    -> 第一视角: {head_img}")
                    if third_img != 'N/A' and third_img is not None:
                        print(f"    -> 第三视角: {third_img}")
                    print(f"    -> 错误: {error_msg[:80]}")
            print("="*80 + "\n")
            
            # 保存测试结果到JSON
            results_json_path = os.path.join(test_output_dir, f"orientation_test_results_frame{test_frame_idx:04d}.json")
            with open(results_json_path, 'w') as f:
                json.dump(test_results, f, indent=2, default=str)
            print(f"测试结果已保存到: {results_json_path}")
            
            return  # 测试模式直接返回，不继续执行正常流程
        
        # ========================================================================
        # 正常模式：使用固定朝向
        # ========================================================================
        fixed_forward_orientation = Rotation1.from_euler('xyz', [np.pi/2, 0, 0]).as_matrix()
        left_wrist_rot_matrix = fixed_forward_orientation
        right_wrist_rot_matrix = fixed_forward_orientation
        print(f"\n步骤6: 设置固定朝向 [π/2, 0, 0]")
        print(f"   旋转矩阵:\n{fixed_forward_orientation}")
            
            # # ========================================================================
            # # 步骤7: 使用Anygrasp信息
            # # ========================================================================
            # # right_wrist_pos = grasp_pos_world
            # # right_wrist_rot_matrix = grasp_rot_world
            # # right_euler = Rotation1.from_matrix(right_wrist_rot_matrix).as_euler('xyz')
            # # print(f"\033[94m   右手欧拉角(世界): {right_euler}\033[0m")
            # # ========================================================================
            # # 步骤7: 使用Anygrasp信息  设置目标位置
            # # ========================================================================
            # print(f"\n步骤7: 最终目标")
            # print(f"   右手位置(世界): {right_wrist_pos}")
            # print(f"   左手位置(世界): {left_wrist_pos}")
            # print("="*80 + "\n")
        
        try:
            left_pos = np.array(left_wrist_pos, dtype=float).flatten()[:3]
            right_pos = np.array(right_wrist_pos, dtype=float).flatten()[:3]
            
            # 处理旋转矩阵
            left_rot_matrix = np.array(left_wrist_rot_matrix, dtype=float).reshape(3, 3)
            right_rot_matrix = np.array(right_wrist_rot_matrix, dtype=float).reshape(3, 3)
            print(f"传入前左手{left_rot_matrix}")
            print(f"\033[92m   get时的右手朝向{right_rot_matrix} \033[0m")
            
            # 补偿0.1m的手腕位置偏移（匹配原始版本）
            # 原始版本: cal_wrist_position = gripper_position - 0.1 * z_axis
            # 这里的z_axis是变换前的z轴，需要从变换后的旋转矩阵恢复
            
            def compensate_wrist_offset(pos, rot_matrix):
                """补偿手腕位置偏移，匹配原始版本的计算"""
                # 变换矩阵（与hdf5_aloha.py中的变换一致）
                R_z_90 = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=np.float64)
                R_x_90 = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float64)
                R_z_270 = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]], dtype=np.float64)
                
                # 逆变换：从变换后的旋转矩阵恢复变换前的z轴
                # 变换公式: rotation_matrix = R_z_90 @ original_rotation @ R_x_90.T @ R_z_270.T
                # 逆变换: original_rotation = R_z_90.T @ rotation_matrix @ R_z_270 @ R_x_90
                original_rotation = R_z_90.T @ rot_matrix @ R_z_270 @ R_x_90
                print(f"\033[94m========= 变换前旋转矩阵:\n{original_rotation} ======= \033[0m")
                original_z_axis = original_rotation[:, 2]
                
                # 应用0.1m的偏移（匹配原始版本的计算）
                # 原始版本: cal_wrist_position = gripper_position - 0.1 * z_axis
                # 预处理版本: wrist_position = gripper_position
                # 所以补偿: compensated_pos = gripper_position - 0.1 * z_axis
                compensated_pos = pos - 0.0 * original_z_axis
                
                print(f"原始位置: {pos}")
                print(f"变换前z轴: {original_z_axis}")
                print(f"补偿后位置: {compensated_pos}")
                
                return original_rotation, compensated_pos
            
            # 应用位置补偿
            # left_rot_matrix_compensated, left_pos_compensated = compensate_wrist_offset(left_pos, left_rot_matrix)
            # right_rot_matrix_compensated, right_pos_compensated = compensate_wrist_offset(right_pos, right_rot_matrix)
            
            # 转换为四元数 (xyzw格式)
            # print(f"    [compensated] 下一步：转换成xyzw 左手{left_rot_matrix_compensated}")
            # print(f"    [Anygrasp] 下一步：转换成xyzw 右手{right_rot_matrix}")
            print(f"    [FIXED] 下一步：转换成xyzw 左手{left_rot_matrix}")
            print(f"    [FIXED] 下一步：转换成xyzw 右手{right_rot_matrix}")
            left_quat_xyzw = Rotation1.from_matrix(left_rot_matrix).as_quat()  # xyzw
            right_quat_xyzw = Rotation1.from_matrix(right_rot_matrix).as_quat()  # xyzw
            
            # （SAPIEN需要xyzw作为输入）
            # left_quat_xyzw = left_quat_scipy
            # right_quat_xyzw = right_quat_scipy
            
            # # # =======================        0120尝试 wxyz ===============================================
            # # # 将 xyzw 转换为 wxyz
            left_quat_wxyz = np.array([left_quat_xyzw[3], left_quat_xyzw[0], left_quat_xyzw[1], left_quat_xyzw[2]])
            right_quat_wxyz = np.array([right_quat_xyzw[3], right_quat_xyzw[0], right_quat_xyzw[1], right_quat_xyzw[2]])
            # left_target_pose = np.concatenate([left_pos_compensated, left_quat_wxyz])
            left_target_pose = np.concatenate([left_pos, left_quat_wxyz])
            # # right_target_pose = np.concatenate([right_pos_compensated, right_quat_wxyz])
            right_target_pose = np.concatenate([right_pos, right_quat_wxyz])
            # # # =======================        0120尝试  =============================================

            # 构造目标姿态 [x, y, z, qx, qy, qz, qw] - 使用补偿后的位置
            # left_quat_xyzw = [9.9740154e-01, -1.9089994e-06, 2.0072575e-06, 7.2043955e-02]
            # right_quat_xyzw = [9.9740154e-01, -1.9089994e-06, 2.0072575e-06, 7.2043955e-02]
            # left_target_pose = np.concatenate([left_pos, left_quat_xyzw])
            # right_target_pose = np.concatenate([right_pos, right_quat_xyzw])
            # ToDebug
            # left_target_pose[2] += 0.1
            # right_target_pose[2] +=0.1
            
            print(f"\033[31m规划左臂路径到目标: {left_target_pose}\033[0m")
            print(f"\033[31m规划右臂路径到目标: {right_target_pose}\033[0m")
            
            # 使用机器人的路径规划功能
            print("########### 左臂: ##############")
            left_plan_result = self.robot.left_plan_path(
                target_pose=left_target_pose.tolist(),
                constraint_pose=None,
                use_point_cloud=False,
                use_attach=False
            )
            
            print("########### 右臂: ##############")
            right_plan_result = self.robot.right_plan_path(
                target_pose=right_target_pose.tolist(),
                constraint_pose=None,
                use_point_cloud=False,
                use_attach=False
            )
            
            print(f"左臂规划结果: {left_plan_result.get('status', 'Unknown')}")
            print(f"右臂规划结果: {right_plan_result.get('status', 'Unknown')}")
            
            left_path = left_plan_result.get('position', [])
            right_path = right_plan_result.get('position', [])
                
            if len(left_path) > 0:
                left_target_joints = left_path[-1]
                self.robot.set_arm_joints(
                    target_position=left_target_joints,
                    target_velocity=[0.0] * len(left_target_joints),
                    arm_tag="left"
                )

            if len(right_path) > 0:
                right_target_joints = right_path[-1]
                self.robot.set_arm_joints(
                    target_position=right_target_joints,
                    target_velocity=[0.0] * len(right_target_joints),
                    arm_tag="right"
                )

            # 施加 torso 固定值后再推进仿真，避免腰部在规划执行中漂移
            self._enforce_fixed_torso()
            for _ in range(10000):
                self._enforce_fixed_torso()
                self.scene.step()
            self._enforce_fixed_torso()
                
            # 验证最终位置
            final_left_pose = self.robot.get_left_ee_pose()
            final_right_pose = self.robot.get_right_ee_pose()
                
            print(f"最终左臂末端位姿: {final_left_pose}")
            print(f"最终右臂末端位姿: {final_right_pose}")
            print(f"左臂规划差异: {final_left_pose - left_target_pose}")
            print(f"右臂规划差异: {final_right_pose - right_target_pose}")

            # 输出3x3旋转矩阵
            from scipy.spatial.transform import Rotation as R
            def pose_to_rotmat(pose):
                # pose: [x, y, z, qw, qx, qy, qz]
                quat = pose[3:7]  # wxyz
                quat = np.array([quat[1], quat[2], quat[3], quat[0]])  # xyzw

                rot = R.from_quat(quat)
                return rot.as_matrix()

            left_rotmat = pose_to_rotmat(final_left_pose)
            right_rotmat = pose_to_rotmat(final_right_pose)
            print(f"左臂末端旋转矩阵:\n{left_rotmat}")
            print(f"\033[92m 右臂末端旋转矩阵:\n{right_rotmat}\033[0m")
            right_euler = R.from_matrix(right_rotmat).as_euler('zyx', degrees=True)
            print(f"\033[94m 右臂末端欧拉角 (ZYX, 度): {right_euler}\033[0m")
            self.left_eelink = self.robot.right_entity.find_link_by_name("left_gripper_link")
            left_eelink_pose = self.left_eelink.get_entity_pose() # 当做wxyz处理
            rot = R.from_quat([left_eelink_pose.q[1], left_eelink_pose.q[2], left_eelink_pose.q[3], left_eelink_pose.q[0]])
            rot_matrix = rot.as_matrix()
            print(f"\033[92m 左臂末端link位姿:\nPosition: {left_eelink_pose.p}, Quaternion(wxyz): {left_eelink_pose.q}\033[0m")
            print(f"\033[92m 左臂末端link旋转矩阵:\n{rot_matrix}\033[0m")
            left_euler = R.from_matrix(rot_matrix).as_euler('zyx', degrees=True)
            print(f"\033[94m 左臂末端link欧拉角 (ZYX, 度): {left_euler}\033[0m")

            self.right_eelink = self.robot.right_entity.find_link_by_name("right_gripper_link")
            right_eelink_pose = self.right_eelink.get_entity_pose()
            rot = R.from_quat([right_eelink_pose.q[1], right_eelink_pose.q[2], right_eelink_pose.q[3], right_eelink_pose.q[0]])
            rot_matrix = rot.as_matrix()
            print(f"\033[92m 右臂末端link位姿:\nPosition: {right_eelink_pose.p}, Quaternion(wxyz): {right_eelink_pose.q}\033[0m")
            print(f"\033[92m 右臂末端link旋转矩阵:\n{rot_matrix}\033[0m")
            right_euler = R.from_matrix(rot_matrix).as_euler('zyx', degrees=True)
            print(f"\033[94m 右臂末端link欧拉角 (ZYX, 度): {right_euler}\033[0m")

            
            # 保存最终位置
            self._final_left_pose = final_left_pose
            self._final_right_pose = final_right_pose
            
            # 保存目标位姿用于可视化（世界坐标系）
            self._left_target_pose_world = left_target_pose
            self._right_target_pose_world = right_target_pose
                
        except Exception as e:
            print(f"设置手臂位姿时出错: {e}")
            import traceback
            traceback.print_exc()

    def set_gripper_direct(self, left_gripper_value, right_gripper_value, gripper_eps=0.1):
        """
        直接设置夹爪值（使用预处理数据）
        
        Args:
            left_gripper_value: 左手夹爪值 (0-1)
            right_gripper_value: 右手夹爪值 (0-1)
            gripper_eps: 夹爪控制精度
        """
        if hasattr(self, 'robot') and self.robot is not None:
            left_gripper_value = max(0.0, min(1.0, left_gripper_value))
            right_gripper_value = max(0.0, min(1.0, right_gripper_value))
            
            self.robot.set_gripper(left_gripper_value, "left", gripper_eps)
            self.robot.set_gripper(right_gripper_value, "right", gripper_eps)
            
            print(f"设置夹爪: 左={left_gripper_value:.3f}, 右={right_gripper_value:.3f}")
    
    def _get_gripper_links(self, arm="left"):
        """
        获取夹爪的所有links
        
        Args:
            arm: "left" 或 "right"
            
        Returns:
            list: 夹爪相关的links列表
        """
        links = []
        try:
            if arm == "left":
                entity = self.robot.left_entity
            else:
                entity = self.robot.right_entity
            
            # 获取所有links
            all_links = entity.get_links()
            
            # 筛选夹爪相关的links（通常包含"gripper"或"finger"关键词）
            for link in all_links:
                link_name = link.get_name().lower()
                # print(f"link_name: {link_name}")
                if "gripper" in link_name or "finger" in link_name or "link7" in link_name or "link8" in link_name:
                    links.append(link)
            
        except Exception as e:
            print(f"获取夹爪links时出错: {e}")
        
        return links
    
    # def close_grippers_and_detect_contact(self, arm="both", close_value=0.0):
    #     """
    #     收拢夹爪并检测与物体的接触
        
    #     Args:
    #         arm: "left", "right", 或 "both"
    #         close_value: 夹爪收拢值（0.0为完全闭合）
            
    #     Returns:
    #         dict: 包含接触检测结果
    #     """
    #     if self.robot is None or self.model_actor is None:
    #         return {"left_contact": False, "right_contact": False, "contacts": [], "grasped": False}
        
    #     # 记录收拢前的夹爪值
    #     original_left_gripper = self.robot.left_gripper_val
    #     original_right_gripper = self.robot.right_gripper_val
        
    #     # 收拢夹爪
    #     if arm == "both" or arm == "left":
    #         self.robot.set_gripper(close_value, "left", gripper_eps=0.01)
    #     if arm == "both" or arm == "right":
    #         self.robot.set_gripper(close_value, "right", gripper_eps=0.01)
        
    #     # 等待夹爪收拢完成
    #     for _ in range(500):
    #         self.scene.step()
        
    #     # 检测接触
    #     contacts = self.scene.get_contacts()
        
    #     left_contact = False
    #     right_contact = False
    #     contact_details = []
        
    #     # 获取夹爪的links
    #     left_gripper_links = self._get_gripper_links("left")
    #     right_gripper_links = self._get_gripper_links("right")
        
    #     # 检查每个接触点
    #     for contact in contacts:
    #         actor1 = contact.actor0
    #         actor2 = contact.actor1
            
    #         # 检查是否是夹爪与目标物体的接触
    #         if actor1 == self.model_actor or actor2 == self.model_actor:
    #             other_actor = actor2 if actor1 == self.model_actor else actor1
                
    #             # 检查是否是左手夹爪
    #             if any(link == other_actor for link in left_gripper_links):
    #                 left_contact = True
    #                 contact_details.append({
    #                     "arm": "left",
    #                     "actor": other_actor.get_name(),
    #                     "points": len(contact.points)
    #                 })
                
    #             # 检查是否是右手夹爪
    #             if any(link == other_actor for link in right_gripper_links):
    #                 right_contact = True
    #                 contact_details.append({
    #                     "arm": "right",
    #                     "actor": other_actor.get_name(),
    #                     "points": len(contact.points)
    #                 })
        
    #     # 恢复原始夹爪值
    #     if arm == "both" or arm == "left":
    #         self.robot.set_gripper(original_left_gripper, "left", gripper_eps=0.1)
    #     if arm == "both" or arm == "right":
    #         self.robot.set_gripper(original_right_gripper, "right", gripper_eps=0.1)
        
    #     # 等待恢复
    #     for _ in range(100):
    #         self.scene.step()
        
    #     result = {
    #         "left_contact": left_contact,
    #         "right_contact": right_contact,
    #         "contact_details": contact_details,
    #         "grasped": left_contact or right_contact  # 任意一只手有接触就认为抓住了
    #     }
        
    #     return result
    
    # def close_grippers_and_detect_contact(self, arm="both", close_value=0.0,
    #                                     settle_steps=20, min_impulse=0.0):
    #     """
    #     收拢夹爪并检测与目标物体(self.model_actor)的接触（SAPIEN 3.x 版）
    #     """
    #     if self.robot is None or self.model_actor is None:
    #         return {"left_contact": False, "right_contact": False,
    #                 "contact_details": [], "grasped": False}

    #     # 记录原始夹爪值
    #     original_left_gripper = getattr(self.robot, "left_gripper_val", None)
    #     original_right_gripper = getattr(self.robot, "right_gripper_val", None)

    #     # 收拢夹爪
    #     if arm in ("both", "left"):
    #         self.robot.set_gripper(close_value, "left", gripper_eps=0.01)
    #     if arm in ("both", "right"):
    #         self.robot.set_gripper(close_value, "right", gripper_eps=0.01)

    #     # 让接触生成/稳定
    #     for _ in range(settle_steps):
    #         self.scene.step()

    #     # === 接触检测（3.x API） ===
    #     left_contact = False
    #     right_contact = False
    #     contact_details = []

    #     # 确保这些是 Entity（和 contact.bodies[*].entity 可直接比较）
    #     left_gripper_links = set(self._get_gripper_links("left"))   # 返回 Entity 列表更稳
    #     right_gripper_links = set(self._get_gripper_links("right"))

    #     target_entity = self.model_actor  # 你的“model_actor”实为 Entity

    #     for contact in self.scene.get_contacts():
    #         bodies = contact.bodies
    #         if not bodies or len(bodies) != 2:
    #             continue
    #         e0 = bodies[0].entity
    #         e1 = bodies[1].entity

    #         if e0 is target_entity or e1 is target_entity:
    #             other_entity = e1 if (e0 is target_entity) else e0

    #             # 计算这对接触的总冲量（可做强度阈值）
    #             total_impulse = 0.0
    #             for p in contact.points:
    #                 # p.impulse 是作用在 bodies[0] 上的冲量（F*dt）
    #                 # 只看幅值即可
    #                 total_impulse += float(np.linalg.norm(p.impulse))

    #             if other_entity in left_gripper_links and total_impulse >= min_impulse:
    #                 left_contact = True
    #                 contact_details.append({
    #                     "arm": "left",
    #                     "other": other_entity.name,
    #                     "num_points": len(contact.points),
    #                     "total_impulse": total_impulse,
    #                 })

    #             if other_entity in right_gripper_links and total_impulse >= min_impulse:
    #                 right_contact = True
    #                 contact_details.append({
    #                     "arm": "right",
    #                     "other": other_entity.name,
    #                     "num_points": len(contact.points),
    #                     "total_impulse": total_impulse,
    #                 })

    #     # 复位夹爪（可选）
    #     if original_left_gripper is not None and arm in ("both", "left"):
    #         self.robot.set_gripper(original_left_gripper, "left", gripper_eps=0.1)
    #     if original_right_gripper is not None and arm in ("both", "right"):
    #         self.robot.set_gripper(original_right_gripper, "right", gripper_eps=0.1)
    #     for _ in range(5):
    #         self.scene.step()

    #     return {
    #         "left_contact": left_contact,
    #         "right_contact": right_contact,
    #         "contact_details": contact_details,
    #         # 你原来是“任一手就算抓住”，如果需要“双指同时接触”就改成 and
    #         "grasped": (left_contact or right_contact)
    #     }


    def render_frame(self, frame_idx=None):
        """
        渲染当前帧
        
        Args:
            frame_idx: 帧索引，用于获取相机坐标系下的位姿数据
        
        Returns:
            tuple: (RGB图像 (H, W, 3), 深度图像 (H, W))
        """
        # 渲染前再固定一次 torso，确保视频中腰部不随物理迭代漂移
        self._enforce_fixed_torso()
        self.scene.step()
        self.scene.update_render()
        
        # 获取相机图像
        self.camera.take_picture()
        camera_rgba = self.camera.get_picture("Color")
        camera_rgba_img = (camera_rgba * 255).clip(0, 255).astype("uint8")
        
        # 获取RGB图像
        rgb = camera_rgba_img[:, :, :3]
        
        # 获取深度图像（参考 camera.py 的 get_depth 实现）
        # 1. 获取Position图像的Z通道
        position = self.camera.get_picture("Position")
        # 2. Z轴正方向指向相机后方，取负数得到正深度值
        depth = -position[..., 2]
        # 3. 转换为毫米单位（与camera.py保持一致）
        depth_image = (depth * 1000.0).astype(np.float64)
        
        # # 如果提供了帧索引且模型位姿数据存在，添加可视化
        # if frame_idx is not None and self.model_poses is not None:
        #     rgb = self._add_pose_visualization(rgb, frame_idx)
        
        # 如果有最佳抓取数据，在图像上可视化
        if self.best_grasp is not None:
            rgb = self.visualize_best_grasp_on_image(rgb)
        
        # 如果有目标位姿数据，在图像上可视化
        if hasattr(self, '_left_target_pose_world') and hasattr(self, '_right_target_pose_world'):
            rgb = self.visualize_target_poses_on_image(rgb)
        
        return rgb, depth_image
    
    def render_third_camera_frame(self, frame_idx=None):
        """
        渲染observer_camera当前帧
        
        Args:
            frame_idx: 帧索引，用于获取相机坐标系下的位姿数据
        
        Returns:
            tuple: (RGB图像 (H, W, 3), 深度图像 (H, W)) 或 (None, None)
        """
        if not hasattr(self, 'observer_camera') or self.observer_camera is None:
            return None, None
        
        # 独立渲染第三视角时同样固定 torso
        self._enforce_fixed_torso()
        self.scene.step()
        self.scene.update_render()
        
        # 获取observer_camera图像
        self.observer_camera.take_picture()
        observer_camera_rgba = self.observer_camera.get_picture("Color")
        observer_camera_rgba_img = (observer_camera_rgba * 255).clip(0, 255).astype("uint8")
        
        # 获取RGB图像
        rgb = observer_camera_rgba_img[:, :, :3]
        
        # 获取深度图像（参考 camera.py 的 get_depth 实现）
        # 1. 获取Position图像的Z通道
        position = self.observer_camera.get_picture("Position")
        # 2. Z轴正方向指向相机后方，取负数得到正深度值
        depth = -position[..., 2]
        # 3. 转换为毫米单位（与camera.py保持一致）
        depth_image = (depth * 1000.0).astype(np.float64)
        
        # 如果有目标位姿数据，在图像上可视化
        if hasattr(self, '_left_target_pose_world') and hasattr(self, '_right_target_pose_world'):
            rgb = self.visualize_target_poses_on_third_image(rgb)

        # 在third视角叠加头部/腕部相机位姿，用于调试相机安装方向与位置
        if self.third_cam_show_link_cams:
            rgb = self.visualize_link_cameras_on_third_image(rgb)
        
        return rgb, depth_image
    
    def render_zed_camera_frame(self, frame_idx=None):
        """
        渲染zed_camera（头部相机）当前帧
        
        Args:
            frame_idx: 帧索引，用于获取相机坐标系下的位姿数据
        
        Returns:
            tuple: (RGB图像 (H, W, 3), 深度图像 (H, W)) 或 (None, None)
        """
        if not hasattr(self, 'zed_camera') or self.zed_camera is None:
            return None, None
        
        self._enforce_fixed_torso()
        self.scene.step()
        self.scene.update_render()
        
        self.zed_camera.take_picture()
        zed_camera_rgba = self.zed_camera.get_picture("Color")
        zed_camera_rgba_img = (zed_camera_rgba * 255).clip(0, 255).astype("uint8")
        
        rgb = zed_camera_rgba_img[:, :, :3]
        
        position = self.zed_camera.get_picture("Position")
        depth = -position[..., 2]
        depth_image = (depth * 1000.0).astype(np.float64)
        
        return rgb, depth_image
    
    def render_left_wrist_camera_frame(self, frame_idx=None):
        """
        渲染left_wrist_camera（左手腕相机）当前帧
        
        Args:
            frame_idx: 帧索引，用于获取相机坐标系下的位姿数据
        
        Returns:
            tuple: (RGB图像 (H, W, 3), 深度图像 (H, W)) 或 (None, None)
        """
        if not hasattr(self, 'left_wrist_camera') or self.left_wrist_camera is None:
            return None, None
        
        self._enforce_fixed_torso()
        self.scene.step()
        self.scene.update_render()
        
        self.left_wrist_camera.take_picture()
        left_wrist_camera_rgba = self.left_wrist_camera.get_picture("Color")
        left_wrist_camera_rgba_img = (left_wrist_camera_rgba * 255).clip(0, 255).astype("uint8")
        
        rgb = left_wrist_camera_rgba_img[:, :, :3]
        
        position = self.left_wrist_camera.get_picture("Position")
        depth = -position[..., 2]
        depth_image = (depth * 1000.0).astype(np.float64)
        
        return rgb, depth_image
    
    def render_right_wrist_camera_frame(self, frame_idx=None):
        """
        渲染right_wrist_camera（右手腕相机）当前帧
        
        Args:
            frame_idx: 帧索引，用于获取相机坐标系下的位姿数据
        
        Returns:
            tuple: (RGB图像 (H, W, 3), 深度图像 (H, W)) 或 (None, None)
        """
        if not hasattr(self, 'right_wrist_camera') or self.right_wrist_camera is None:
            return None, None
        
        self._enforce_fixed_torso()
        self.scene.step()
        self.scene.update_render()
        
        self.right_wrist_camera.take_picture()
        right_wrist_camera_rgba = self.right_wrist_camera.get_picture("Color")
        right_wrist_camera_rgba_img = (right_wrist_camera_rgba * 255).clip(0, 255).astype("uint8")
        
        rgb = right_wrist_camera_rgba_img[:, :, :3]
        
        position = self.right_wrist_camera.get_picture("Position")
        depth = -position[..., 2]
        depth_image = (depth * 1000.0).astype(np.float64)
        
        return rgb, depth_image
    
    # def _add_pose_visualization(self, rgb_image, frame_idx):
    #     """
    #     在图像上添加相机坐标系下的位姿可视化（参考vis_pose_on_video.py）
        
    #     Args:
    #         rgb_image: RGB图像
    #         frame_idx: 帧索引
            
    #     Returns:
    #         np.ndarray: 添加了可视化的RGB图像
    #     """
    #     if str(frame_idx) not in self.model_poses:
    #         return rgb_image
        
    #     try:
    #         # 确保图像数组是连续的并且格式正确
    #         if not rgb_image.flags['C_CONTIGUOUS']:
    #             rgb_image = np.ascontiguousarray(rgb_image)
            
    #         # 确保数据类型是uint8
    #         if rgb_image.dtype != np.uint8:
    #             rgb_image = rgb_image.astype(np.uint8)
            
    #         # 获取相机坐标系下的位姿矩阵 (camera->object变换)
    #         cam_T_obj = np.array(self.model_poses[str(frame_idx)])
            
    #         # 使用与vis_pose_on_video.py相同的相机内参
    #         # 这些内参应该与您的相机标定结果匹配
    #         camera_intrinsics = np.array([
    #             [736.6339111328125, 0.0, 960.0],
    #             [0.0, 736.6339111328125, 540.0],
    #             [0.0, 0.0, 1.0]
    #         ], dtype=float)
            
    #         # 投影原点
    #         origin_cam = cam_T_obj[:3, 3]
    #         px, valid = self._project_point(origin_cam, camera_intrinsics)
            
    #         if valid:
    #             # 绘制原点
    #             cv2.circle(rgb_image, px, 6, (0, 255, 255), -1)  # 黄色圆点
    #             cv2.circle(rgb_image, px, 10, (255, 255, 255), 2)  # 白色边框
    #             # 绘制坐标轴
    #             self._draw_axes(rgb_image, camera_intrinsics, cam_T_obj, length=0.1)
    #             # 添加坐标信息
    #             coord_text = f"3D: ({origin_cam[0]:.2f}, {origin_cam[1]:.2f}, {origin_cam[2]:.2f})"
    #             pixel_text = f"Pixel: ({px[0]}, {px[1]})"
    #             cv2.putText(rgb_image, coord_text, (px[0] + 10, px[1] - 25), 
    #                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    #             cv2.putText(rgb_image, pixel_text, (px[0] + 10, px[1] - 10), 
    #                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    #             print(f"帧 {frame_idx}: 相机坐标系位姿可视化 - 位置: {origin_cam}, 像素: {px}")
    #         else:
    #             print(f"帧 {frame_idx}: 原点在相机后方，跳过绘制")
            
    #     except Exception as e:
    #         print(f"添加位姿可视化时出错: {e}")
    #         import traceback
    #         traceback.print_exc()
        
    #     return rgb_image
    
    def _project_point(self, point_cam, intr):
        """将3D点投影到图像平面。返回像素坐标和有效性标志"""
        z = point_cam[2]
        if z <= 1e-6:
            return (0, 0), False
        uv = intr @ point_cam
        u = uv[0] / z
        v = uv[1] / z
        return (int(round(u)), int(round(v))), True
    
    def _get_camera_intrinsics(self):
        """计算相机内参矩阵"""
        # 根据fovy和图像尺寸计算焦距
        fovy_rad = np.deg2rad(self.fovy_deg)
        fy = self.image_height / (2.0 * np.tan(fovy_rad / 2.0))
        fx = fy  # 假设像素是正方形
        cx = self.image_width / 2.0
        cy = self.image_height / 2.0
        
        intrinsics = np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ], dtype=np.float64)
        
        return intrinsics, fx, fy, cx, cy
    
    def visualize_best_grasp_on_image(self, rgb_image):
        """
        在RGB图像上可视化最佳抓取位姿
        
        注意：这里的抓取位姿是在OpenCV相机坐标系下的（anygrasp输出）
        由于投影操作也是在相机坐标系下进行的，所以不需要转换到世界坐标系
        
        Args:
            rgb_image: RGB图像 (numpy array)
            
        Returns:
            np.ndarray: 添加了可视化的RGB图像
        """
        if self.best_grasp is None:
            return rgb_image
        
        # 获取相机内参
        intrinsics, fx, fy, cx, cy = self._get_camera_intrinsics()
        
        # 确保图像数组是连续的并且格式正确
        if not rgb_image.flags['C_CONTIGUOUS']:
            rgb_image = np.ascontiguousarray(rgb_image)
        
        # 确保数据类型是uint8
        if rgb_image.dtype != np.uint8:
            rgb_image = rgb_image.astype(np.uint8)
        
        # 转换为BGR格式用于OpenCV
        vis_img = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
        
        # 获取抓取参数（OpenCV相机坐标系：X右，Y下，Z前）
        # 注意：这里直接使用anygrasp输出的数据，已经在当前相机坐标系下
        center = np.array(self.best_grasp['translation'], dtype=np.float64)
        R = np.array(self.best_grasp['rotation_matrix'], dtype=np.float64)
        width = self.best_grasp['width']
        depth = self.best_grasp['depth']
        score = self.best_grasp['score']
        
        # 根据分数设置颜色（分数高=红色，分数低=蓝色）
        color = (int(255 * (1 - score)), 0, int(255 * score))  # BGR格式
        
        # 定义夹爪关键点（局部坐标系）
        finger_width = 0.004
        depth_base = 0.02
        tail_length = 0.04
        
        # 关键点：左手指、右手指、底部、尾部的端点
        # 左手指前端和后端
        left_finger_front = np.array([depth, -width/2 - finger_width, 0])
        left_finger_back = np.array([-depth_base, -width/2 - finger_width, 0])
        
        # 右手指前端和后端
        right_finger_front = np.array([depth, width/2 + finger_width, 0])
        right_finger_back = np.array([-depth_base, width/2 + finger_width, 0])
        
        # 底部中心点
        bottom_center = np.array([-depth_base, 0, 0])
        
        # 尾部端点
        tail_end = np.array([-depth_base - tail_length, 0, 0])
        
        # 将局部坐标点转换到相机坐标系
        # 注意：这里的R和center都已经在OpenCV相机坐标系下了
        # 所以转换后的points_cam也在OpenCV相机坐标系下
        points_local = [left_finger_front, left_finger_back, right_finger_front, 
                       right_finger_back, bottom_center, tail_end]
        points_cam = [np.dot(R, p) + center for p in points_local]
        
        # 投影到2D图像平面
        # _project_point使用标准的针孔相机模型：u = fx * X/Z + cx, v = fy * Y/Z + cy
        # 这个投影操作是在OpenCV相机坐标系下进行的，所以直接使用points_cam即可
        points_2d = []
        all_valid = True
        for p in points_cam:
            px, valid = self._project_point(p, intrinsics)
            if not valid or px[0] < 0 or px[0] >= vis_img.shape[1] or px[1] < 0 or px[1] >= vis_img.shape[0]:
                all_valid = False
                break
            points_2d.append(px)
        
        if all_valid:
            # 绘制夹爪
            # 左手指线
            cv2.line(vis_img, points_2d[0], points_2d[1], color, 3)
            # 右手指线
            cv2.line(vis_img, points_2d[2], points_2d[3], color, 3)
            # 底部连接线
            cv2.line(vis_img, points_2d[1], points_2d[3], color, 2)
            # 尾部线
            cv2.line(vis_img, points_2d[4], points_2d[5], color, 2)
            
            # 绘制手指端点（圆圈表示两个爪子）
            cv2.circle(vis_img, points_2d[0], 4, color, -1)
            cv2.circle(vis_img, points_2d[2], 4, color, -1)
            
            # 投影抓取中心
            px_center, valid_center = self._project_point(center, intrinsics)
            if valid_center:
                # 绘制中心点
                cv2.circle(vis_img, px_center, 5, (255, 255, 255), -1)
                
                # 标注抓取分数
                text = f"Best Grasp: {score:.3f}"
                # 添加文字背景
                (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(vis_img, (px_center[0] + 8, px_center[1] - text_h - 10), 
                            (px_center[0] + text_w + 15, px_center[1] - 3), (0, 0, 0), -1)
                cv2.putText(vis_img, text, (px_center[0] + 10, px_center[1] - 5), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                print(f"最佳抓取可视化: 中心像素({px_center[0]}, {px_center[1]}), 分数={score:.3f}")
        else:
            print("警告: 抓取位姿在图像外，无法可视化")
        
        # 转换回RGB格式
        vis_img = cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB)
        
        return vis_img
    
    def _clip_line_to_image(self, p1, p2, img_width, img_height):
        """
        将线段裁剪到图像边界内
        
        Args:
            p1: 起点 (x, y)
            p2: 终点 (x, y)
            img_width: 图像宽度
            img_height: 图像高度
            
        Returns:
            tuple: (裁剪后的起点, 裁剪后的终点, 是否有可见部分)
        """
        x1, y1 = float(p1[0]), float(p1[1])
        x2, y2 = float(p2[0]), float(p2[1])
        
        # 如果两个点都在图像内，直接返回
        if (0 <= x1 < img_width and 0 <= y1 < img_height and 
            0 <= x2 < img_width and 0 <= y2 < img_height):
            return p1, p2, True
        
        # Cohen-Sutherland 线段裁剪算法
        INSIDE = 0  # 0000
        LEFT = 1    # 0001
        RIGHT = 2   # 0010
        BOTTOM = 4  # 0100
        TOP = 8     # 1000
        
        def compute_code(x, y):
            code = INSIDE
            if x < 0:
                code |= LEFT
            elif x >= img_width:
                code |= RIGHT
            if y < 0:
                code |= TOP
            elif y >= img_height:
                code |= BOTTOM
            return code
        
        code1 = compute_code(x1, y1)
        code2 = compute_code(x2, y2)
        
        accept = False
        
        while True:
            # 两个端点都在图像内
            if code1 == 0 and code2 == 0:
                accept = True
                break
            # 两个端点都在图像外的同一侧
            elif (code1 & code2) != 0:
                break
            else:
                # 至少有一个端点在外面，需要裁剪
                code_out = code1 if code1 != 0 else code2
                
                # 计算交点
                if code_out & TOP:  # 上边界
                    x = x1 + (x2 - x1) * (0 - y1) / (y2 - y1)
                    y = 0
                elif code_out & BOTTOM:  # 下边界
                    x = x1 + (x2 - x1) * ((img_height - 1) - y1) / (y2 - y1)
                    y = img_height - 1
                elif code_out & RIGHT:  # 右边界
                    y = y1 + (y2 - y1) * ((img_width - 1) - x1) / (x2 - x1)
                    x = img_width - 1
                elif code_out & LEFT:  # 左边界
                    y = y1 + (y2 - y1) * (0 - x1) / (x2 - x1)
                    x = 0
                
                # 更新裁剪后的端点
                if code_out == code1:
                    x1, y1 = x, y
                    code1 = compute_code(x1, y1)
                else:
                    x2, y2 = x, y
                    code2 = compute_code(x2, y2)
        
        if accept:
            return (int(round(x1)), int(round(y1))), (int(round(x2)), int(round(y2))), True
        else:
            return p1, p2, False
    
    def visualize_target_poses_on_image(self, rgb_image):
        """
        在RGB图像上可视化左右目标位姿和实际位姿（从世界坐标系转换）
        
        Args:
            rgb_image: RGB图像 (numpy array)
            
        Returns:
            np.ndarray: 添加了可视化的RGB图像
        """
        if not hasattr(self, '_left_target_pose_world') or not hasattr(self, '_right_target_pose_world'):
            return rgb_image
        
        # 获取相机内参
        intrinsics, fx, fy, cx, cy = self._get_camera_intrinsics()
        
        # 确保图像数组是连续的并且格式正确
        if not rgb_image.flags['C_CONTIGUOUS']:
            rgb_image = np.ascontiguousarray(rgb_image)
        
        # 确保数据类型是uint8
        if rgb_image.dtype != np.uint8:
            rgb_image = rgb_image.astype(np.uint8)
        
        # 转换为BGR格式用于OpenCV
        vis_img = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
        
        # 获取相机在世界坐标系中的位姿
        cam_pose = self.camera.get_entity_pose()  # 使用 get_entity_pose 替代 get_pose
        cam_pos_world = cam_pose.p  # 相机在世界坐标系的位置
        cam_quat_wxyz = cam_pose.q  # 相机在世界坐标系的朝向(wxyz格式)
        
        # 转换四元数为旋转矩阵
        cam_quat_xyzw = np.array([cam_quat_wxyz[1], cam_quat_wxyz[2], cam_quat_wxyz[3], cam_quat_wxyz[0]])
        camera_rotation_matrix = Rotation1.from_quat(cam_quat_xyzw).as_matrix()
        
        # 应用相机坐标系修正矩阵
        R_offset_inv = np.array([
            [0, 0, 1],
            [-1, 0, 0],
            [0, -1, 0]
        ])
        camera_rotation_matrix = camera_rotation_matrix @ R_offset_inv
        
        # 构建世界到相机的变换矩阵
        world_T_cam = np.eye(4)
        world_T_cam[:3, :3] = camera_rotation_matrix
        world_T_cam[:3, 3] = cam_pos_world
        
        # 相机到世界的逆变换
        cam_T_world = np.linalg.inv(world_T_cam)
        
        # 处理左手和右手的目标位姿和实际位姿
        poses_to_visualize = [
            ('Left Target', self._left_target_pose_world, (255, 0, 0)),   # 蓝色（BGR）- 目标位置
            ('Right Target', self._right_target_pose_world, (0, 0, 255)),  # 红色（BGR）- 目标位置
        ]
        
        # 如果有实际位姿，添加到可视化列表
        if hasattr(self, '_final_left_pose') and hasattr(self, '_final_right_pose'):
            poses_to_visualize.extend([
                ('Left Actual', self._final_left_pose, (255, 255, 0)),   # 青色（BGR）- 实际位置
                ('Right Actual', self._final_right_pose, (255, 0, 255))  # 品红色（BGR）- 实际位置
            ])
        
        for label, target_pose_world, color in poses_to_visualize:
            # target_pose_world: [x, y, z, qx, qy, qz, qw] 或 [x, y, z, qw, qx, qy, qz]
            # 统一转换为numpy数组以便处理
            target_pose_world = np.array(target_pose_world)
            pos_world = target_pose_world[:3]
            # 处理不同的四元数格式
            if label.endswith('Actual'):
                # _final_*_pose 是 wxyz 格式: [x, y, z, qw, qx, qy, qz]
                quat_xyzw = np.array([target_pose_world[4], target_pose_world[5], target_pose_world[6], target_pose_world[3]])
            else:
                # _*_target_pose_world 是 wxyz 格式: [x, y, z, qw, qx, qy, qz]
                quat_xyzw = np.array([target_pose_world[4], target_pose_world[5], target_pose_world[6], target_pose_world[3]])
            
            # 构建世界坐标系下的齐次变换矩阵
            T_world_target = np.eye(4)
            T_world_target[:3, 3] = pos_world
            T_world_target[:3, :3] = Rotation1.from_quat(quat_xyzw).as_matrix()
            
            # 转换到相机坐标系
            T_cam_target = cam_T_world @ T_world_target
            
            # 提取相机坐标系下的位置和旋转
            pos_cam = T_cam_target[:3, 3]
            rot_cam = T_cam_target[:3, :3]
            
            # 投影目标位置到图像平面
            px_center, valid_center = self._project_point(pos_cam, intrinsics)
            
            if valid_center and 0 <= px_center[0] < vis_img.shape[1] and 0 <= px_center[1] < vis_img.shape[0]:
                # 绘制目标位置点
                cv2.circle(vis_img, px_center, 8, color, -1)
                cv2.circle(vis_img, px_center, 12, (255, 255, 255), 2)
                
                # 绘制坐标轴
                axis_length = 0.05  # 5cm
                axes_points_local = [
                    np.array([0, 0, 0, 1]),           # 原点
                    np.array([axis_length, 0, 0, 1]), # X轴
                    np.array([0, axis_length, 0, 1]), # Y轴
                    np.array([0, 0, axis_length, 1])  # Z轴
                ]
                
                axes_points_cam = [T_cam_target @ p for p in axes_points_local]
                axes_pixels = []
                axes_valid = []
                
                for i, p in enumerate(axes_points_cam):
                    px, valid = self._project_point(p[:3], intrinsics)
                    axes_pixels.append(px)
                    axes_valid.append(valid)
                    if i == 0:
                        print(f"  {label} - 原点: valid={valid}, px={px}")
                    else:
                        axis_name = ['X', 'Y', 'Z'][i-1]
                        print(f"  {label} - {axis_name}轴端点: valid={valid}, px={px}, cam_pos={p[:3]}")
                
                # 分别绘制每个坐标轴，使用裁剪功能处理超出边界的部分
                if len(axes_pixels) == 4 and axes_valid[0]:  # 原点必须有效（在相机前方）
                    origin_px = axes_pixels[0]
                    
                    # 定义坐标轴颜色和名称
                    axes_info = [
                        (1, 'X', (0, 0, 255)),    # X轴 - 红色 (BGR)
                        (2, 'Y', (0, 255, 0)),    # Y轴 - 绿色 (BGR)
                        (3, 'Z', (255, 0, 0))     # Z轴 - 蓝色 (BGR)
                    ]
                    
                    for idx, axis_name, axis_color in axes_info:
                        if axes_valid[idx]:  # 端点在相机前方
                            # 裁剪线段到图像边界
                            clipped_start, clipped_end, has_visible = self._clip_line_to_image(
                                origin_px, axes_pixels[idx], 
                                vis_img.shape[1], vis_img.shape[0]
                            )
                            
                            if has_visible:
                                cv2.line(vis_img, clipped_start, clipped_end, axis_color, 2)
                                if clipped_end != axes_pixels[idx]:
                                    print(f"  {label} - {axis_name}轴已绘制（裁剪: {origin_px} -> {clipped_end}）")
                                else:
                                    print(f"  {label} - {axis_name}轴已绘制（完整）")
                            else:
                                print(f"  {label} - {axis_name}轴跳过（完全在图像外）")
                        else:
                            print(f"  {label} - {axis_name}轴跳过（端点在相机后方）")
                else:
                    print(f"  {label} - 坐标轴未绘制（原点在相机后方或像素数据不完整）")
                
                # 添加标签
                text = f"{label}"
                # 添加文字背景
                (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                cv2.rectangle(vis_img, (px_center[0] + 15, px_center[1] - text_h - 5), 
                            (px_center[0] + text_w + 20, px_center[1]), (0, 0, 0), -1)
                cv2.putText(vis_img, text, (px_center[0] + 17, px_center[1] - 3), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                print(f"{label}位姿可视化: 世界坐标{pos_world}, 相机坐标{pos_cam}, 像素({px_center[0]}, {px_center[1]})")
            else:
                print(f"警告: {label}位姿在图像外或相机后方，无法可视化")
        
        # 转换回RGB格式
        vis_img = cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB)
        
        return vis_img
    
    def visualize_target_poses_on_third_image(self, rgb_image):
        """
        在third_camera的RGB图像上可视化左右目标位姿和实际位姿（从世界坐标系转换）
        
        Args:
            rgb_image: RGB图像 (numpy array)
            
        Returns:
            np.ndarray: 添加了可视化的RGB图像
        """
        if not hasattr(self, '_left_target_pose_world') or not hasattr(self, '_right_target_pose_world'):
            return rgb_image
        
        if not hasattr(self, 'third_camera') or self.third_camera is None:
            return rgb_image
        
        # 获取相机内参
        intrinsics, fx, fy, cx, cy = self._get_camera_intrinsics()
        
        # 确保图像数组是连续的并且格式正确
        if not rgb_image.flags['C_CONTIGUOUS']:
            rgb_image = np.ascontiguousarray(rgb_image)
        
        # 确保数据类型是uint8
        if rgb_image.dtype != np.uint8:
            rgb_image = rgb_image.astype(np.uint8)
        
        # 转换为BGR格式用于OpenCV
        vis_img = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
        
        # 获取third_camera在世界坐标系中的位姿
        cam_pose = self.third_camera.get_entity_pose()  # 使用 get_entity_pose 替代 get_pose
        cam_pos_world = cam_pose.p  # 相机在世界坐标系的位置
        cam_quat_wxyz = cam_pose.q  # 相机在世界坐标系的朝向(wxyz格式)
        
        # 转换四元数为旋转矩阵
        cam_quat_xyzw = np.array([cam_quat_wxyz[1], cam_quat_wxyz[2], cam_quat_wxyz[3], cam_quat_wxyz[0]])
        camera_rotation_matrix = Rotation1.from_quat(cam_quat_xyzw).as_matrix()
        
        # 应用相机坐标系修正矩阵
        R_offset_inv = np.array([
            [0, 0, 1],
            [-1, 0, 0],
            [0, -1, 0]
        ])
        camera_rotation_matrix = camera_rotation_matrix @ R_offset_inv
        
        # 构建世界到相机的变换矩阵
        world_T_cam = np.eye(4)
        world_T_cam[:3, :3] = camera_rotation_matrix
        world_T_cam[:3, 3] = cam_pos_world
        
        # 相机到世界的逆变换
        cam_T_world = np.linalg.inv(world_T_cam)
        
        # 处理左手和右手的目标位姿和实际位姿
        poses_to_visualize = [
            ('Left Target', self._left_target_pose_world, (255, 0, 0)),   # 蓝色（BGR）- 目标位置
            ('Right Target', self._right_target_pose_world, (0, 0, 255)),  # 红色（BGR）- 目标位置
        ]
        
        # 如果有实际位姿，添加到可视化列表
        if hasattr(self, '_final_left_pose') and hasattr(self, '_final_right_pose'):
            poses_to_visualize.extend([
                ('Left Actual', self._final_left_pose, (255, 255, 0)),   # 青色（BGR）- 实际位置
                ('Right Actual', self._final_right_pose, (255, 0, 255))  # 品红色（BGR）- 实际位置
            ])
        
        for label, target_pose_world, color in poses_to_visualize:
            # target_pose_world: [x, y, z, qx, qy, qz, qw] 或 [x, y, z, qw, qx, qy, qz]
            # 统一转换为numpy数组以便处理
            target_pose_world = np.array(target_pose_world)
            pos_world = target_pose_world[:3]
            # 处理不同的四元数格式
            if label.endswith('Actual'):
                # _final_*_pose 是 wxyz 格式: [x, y, z, qw, qx, qy, qz]
                quat_xyzw = np.array([target_pose_world[4], target_pose_world[5], target_pose_world[6], target_pose_world[3]])
            else:
                # _*_target_pose_world 是 wxyz 格式: [x, y, z, qw, qx, qy, qz]
                quat_xyzw = np.array([target_pose_world[4], target_pose_world[5], target_pose_world[6], target_pose_world[3]])
            
            # 构建世界坐标系下的齐次变换矩阵
            T_world_target = np.eye(4)
            T_world_target[:3, 3] = pos_world
            T_world_target[:3, :3] = Rotation1.from_quat(quat_xyzw).as_matrix()
            
            # 转换到相机坐标系
            T_cam_target = cam_T_world @ T_world_target
            
            # 提取相机坐标系下的位置和旋转
            pos_cam = T_cam_target[:3, 3]
            rot_cam = T_cam_target[:3, :3]
            
            # 投影目标位置到图像平面
            px_center, valid_center = self._project_point(pos_cam, intrinsics)
            
            if valid_center and 0 <= px_center[0] < vis_img.shape[1] and 0 <= px_center[1] < vis_img.shape[0]:
                # 绘制目标位置点
                cv2.circle(vis_img, px_center, 8, color, -1)
                cv2.circle(vis_img, px_center, 12, (255, 255, 255), 2)
                
                # 绘制坐标轴
                axis_length = 0.05  # 5cm
                axes_points_local = [
                    np.array([0, 0, 0, 1]),           # 原点
                    np.array([axis_length, 0, 0, 1]), # X轴
                    np.array([0, axis_length, 0, 1]), # Y轴
                    np.array([0, 0, axis_length, 1])  # Z轴
                ]
                
                axes_points_cam = [T_cam_target @ p for p in axes_points_local]
                axes_pixels = []
                axes_valid = []
                
                for i, p in enumerate(axes_points_cam):
                    px, valid = self._project_point(p[:3], intrinsics)
                    axes_pixels.append(px)
                    axes_valid.append(valid)
                    if i == 0:
                        print(f"  [Third] {label} - 原点: valid={valid}, px={px}")
                    else:
                        axis_name = ['X', 'Y', 'Z'][i-1]
                        print(f"  [Third] {label} - {axis_name}轴端点: valid={valid}, px={px}, cam_pos={p[:3]}")
                
                # 分别绘制每个坐标轴，使用裁剪功能处理超出边界的部分
                if len(axes_pixels) == 4 and axes_valid[0]:  # 原点必须有效（在相机前方）
                    origin_px = axes_pixels[0]
                    
                    # 定义坐标轴颜色和名称
                    axes_info = [
                        (1, 'X', (0, 0, 255)),    # X轴 - 红色 (BGR)
                        (2, 'Y', (0, 255, 0)),    # Y轴 - 绿色 (BGR)
                        (3, 'Z', (255, 0, 0))     # Z轴 - 蓝色 (BGR)
                    ]
                    
                    for idx, axis_name, axis_color in axes_info:
                        if axes_valid[idx]:  # 端点在相机前方
                            # 裁剪线段到图像边界
                            clipped_start, clipped_end, has_visible = self._clip_line_to_image(
                                origin_px, axes_pixels[idx], 
                                vis_img.shape[1], vis_img.shape[0]
                            )
                            
                            if has_visible:
                                cv2.line(vis_img, clipped_start, clipped_end, axis_color, 2)
                                if clipped_end != axes_pixels[idx]:
                                    print(f"  [Third] {label} - {axis_name}轴已绘制（裁剪: {origin_px} -> {clipped_end}）")
                                else:
                                    print(f"  [Third] {label} - {axis_name}轴已绘制（完整）")
                            else:
                                print(f"  [Third] {label} - {axis_name}轴跳过（完全在图像外）")
                        else:
                            print(f"  [Third] {label} - {axis_name}轴跳过（端点在相机后方）")
                else:
                    print(f"  [Third] {label} - 坐标轴未绘制（原点在相机后方或像素数据不完整）")
                
                # 添加标签
                text = f"{label}"
                # 添加文字背景
                (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                cv2.rectangle(vis_img, (px_center[0] + 15, px_center[1] - text_h - 5), 
                            (px_center[0] + text_w + 20, px_center[1]), (0, 0, 0), -1)
                cv2.putText(vis_img, text, (px_center[0] + 17, px_center[1] - 3), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                print(f"[Third] {label}位姿可视化: 世界坐标{pos_world}, 相机坐标{pos_cam}, 像素({px_center[0]}, {px_center[1]})")
            else:
                print(f"[Third] 警告: {label}位姿在图像外或相机后方，无法可视化")
        
        # 转换回RGB格式
        vis_img = cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB)
        
        return vis_img

    def _get_world_to_camera_transform(self, camera_entity):
        """获取世界坐标到指定相机坐标系的变换矩阵。"""
        cam_pose = camera_entity.get_entity_pose()
        cam_pos_world = cam_pose.p
        cam_quat_wxyz = cam_pose.q

        cam_quat_xyzw = np.array([cam_quat_wxyz[1], cam_quat_wxyz[2], cam_quat_wxyz[3], cam_quat_wxyz[0]])
        camera_rotation_matrix = Rotation1.from_quat(cam_quat_xyzw).as_matrix()

        # 与现有投影可视化逻辑保持一致
        R_offset_inv = np.array([
            [0, 0, 1],
            [-1, 0, 0],
            [0, -1, 0]
        ])
        camera_rotation_matrix = camera_rotation_matrix @ R_offset_inv

        world_T_cam = np.eye(4)
        world_T_cam[:3, :3] = camera_rotation_matrix
        world_T_cam[:3, 3] = cam_pos_world
        return np.linalg.inv(world_T_cam)

    def _draw_world_pose_on_bgr(
        self,
        vis_img,
        intrinsics,
        cam_T_world,
        pose_world,
        label,
        color,
        axis_length=0.06,
        view_dir_length=0.12,
    ):
        """将一个世界坐标系下的pose绘制到BGR图像。"""
        T_world = np.eye(4)
        T_world[:3, 3] = pose_world.p
        quat = pose_world.q  # wxyz
        quat_xyzw = np.array([quat[1], quat[2], quat[3], quat[0]], dtype=float)
        T_world[:3, :3] = Rotation1.from_quat(quat_xyzw).as_matrix()

        T_cam = cam_T_world @ T_world
        origin_cam = T_cam[:3, 3]
        px_center, valid_center = self._project_point(origin_cam, intrinsics)
        if not valid_center:
            return

        h, w = vis_img.shape[:2]
        if not (0 <= px_center[0] < w and 0 <= px_center[1] < h):
            return

        cv2.circle(vis_img, px_center, 6, color, -1)
        cv2.circle(vis_img, px_center, 9, (255, 255, 255), 1)

        axes_points_local = [
            np.array([0, 0, 0, 1]),
            np.array([axis_length, 0, 0, 1]),
            np.array([0, axis_length, 0, 1]),
            np.array([0, 0, axis_length, 1]),
        ]
        axes_pixels = []
        axes_valid = []
        for p in axes_points_local:
            p_cam = T_cam @ p
            px, ok = self._project_point(p_cam[:3], intrinsics)
            axes_pixels.append(px)
            axes_valid.append(ok)

        axes_info = [
            (1, (0, 0, 255)),   # X red
            (2, (0, 255, 0)),   # Y green
            (3, (255, 0, 0)),   # Z blue
        ]
        for idx, axis_color in axes_info:
            if not axes_valid[idx]:
                continue
            clipped_start, clipped_end, has_visible = self._clip_line_to_image(
                axes_pixels[0], axes_pixels[idx], w, h
            )
            if has_visible:
                cv2.line(vis_img, clipped_start, clipped_end, axis_color, 2)

        # 相机前向线（SAPIEN相机前向为局部 -Z）
        view_end_local = np.array([0, 0, -view_dir_length, 1])
        view_end_cam = T_cam @ view_end_local
        px_view_end, valid_view = self._project_point(view_end_cam[:3], intrinsics)
        if valid_view:
            clipped_start, clipped_end, has_visible = self._clip_line_to_image(
                axes_pixels[0], px_view_end, w, h
            )
            if has_visible:
                cv2.line(vis_img, clipped_start, clipped_end, (0, 255, 255), 2)

        text = label
        (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 2)
        x0 = min(max(px_center[0] + 10, 0), max(0, w - text_w - 4))
        y0 = min(max(px_center[1] - 6, text_h + 4), h - 2)
        cv2.rectangle(vis_img, (x0 - 2, y0 - text_h - 2), (x0 + text_w + 2, y0 + 2), (0, 0, 0), -1)
        cv2.putText(vis_img, text, (x0, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2, cv2.LINE_AA)

    def visualize_link_cameras_on_third_image(self, rgb_image):
        """在third camera图像上叠加头部/腕部相机位姿。"""
        if not hasattr(self, "third_camera") or self.third_camera is None:
            return rgb_image

        intrinsics, _, _, _, _ = self._get_camera_intrinsics()
        cam_T_world = self._get_world_to_camera_transform(self.third_camera)

        if not rgb_image.flags["C_CONTIGUOUS"]:
            rgb_image = np.ascontiguousarray(rgb_image)
        if rgb_image.dtype != np.uint8:
            rgb_image = rgb_image.astype(np.uint8)
        vis_img = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)

        camera_poses = [
            ("HeadCam", self.zed_camera.get_entity_pose(), (0, 255, 255)),
            ("LeftWristCam", self.left_wrist_camera.get_entity_pose(), (0, 165, 255)),
            ("RightWristCam", self.right_wrist_camera.get_entity_pose(), (255, 0, 255)),
        ]
        for label, pose_world, color in camera_poses:
            self._draw_world_pose_on_bgr(
                vis_img=vis_img,
                intrinsics=intrinsics,
                cam_T_world=cam_T_world,
                pose_world=pose_world,
                label=label,
                color=color,
            )

        return cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB)
    
    # def _draw_axes(self, image, intr, cam_T_obj, length=0.1):
    #     """绘制坐标轴"""
    #     origin = cam_T_obj[:3, 3]
    #     points = np.stack([origin,
    #                        origin + cam_T_obj[:3, 0] * length,
    #                        origin + cam_T_obj[:3, 1] * length,
    #                        origin + cam_T_obj[:3, 2] * length], axis=0)
    #     pixels = []
    #     valid = []
    #     for pt in points:
    #         px, ok = self._project_point(pt, intr)
    #         pixels.append(px)
    #         valid.append(ok)
    #     if not all(valid):
    #         return
    #     origin_px = pixels[0]
    #     cv2.line(image, origin_px, pixels[1], (0, 0, 255), 2)  # X轴 (红色)
    #     cv2.line(image, origin_px, pixels[2], (0, 255, 0), 2)  # Y轴 (绿色)
    #     cv2.line(image, origin_px, pixels[3], (255, 0, 0), 2)  # Z轴 (蓝色)
    
    def save_image(self, image: np.ndarray, filepath: str):
        """保存图像到文件"""
        cv2.imwrite(filepath, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        print(f"图像已保存到: {filepath}")
    
    def close(self):
        """清理资源"""
        if hasattr(self, 'robot') and self.robot is not None:
            if hasattr(self.robot, 'communication_flag') and self.robot.communication_flag:
                if hasattr(self.robot, 'left_conn') and self.robot.left_conn:
                    self.robot.left_conn.close()
                if hasattr(self.robot, 'right_conn') and self.robot.right_conn:
                    self.robot.right_conn.close()
                if hasattr(self.robot, 'left_proc') and self.robot.left_proc.is_alive():
                    self.robot.left_proc.terminate()
                if hasattr(self.robot, 'right_proc') and self.robot.right_proc.is_alive():
                    self.robot.right_proc.terminate()
        
        if hasattr(self, 'scene'):
            for actor in self.scene.get_all_actors():
                self.scene.remove_actor(actor)
        
        sapien_clear_cache()
        print("简化版机器人渲染器已关闭。")
    
    def __del__(self):
        """析构函数"""
        self.close()


def generate_robot_video_from_preprocessed(json_path: str, 
                                           output_video_path: str = "robot_animation.mp4",
                                           fps: int = 30,
                                           model_path: str = None,
                                           poses_path: str = None,
                                           model_path2: str = None,
                                           poses_path2: str = None,
                                           depth_scale: float = 1.0,
                                           depth_offset: float = 0.0,
                                           grasp_json_path: str = None,
                                           link_cam_debug_enable: bool = False,
                                           link_cam_debug_rot_xyz_deg=(0.0, 0.0, 0.0),
                                           link_cam_debug_forward: float = 0.0,
                                           link_cam_debug_right: float = 0.0,
                                           link_cam_debug_up: float = 0.0,
                                           link_cam_axis_mode: str = "none",
                                           link_cam_debug_apply_to: str = "all",
                                           third_cam_show_link_cams: bool = True,
                                           orientation_test_mode: bool = False):
    """
    从预处理的JSON数据生成机器人动作视频
    
    Args:
        json_path: 预处理的JSON数据文件路径
        output_video_path: 输出视频文件路径
        fps: 视频帧率
        depth_scale: 深度缩放因子（调整深度估计误差，物体尺寸会相应调整）
        depth_offset: 深度偏移量（米）
        grasp_json_path: anygrasp抓取数据JSON路径
        link_cam_debug_enable: 是否启用头部/腕部相机调试偏置
        link_cam_debug_rot_xyz_deg: 相机局部RPY旋转（度）
        link_cam_debug_forward/right/up: 按相机局部坐标平移（米）
        link_cam_axis_mode: 轴修正模式（none/yaw_p90/.../swap_xy等）
        link_cam_debug_apply_to: 应用对象，all/head/left/right，可逗号分隔
        third_cam_show_link_cams: 是否在third视角叠加相机位姿可视化
        orientation_test_mode: 是否启用朝向测试模式（测试多个欧拉角组合）
    """
    
    # 开始计时
    total_start_time = time.time()
    current_time = total_start_time
    
    print("\n" + "="*80)
    print("开始时间统计")
    print("="*80)
    
    # 加载预处理的JSON数据
    step_start = time.time()
    with open(json_path, "r") as f:
        data = json.load(f)
    step_end = time.time()
    step_duration = step_end - step_start
    current_time = step_end
    elapsed_time = current_time - total_start_time
    print(f"[时间统计] 加载JSON数据: {step_duration:.4f}秒 | 累计时间: {elapsed_time:.4f}秒")
    
    # 从metadata获取配置信息
    step_start = time.time()
    metadata = data["metadata"]
    image_width = metadata["image_width"]
    image_height = metadata["image_height"]
    fovy_deg = metadata["fovy_deg"]
    num_frames = 1 #metadata["num_frames"]
    lowerest_pose = metadata["lowerest_pose"]
    step_end = time.time()
    step_duration = step_end - step_start
    current_time = step_end
    elapsed_time = current_time - total_start_time
    print(f"[时间统计] 提取metadata: {step_duration:.4f}秒 | 累计时间: {elapsed_time:.4f}秒")
    
    print(f"加载预处理数据: {json_path}")
    print(f"总帧数: {num_frames}")
    print(f"图像尺寸: {image_width}x{image_height}")
    print(f"视场角: {fovy_deg:.2f}°")
    
    # 初始化渲染器
    step_start = time.time()
    renderer = SimplifiedRobotRenderer(
        image_width=image_width,
        image_height=image_height,
        enable_viewer=False,
        fovy_deg=fovy_deg,
        arms_z_offset=0.9,
        model_path=model_path,
        poses_path=poses_path,
        model_path2=model_path2,
        poses_path2=poses_path2,
        depth_scale=depth_scale,
        depth_offset=depth_offset,
        grasp_json_path=grasp_json_path,
        link_cam_debug_enable=link_cam_debug_enable,
        link_cam_debug_rot_xyz_deg=link_cam_debug_rot_xyz_deg,
        link_cam_debug_forward=link_cam_debug_forward,
        link_cam_debug_right=link_cam_debug_right,
        link_cam_debug_up=link_cam_debug_up,
        link_cam_axis_mode=link_cam_axis_mode,
        link_cam_debug_apply_to=link_cam_debug_apply_to,
        third_cam_show_link_cams=third_cam_show_link_cams,
    )
    step_end = time.time()
    step_duration = step_end - step_start
    current_time = step_end
    elapsed_time = current_time - total_start_time
    print(f"[时间统计] 初始化渲染器: {step_duration:.4f}秒 | 累计时间: {elapsed_time:.4f}秒")
    print(f"深度调整参数: depth_scale={depth_scale}, depth_offset={depth_offset}")
    
    # 创建桌子
    lower_height = 0.05
    height_attempt = lowerest_pose[2] - lower_height
    # *** 注释掉桌子的加入逻辑 ***
    # renderer._create_table(height=height_attempt)
    # print(f"桌子高度尝试: {height_attempt}")
    
    # 初始化视频编写器
    step_start = time.time()
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (image_width, image_height))
    
    output_dir = os.path.dirname(output_video_path)
    output_filename = os.path.basename(output_video_path)
    # 只在third_person_view为True时创建third_camera相关视频文件
    if hasattr(renderer, 'third_person_view') and renderer.third_person_view:
        third_output_filename = "third" + output_filename
        third_output_video_path = os.path.join(output_dir, third_output_filename)
        third_video_writer = cv2.VideoWriter(third_output_video_path, fourcc, fps, (image_width, image_height))
        third_depth_output_filename = "third_depth" + output_filename
        third_depth_output_video_path = os.path.join(output_dir, third_depth_output_filename)
        third_depth_video_writer = cv2.VideoWriter(third_depth_output_video_path, fourcc, fps, (image_width, image_height), isColor=False)
    else:
        third_video_writer = None
        third_depth_video_writer = None
    # 创建深度视频路径（在文件名前加"depth")
    # 注意：深度视频使用灰度格式保存原始深度值（单通道）
    depth_output_filename = "depth" + output_filename
    depth_output_video_path = os.path.join(output_dir, depth_output_filename)
    depth_video_writer = cv2.VideoWriter(depth_output_video_path, fourcc, fps, (image_width, image_height), isColor=False)
    
    # 创建帧图片保存目录（在视频目录下）
    frames_dir = os.path.join(output_dir, "frames_debug-orientation_test" if orientation_test_mode else "frames")
    if not os.path.exists(frames_dir):
        os.makedirs(frames_dir)
    
    step_end = time.time()
    step_duration = step_end - step_start
    current_time = step_end
    elapsed_time = current_time - total_start_time
    print(f"[时间统计] 初始化视频编写器: {step_duration:.4f}秒 | 累计时间: {elapsed_time:.4f}秒")
    if hasattr(renderer, 'third_person_view') and renderer.third_person_view:
        print(f"third_camera视频将保存到: {third_output_video_path}")
        print(f"third_camera深度视频将保存到: {third_depth_output_video_path}")
    print(f"深度视频将保存到: {depth_output_video_path}")
    print(f"帧图片将保存到: {frames_dir}")
    
    # 初始化路径规划器（只需要设置一次，使用第一帧的基座位置）
    if num_frames > 0:
        step_start = time.time()
        first_frame_data = data["frames"][0]
        first_robot_base_pos = first_frame_data["robot_base"]["position"]
        first_robot_base_quat = first_frame_data["robot_base"]["quaternion_wxyz"]
        first_camera_position = first_frame_data["camera"]["position"]
        first_camera_rotation = first_frame_data["camera"]["rotation_matrix"]
        
        # 设置第一帧的基座位置
        renderer.set_robot_base_pose_direct(first_robot_base_pos, first_robot_base_quat)
        
        # 设置第一帧的相机位置（用于计算物体位置）
        first_camera_position[2] += 0.4 # 人型机器人高大 
        renderer.set_camera_pose_direct(first_camera_position, first_camera_rotation)
        
        # 初始化路径规划器（只需要设置一次）
        renderer.robot.set_planner(renderer.scene)
        
        # *** Model1将按照JSON数据每帧更新位置（已禁用固定位姿） ***
        # if renderer.model_actor is not None:
        #     renderer.initialize_fixed_model_pose(frame_idx=0, table_height=height_attempt)
        #     print("已初始化固定模型位置（使用第一帧，放置在桌面上）")
        
        step_end = time.time()
        step_duration = step_end - step_start
        current_time = step_end
        elapsed_time = current_time - total_start_time
        print(f"[时间统计] 初始化路径规划器和模型位置: {step_duration:.4f}秒 | 累计时间: {elapsed_time:.4f}秒")
    
    # 用于统计每帧的平均时间
    frame_times = {
        "提取数据": [],
        "设置机器人基座": [],
        "设置相机位姿": [],
        "设置夹爪": [],
        "设置手臂位姿": [],
        "设置模型位姿1": [],
        "设置模型位姿2": [],
        "抓取检测": [],
        "渲染帧": [],
        "图像处理和写入": [],
    }
    
    # 用于存储抓取检测结果
    grasp_detection_results = []
    
    try:
        # 处理每一帧
        # num_frames = 2
        for frame_idx in range(num_frames):
            frame_start_time = time.time()
            print(f"\n=== 处理第 {frame_idx + 1}/{num_frames} 帧 ===")
            
            # 提取数据
            step_start = time.time()
            frame_data = data["frames"][frame_idx]
            
            # 提取相机数据
            camera_position = frame_data["camera"]["position"]
            camera_rotation = frame_data["camera"]["rotation_matrix"]
            
            # 提取机器人基座数据
            robot_base_pos = frame_data["robot_base"]["position"]
            robot_base_quat = frame_data["robot_base"]["quaternion_wxyz"]
            
            # 提取左右手数据
            left_wrist_pos = frame_data["left_hand"]["wrist_position"]
            left_wrist_rot = frame_data["left_hand"]["wrist_rotation_matrix"]
            left_gripper_value = frame_data["left_hand"]["gripper_value"]
            
            right_wrist_pos = frame_data["right_hand"]["wrist_position"]
            right_wrist_rot = frame_data["right_hand"]["wrist_rotation_matrix"]
            right_gripper_value = frame_data["right_hand"]["gripper_value"]
            step_end = time.time()
            step_duration = step_end - step_start
            frame_times["提取数据"].append(step_duration)
            current_time = step_end
            elapsed_time = current_time - total_start_time
            print(f"[时间统计] 帧{frame_idx+1} - 提取数据: {step_duration:.4f}秒 | 累计时间: {elapsed_time:.4f}秒")
            
            print(f"    相机位置: {camera_position}")
            print(f"    相机朝向: {camera_rotation}")
            print(f"    机器人基座位置: {robot_base_pos}")
            print(f"    左腕位置: {left_wrist_pos}")
            print(f"    左腕朝向:{left_wrist_rot}")
            print(f"    右腕位置: {right_wrist_pos}")
            print(f"    右腕朝向: {right_wrist_rot}")
            print(f"    左手夹爪值: {left_gripper_value:.3f}")
            print(f"    右手夹爪值: {right_gripper_value:.3f}")
            
            # 设置机器人基座位置（每帧都设置，因为可能会移动）
            step_start = time.time()
            renderer.set_robot_base_pose_direct(robot_base_pos, robot_base_quat)
            step_end = time.time()
            step_duration = step_end - step_start
            frame_times["设置机器人基座"].append(step_duration)
            current_time = step_end
            elapsed_time = current_time - total_start_time
            print(f"[时间统计] 帧{frame_idx+1} - 设置机器人基座: {step_duration:.4f}秒 | 累计时间: {elapsed_time:.4f}秒")
            
            # 设置相机位姿
            step_start = time.time()
            renderer.set_camera_pose_direct(camera_position, camera_rotation)
            step_end = time.time()
            step_duration = step_end - step_start
            frame_times["设置相机位姿"].append(step_duration)
            current_time = step_end
            elapsed_time = current_time - total_start_time
            print(f"[时间统计] 帧{frame_idx+1} - 设置相机位姿: {step_duration:.4f}秒 | 累计时间: {elapsed_time:.4f}秒")
            
            # 注意：路径规划器已在初始化时设置，不需要每帧重新设置
            
            # 设置夹爪和手臂位姿
            # 如果有best_grasp，只在第一帧执行到抓取位置，后续帧保持不动
            # if frame_idx == 0 or renderer.best_grasp is None:
            # 设置夹爪
            step_start = time.time()
            renderer.set_gripper_direct(left_gripper_value, right_gripper_value)
            step_end = time.time()
            step_duration = step_end - step_start
            frame_times["设置夹爪"].append(step_duration)
            current_time = step_end
            elapsed_time = current_time - total_start_time
            print(f"[时间统计] 帧{frame_idx+1} - 设置夹爪: {step_duration:.4f}秒 | 累计时间: {elapsed_time:.4f}秒")
            
            # 设置手臂位姿（如果有best_grasp，set_arm_poses_direct会自动使用抓取位姿）
            step_start = time.time()
            renderer.set_arm_poses_direct(
                left_wrist_pos=left_wrist_pos,
                left_wrist_rot_matrix=left_wrist_rot,
                right_wrist_pos=right_wrist_pos,
                right_wrist_rot_matrix=right_wrist_rot,
                orientation_test=orientation_test_mode,
                test_frame_idx=frame_idx,
                test_output_dir=frames_dir if orientation_test_mode else None,
            )
            
            # 如果是测试模式，测试完成后直接返回，不继续处理后续帧
            if orientation_test_mode:
                print("\n朝向测试模式完成，退出渲染流程")
                renderer.close()
                return
            
            step_end = time.time()
            step_duration = step_end - step_start
            frame_times["设置手臂位姿"].append(step_duration)
            current_time = step_end
            elapsed_time = current_time - total_start_time
            print(f"[时间统计] 帧{frame_idx+1} - 设置手臂位姿: {step_duration:.4f}秒 | 累计时间: {elapsed_time:.4f}秒")
            
            if renderer.best_grasp is not None and frame_idx == 0:
                print("✓ 机械臂已移动到最佳抓取位置（将在后续帧保持不动）")
            # else:
            #     print("机械臂保持在抓取位置（不更新）")
            #     # 保持时间统计的一致性
            #     frame_times["设置夹爪"].append(0.0)
            #     frame_times["设置手臂位姿"].append(0.0)
            
            # 设置模型位姿（如果模型已加载）
            if renderer.model_actor is not None:
                step_start = time.time()
                renderer.set_model_pose_from_camera_coords(frame_idx)
                step_end = time.time()
                step_duration = step_end - step_start
                frame_times["设置模型位姿1"].append(step_duration)
                current_time = step_end
                elapsed_time = current_time - total_start_time
                print(f"[时间统计] 帧{frame_idx+1} - 设置模型位姿1: {step_duration:.4f}秒 | 累计时间: {elapsed_time:.4f}秒")
            else:
                # 即使没有执行，也要更新当前时间以保持累计时间的准确性
                current_time = time.time()
            
            # 设置第二个模型位姿（如果第二个模型已加载）
            if renderer.model_actor2 is not None:
                step_start = time.time()
                renderer.set_model_pose_from_camera_coords2(frame_idx)
                step_end = time.time()
                step_duration = step_end - step_start
                frame_times["设置模型位姿2"].append(step_duration)
                current_time = step_end
                elapsed_time = current_time - total_start_time
                print(f"[时间统计] 帧{frame_idx+1} - 设置模型位姿2: {step_duration:.4f}秒 | 累计时间: {elapsed_time:.4f}秒")
            else:
                # 即使没有执行，也要更新当前时间以保持累计时间的准确性
                current_time = time.time()

            # *** 抓取检测相关功能已注释 ***
            # if renderer.model_actor is not None:
            #     step_start = time.time()
            #     print(f"\n>>> 帧 {frame_idx}: 开始抓取检测...")
            #     contact_result = renderer.close_grippers_and_detect_contact(arm="both", close_value=0.0)
            #     # 记录结果，包含物体位置
            #     model_position = None
            #     if renderer.fixed_model_pose is not None:
            #         model_position = renderer.fixed_model_pose.p.tolist()
            #     frame_grasp_result = {
            #         "frame": frame_idx,
            #         "left_contact": contact_result["left_contact"],
            #         "right_contact": contact_result["right_contact"],
            #         "grasped": contact_result["grasped"],
            #         "contact_details": contact_result["contact_details"],
            #         "model_position": model_position,  # 记录物体位置
            #         "depth_scale": depth_scale,  # 记录深度参数
            #         "depth_offset": depth_offset
            #     }
            #     grasp_detection_results.append(frame_grasp_result)
            #     # 打印结果
            #     print(f">>> 帧 {frame_idx} 抓取检测结果:")
            #     print(f"    左手接触: {contact_result['left_contact']}")
            #     print(f"    右手接触: {contact_result['right_contact']}")
            #     print(f"    是否抓住: {contact_result['grasped']}")
            #     if model_position:
            #         print(f"    物体位置: {model_position}")
            #     if contact_result['contact_details']:
            #         print(f"    接触详情:")
            #         for detail in contact_result['contact_details']:
            #             print(f"      - {detail['arm']}手: {detail.get('other', detail.get('actor', 'unknown'))}")
            #     step_end = time.time()
            #     step_duration = step_end - step_start
            #     frame_times["抓取检测"].append(step_duration)
            #     current_time = step_end
            #     elapsed_time = current_time - total_start_time
            #     print(f"[时间统计] 帧{frame_idx+1} - 抓取检测: {step_duration:.4f}秒 | 累计时间: {elapsed_time:.4f}秒")
            # else:
            #     current_time = time.time()
            
            # 每帧先同步robot-link相机位姿，避免腕部/头部相机滞后
            renderer.update_robot_link_cameras()

            # 渲染当前帧
            step_start = time.time()
            rgb_image, depth_image = renderer.render_frame(frame_idx)
            step_end = time.time()
            step_duration = step_end - step_start
            frame_times["渲染帧"].append(step_duration)
            current_time = step_end
            elapsed_time = current_time - total_start_time
            print(f"[时间统计] 帧{frame_idx+1} - 渲染帧: {step_duration:.4f}秒 | 累计时间: {elapsed_time:.4f}秒")
            
            # 渲染third_camera帧（可选）
            if hasattr(renderer, 'third_person_view') and renderer.third_person_view and hasattr(renderer, 'third_camera') and renderer.third_camera is not None:
                step_start = time.time()
                third_rgb_image, third_depth_image = renderer.render_third_camera_frame(frame_idx)
                step_end = time.time()
                step_duration = step_end - step_start
                current_time = step_end
                elapsed_time = current_time - total_start_time
                print(f"[时间统计] 帧{frame_idx+1} - 渲染third_camera帧: {step_duration:.4f}秒 | 累计时间: {elapsed_time:.4f}秒")
            else:
                third_rgb_image, third_depth_image = None, None

            # 转换为BGR格式
            step_start = time.time()
            bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
            # 左上角叠加帧号信息
            frame_text = f"Frame {frame_idx + 1}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1
            thickness = 1
            x0, y0 = 10, 10
            (text_size, baseline) = cv2.getTextSize(frame_text, font, font_scale, thickness)
            cv2.putText(bgr_image, frame_text, (x0, y0 + text_size[1] + 2), 
                       font, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)
            video_writer.write(bgr_image)
            # ==================== 深度图像处理和分析 ====================
            # 检查深度图像是否有无效值
            has_nan = np.isnan(depth_image).any()
            has_inf = np.isinf(depth_image).any()
            if has_nan or has_inf:
                print(f"[警告] 帧{frame_idx} - 主相机深度图像包含无效值: NaN={has_nan}, Inf={has_inf}")
                # 替换无效值为0
                depth_image = np.nan_to_num(depth_image, nan=0.0, posinf=0.0, neginf=0.0)
            
            # 分析深度值范围（注意：depth_image现在是毫米单位）
            depth_min_mm = depth_image.min()
            depth_max_mm = depth_image.max()
            depth_mean_mm = depth_image.mean()
            depth_median_mm = np.median(depth_image)
            
            # 转换为米单位用于显示
            depth_min = depth_min_mm / 1000.0
            depth_max = depth_max_mm / 1000.0
            depth_mean = depth_mean_mm / 1000.0
            depth_median = depth_median_mm / 1000.0
            
            # 打印深度分析信息
            print(f"\n[深度分析] 帧{frame_idx} - 主相机:")
            print(f"  最近距离: {depth_min:.4f}m ({depth_min_mm:.2f}mm)")
            print(f"  最远距离: {depth_max:.4f}m ({depth_max_mm:.2f}mm)")
            print(f"  平均深度: {depth_mean:.4f}m")
            print(f"  中位深度: {depth_median:.4f}m")
            print(f"  数据类型: {depth_image.dtype}, 形状: {depth_image.shape}")
            
            # 保存深度图像（多种格式）
            # 1. uint16格式（毫米单位，用于快速加载）
            depth_uint16 = depth_image.astype(np.uint16)
            depth_frame_path = os.path.join(frames_dir, f"depth_{frame_idx:04d}.png")
            cv2.imwrite(depth_frame_path, depth_uint16)
            
            # 2. 以米为单位的原始float32格式（.npy文件，保留完整精度）
            depth_meters = depth_image / 1000.0
            depth_meters_path = os.path.join(frames_dir, f"depth_meters_{frame_idx:04d}.npy")
            np.save(depth_meters_path, depth_meters.astype(np.float32))
            
            # 为视频写入，归一化到0-255的灰度图
            if depth_max_mm > depth_min_mm:
                depth_normalized = ((depth_image - depth_min_mm) / (depth_max_mm - depth_min_mm) * 255).astype(np.uint8)
            else:
                depth_normalized = np.zeros_like(depth_image, dtype=np.uint8)
            
            # 写入深度视频（灰度图，单通道）
            depth_video_writer.write(depth_normalized)
            
            # 保存每一帧RGB图片到视频目录
            rgb_frame_path = os.path.join(frames_dir, f"rgb_{frame_idx:04d}.png")
            cv2.imwrite(rgb_frame_path, bgr_image)
            
            # 如果是第一帧且有抓取可视化，额外保存一张高亮图
            if frame_idx == 0 and renderer.best_grasp is not None:
                grasp_vis_path = os.path.join(frames_dir, "best_grasp_visualization.png")
                cv2.imwrite(grasp_vis_path, bgr_image)
                print(f"[抓取可视化] 最佳抓取可视化已保存到: {grasp_vis_path}")
            
            # 处理和保存third_camera的图像和视频（可选）
            if hasattr(renderer, 'third_person_view') and renderer.third_person_view and hasattr(renderer, 'third_camera') and renderer.third_camera is not None and third_rgb_image is not None:
                third_bgr_image = cv2.cvtColor(third_rgb_image, cv2.COLOR_RGB2BGR)
                # 左上角叠加帧号信息
                cv2.putText(third_bgr_image, frame_text, (x0, y0 + text_size[1] + 2), 
                           font, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)
                # 写入third_camera视频
                third_video_writer.write(third_bgr_image)
                # 保存third_camera的RGB图片
                third_rgb_frame_path = os.path.join(frames_dir, f"third_rgb_{frame_idx:04d}.png")
                cv2.imwrite(third_rgb_frame_path, third_bgr_image)
                # ==================== Third相机深度图像处理和分析 ====================
                if third_depth_image is not None:
                    # 检查深度图像是否有无效值
                    has_nan = np.isnan(third_depth_image).any()
                    has_inf = np.isinf(third_depth_image).any()
                    if has_nan or has_inf:
                        print(f"[警告] 帧{frame_idx} - Third相机深度图像包含无效值: NaN={has_nan}, Inf={has_inf}")
                        # 替换无效值为0
                        third_depth_image = np.nan_to_num(third_depth_image, nan=0.0, posinf=0.0, neginf=0.0)
                    # 分析深度值范围（注意：third_depth_image现在是毫米单位）
                    third_depth_min_mm = third_depth_image.min()
                    third_depth_max_mm = third_depth_image.max()
                    third_depth_mean_mm = third_depth_image.mean()
                    third_depth_median_mm = np.median(third_depth_image)
                    # 转换为米单位用于显示
                    third_depth_min = third_depth_min_mm / 1000.0
                    third_depth_max = third_depth_max_mm / 1000.0
                    third_depth_mean = third_depth_mean_mm / 1000.0
                    third_depth_median = third_depth_median_mm / 1000.0
                    # 打印深度分析信息
                    print(f"[深度分析] 帧{frame_idx} - Third相机:")
                    print(f"  最近距离: {third_depth_min:.4f}m ({third_depth_min_mm:.2f}mm)")
                    print(f"  最远距离: {third_depth_max:.4f}m ({third_depth_max_mm:.2f}mm)")
                    print(f"  平均深度: {third_depth_mean:.4f}m")
                    print(f"  中位深度: {third_depth_median:.4f}m")
                    print(f"  数据类型: {third_depth_image.dtype}, 形状: {third_depth_image.shape}")
                    print(f"  平均深度: {third_depth_mean:.4f}m")
                    print(f"  中位深度: {third_depth_median:.4f}m")
                    # 保存深度图像（多种格式）
                    # 1. uint16格式（毫米单位）
                    third_depth_uint16 = third_depth_image.astype(np.uint16)
                    third_depth_frame_path = os.path.join(frames_dir, f"third_depth_{frame_idx:04d}.png")
                    cv2.imwrite(third_depth_frame_path, third_depth_uint16)
                    # 2. 以米为单位的原始float32格式（.npy文件）
                    third_depth_meters = third_depth_image / 1000.0
                    third_depth_meters_path = os.path.join(frames_dir, f"third_depth_meters_{frame_idx:04d}.npy")
                    np.save(third_depth_meters_path, third_depth_meters.astype(np.float32))
                    # 为视频写入，归一化到0-255的灰度图
                    if third_depth_max_mm > third_depth_min_mm:
                        third_depth_normalized = ((third_depth_image - third_depth_min_mm) / (third_depth_max_mm - third_depth_min_mm) * 255).astype(np.uint8)
                    else:
                        third_depth_normalized = np.zeros_like(third_depth_image, dtype=np.uint8)
                    # 写入third_camera深度视频（灰度图，单通道）
                    third_depth_video_writer.write(third_depth_normalized)
            
            step_end = time.time()
            step_duration = step_end - step_start
            frame_times["图像处理和写入"].append(step_duration)
            current_time = step_end
            elapsed_time = current_time - total_start_time
            print(f"[时间统计] 帧{frame_idx+1} - 图像处理和写入: {step_duration:.4f}秒 | 累计时间: {elapsed_time:.4f}秒")
            
            frame_duration = step_end - frame_start_time
            print(f"[时间统计] 帧{frame_idx+1} - 总计: {frame_duration:.4f}秒")
            print(f"Frame {frame_idx} RGB saved to {rgb_frame_path}")
            print(f"Frame {frame_idx} Depth saved to {depth_frame_path}")
            print(f"Frame {frame_idx} Depth (meters) saved to {depth_meters_path}\n")
        
        total_end_time = time.time()
        total_duration = total_end_time - total_start_time
        
        print(f"\n视频生成完成！保存到: {output_video_path}")
        if hasattr(renderer, 'third_person_view') and renderer.third_person_view:
            print(f"third_camera视频生成完成！保存到: {third_output_video_path}")
            print(f"third_camera深度视频生成完成！保存到: {third_depth_output_video_path}")
        print(f"深度视频生成完成！保存到: {depth_output_video_path}")
        print(f"视频参数: {num_frames}帧, {fps}fps, 分辨率{image_width}x{image_height}")
        
        # 保存抓取检测结果
        if grasp_detection_results:
            grasp_results_path = output_video_path.replace('.mp4', '_grasp_detection.json')
            with open(grasp_results_path, 'w') as f:
                json.dump(grasp_detection_results, f, indent=2)
            print(f"\n抓取检测结果已保存到: {grasp_results_path}")
            
            # 打印抓取检测摘要
            print(f"\n{'='*80}")
            print("抓取检测摘要")
            print('='*80)
            grasped_frames = [r for r in grasp_detection_results if r['grasped']]
            if grasped_frames:
                print(f"成功抓取的帧数: {len(grasped_frames)}/{len(grasp_detection_results)}")
                print(f"成功抓取的帧: {[r['frame'] for r in grasped_frames]}")
                print(f"第一次成功抓取: 帧 {grasped_frames[0]['frame']}")
                
                # 统计左右手抓取情况
                left_grasps = [r for r in grasped_frames if r['left_contact']]
                right_grasps = [r for r in grasped_frames if r['right_contact']]
                both_grasps = [r for r in grasped_frames if r['left_contact'] and r['right_contact']]
                
                print(f"左手抓取: {len(left_grasps)} 帧")
                print(f"右手抓取: {len(right_grasps)} 帧")
                print(f"双手同时抓取: {len(both_grasps)} 帧")
            else:
                print("未检测到任何成功的抓取")
            print('='*80)
        
        # 打印时间统计摘要
        print("\n" + "="*80)
        print("时间统计摘要")
        print("="*80)
        print(f"总运行时间: {total_duration:.4f}秒 ({total_duration/60:.2f}分钟)")
        print("\n各步骤时间统计:")
        print("-" * 80)
        
        # 打印初始化阶段的时间
        init_times = []
        print("\n[初始化阶段]")
        # 这些时间已经在前面记录了
        
        # 打印每帧处理的时间统计
        print(f"\n[每帧处理阶段 - 共{num_frames}帧]")
        for step_name, times_list in frame_times.items():
            if len(times_list) > 0:
                avg_time = sum(times_list) / len(times_list)
                total_step_time = sum(times_list)
                max_time = max(times_list)
                min_time = min(times_list)
                percentage = (total_step_time / total_duration) * 100 if total_duration > 0 else 0
                print(f"  {step_name:20s}: 平均={avg_time:.4f}秒, 总计={total_step_time:.4f}秒, "
                      f"最大={max_time:.4f}秒, 最小={min_time:.4f}秒, 占比={percentage:.2f}%")
        
        print("\n" + "="*80)
        
    except Exception as e:
        print(f"视频生成过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        # 清理资源
        step_start = time.time()
        video_writer.release()
        if hasattr(renderer, 'third_person_view') and renderer.third_person_view:
            if 'third_video_writer' in locals() and third_video_writer is not None:
                third_video_writer.release()
            if 'third_depth_video_writer' in locals() and third_depth_video_writer is not None:
                third_depth_video_writer.release()
        if 'depth_video_writer' in locals() and depth_video_writer is not None:
            depth_video_writer.release()
        renderer.close()
        step_end = time.time()
        step_duration = step_end - step_start
        current_time = step_end
        elapsed_time = current_time - total_start_time
        print(f"[时间统计] 清理资源: {step_duration:.4f}秒 | 累计时间: {elapsed_time:.4f}秒")


def create_side_by_side_video(original_video_path: str, 
                              generated_video_path: str, 
                              output_path: str, 
                              fps: int = 30):
    """
    创建原始视频和生成视频并排的对比视频
    
    Args:
        original_video_path: 原始视频路径
        generated_video_path: 生成的机器人视频路径
        output_path: 输出并排视频路径
        fps: 输出视频的帧率
    """
    # 打开原始视频和生成的视频
    cap_original = cv2.VideoCapture(original_video_path)
    cap_generated = cv2.VideoCapture(generated_video_path)
    
    # 检查视频是否打开成功
    if not cap_original.isOpened() or not cap_generated.isOpened():
        print(f"错误: 无法打开视频文件")
        if not cap_original.isOpened():
            print(f"  - 原始视频文件不存在或损坏: {original_video_path}")
        if not cap_generated.isOpened():
            print(f"  - 生成的视频文件不存在或损坏: {generated_video_path}")
        return
    
    # 获取视频尺寸和帧数
    width_orig = int(cap_original.get(cv2.CAP_PROP_FRAME_WIDTH))
    height_orig = int(cap_original.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width_gen = int(cap_generated.get(cv2.CAP_PROP_FRAME_WIDTH))
    height_gen = int(cap_generated.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count_orig = int(cap_original.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_count_gen = int(cap_generated.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"原始视频: {width_orig}x{height_orig}, {frame_count_orig} 帧")
    print(f"生成视频: {width_gen}x{height_gen}, {frame_count_gen} 帧")
    
    # 计算目标尺寸
    target_height = max(height_orig, height_gen)
    combined_width = width_orig + width_gen
    
    # 创建视频写入器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (combined_width, target_height))
    
    # 计算处理的帧数
    frame_count = min(frame_count_orig, frame_count_gen)
    
    print(f"开始创建并排视频，总共 {frame_count} 帧...")
    
    # 处理每一帧
    for i in range(frame_count):
        # 读取原始视频帧
        ret_orig, frame_orig = cap_original.read()
        if not ret_orig:
            break
        
        # 读取生成的视频帧
        ret_gen, frame_gen = cap_generated.read()
        if not ret_gen:
            break
        
        # 调整尺寸
        if height_orig != target_height:
            frame_orig = cv2.resize(frame_orig, (width_orig, target_height))
        if height_gen != target_height:
            frame_gen = cv2.resize(frame_gen, (width_gen, target_height))
        
        # 创建并排帧
        combined_frame = np.hstack((frame_orig, frame_gen))
        
        # 添加帧号信息
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(combined_frame, f"Frame: {i+1}/{frame_count}", 
                   (10, 30), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
        
        # 写入帧
        out.write(combined_frame)
        
        # 每10帧显示一次进度
        if i % 10 == 0:
            print(f"处理中: {i+1}/{frame_count} 帧 ({(i+1)/frame_count*100:.1f}%)")
    
    # 释放资源
    cap_original.release()
    cap_generated.release()
    out.release()
    
    print(f"并排视频创建完成，保存到: {output_path}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="从预处理JSON生成机器人视频")
    parser.add_argument("--task", "-t", type=str, default="assemble_disassemble_furniture_bench_lamp",
                       help="任务名称")
    parser.add_argument("--id", "-i", type=str, default="0",
                       help="视频ID")
    parser.add_argument("--fps", "-f", type=int, default=5,
                       help="视频帧率 (默认: 5)")
    parser.add_argument("--pure", type=int, default=1,
                       help="是否纯渲染[MLP的结果] (默认: 1=True) 0=False,使用hdf5_aloha.py生成的json")
    parser.add_argument("--lg", type=int, default=2,
                       help="左手手指配置 (用于文件名)")
    parser.add_argument("--rg", type=int, default=2,
                       help="右手手指配置 (用于文件名)")
    parser.add_argument("--mode", "-m", type=int, choices=[2, 3], default=2,
                       help="运行模式: 2=视频生成, 3=创建并排视频")
    # 按固定格式通过 session 自动构建路径
    parser.add_argument("--session", type=str, default="session_1029163454",
                       help="会话ID，用于自动构建模型与位姿路径，如 session_XXXXXXXXXX")
    parser.add_argument("--session2", type=str, default=None,
                       help="第二个会话ID，用于自动构建第二个模型与位姿路径，如 session_XXXXXXXXXX")
    parser.add_argument("--depth_scale", type=float, default=1.0,
                       help="深度缩放因子（调整深度估计误差，物体尺寸会相应缩放，默认1.0）")
    parser.add_argument("--depth_offset", type=float, default=0.0,
                       help="深度偏移量（米，正值使物体远离相机，默认0.0）")
    parser.add_argument("--test_orientation", action="store_true",
                       help="启用朝向测试模式（测试多个欧拉角组合并保存图片）")
    parser.add_argument("--link_cam_debug_enable", type=int, default=0,
                       help="启用头部/腕部相机调试偏置: 0=关闭, 1=开启")
    parser.add_argument("--link_cam_debug_rot_x_deg", type=float, default=0.0,
                       help="相机调试旋转X（度，局部坐标）")
    parser.add_argument("--link_cam_debug_rot_y_deg", type=float, default=0.0,
                       help="相机调试旋转Y（度，局部坐标）")
    parser.add_argument("--link_cam_debug_rot_z_deg", type=float, default=0.0,
                       help="相机调试旋转Z（度，局部坐标）")
    parser.add_argument("--link_cam_debug_forward", type=float, default=0.0,
                       help="沿相机原始前向平移（米，正值前进）")
    parser.add_argument("--link_cam_debug_right", type=float, default=0.0,
                       help="沿相机原始右向平移（米）")
    parser.add_argument("--link_cam_debug_up", type=float, default=0.0,
                       help="沿相机原始上向平移（米）")
    parser.add_argument("--link_cam_axis_mode", type=str, default="none",
                       help="相机轴修正模式: none/yaw_p90/yaw_n90/pitch_p90/pitch_n90/roll_p90/roll_n90/swap_xy/swap_xz/swap_yz")
    parser.add_argument("--link_cam_debug_apply_to", type=str, default="all",
                       help="相机调试应用对象: all/head/left/right，可逗号分隔")
    parser.add_argument("--third_cam_show_link_cams", type=int, default=1,
                       help="是否在third视角叠加头部/腕部相机位姿: 0=否, 1=是")
    args = parser.parse_args()
    # 基于 session 构建固定格式路径
    base_dir = "/data1/zjyang/program/OnePoseviaGen/temp_local/pour/"
    # base_dir = "/data1/zjyang/program/OnePoseviaGen/temp_local"
    # /data1/zjyang/program/OnePoseviaGen/temp_local/pour/pour_37_left_bottle_right_cup/object_1/pose_result/poses.json
    # /data1/zjyang/program/OnePoseviaGen/temp_local/pour/pour_37_left_bottle_right_cup/object_1/model/mid_files/scaled_mesh.obj
    session_dir = os.path.join(base_dir, args.session)
    obj1_dir = os.path.join(session_dir, "object_1")
    model_path = f"{obj1_dir}/model/mid_files/scaled_mesh.obj"
    poses_path = f"{obj1_dir}/pose_result/poses.json"
    print(f"model_path: {model_path}")
    print(f"poses_path: {poses_path}")
    
    # 基于 session2 构建第二个模型的固定格式路径（如果提供了 session2）
    obj2_dir = os.path.join(session_dir, "object_2")
    model_path2 = f"{obj2_dir}/model/mid_files/scaled_mesh.obj"
    poses_path2 = f"{obj2_dir}/pose_result/poses.json"
    print(f"model_path2: {model_path2}")
    print(f"poses_path2: {poses_path2}")
    # model_path2 = None
    # poses_path2 = None
    # if args.session2 is not None:
    #     session_dir2 = os.path.join(base_dir, args.session2)
    #     model_path2 = f"{session_dir2}/model/mid_files/scaled_mesh.obj"
    #     poses_path2 = f"{session_dir2}/pose_result/poses.json"
    #     print(f"model_path2: {model_path2}")
    #     print(f"poses_path2: {poses_path2}")
    if args.pure == 1:
        args.pure = True
    else:
        args.pure = False
    print(f"args.pure: {args.pure}")
    if args.pure: # MLP 结果
        # json_path = f"/data1/zjyang/program/third/RoboTwin/data_process/results_raw_rot/{args.task}/{args.task}_{args.id}.json"
        json_path = f"/data1/zjyang/program/third/RoboTwin/data_process/results/{args.task}/{args.task}_{args.id}.json"
    else:
        # 构建文件路径（与hdf5_aloha.py对齐）
        # json_path = f"/data1/zjyang/program/third/RoboTwin/data_process/processed/{args.task}_{args.id}_lg{args.lg}_rg{args.rg}.json"
        # json_path = f"/data1/zjyang/program/third/RoboTwin/data_process/processed/world_rot_no_basepose_backward/pour_0_lg2_rg2_sfNone_efNone.json"
        # json_path = f"/data1/zjyang/program/third/RoboTwin/data_process/processed/world_rot_debug/pour_0_lg2_rg2_sfNone_efNone.json"
        json_path = f"/data1/zjyang/program/third/RoboTwin/data_process/processed/world_raw_rot_h09_base040-090/pour_0_lg2_rg2_sfNone_efNone.json"
    original_video_path = f"/data1/zjyang/program/egodex/egodex_stored/{args.task}/{args.id}.mp4"
        
    # 输出视频路径
    output_video_dir = f"/data1/zjyang/program/third/RoboTwin/code_painting/{args.task}/debug_depth"
    if args.pure:
        output_video_path = f"{output_video_dir}/{args.task}_{args.id}_{args.fps}fps.mp4"
        side_by_side_path = f"{output_video_dir}/{args.id}_{args.fps}fps_side_by_side.mp4"
    else:   
        output_video_path = f"{output_video_dir}/{args.task}_{args.id}_lg{args.lg}_rg{args.rg}_{args.fps}fps.mp4"
        side_by_side_path = f"{output_video_dir}/{args.id}_lg{args.lg}_rg{args.rg}_side_by_side_{args.fps}fps.mp4"
    
    if not os.path.exists(output_video_dir):
        os.makedirs(output_video_dir)
    
    # 设置抓取JSON路径
    grasp_json_path = "/data1/zjyang/program/third/RoboTwin/code_painting/pour/debug_depth/frames/grasp_can.json"
    if not os.path.exists(grasp_json_path):
        print(f"警告: 抓取JSON文件不存在: {grasp_json_path}")
        grasp_json_path = None
    else:
        print(f"使用抓取数据: {grasp_json_path}")
    
    if args.mode == 2:
        # 视频生成模式
        if args.test_orientation:
            print(f"模式2: 朝向测试模式 - 测试多个欧拉角组合")
        else:
            print(f"模式2: 从预处理JSON生成机器人视频")
        print(f"输入JSON: {json_path}")
        print(f"输出视频: {output_video_path}")
        
        if not os.path.exists(json_path):
            print(f"错误: JSON文件不存在: {json_path}")
            print(f"请先运行 hdf5_aloha.py 生成预处理数据")
            return
        
        generate_robot_video_from_preprocessed(
            json_path=json_path,
            output_video_path=output_video_path,
            fps=args.fps,
            model_path=model_path,
            poses_path=poses_path,
            model_path2=model_path2,
            poses_path2=poses_path2,
            depth_scale=args.depth_scale,
            depth_offset=args.depth_offset,
            grasp_json_path=grasp_json_path,
            link_cam_debug_enable=(args.link_cam_debug_enable == 1),
            link_cam_debug_rot_xyz_deg=(
                args.link_cam_debug_rot_x_deg,
                args.link_cam_debug_rot_y_deg,
                args.link_cam_debug_rot_z_deg,
            ),
            link_cam_debug_forward=args.link_cam_debug_forward,
            link_cam_debug_right=args.link_cam_debug_right,
            link_cam_debug_up=args.link_cam_debug_up,
            link_cam_axis_mode=args.link_cam_axis_mode,
            link_cam_debug_apply_to=args.link_cam_debug_apply_to,
            third_cam_show_link_cams=(args.third_cam_show_link_cams == 1),
            orientation_test_mode=args.test_orientation
        )
        
    elif args.mode == 3:
        # 创建并排视频模式
        print(f"模式3: 创建并排对比视频")
        print(f"原始视频: {original_video_path}")
        print(f"生成视频: {output_video_path}")
        print(f"输出视频: {side_by_side_path}")
        
        if not os.path.exists(original_video_path):
            print(f"错误: 原始视频不存在: {original_video_path}")
            return
        
        if not os.path.exists(output_video_path):
            print(f"错误: 生成的视频不存在: {output_video_path}")
            print(f"请先运行模式2生成机器人视频")
            return
        
        create_side_by_side_video(
            original_video_path=original_video_path,
            generated_video_path=output_video_path,
            output_path=side_by_side_path,
            fps=args.fps
        )


if __name__ == "__main__":
    main()