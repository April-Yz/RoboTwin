import os
import sys
import time
from argparse import ArgumentParser

import numpy as np
import sapien.core as sapien
import transforms3d as t3d
import yaml

sys.path.append("./")

from script.collect_data import class_decorator, get_embodiment_config
from envs import CONFIGS_PATH


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--task_name", default="pick_diverse_bottles_piper")
    parser.add_argument("--task_config", default="demo_clean_piper_calibrated_viewer")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max_seed_tries", type=int, default=50)
    parser.add_argument("--render_freq", type=int, default=1)
    parser.add_argument("--show_axes", type=int, default=1)
    parser.add_argument("--hold", type=int, default=1)
    args_cli = parser.parse_args()

    config_path = f"./task_config/{args_cli.task_config}.yml"
    with open(config_path, "r", encoding="utf-8") as f:
        args = yaml.load(f.read(), Loader=yaml.FullLoader)

    args["task_name"] = args_cli.task_name
    args["task_config"] = args_cli.task_config
    args["render_freq"] = args_cli.render_freq
    args["need_plan"] = False
    args["save_data"] = False
    args["collect_data"] = False
    args["skip_planner"] = True

    with open(os.path.join(CONFIGS_PATH, "_embodiment_config.yml"), "r", encoding="utf-8") as f:
        embodiment_types = yaml.load(f.read(), Loader=yaml.FullLoader)

    embodiment = args["embodiment"]

    def embodiment_file(name):
        return embodiment_types[name]["file_path"]

    if len(embodiment) == 1:
        args["left_robot_file"] = embodiment_file(embodiment[0])
        args["right_robot_file"] = embodiment_file(embodiment[0])
        args["dual_arm_embodied"] = True
        embodiment_name = str(embodiment[0])
    elif len(embodiment) == 3:
        args["left_robot_file"] = embodiment_file(embodiment[0])
        args["right_robot_file"] = embodiment_file(embodiment[1])
        args["embodiment_dis"] = embodiment[2]
        args["dual_arm_embodied"] = False
        embodiment_name = f"{embodiment[0]}+{embodiment[1]}"
    else:
        raise ValueError("embodiment must contain either 1 or 3 values")

    args["left_embodiment_config"] = get_embodiment_config(args["left_robot_file"])
    args["right_embodiment_config"] = get_embodiment_config(args["right_robot_file"])
    args["embodiment_name"] = embodiment_name

    print(f"[viewer-scene] task={args_cli.task_name} config={args_cli.task_config} seed_start={args_cli.seed}")
    print(f"[viewer-scene] embodiment={embodiment_name}")
    print("[viewer-scene] loading scene only; planner/play_once are intentionally skipped")

    task = None
    loaded_seed = None
    for seed in range(args_cli.seed, args_cli.seed + args_cli.max_seed_tries):
        task = class_decorator(args_cli.task_name)
        try:
            task.setup_demo(now_ep_num=0, seed=seed, **args)
            loaded_seed = seed
            break
        except Exception as exc:
            print(f"[viewer-scene] skip seed={seed}: {exc}")
            try:
                task.close_env()
            except Exception:
                pass
            task = None

    if task is None or loaded_seed is None:
        raise RuntimeError(
            f"failed to load a stable scene from seed {args_cli.seed} "
            f"to {args_cli.seed + args_cli.max_seed_tries - 1}"
        )

    print(f"[viewer-scene] loaded stable scene seed={loaded_seed}")
    if args_cli.show_axes:
        add_scene_debug_axes(task)
        print("[viewer-scene] debug axes: bottle centers and left/right env target poses")
    if not args_cli.hold:
        task._update_render()
        viewer = getattr(task, "viewer", None)
        if viewer is not None:
            viewer.render()
        task.close_env()
        print("[viewer-scene] hold=0, rendered one frame and exited")
        return
    print("[viewer-scene] viewer loaded; close the SAPIEN window or press Ctrl-C to exit")
    try:
        while True:
            task._update_render()
            viewer = getattr(task, "viewer", None)
            if viewer is None or getattr(viewer, "window", None) is None:
                print("[viewer-scene] viewer window closed")
                break
            try:
                viewer.render()
            except AttributeError as exc:
                if "should_close" not in str(exc):
                    raise
                print("[viewer-scene] viewer window closed")
                break
            time.sleep(0.02)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            task.close_env()
        except Exception:
            pass


def add_visual_box(scene, pose, half_size, color, name):
    entity = sapien.Entity()
    entity.set_name(name)
    entity.set_pose(pose)
    material = sapien.render.RenderMaterial(base_color=[float(color[0]), float(color[1]), float(color[2]), 1.0])
    render_body = sapien.render.RenderBodyComponent()
    render_body.attach(sapien.render.RenderShapeBox(half_size, material))
    entity.add_component(render_body)
    scene.add_entity(entity)
    return entity


def add_axis_marker(scene, pose, name, length=0.10, thickness=0.006):
    pos = np.asarray(pose.p, dtype=np.float64)
    rot = t3d.quaternions.quat2mat(np.asarray(pose.q, dtype=np.float64))
    axis_align = [
        np.eye(3),
        t3d.axangles.axangle2mat([0, 0, 1], np.pi / 2.0),
        t3d.axangles.axangle2mat([0, 1, 0], -np.pi / 2.0),
    ]
    colors = [(1.0, 0.05, 0.05), (0.05, 0.8, 0.05), (0.05, 0.2, 1.0)]
    for idx, color in enumerate(colors):
        axis_dir = rot[:, idx]
        axis_pose = sapien.Pose(
            pos + axis_dir * (length / 2.0),
            t3d.quaternions.mat2quat(rot @ axis_align[idx]),
        )
        add_visual_box(
            scene,
            axis_pose,
            [length / 2.0, thickness, thickness],
            color,
            f"{name}_axis_{idx}",
        )
    add_visual_box(scene, sapien.Pose(pos), [thickness * 1.5] * 3, (1.0, 1.0, 1.0), f"{name}_origin")


def pose_from_list(values):
    arr = np.asarray(values, dtype=np.float64)
    if arr.shape[0] >= 7:
        return sapien.Pose(arr[:3], arr[3:7])
    return sapien.Pose(arr[:3])


def add_scene_debug_axes(task):
    if hasattr(task, "bottle1"):
        add_axis_marker(task.scene, sapien.Pose(task.bottle1.get_pose().p), "left_bottle_center", length=0.07)
    if hasattr(task, "bottle2"):
        add_axis_marker(task.scene, sapien.Pose(task.bottle2.get_pose().p), "right_bottle_center", length=0.07)
    if hasattr(task, "left_target_pose"):
        add_axis_marker(task.scene, pose_from_list(task.left_target_pose), "left_place_target", length=0.10)
    if hasattr(task, "right_target_pose"):
        add_axis_marker(task.scene, pose_from_list(task.right_target_pose), "right_place_target", length=0.10)


if __name__ == "__main__":
    main()
