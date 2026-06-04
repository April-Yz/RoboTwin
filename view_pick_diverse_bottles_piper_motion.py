"""Viewer-only entrypoint for the O.0 Piper/Pika motion baseline.

This script intentionally does not call script/collect_data.py, because that
entrypoint reuses seed.txt and can return immediately when enough viewer seeds
already exist. Here each run searches for a stable scene, executes play_once()
once in the interactive viewer, and optionally holds the window open.
"""

import os
import sys
import time
from argparse import ArgumentParser

import yaml

sys.path.append("./")

from script.collect_data import class_decorator, get_embodiment_config
from envs import CONFIGS_PATH
from view_pick_diverse_bottles_piper_scene import add_scene_debug_axes


def build_args(task_name, task_config, render_freq):
    config_path = f"./task_config/{task_config}.yml"
    with open(config_path, "r", encoding="utf-8") as f:
        args = yaml.load(f.read(), Loader=yaml.FullLoader)

    args["task_name"] = task_name
    args["task_config"] = task_config
    args["render_freq"] = render_freq
    args["need_plan"] = True
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
    return args, embodiment_name


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--task_name", default="pick_diverse_bottles_piper_motion")
    parser.add_argument("--task_config", default="demo_clean_piper_motion_viewer")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max_seed_tries", type=int, default=50)
    parser.add_argument("--render_freq", type=int, default=1)
    parser.add_argument("--show_axes", type=int, default=1)
    parser.add_argument("--hold", type=int, default=1)
    args_cli = parser.parse_args()

    args, embodiment_name = build_args(args_cli.task_name, args_cli.task_config, args_cli.render_freq)

    print(f"[motion-viewer] task={args_cli.task_name} config={args_cli.task_config} seed_start={args_cli.seed}")
    print(f"[motion-viewer] embodiment={embodiment_name}")
    print("[motion-viewer] executing play_once once; collect_data.py seed.txt is intentionally bypassed")

    task = None
    loaded_seed = None
    for seed in range(args_cli.seed, args_cli.seed + args_cli.max_seed_tries):
        task = class_decorator(args_cli.task_name)
        try:
            task.setup_demo(now_ep_num=0, seed=seed, **args)
            loaded_seed = seed
            break
        except Exception as exc:
            print(f"[motion-viewer] skip seed={seed}: {exc}")
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

    print(f"[motion-viewer] loaded stable scene seed={loaded_seed}")
    if args_cli.show_axes:
        add_scene_debug_axes(task)
        print("[motion-viewer] debug axes: bottle centers and left/right env target poses")

    try:
        task.play_once()
        print("[motion-viewer] play_once finished")
        if args_cli.hold:
            print("[motion-viewer] holding viewer; close the SAPIEN window or press Ctrl-C to exit")
            while True:
                task._update_render()
                viewer = getattr(task, "viewer", None)
                if viewer is None or getattr(viewer, "window", None) is None:
                    print("[motion-viewer] viewer window closed")
                    break
                try:
                    viewer.render()
                except AttributeError as exc:
                    if "should_close" not in str(exc):
                        raise
                    print("[motion-viewer] viewer window closed")
                    break
                time.sleep(0.02)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            task.close_env()
        except Exception:
            pass


if __name__ == "__main__":
    main()
