"""Motion viewer for Piper IK tasks (V1-V4).

Usage:
  python view_pick_diverse_bottles_piper_ik_motion.py --ik_version v1 --seed 0
  python view_pick_diverse_bottles_piper_ik_motion.py --ik_version v2 --seed 0
  python view_pick_diverse_bottles_piper_ik_motion.py --ik_version v3 --seed 0
  python view_pick_diverse_bottles_piper_ik_motion.py --ik_version v4 --seed 0
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


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--task_name", default="pick_diverse_bottles_piper_ik")
    parser.add_argument("--ik_version", default="v1", choices=["v1", "v2", "v3", "v4"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max_seed_tries", type=int, default=50)
    parser.add_argument("--render_freq", type=int, default=1)
    parser.add_argument("--show_axes", type=int, default=1)
    parser.add_argument("--hold", type=int, default=1)
    args_cli = parser.parse_args()

    task_config = f"demo_piper_ik_{args_cli.ik_version}"
    config_path = f"./task_config/{task_config}.yml"
    if not os.path.exists(config_path):
        print(f"[motion-viewer] config not found: {config_path}")
        sys.exit(1)

    with open(config_path, "r", encoding="utf-8") as f:
        args = yaml.load(f.read(), Loader=yaml.FullLoader)

    args["task_name"] = args_cli.task_name
    args["task_config"] = task_config
    args["render_freq"] = args_cli.render_freq
    args["need_plan"] = True
    args["save_data"] = False
    args["collect_data"] = False
    args["skip_planner"] = True

    with open(os.path.join(CONFIGS_PATH, "_embodiment_config.yml"), "r", encoding="utf-8") as f:
        embodiment_types = yaml.load(f.read(), Loader=yaml.FullLoader)

    embodiment = args["embodiment"]

    def embodiment_file(name):
        return embodiment_types[name]["file_path"]

    if len(embodiment) == 3:
        args["left_robot_file"] = embodiment_file(embodiment[0])
        args["right_robot_file"] = embodiment_file(embodiment[1])
        args["embodiment_dis"] = embodiment[2]
        args["dual_arm_embodied"] = False
        embodiment_name = f"{embodiment[0]}+{embodiment[1]}"
    else:
        raise ValueError("embodiment must contain 3 values")

    args["left_embodiment_config"] = get_embodiment_config(args["left_robot_file"])
    args["right_embodiment_config"] = get_embodiment_config(args["right_robot_file"])
    args["embodiment_name"] = embodiment_name

    print(f"[motion-viewer] task={args_cli.task_name} ik_version={args_cli.ik_version} seed_start={args_cli.seed}")
    print(f"[motion-viewer] embodiment={embodiment_name}")
    print("[motion-viewer] executing play_once with Piper IK planner")

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

    try:
        task.play_once()
        print("[motion-viewer] play_once finished")
        if args_cli.hold:
            print("[motion-viewer] holding viewer; close window or Ctrl-C to exit")
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
