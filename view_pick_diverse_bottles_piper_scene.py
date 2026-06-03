import os
import sys
import time
from argparse import ArgumentParser

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
    print("[viewer-scene] loading scene only; play_once/planning is intentionally skipped")

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


if __name__ == "__main__":
    main()
