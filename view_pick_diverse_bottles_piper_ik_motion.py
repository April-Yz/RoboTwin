"""Motion viewer for Piper IK tasks (V1-V4).

Usage:
  # 单次运行
  python view_pick_diverse_bottles_piper_ik_motion.py --ik_version v1 --seed 0

  # 连续运行 10 个 episode（自动找 stable seed，episode 间延时 2s）
  python view_pick_diverse_bottles_piper_ik_motion.py --ik_version v1 --num_episodes 10 --episode_delay 2.0
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


def run_one_episode(task_cls, build_args, seed_start, max_tries, show_axes, ik_version,
                    step_mode=False, require_success=True):
    """Load stable scenes until one completes and, by default, physically succeeds."""
    task = None
    for seed in range(seed_start, seed_start + max_tries):
        task = task_cls()
        try:
            task.setup_demo(now_ep_num=0, seed=seed, **build_args)
            if show_axes:
                add_scene_debug_axes(task)
            if build_args["render_freq"]:
                print("[motion-viewer] waiting for viewer window to open...")
                viewer_ready = False
                for i in range(600):  # 最多等 12 秒
                    task._update_render()
                    viewer = getattr(task, "viewer", None)
                    if viewer is not None:
                        window = getattr(viewer, "window", None)
                        if window is not None:
                            if not viewer_ready:
                                print(f"[motion-viewer] viewer window ready after {i+1} frames")
                                viewer_ready = True
                            if i > 5:
                                break
                    time.sleep(0.02)
                if not viewer_ready:
                    print("[motion-viewer] WARNING: viewer window not detected, proceeding anyway")
            else:
                print("[motion-viewer] headless validation mode")
            if step_mode:
                task._step_mode = True
            task.play_once()
            physical_success = bool(task.plan_success and task.check_success())
            if require_success and not physical_success:
                print(f"[motion-viewer] seed={seed} completed but physical task failed")
                task.close_env()
                task = None
                continue
            print(f"[motion-viewer] seed={seed} physical_success={physical_success}")
            return True, seed, task
        except Exception as exc:
            print(f"[motion-viewer] seed={seed} FAILED: {exc}")
            try:
                task.close_env()
            except Exception:
                pass
            task = None
    return False, seed_start, None


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--task_name", default="pick_diverse_bottles_piper_ik")
    parser.add_argument("--ik_version", default="v1", choices=["v1", "v2", "v3", "v4"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max_seed_tries", type=int, default=50)
    parser.add_argument("--num_episodes", type=int, default=1,
                        help="连续运行 episode 数量（>1 时自动循环）")
    parser.add_argument("--episode_delay", type=float, default=2.0,
                        help="episode 间保持 viewer 的秒数")
    parser.add_argument("--render_freq", type=int, default=1)
    parser.add_argument("--show_axes", type=int, default=1)
    parser.add_argument("--hold", type=int, default=1)
    parser.add_argument("--step_mode", type=int, default=0,
                        help="逐步确认模式：每个动作后等待终端回车才继续（1=启用）")
    parser.add_argument("--require_success", type=int, default=1,
                        help="1=跳过物理抓放失败的 seed；0=只要求轨迹执行完成")
    args_cli = parser.parse_args()

    task_config = f"demo_piper_ik_seq_{args_cli.ik_version}"
    config_path = f"./task_config/{task_config}.yml"
    if not os.path.exists(config_path):
        print(f"[motion-viewer] config not found: {config_path}")
        sys.exit(1)

    with open(config_path, "r", encoding="utf-8") as f:
        build_args = yaml.load(f.read(), Loader=yaml.FullLoader)

    build_args["task_name"] = args_cli.task_name
    build_args["task_config"] = task_config
    build_args["render_freq"] = args_cli.render_freq
    build_args["need_plan"] = True
    build_args["save_data"] = False
    build_args["collect_data"] = False
    build_args["skip_planner"] = True
    build_args["save_all_episodes"] = False

    with open(os.path.join(CONFIGS_PATH, "_embodiment_config.yml"), "r", encoding="utf-8") as f:
        embodiment_types = yaml.load(f.read(), Loader=yaml.FullLoader)

    embodiment = build_args["embodiment"]

    def embodiment_file(name):
        return embodiment_types[name]["file_path"]

    if len(embodiment) == 3:
        build_args["left_robot_file"] = embodiment_file(embodiment[0])
        build_args["right_robot_file"] = embodiment_file(embodiment[1])
        build_args["embodiment_dis"] = embodiment[2]
        build_args["dual_arm_embodied"] = False
        embodiment_name = f"{embodiment[0]}+{embodiment[1]}"
    else:
        raise ValueError("embodiment must contain 3 values")

    build_args["left_embodiment_config"] = get_embodiment_config(build_args["left_robot_file"])
    build_args["right_embodiment_config"] = get_embodiment_config(build_args["right_robot_file"])
    build_args["embodiment_name"] = embodiment_name

    print(f"[motion-viewer] task={args_cli.task_name} ik_version={args_cli.ik_version} "
          f"num_episodes={args_cli.num_episodes} seed_start={args_cli.seed}")
    print(f"[motion-viewer] embodiment={embodiment_name}")

    import importlib
    envs_module = importlib.import_module(f"envs.{args_cli.task_name}")
    task_cls = getattr(envs_module, args_cli.task_name)

    success_count = 0
    next_seed = args_cli.seed

    for ep in range(args_cli.num_episodes):
        print(f"\n[motion-viewer] === Episode {ep+1}/{args_cli.num_episodes} (seed_start={next_seed}) ===")

        ok, used_seed, task = run_one_episode(
            task_cls, build_args, next_seed, args_cli.max_seed_tries,
            args_cli.show_axes, args_cli.ik_version, args_cli.step_mode,
            bool(args_cli.require_success),
        )

        if not ok:
            print(f"[motion-viewer] Episode {ep+1}: ALL SEEDS FAILED from {next_seed}")
            break

        success_count += 1
        next_seed = used_seed + 1
        print(f"[motion-viewer] Episode {ep+1}: seed={used_seed} FINISHED (total_ok={success_count})")

        # 短暂保持 viewer 显示结果
        hold_start = time.time()
        while time.time() - hold_start < args_cli.episode_delay:
            task._update_render()
            viewer = getattr(task, "viewer", None)
            if viewer is None or getattr(viewer, "window", None) is None:
                break
            try:
                viewer.render()
            except AttributeError:
                break
            time.sleep(0.02)

        # 关闭当前 episode 的 viewer
        try:
            task.close_env()
        except Exception:
            pass

    print(f"\n[motion-viewer] DONE: {success_count}/{args_cli.num_episodes} episodes succeeded")

    # 最后一个 episode 保持 viewer（如果 hold=1 且只有 1 个 episode）
    if args_cli.num_episodes == 1 and success_count == 1:
        print("[motion-viewer] single-episode mode: viewer already closed")


if __name__ == "__main__":
    main()
