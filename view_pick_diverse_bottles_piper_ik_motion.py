"""Motion viewer for Piper IK tasks (V1-V4).

Usage:
  # 单次运行
  python view_pick_diverse_bottles_piper_ik_motion.py --ik_version v1 --seed 0

  # 连续运行 10 个 episode（自动找 stable seed，episode 间延时 2s）
  python view_pick_diverse_bottles_piper_ik_motion.py --ik_version v1 --num_episodes 10 --episode_delay 2.0
"""

import os
import re
import sys
import time
from argparse import ArgumentParser

import yaml

sys.path.append("./")

from script.collect_data import class_decorator, get_embodiment_config
from envs import CONFIGS_PATH
from view_pick_diverse_bottles_piper_scene import add_scene_debug_axes


def configure_camera_frustums(task, enabled):
    """Toggle SAPIEN camera frustums and report the debug cameras being drawn."""
    viewer = getattr(task, "viewer", None)
    if viewer is None:
        if enabled:
            raise RuntimeError("--show_camera_frustums requires --render_freq > 0")
        return

    viewer.control_window.show_camera_linesets = bool(enabled)
    if not enabled:
        return

    camera_names = []
    for camera in viewer.cameras:
        entity = getattr(camera, "entity", None)
        camera_names.append(getattr(entity, "name", "<unnamed>"))
    required = {"left_camera", "right_camera", "head_camera"}
    missing = required.difference(camera_names)
    if missing:
        raise RuntimeError(
            "Camera frustums are enabled but required cameras are missing: "
            + ", ".join(sorted(missing))
        )
    print(
        "[motion-viewer] camera frustums enabled: "
        "left_camera, right_camera, head_camera "
        f"(all viewer cameras={camera_names})"
    )


def run_one_episode(task_cls, build_args, seed_start, max_tries, show_axes, ik_version,
                    show_camera_frustums=False, step_mode=False, require_success=True):
    """Load stable scenes until one completes and, by default, physically succeeds."""
    task = None
    for seed in range(seed_start, seed_start + max_tries):
        task = task_cls()
        try:
            task.setup_demo(now_ep_num=0, seed=seed, **build_args)
            if show_axes:
                add_scene_debug_axes(task)
            configure_camera_frustums(task, show_camera_frustums)
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
    parser.add_argument(
        "--show_camera_frustums",
        type=int,
        default=0,
        help="1=在 SAPIEN viewer 中显示所有相机框线（含左右 wrist 和 head）",
    )
    parser.add_argument("--hold", type=int, default=1)
    parser.add_argument("--step_mode", type=int, default=0,
                        help="逐步确认模式：每个动作后等待终端回车才继续（1=启用）")
    parser.add_argument("--require_success", type=int, default=1,
                        help="1=跳过物理抓放失败的 seed；0=只要求轨迹执行完成")
    parser.add_argument("--wrist_preview", type=int, default=0,
                        help="1=额外显示左右 wrist RGB 实时拼接窗口")
    parser.add_argument("--wrist_left_forward_offset_m", type=float, default=None,
                        help="覆盖左 wrist 沿相机前进轴的仿真偏移（米）")
    parser.add_argument("--wrist_right_forward_offset_m", type=float, default=None,
                        help="覆盖右 wrist 沿相机前进轴的仿真偏移（米）")
    parser.add_argument("--wrist_left_roll_deg", type=float, default=None,
                        help="覆盖左 wrist 绕光轴的有符号校正角（度）")
    parser.add_argument("--wrist_right_roll_deg", type=float, default=None,
                        help="覆盖右 wrist 绕光轴的有符号校正角（度）")
    parser.add_argument("--wrist_debug_record", type=int, default=0,
                        help="1=保存左右 wrist 原始视频、带标签拼接视频和参数 JSON")
    parser.add_argument("--wrist_debug_tag", type=str, default="",
                        help="debug 输出标签，只允许字母、数字、下划线和短横线")
    parser.add_argument("--wrist_debug_dir", type=str,
                        default="data/wrist_camera_debug",
                        help="wrist debug 视频输出根目录")
    parser.add_argument("--wrist_debug_fps", type=float, default=30.0,
                        help="debug MP4 回放帧率")
    parser.add_argument("--task_config", type=str, default="",
                        help="覆盖自动推断的 config 名称 (例如 demo_piper_ik_foundation_v1)")
    parser.add_argument("--foundation_id", type=int, default=-1,
                        help="O.1: override foundation_input_<ID> without editing YAML")
    parser.add_argument("--foundation_frame", type=int, default=-1,
                        help="O.1: override the FoundationPose frame without editing YAML")
    parser.add_argument("--foundation_mode", default="",
                        choices=["", "o1", "o1.1", "o1.2"],
                        help="O.1 mode: frame 0, annotated object frame, or annotated EE action")
    args_cli = parser.parse_args()

    if args_cli.task_config:
        task_config = args_cli.task_config
    elif "foundation" in args_cli.task_name:
        task_config = f"demo_piper_ik_foundation_{args_cli.ik_version}"
    else:
        task_config = f"demo_piper_ik_seq_{args_cli.ik_version}"
    config_path = f"./task_config/{task_config}.yml"
    if not os.path.exists(config_path):
        print(f"[motion-viewer] config not found: {config_path}")
        sys.exit(1)

    with open(config_path, "r", encoding="utf-8") as f:
        build_args = yaml.load(f.read(), Loader=yaml.FullLoader)

    if args_cli.foundation_id >= 0:
        current_dir = build_args.get("foundation_input_dir")
        if not current_dir:
            raise ValueError("--foundation_id requires foundation_input_dir in the selected config")
        input_root = os.path.dirname(current_dir.rstrip("/"))
        build_args["foundation_input_dir"] = os.path.join(
            input_root, f"foundation_input_{args_cli.foundation_id}"
        )
    if args_cli.foundation_frame >= 0:
        build_args["foundation_frame"] = args_cli.foundation_frame
    if args_cli.foundation_mode:
        build_args["foundation_mode"] = args_cli.foundation_mode
    build_args.setdefault("camera", {})["wrist_camera_preview"] = bool(
        args_cli.wrist_preview
    )
    tuning = build_args["camera"].setdefault("wrist_camera_tuning", {})
    for side, forward_offset, roll_deg in (
        ("left", args_cli.wrist_left_forward_offset_m, args_cli.wrist_left_roll_deg),
        ("right", args_cli.wrist_right_forward_offset_m, args_cli.wrist_right_roll_deg),
    ):
        side_tuning = tuning.setdefault(side, {})
        if forward_offset is not None:
            side_tuning["forward_offset_m"] = forward_offset
        if roll_deg is not None:
            side_tuning["image_roll_deg"] = roll_deg
    if args_cli.wrist_preview and not build_args["camera"].get(
        "collect_wrist_camera", False
    ):
        raise ValueError("--wrist_preview requires collect_wrist_camera: true")

    if args_cli.wrist_debug_record:
        if not build_args["camera"].get("collect_wrist_camera", False):
            raise ValueError("--wrist_debug_record requires collect_wrist_camera: true")
        if args_cli.num_episodes != 1:
            raise ValueError("--wrist_debug_record currently requires --num_episodes 1")
        debug_tag = args_cli.wrist_debug_tag or time.strftime("%Y%m%d_%H%M%S")
        if not re.fullmatch(r"[A-Za-z0-9_-]+", debug_tag):
            raise ValueError("--wrist_debug_tag only accepts letters, digits, _ and -")
        debug_output_dir = os.path.abspath(
            os.path.join(args_cli.wrist_debug_dir, debug_tag)
        )
        if os.path.exists(debug_output_dir) and os.listdir(debug_output_dir):
            raise FileExistsError(
                f"Wrist debug output already exists and is not empty: {debug_output_dir}"
            )
        build_args["camera"]["wrist_camera_debug_record_dir"] = debug_output_dir
        build_args["camera"]["wrist_camera_debug_fps"] = args_cli.wrist_debug_fps
        build_args["camera"]["wrist_camera_debug_context"] = {
            "task_name": args_cli.task_name,
            "task_config": task_config,
            "ik_version": args_cli.ik_version,
            "seed": args_cli.seed,
            "foundation_id": args_cli.foundation_id,
            "foundation_frame": args_cli.foundation_frame,
            "foundation_mode": args_cli.foundation_mode,
        }

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
    if args_cli.wrist_preview:
        print("[motion-viewer] wrist preview enabled: left/right RGB mosaic")
    if tuning:
        print(f"[motion-viewer] wrist tuning={tuning}")

    if args_cli.wrist_debug_record:
        print(
            "[motion-viewer] wrist debug recording="
            f"{build_args['camera']['wrist_camera_debug_record_dir']}"
        )

    import importlib
    envs_module = importlib.import_module(f"envs.{args_cli.task_name}")
    task_cls = getattr(envs_module, args_cli.task_name)

    success_count = 0
    next_seed = args_cli.seed

    for ep in range(args_cli.num_episodes):
        print(f"\n[motion-viewer] === Episode {ep+1}/{args_cli.num_episodes} (seed_start={next_seed}) ===")

        ok, used_seed, task = run_one_episode(
            task_cls, build_args, next_seed, args_cli.max_seed_tries,
            args_cli.show_axes, args_cli.ik_version,
            bool(args_cli.show_camera_frustums), args_cli.step_mode,
            bool(args_cli.require_success),
        )

        if not ok:
            print(f"[motion-viewer] Episode {ep+1}: ALL SEEDS FAILED from {next_seed}")
            break

        success_count += 1
        next_seed = used_seed + 1
        print(f"[motion-viewer] Episode {ep+1}: seed={used_seed} FINISHED (total_ok={success_count})")

        # 单 episode 的 hold=1 会保留最终状态，直到用户关窗或 Ctrl-C。
        hold_forever = bool(
            args_cli.hold and args_cli.num_episodes == 1 and args_cli.render_freq
        )
        hold_start = time.time()
        if hold_forever:
            print("[motion-viewer] hold=1; close the SAPIEN window or press Ctrl-C to exit")
        try:
            while hold_forever or time.time() - hold_start < args_cli.episode_delay:
                task._update_render()
                viewer = getattr(task, "viewer", None)
                if viewer is None or getattr(viewer, "window", None) is None:
                    break
                try:
                    viewer.render()
                except AttributeError:
                    break
                time.sleep(0.02)
        except KeyboardInterrupt:
            print("[motion-viewer] interrupted; closing viewer")

        # 关闭当前 episode 的 viewer
        try:
            task.close_env()
        except KeyboardInterrupt:
            print("[motion-viewer] cleanup interrupted; viewer process is exiting")
        except Exception:
            pass

    print(f"\n[motion-viewer] DONE: {success_count}/{args_cli.num_episodes} episodes succeeded")

    if args_cli.num_episodes == 1 and success_count == 1:
        print("[motion-viewer] single-episode viewer closed")


if __name__ == "__main__":
    main()
