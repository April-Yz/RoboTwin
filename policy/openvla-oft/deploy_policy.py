import os
import numpy as np
from dataclasses import dataclass

from prismatic.vla.constants import NUM_ACTIONS_CHUNK, PROPRIO_DIM
from experiments.robot.openvla_utils import (
    get_vla,
    get_processor,
    get_action_head,
    get_proprio_projector,
    get_vla_action,
)


@dataclass
class InferenceConfig:
    pretrained_checkpoint: str
    use_l1_regression: bool = True
    use_diffusion: bool = False
    use_film: bool = True
    use_proprio: bool = True
    load_in_8bit: bool = False
    load_in_4bit: bool = False
    num_images_in_input: int = 3
    center_crop: bool = True
    unnorm_key: str = ""
    num_open_loop_steps: int = NUM_ACTIONS_CHUNK
    lora_rank: int = 32
    log_eval_action_stats: bool = True
    eval_action_stats_freq: int = 10
    eval_action_stats_print_first_n_steps: int = 20


def encode_obs(obs: dict) -> dict:
    return {
        "full_image": obs["observation"]["head_camera"]["rgb"],
        "left_wrist_image": obs["observation"]["left_camera"]["rgb"],
        "right_wrist_image": obs["observation"]["right_camera"]["rgb"],
        "state": obs["joint_action"]["vector"],
        "instruction": obs["language"],
    }


class Model:
    def __init__(self, cfg: InferenceConfig):
        self.cfg = cfg
        self.vla = get_vla(cfg)
        self.processor = get_processor(cfg)
        self.action_head = None
        if cfg.use_l1_regression or cfg.use_diffusion:
            self.action_head = get_action_head(cfg, self.vla.llm_dim)
        self.proprio_projector = None
        if cfg.use_proprio:
            self.proprio_projector = get_proprio_projector(
                cfg, self.vla.llm_dim, PROPRIO_DIM
            )
        self.last_action_diagnostics = None
        self.global_eval_step = 0
        self.episode_action_count = 0
        self.episode_joint_delta_norm = 0.0
        self.episode_gripper_open_close_changes = 0
        self._last_gripper_sign = None
        self._print_eval_config_summary()

    def get_action(self, observation: dict):
        obs = encode_obs(observation)
        if self.cfg.log_eval_action_stats:
            actions, diagnostics = get_vla_action(
                cfg=self.cfg,
                vla=self.vla,
                processor=self.processor,
                obs=obs,
                task_label=obs["instruction"],
                action_head=self.action_head,
                proprio_projector=self.proprio_projector,
                use_film=self.cfg.use_film,
                return_diagnostics=True,
            )
            self.last_action_diagnostics = diagnostics
        else:
            actions = get_vla_action(
                cfg=self.cfg,
                vla=self.vla,
                processor=self.processor,
                obs=obs,
                task_label=obs["instruction"],
                action_head=self.action_head,
                proprio_projector=self.proprio_projector,
                use_film=self.cfg.use_film,
            )
            self.last_action_diagnostics = None
        return actions

    def _print_eval_config_summary(self):
        checkpoint_path = self.cfg.pretrained_checkpoint
        has_lora_adapter = os.path.isdir(os.path.join(checkpoint_path, "lora_adapter"))
        has_config_json = os.path.isfile(os.path.join(checkpoint_path, "config.json"))
        print("[EvalConfig] checkpoint_path:", checkpoint_path)
        print(
            "[EvalConfig] "
            f"use_l1_regression={self.cfg.use_l1_regression} "
            f"use_diffusion={self.cfg.use_diffusion} "
            f"use_proprio={self.cfg.use_proprio} "
            f"use_film={self.cfg.use_film} "
            f"num_images_in_input={self.cfg.num_images_in_input} "
            f"unnorm_key={self.cfg.unnorm_key}"
        )
        print(
            "[EvalCheckpoint] "
            f"has_config_json={has_config_json} "
            f"has_lora_adapter={has_lora_adapter}"
        )

    @staticmethod
    def _vector_stats(prefix: str, values: np.ndarray) -> dict:
        values = np.asarray(values, dtype=np.float32)
        stats = {
            f"{prefix}_mean": float(values.mean()),
            f"{prefix}_std": float(values.std()),
            f"{prefix}_abs_mean": float(np.abs(values).mean()),
            f"{prefix}_abs_max": float(np.abs(values).max()),
        }
        dim_abs_mean = np.abs(values).mean(axis=0)
        for dim_idx, value in enumerate(dim_abs_mean):
            stats[f"{prefix}_dim_abs_mean_{dim_idx}"] = float(value)
        return stats

    @staticmethod
    def _format_stats(stats: dict, prefix: str) -> str:
        dim_items = sorted(
            (key, value) for key, value in stats.items() if key.startswith(f"{prefix}_dim_abs_mean_")
        )
        dim_summary = ", ".join(f"d{idx.split('_')[-1]}={value:.4f}" for idx, value in dim_items)
        return (
            f"{prefix}_abs_mean={stats[f'{prefix}_abs_mean']:.4f} "
            f"{prefix}_abs_max={stats[f'{prefix}_abs_max']:.4f} "
            f"{prefix}_std={stats[f'{prefix}_std']:.4f} "
            f"[{dim_summary}]"
        )

    @staticmethod
    def _infer_gripper_indices(action_dim: int) -> tuple[int, int]:
        left_gripper_idx = max(action_dim // 2 - 1, 0)
        right_gripper_idx = action_dim - 1
        return left_gripper_idx, right_gripper_idx

    def reset_episode_diagnostics(self):
        self.last_action_diagnostics = None
        self.episode_action_count = 0
        self.episode_joint_delta_norm = 0.0
        self.episode_gripper_open_close_changes = 0
        self._last_gripper_sign = None

    def record_executed_action(
        self,
        raw_action: np.ndarray,
        denorm_action: np.ndarray,
        executed_action: np.ndarray,
        observation_before: dict | None = None,
        observation_after: dict | None = None,
    ):
        raw_action = np.asarray(raw_action, dtype=np.float32)
        denorm_action = np.asarray(denorm_action, dtype=np.float32)
        executed_action = np.asarray(executed_action, dtype=np.float32)

        self.global_eval_step += 1
        self.episode_action_count += 1

        raw_stats = self._vector_stats("raw_action", raw_action[None, :])
        denorm_stats = self._vector_stats("denorm_action", denorm_action[None, :])
        executed_stats = self._vector_stats("executed_action", executed_action[None, :])

        should_print = (
            self.global_eval_step <= self.cfg.eval_action_stats_print_first_n_steps
            or self.global_eval_step % self.cfg.eval_action_stats_freq == 0
        )
        if should_print:
            print(
                f"[EvalAction][step={self.global_eval_step}] "
                f"{self._format_stats(raw_stats, 'raw_action')} | "
                f"{self._format_stats(denorm_stats, 'denorm_action')} | "
                f"{self._format_stats(executed_stats, 'executed_action')}"
            )

        if observation_before is not None and observation_after is not None:
            before_joint = np.asarray(observation_before["joint_action"]["vector"], dtype=np.float32)
            after_joint = np.asarray(observation_after["joint_action"]["vector"], dtype=np.float32)
            self.episode_joint_delta_norm += float(np.linalg.norm(after_joint - before_joint))

        left_gripper_idx, right_gripper_idx = self._infer_gripper_indices(executed_action.shape[0])
        current_gripper_sign = np.sign(executed_action[[left_gripper_idx, right_gripper_idx]])
        if self._last_gripper_sign is not None:
            self.episode_gripper_open_close_changes += int(
                np.any(current_gripper_sign != self._last_gripper_sign)
            )
        self._last_gripper_sign = current_gripper_sign

    def finish_episode_diagnostics(self, success: bool, step_limit: int):
        print(
            "[EvalEpisode] "
            f"episode_len={self.episode_action_count} "
            f"success={int(success)} "
            f"joint_delta_norm={self.episode_joint_delta_norm:.4f} "
            f"gripper_open_close_changes={self.episode_gripper_open_close_changes} "
            f"step_limit={step_limit}"
        )


def get_model(usr_args: dict):
    config_args = {
        "pretrained_checkpoint": usr_args["checkpoint_path"],
        "use_l1_regression": usr_args.get("use_l1_regression", True),
        "use_diffusion": usr_args.get("use_diffusion", False),
        "use_film": usr_args.get("use_film", True),
        "use_proprio": usr_args.get("use_proprio", True),
        "load_in_8bit": usr_args.get("load_in_8bit", False),
        "load_in_4bit": usr_args.get("load_in_4bit", False),
        "num_images_in_input": usr_args.get("num_images_in_input", 3),
        "center_crop": usr_args.get("center_crop", True),
        "unnorm_key": usr_args["unnorm_key"],
        "num_open_loop_steps": usr_args.get("num_open_loop_steps", NUM_ACTIONS_CHUNK),
        "lora_rank": usr_args.get("lora_rank", 32),
        "log_eval_action_stats": usr_args.get("log_eval_action_stats", True),
        "eval_action_stats_freq": usr_args.get("eval_action_stats_freq", 10),
        "eval_action_stats_print_first_n_steps": usr_args.get("eval_action_stats_print_first_n_steps", 20),
    }

    cfg = InferenceConfig(**config_args)
    return Model(cfg)


def reset_model(model=None):
    if model is not None and hasattr(model, "reset_episode_diagnostics"):
        model.reset_episode_diagnostics()


def eval(TASK_ENV, model: Model, observation: dict):
    observation["language"] = TASK_ENV.get_instruction()

    actions = model.get_action(observation)
    diagnostics = model.last_action_diagnostics or {}
    raw_actions = np.asarray(diagnostics.get("raw_actions", actions), dtype=np.float32)
    denorm_actions = np.asarray(diagnostics.get("denorm_actions", actions), dtype=np.float32)

    for action_idx, action in enumerate(actions):
        observation_before = observation
        TASK_ENV.take_action(action)
        observation = TASK_ENV.get_obs()
        if model.cfg.log_eval_action_stats:
            raw_action = raw_actions[action_idx]
            denorm_action = denorm_actions[action_idx]
            model.record_executed_action(
                raw_action=raw_action,
                denorm_action=denorm_action,
                executed_action=np.asarray(action, dtype=np.float32),
                observation_before=observation_before,
                observation_after=observation,
            )
