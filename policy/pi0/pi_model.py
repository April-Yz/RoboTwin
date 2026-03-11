#!/usr/bin/python3
# -- coding: UTF-8

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np

from attention_visualizer import save_attention_visualizations
from openpi.policies import policy_config as _policy_config
from openpi.training import config as _config


class PI0:

    def __init__(
        self,
        train_config_name: str,
        model_name: str,
        checkpoint_id: int,
        pi0_step: int,
        *,
        attn_vis_enable: bool = False,
        attn_vis_every_n_steps: int = 20,
        attn_vis_max_images_per_episode: int = 6,
        attn_vis_overlay_alpha: float = 0.45,
        attn_vis_save_dir: str | None = None,
    ):
        self.train_config_name = train_config_name
        self.model_name = model_name
        self.checkpoint_id = checkpoint_id
        self.pi0_step = pi0_step

        config = _config.get_config(self.train_config_name)
        ckpt_dir = f"policy/pi0/checkpoints/{self.train_config_name}/{self.model_name}/{self.checkpoint_id}"
        asset_dir = os.path.join(ckpt_dir, "assets")
        assets = sorted(os.listdir(asset_dir))
        if not assets:
            raise FileNotFoundError(f"No assets found at: {asset_dir}")
        asset_id = assets[0]

        self.policy = _policy_config.create_trained_policy(config, ckpt_dir, robotwin_repo_id=asset_id)
        print("loading model success!")

        self.img_size = (224, 224)
        self.observation_window: dict[str, Any] | None = None
        self.latest_rgb_images: dict[str, np.ndarray] = {}
        self.instruction: str | None = None

        self.attn_vis_enable = bool(attn_vis_enable)
        self.attn_vis_every_n_steps = max(1, int(attn_vis_every_n_steps))
        self.attn_vis_max_images_per_episode = max(1, int(attn_vis_max_images_per_episode))
        self.attn_vis_overlay_alpha = float(attn_vis_overlay_alpha)
        self.attn_vis_save_dir = Path(attn_vis_save_dir) if attn_vis_save_dir else None
        self._episode_vis_count: dict[int, int] = {}

    def set_img_size(self, img_size):
        self.img_size = img_size

    def set_language(self, instruction):
        self.instruction = instruction
        print(f"successfully set instruction:{instruction}")

    def update_observation_window(self, img_arr, state):
        img_front, img_right, img_left, _ = (
            img_arr[0],
            img_arr[1],
            img_arr[2],
            state,
        )

        self.latest_rgb_images = {
            "base_0_rgb": np.asarray(img_front).copy(),
            "left_wrist_0_rgb": np.asarray(img_left).copy(),
            "right_wrist_0_rgb": np.asarray(img_right).copy(),
            "cam_high": np.asarray(img_front).copy(),
            "cam_left_wrist": np.asarray(img_left).copy(),
            "cam_right_wrist": np.asarray(img_right).copy(),
        }

        img_front = np.transpose(img_front, (2, 0, 1))
        img_right = np.transpose(img_right, (2, 0, 1))
        img_left = np.transpose(img_left, (2, 0, 1))

        self.observation_window = {
            "state": state,
            "images": {
                "cam_high": img_front,
                "cam_left_wrist": img_left,
                "cam_right_wrist": img_right,
            },
            "prompt": self.instruction,
        }

    def _should_visualize(self, episode_id: int, env_step: int) -> bool:
        if not self.attn_vis_enable:
            return False
        if self.attn_vis_save_dir is None:
            return False
        if env_step % self.attn_vis_every_n_steps != 0:
            return False

        count = self._episode_vis_count.get(episode_id, 0)
        return count < self.attn_vis_max_images_per_episode

    def _save_attention(self, episode_id: int, env_step: int, attention: dict[str, Any]) -> None:
        if self.attn_vis_save_dir is None:
            return
        prefix = f"episode{episode_id:04d}_step{env_step:05d}"
        save_attention_visualizations(
            images=self.latest_rgb_images,
            attention=attention,
            save_dir=self.attn_vis_save_dir / f"episode{episode_id:04d}",
            file_prefix=prefix,
            overlay_alpha=self.attn_vis_overlay_alpha,
        )
        self._episode_vis_count[episode_id] = self._episode_vis_count.get(episode_id, 0) + 1

    def get_action(self, episode_id: int | None = None, env_step: int | None = None):
        assert self.observation_window is not None, "update observation_window first!"

        if episode_id is None or env_step is None or not self._should_visualize(episode_id, env_step):
            return self.policy.infer(self.observation_window)["actions"]

        try:
            outputs = self.policy.infer_with_attention(self.observation_window)
            actions = outputs["actions"]
            attention = outputs.get("attention")
            if isinstance(attention, dict):
                self._save_attention(episode_id, env_step, attention)
            return actions
        except Exception as e:
            print(f"[attention-vis] fallback to normal inference: {e}")
            return self.policy.infer(self.observation_window)["actions"]

    def reset_obsrvationwindows(self):
        self.instruction = None
        self.observation_window = None
        self.latest_rgb_images = {}
        print("successfully unset obs and language intruction")
