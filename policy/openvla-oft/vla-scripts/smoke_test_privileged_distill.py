"""CPU smoke test for privileged distillation plumbing."""

from __future__ import annotations

import tensorflow as tf
import torch

from prismatic.models.future_projector import FutureObservationProjector
from prismatic.training.distill_losses import compute_action_distill_loss
from prismatic.util.data_utils import PaddedCollatorForActionPrediction
from prismatic.vla.datasets.rlds.traj_transforms import chunk_act_obs


def check_traj_chunking() -> None:
    traj = {
        "action": tf.constant([[0.0], [1.0], [2.0], [3.0]], dtype=tf.float32),
        "observation": {
            "image_primary": tf.constant([[10], [20], [30], [40]], dtype=tf.int32),
            "timestep": tf.constant([0, 1, 2, 3], dtype=tf.int32),
            "pad_mask_dict": {"image_primary": tf.constant([True, True, True, True])},
        },
        "task": {"language_instruction": tf.constant([b"a", b"a", b"a", b"a"])},
        "dataset_name": tf.constant([b"dummy", b"dummy", b"dummy", b"dummy"]),
        "absolute_action_mask": tf.constant([[False], [False], [False], [False]]),
    }

    out = chunk_act_obs(
        traj,
        window_size=1,
        future_action_window_size=1,
        future_observation_window_size=2,
    )

    obs = out["observation"]["image_primary"].numpy().tolist()
    future_mask = out["observation"]["future_pad_mask"].numpy().tolist()

    assert obs == [[[10], [20], [30]], [[20], [30], [40]], [[30], [40], [40]]]
    assert future_mask == [[True, True, True], [True, True, True], [True, True, False]]


def check_future_projector_and_loss() -> None:
    projector = FutureObservationProjector(llm_dim=16)
    future_patches = torch.randn(2, 4, 8, 16)
    future_mask = torch.tensor([[True, True, True, False], [True, False, False, False]])
    future_token = projector(future_patches, future_mask=future_mask)
    assert future_token.shape == (2, 1, 16)

    student_actions = torch.randn(2, 8, 7)
    teacher_actions = torch.randn(2, 8, 7)
    loss = compute_action_distill_loss(student_actions, teacher_actions)
    assert loss.ndim == 0


def check_collator() -> None:
    collator = PaddedCollatorForActionPrediction(model_max_length=16, pad_token_id=0)
    instances = [
        {
            "pixel_values": torch.randn(3, 224, 224),
            "future_pixel_values": torch.randn(4, 3, 224, 224),
            "future_mask": torch.tensor([True, True, False, False]),
            "input_ids": torch.tensor([1, 2, 3]),
            "labels": torch.tensor([1, 2, 3]),
            "actions": torch.randn(8, 7).numpy(),
            "dataset_name": b"dummy",
        },
        {
            "pixel_values": torch.randn(3, 224, 224),
            "future_pixel_values": torch.randn(4, 3, 224, 224),
            "future_mask": torch.tensor([True, True, True, False]),
            "input_ids": torch.tensor([1, 2]),
            "labels": torch.tensor([1, 2]),
            "actions": torch.randn(8, 7).numpy(),
            "dataset_name": b"dummy",
        },
    ]
    batch = collator(instances)
    assert batch["future_pixel_values"].shape == (2, 4, 3, 224, 224)
    assert batch["future_mask"].shape == (2, 4)


def main() -> None:
    check_traj_chunking()
    check_future_projector_and_loss()
    check_collator()
    print("privileged distillation smoke test: PASS")


if __name__ == "__main__":
    main()
