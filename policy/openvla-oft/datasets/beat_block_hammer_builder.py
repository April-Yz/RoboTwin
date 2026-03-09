from typing import Iterator, Tuple, Any
import argparse
import glob
import os

import h5py
import numpy as np
import tensorflow_datasets as tfds

from datasets.conversion_utils import MultiThreadedDatasetBuilder


def _generate_examples(paths) -> Iterator[Tuple[str, Any]]:
    for path in paths:
        with h5py.File(path, "r") as f:
            required_keys = [
                "/head_camera_image",
                "/left_wrist_image",
                "/right_wrist_image",
                "/low_cam_image",
                "/action",
                "/seen",
            ]
            if not all(k in f for k in required_keys):
                continue

            total_steps = f["/action"].shape[0]
            actions = f["/action"][1:].astype(np.float32)
            head = f["/head_camera_image"][: total_steps - 1].astype(np.uint8)
            left = f["/left_wrist_image"][: total_steps - 1].astype(np.uint8)
            right = f["/right_wrist_image"][: total_steps - 1].astype(np.uint8)
            low = f["/low_cam_image"][: total_steps - 1].astype(np.uint8)
            states = f["/action"][: total_steps - 1].astype(np.float32)
            seen = [s.decode("utf-8") if isinstance(s, bytes) else s for s in f["/seen"][()]]

            if not seen:
                continue

            episode_len = total_steps - 1
            if not (
                head.shape[0]
                == left.shape[0]
                == right.shape[0]
                == low.shape[0]
                == episode_len
                == states.shape[0]
                == actions.shape[0]
            ):
                continue

            steps = []
            for i in range(episode_len):
                steps.append(
                    {
                        "observation": {
                            "image": head[i],
                            "left_wrist_image": left[i],
                            "right_wrist_image": right[i],
                            "low_cam_image": low[i],
                            "state": states[i],
                        },
                        "action": actions[i],
                        "discount": np.float32(1.0),
                        "reward": np.float32(1.0 if i == episode_len - 1 else 0.0),
                        "is_first": np.bool_(i == 0),
                        "is_last": np.bool_(i == episode_len - 1),
                        "is_terminal": np.bool_(i == episode_len - 1),
                        "language_instruction": seen,
                    }
                )

            yield path, {"steps": steps, "episode_metadata": {"file_path": path}}


class aloha_beat_block_hammer_builder(MultiThreadedDatasetBuilder):
    VERSION = tfds.core.Version("1.0.0")
    RELEASE_NOTES = {"1.0.0": "Initial RoboTwin beat_block_hammer dataset release."}

    N_WORKERS = 1
    MAX_PATHS_IN_MEMORY = 100
    PARSE_FCN = _generate_examples
    INPUT_DIR = None

    def _info(self) -> tfds.core.DatasetInfo:
        return self.dataset_info_from_configs(
            features=tfds.features.FeaturesDict(
                {
                    "steps": tfds.features.Dataset(
                        {
                            "observation": tfds.features.FeaturesDict(
                                {
                                    "image": tfds.features.Image(shape=(256, 256, 3), dtype=np.uint8, encoding_format="jpeg"),
                                    "left_wrist_image": tfds.features.Image(
                                        shape=(256, 256, 3), dtype=np.uint8, encoding_format="jpeg"
                                    ),
                                    "right_wrist_image": tfds.features.Image(
                                        shape=(256, 256, 3), dtype=np.uint8, encoding_format="jpeg"
                                    ),
                                    "low_cam_image": tfds.features.Image(
                                        shape=(256, 256, 3), dtype=np.uint8, encoding_format="jpeg"
                                    ),
                                    "state": tfds.features.Tensor(shape=(14,), dtype=np.float32),
                                }
                            ),
                            "action": tfds.features.Tensor(shape=(14,), dtype=np.float32),
                            "discount": tfds.features.Scalar(dtype=np.float32),
                            "reward": tfds.features.Scalar(dtype=np.float32),
                            "is_first": tfds.features.Scalar(dtype=np.bool_),
                            "is_last": tfds.features.Scalar(dtype=np.bool_),
                            "is_terminal": tfds.features.Scalar(dtype=np.bool_),
                            "language_instruction": tfds.features.Sequence(tfds.features.Text()),
                        }
                    ),
                    "episode_metadata": tfds.features.FeaturesDict({"file_path": tfds.features.Text()}),
                }
            )
        )

    def _split_paths(self):
        if self.INPUT_DIR is None:
            raise ValueError("Set INPUT_DIR before calling download_and_prepare().")
        train_files = glob.glob(os.path.join(self.INPUT_DIR, "train", "*.hdf5"))
        val_files = glob.glob(os.path.join(self.INPUT_DIR, "val", "*.hdf5"))
        return {"train": sorted(train_files), "val": sorted(val_files)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True, help="Preprocessed ALOHA-style directory containing train/val.")
    parser.add_argument(
        "--data_dir",
        default=None,
        help="Optional TFDS output root. If omitted, tfds default location is used.",
    )
    args = parser.parse_args()

    aloha_beat_block_hammer_builder.INPUT_DIR = os.path.abspath(args.input_dir)
    builder = aloha_beat_block_hammer_builder(data_dir=args.data_dir)
    builder.download_and_prepare()


if __name__ == "__main__":
    main()
