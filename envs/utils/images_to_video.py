import cv2
import numpy as np
import os
import subprocess
import pickle
import pdb


def _ffmpeg_env() -> dict:
    env = os.environ.copy()
    ld_library_path = env.get("LD_LIBRARY_PATH", "")
    if ld_library_path:
        filtered = [p for p in ld_library_path.split(":") if p and "/ssd/local_install/lib" not in p]
        env["LD_LIBRARY_PATH"] = ":".join(filtered)
    return env


def images_to_video(imgs: np.ndarray, out_path: str, fps: float = 30.0, is_rgb: bool = True) -> None:
    if (not isinstance(imgs, np.ndarray) or imgs.ndim != 4 or imgs.shape[3] not in (3, 4)):
        raise ValueError("imgs must be a numpy.ndarray of shape (N, H, W, C), with C equal to 3 or 4.")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    n_frames, H, W, C = imgs.shape
    if C == 3:
        pixel_format = "rgb24" if is_rgb else "bgr24"
    else:
        pixel_format = "rgba"
    ffmpeg = subprocess.Popen(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-pixel_format",
            pixel_format,
            "-video_size",
            f"{W}x{H}",
            "-framerate",
            str(fps),
            "-i",
            "-",
            "-pix_fmt",
            "yuv420p",
            "-vcodec",
            "libx264",
            "-crf",
            "23",
            f"{out_path}",
        ],
        stdin=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=_ffmpeg_env(),
    )
    try:
        ffmpeg.stdin.write(imgs.tobytes())
        ffmpeg.stdin.close()
    except BrokenPipeError:
        stderr = ffmpeg.stderr.read().decode("utf-8", errors="replace").strip()
        ffmpeg.wait()
        raise IOError(f"ffmpeg pipe failed: {stderr}")

    stderr = ffmpeg.stderr.read().decode("utf-8", errors="replace").strip()
    if ffmpeg.wait() != 0:
        raise IOError(f"Cannot open ffmpeg. {stderr}")

    print(
        f"🎬 Video is saved to `{out_path}`, containing \033[94m{n_frames}\033[0m frames at {W}×{H} resolution and {fps} FPS."
    )
