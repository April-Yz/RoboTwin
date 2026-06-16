"""O.2: FoundationPose pnp_tray replay with Piper Cartesian IK."""

from .pick_diverse_bottles_piper_ik_foundation import (
    pick_diverse_bottles_piper_ik_foundation,
)


class pnp_tray_piper_ik_foundation(pick_diverse_bottles_piper_ik_foundation):
    """Pick the annotated cup and bottle, move to the second keyframe, then release."""

    FOUNDATION_OBJECT_KEYS = {
        "left": "left_dark_red_cup",
        "right": "right_bottle",
    }
    FOUNDATION_ACTOR_IDS = {
        "left": "foundation_left_dark_red_cup",
        "right": "foundation_right_bottle",
    }
    FOUNDATION_DEFAULT_ANNOTATION_JSON = (
        "code_painting/h2o_manual_review/pnp_tray/hand_keyframes_all.json"
    )
    FOUNDATION_DEFAULT_HAND_TARGETS_ROOT = "code_painting/human_replay/h2_pure_d435/pnp_tray"
    FOUNDATION_DEFAULT_HAND_TARGETS_PATTERN = "id{episode}_d435_z005/world_targets_and_status.npz"
    FOUNDATION_DEFAULT_LEFT_DESCRIPTION = "left dark red cup"
    FOUNDATION_DEFAULT_RIGHT_DESCRIPTION = "right bottle"
    FOUNDATION_DEFAULT_OPEN_AFTER_ACTION = True
