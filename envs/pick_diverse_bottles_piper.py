"""O.0 Piper data-collection wrapper for pick_diverse_bottles.

This task intentionally reuses the original `pick_diverse_bottles` random
bottle sampling and scripted demo logic. It exists as a separate task name so
Piper-specific data collection can run through `collect_data.sh` without
modifying the original ALOHA/AgileX task implementation.
"""

from .pick_diverse_bottles import pick_diverse_bottles


class pick_diverse_bottles_piper(pick_diverse_bottles):
    """Piper-suffixed task alias using the original pick_diverse_bottles logic."""

    pass
