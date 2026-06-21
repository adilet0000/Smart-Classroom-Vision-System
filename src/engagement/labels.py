"""
Central definition of all engagement labels used throughout the system.

All components (scorer, temporal tracker, CSV logger, overlay) must import
label constants from here. This prevents label string drift between modules.
"""
from __future__ import annotations

# ── Canonical label strings ────────────────────────────────────────────────
LABEL_ATTENTIVE    = "attentive"
LABEL_DISTRACTED   = "distracted"
LABEL_LOOKING_AWAY = "looking_away"
LABEL_LOOKING_DOWN = "looking_down"
LABEL_EYES_CLOSED  = "eyes_closed"
LABEL_HEAD_TILTED  = "head_tilted"
LABEL_LEANING      = "leaning"

# ── Ordered set of all valid labels ───────────────────────────────────────
ALL_LABELS: tuple[str, ...] = (
    LABEL_ATTENTIVE,
    LABEL_DISTRACTED,
    LABEL_LOOKING_AWAY,
    LABEL_LOOKING_DOWN,
    LABEL_EYES_CLOSED,
    LABEL_HEAD_TILTED,
    LABEL_LEANING,
)

# ── Labels considered "on-task" for attentive_ratio computation ───────────
# Only ATTENTIVE counts as on-task. Adjust here if the definition changes.
ON_TASK_LABELS: frozenset[str] = frozenset({LABEL_ATTENTIVE})

# ── Display colours per label (BGR for OpenCV) ────────────────────────────
LABEL_COLOURS: dict[str, tuple[int, int, int]] = {
    LABEL_ATTENTIVE:    (80, 200, 80),    # green
    LABEL_DISTRACTED:   (0, 165, 255),    # orange
    LABEL_LOOKING_AWAY: (0, 100, 255),    # red-orange
    LABEL_LOOKING_DOWN: (0, 60, 200),     # red
    LABEL_EYES_CLOSED:  (200, 60, 200),   # purple
    LABEL_HEAD_TILTED:  (180, 180, 0),    # yellow
    LABEL_LEANING:      (120, 180, 200),  # teal
}

_LABEL_SET: frozenset[str] = frozenset(ALL_LABELS)


def is_valid_label(label: str) -> bool:
    return label in _LABEL_SET
