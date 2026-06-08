"""Deterministic mark-shape mask generators for the PARF redesign.

These complement the stochastic MAT-style ``RandomMask`` (datasets/mask_generator_512.py)
with three hand-placeable mark shapes used by the "Create Input" step: Cross, Rect, and
Scribble.

Convention (identical to the rest of the codebase): every generator returns a
``(size, size)`` float32 array where ``1.0`` marks a kept pixel and ``0.0`` marks a hole.
The hole is centered; callers reposition it with ``centered_shift`` and rescale/pad with
``resize_mask``. ``scale`` in (0, 1] controls how large the mark is relative to ``size``.

This module deliberately depends only on numpy + PIL (no streamlit/torch) so the shape
generators can be imported and unit-tested in isolation.
"""

from __future__ import annotations

import numpy as np
import PIL.Image
import PIL.ImageDraw

from datasets.mask_generator_512 import RandomMask

KEEP = 1.0
HOLE = 0.0


def _blank(size: int) -> np.ndarray:
    """All-keep canvas of shape (size, size)."""
    return np.ones((size, size), dtype=np.float32)


def _clip_scale(scale: float) -> float:
    return float(np.clip(scale, 0.05, 1.0))


def cross_mask(size: int, scale: float = 1.0) -> np.ndarray:
    """Centered plus/cross-shaped hole. Each arm spans ``scale * size``."""
    size = int(size)
    scale = _clip_scale(scale)
    mask = _blank(size)
    center = size / 2.0
    span = scale * size
    thickness = max(1, int(round(0.22 * span)))
    half_span = span / 2.0
    half_t = thickness / 2.0

    def _bounds(lo: float, hi: float):
        return max(0, int(round(lo))), min(size, int(round(hi)))

    # horizontal arm
    y0, y1 = _bounds(center - half_t, center + half_t)
    x0, x1 = _bounds(center - half_span, center + half_span)
    mask[y0:y1, x0:x1] = HOLE
    # vertical arm
    y0, y1 = _bounds(center - half_span, center + half_span)
    x0, x1 = _bounds(center - half_t, center + half_t)
    mask[y0:y1, x0:x1] = HOLE
    return mask


def rect_mask(size: int, scale: float = 1.0, aspect: float = 0.72) -> np.ndarray:
    """Centered rectangular hole, ``scale * size`` wide and ``scale * size * aspect`` tall."""
    size = int(size)
    scale = _clip_scale(scale)
    mask = _blank(size)
    center = size / 2.0
    width = scale * size
    height = scale * size * aspect

    x0 = max(0, int(round(center - width / 2.0)))
    x1 = min(size, int(round(center + width / 2.0)))
    y0 = max(0, int(round(center - height / 2.0)))
    y1 = min(size, int(round(center + height / 2.0)))
    if x1 <= x0:
        x1 = min(size, x0 + 1)
    if y1 <= y0:
        y1 = min(size, y0 + 1)
    mask[y0:y1, x0:x1] = HOLE
    return mask


def scribble_mask(size: int, scale: float = 1.0, seed: int = 0) -> np.ndarray:
    """Centered freehand scribble hole. Reproducible for a given ``seed``.

    A short poly-line of thick, round-jointed strokes is drawn within a centered box
    of side ``scale * size``. Both the spread and the stroke width grow with ``scale``,
    so the masked area increases monotonically with ``scale``.
    """
    size = int(size)
    scale = _clip_scale(scale)
    rng = np.random.RandomState(int(seed) & 0x7FFFFFFF)

    img = PIL.Image.new("L", (size, size), color=255)  # 255 = keep
    draw = PIL.ImageDraw.Draw(img)

    center = size / 2.0
    span = scale * size
    half = span / 2.0
    width = max(2, int(round(0.12 * span)))

    points = [
        (center + rng.uniform(-half, half), center + rng.uniform(-half, half))
        for _ in range(6)
    ]
    for (x0, y0), (x1, y1) in zip(points[:-1], points[1:]):
        draw.line((x0, y0, x1, y1), fill=0, width=width)  # 0 = hole
    radius = width / 2.0
    for (px, py) in points:  # round the joints so the stroke stays connected
        draw.ellipse((px - radius, py - radius, px + radius, py + radius), fill=0)

    arr = np.array(img, dtype=np.uint8)
    return (arr > 0).astype(np.float32)


# ---------------------------------------------------------------------------
# Random shape — MAT face-focused test spec.
# Mirrors collab/eval_phase1_helpers/create_face_focused_masks.py: a full-image
# RandomMask is shifted so its hole centroid lands on a target, retrying until the
# masked-area ratio falls in range. Size is controlled by hole-ratio, never by a
# geometric resize.
# ---------------------------------------------------------------------------
def _centered_shift(mask: np.ndarray, target_cy: int, target_cx: int) -> np.ndarray:
    """Translate the hole region so its centroid lands at (target_cy, target_cx)."""
    hole = np.argwhere(mask == 0)
    if hole.size == 0:
        return mask
    cy, cx = hole.mean(axis=0)
    dy = int(round(target_cy - cy))
    dx = int(round(target_cx - cx))
    out = np.ones_like(mask, dtype=np.float32)
    ny = hole[:, 0] + dy
    nx = hole[:, 1] + dx
    valid = (ny >= 0) & (ny < mask.shape[0]) & (nx >= 0) & (nx < mask.shape[1])
    out[ny[valid], nx[valid]] = HOLE
    return out


def random_hole_range_for_scale(
    scale: float,
    min_ratio: float = 0.10,
    max_ratio: float = 0.36,
    half_width: float = 0.035,
) -> tuple:
    """Map the 0.25-1.0 scale slider to a (lo, hi) hole-ratio window.

    scale 0.25 -> ~10% masked, scale 1.0 -> ~36% masked, matching the small/large
    hole ranges used by the thesis eval mask set.
    """
    scale = float(np.clip(scale, 0.25, 1.0))
    center = min_ratio + (scale - 0.25) / 0.75 * (max_ratio - min_ratio)
    lo = max(0.02, center - half_width)
    hi = center + half_width
    return (lo, hi)


def _select_random_pattern(size, hole_range, seed, max_tries: int = 64) -> np.ndarray:
    """Pick a RandomMask candidate whose *centered* hole-ratio is in range.

    The acceptance ratio is measured on the centered pattern, so selection depends only on
    ``size``, ``hole_range`` and ``seed`` — never on where the mark will be placed. Returns the
    centered ``(size, size)`` float32 pattern (1=keep, 0=hole).
    """
    size = int(size)
    lo, hi = hole_range
    target_ratio = 0.5 * (lo + hi)
    center = size // 2
    best = None
    best_dist = float("inf")
    for attempt in range(max_tries):
        np.random.seed(int(seed) + attempt)
        base = RandomMask(size, hole_range=[0, 1])[0].astype(np.float32)
        centered = _centered_shift(base, center, center)
        ratio = float((centered == 0).mean())
        if lo <= ratio <= hi:
            return centered
        dist = abs(ratio - target_ratio)
        if dist < best_dist:
            best_dist = dist
            best = centered
    return best if best is not None else np.ones((size, size), dtype=np.float32)


def focused_random_mask(size, target_center, hole_range, seed, max_tries: int = 64) -> np.ndarray:
    """MAT face-focused random mask: select a position-independent pattern, then translate it.

    The pattern is chosen by its centered hole-ratio (depends only on seed + hole_range), then
    shifted to ``target_center``. Moving the target therefore TRANSLATES the same pattern instead
    of regenerating a new one. Returns ``(size, size)`` float32 (1=keep, 0=hole).
    """
    pattern = _select_random_pattern(size, hole_range, seed, max_tries)
    target_cy, target_cx = target_center
    return _centered_shift(pattern, int(target_cy), int(target_cx))
