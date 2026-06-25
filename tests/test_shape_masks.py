"""Tests for the deterministic mark-shape mask generators (Cross / Rect / Scribble).

Convention under test (matches datasets/mask_generator_512.py): each generator
returns a ``(size, size)`` float32 array where ``1.0`` = keep and ``0.0`` = hole,
with the hole centered. ``scale`` in (0, 1] controls how large the mark is.

Runnable as a plain script (no pytest dependency, mirroring smoke_test_ffl.py):

    .venv/bin/python tests/test_shape_masks.py
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # THESIS repo root

from datasets.shape_masks import (
    _centered_shift,
    _select_random_pattern,
    cross_mask,
    focused_random_mask,
    random_hole_range_for_scale,
    rect_mask,
    scribble_mask,
)

SIZE = 128


def _check_binary_keep_hole(mask):
    assert mask.shape == (SIZE, SIZE), f"shape {mask.shape}"
    assert mask.dtype == np.float32, f"dtype {mask.dtype}"
    uniq = set(np.unique(mask).tolist())
    assert uniq <= {0.0, 1.0}, f"non-binary values {uniq}"
    assert (mask == 0).any(), "expected at least one hole pixel"
    assert (mask == 1).any(), "expected at least one keep pixel"


def _hole_centroid(mask):
    ys, xs = np.where(mask == 0)
    return ys.mean(), xs.mean()


def test_cross_mask_binary_and_centered():
    m = cross_mask(SIZE, scale=0.8)
    _check_binary_keep_hole(m)
    cy, cx = _hole_centroid(m)
    assert abs(cy - SIZE / 2) < SIZE * 0.1, f"cy {cy}"
    assert abs(cx - SIZE / 2) < SIZE * 0.1, f"cx {cx}"


def test_cross_has_vertical_and_horizontal_arms():
    m = cross_mask(SIZE, scale=0.8)
    mid = SIZE // 2
    assert (m[mid, :] == 0).sum() > SIZE * 0.3, "horizontal arm too short"
    assert (m[:, mid] == 0).sum() > SIZE * 0.3, "vertical arm too short"


def test_rect_mask_is_filled_block():
    m = rect_mask(SIZE, scale=0.5)
    _check_binary_keep_hole(m)
    ys, xs = np.where(m == 0)
    block = m[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    assert (block == 0).mean() > 0.95, "rect hole is not a solid block"


def test_scribble_mask_binary_and_reproducible():
    a = scribble_mask(SIZE, scale=0.8, seed=7)
    b = scribble_mask(SIZE, scale=0.8, seed=7)
    _check_binary_keep_hole(a)
    assert np.array_equal(a, b), "same seed must reproduce identical mask"
    c = scribble_mask(SIZE, scale=0.8, seed=8)
    assert not np.array_equal(a, c), "different seeds should differ"


def test_hole_area_increases_with_scale():
    for gen in (cross_mask, rect_mask):
        small = int((gen(SIZE, scale=0.3) == 0).sum())
        large = int((gen(SIZE, scale=0.9) == 0).sum())
        assert large > small, f"{gen.__name__}: {small} !< {large}"
    small = int((scribble_mask(SIZE, scale=0.3, seed=1) == 0).sum())
    large = int((scribble_mask(SIZE, scale=0.9, seed=1) == 0).sum())
    assert large > small, f"scribble: {small} !< {large}"


def test_random_hole_range_monotonic_and_bounds():
    lo_s, hi_s = random_hole_range_for_scale(0.25)
    lo_l, hi_l = random_hole_range_for_scale(1.0)
    assert lo_s < hi_s and lo_l < hi_l, "range must be ordered"
    center_small = 0.5 * (lo_s + hi_s)
    center_large = 0.5 * (lo_l + hi_l)
    assert center_large > center_small, "larger scale must mask more"
    assert abs(center_small - 0.10) < 0.02, f"low end {center_small}"
    assert abs(center_large - 0.36) < 0.02, f"high end {center_large}"


def test_focused_random_mask_binary_and_in_range():
    S = 512
    hole_range = (0.20, 0.27)
    m = focused_random_mask(S, (S // 2, S // 2), hole_range, seed=5)
    assert m.shape == (S, S) and m.dtype == np.float32
    uniq = set(np.unique(m).tolist())
    assert uniq <= {0.0, 1.0}, f"non-binary {uniq}"
    assert (m == 0).any() and (m == 1).any()
    ratio = float((m == 0).mean())
    assert hole_range[0] - 0.03 <= ratio <= hole_range[1] + 0.03, f"hole ratio {ratio}"


def test_focused_random_mask_position_directional_and_reproducible():
    # The MAT centered_shift clips holes that leave the frame, so the absolute centroid
    # drifts; what the spec guarantees is that moving the target moves the mask the same
    # way. Assert directional ordering plus reproducibility.
    S = 512
    rng = (0.20, 0.27)
    top = focused_random_mask(S, (150, 256), rng, seed=5)
    mid = focused_random_mask(S, (256, 256), rng, seed=5)
    bot = focused_random_mask(S, (362, 256), rng, seed=5)
    cy_top = np.where(top == 0)[0].mean()
    cy_mid = np.where(mid == 0)[0].mean()
    cy_bot = np.where(bot == 0)[0].mean()
    assert cy_top < cy_mid < cy_bot, f"y ordering broke: {cy_top:.0f}, {cy_mid:.0f}, {cy_bot:.0f}"

    again = focused_random_mask(S, (256, 256), rng, seed=5)
    assert np.array_equal(mid, again), "same seed must reproduce"
    other = focused_random_mask(S, (256, 256), rng, seed=6)
    assert not np.array_equal(mid, other), "different seed should differ"


def test_focused_random_mask_ratio_preserved_after_shift():
    # Moving the target center searches for a seed that maintains the ratio after shift.
    # Verify that even when shifted towards borders (e.g. 150, 360), the resulting mask
    # has a hole ratio within the target range (with minor tolerance).
    S = 512
    rng = (0.20, 0.27)
    for cy in [256, 150, 360]:
        m = focused_random_mask(S, (cy, 256), rng, seed=11, max_tries=256)
        ratio = float((m == 0).mean())
        assert rng[0] - 0.03 <= ratio <= rng[1] + 0.03, f"ratio {ratio} for cy {cy} was out of range"


def _run():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failures = 0
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
        except Exception as exc:  # noqa: BLE001 - test runner reports any failure
            failures += 1
            print(f"FAIL  {t.__name__}: {type(exc).__name__}: {exc}")
    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(_run())
