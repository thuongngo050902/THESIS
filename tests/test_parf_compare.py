"""Tests for the pure PARF compare-view logic (no Streamlit/torch deps).

Runnable as a plain script (mirrors tests/test_shape_masks.py):

    ../.venv/bin/python tests/test_parf_compare.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # THESIS repo root

from demo_ui.parf_compare import (
    ALL_KEYS,
    COMPARE_LABELS,
    OUTPUT_KEY,
    REFERENCE_KEYS,
    can_move,
    compare_items,
    reorder,
    selected_keys,
)

ALL_ON = {k: True for k in REFERENCE_KEYS}


def test_output_always_first_and_present():
    items = compare_items(ALL_ON, list(ALL_KEYS), set())
    assert items[0] == OUTPUT_KEY
    none_on = compare_items({k: False for k in REFERENCE_KEYS}, list(ALL_KEYS), set())
    assert none_on == [OUTPUT_KEY]


def test_unchecked_reference_excluded():
    cmp = {"masked": True, "mat": False, "coarse": True, "origin": False}
    assert compare_items(cmp, list(ALL_KEYS), set()) == ["output", "masked", "coarse"]


def test_removed_reference_excluded():
    items = compare_items(ALL_ON, list(ALL_KEYS), {"mat"})
    assert "mat" not in items and items[0] == "output"
    assert set(items) == set(ALL_KEYS) - {"mat"}  # the other three remain


def test_output_never_removed():
    items = compare_items(ALL_ON, list(ALL_KEYS), {"output"})
    assert items[0] == "output"


def test_order_is_respected():
    order = ["output", "origin", "coarse", "mat", "masked"]
    assert compare_items(ALL_ON, order, set()) == order


def test_newly_checked_key_appended_canonically():
    order = ["output", "masked", "mat", "origin"]  # 'coarse' missing
    items = compare_items(ALL_ON, order, set())
    assert set(items) == set(ALL_KEYS)
    assert items[:4] == order
    assert items[-1] == "coarse"


def test_reorder_swaps_neighbors():
    items = ["output", "masked", "mat"]
    assert reorder(items, "masked", 1) == ["output", "mat", "masked"]
    assert reorder(items, "mat", -1) == ["output", "mat", "masked"]


def test_reorder_clamps_at_ends():
    items = ["output", "masked", "mat"]
    assert reorder(items, "output", -1) == items
    assert reorder(items, "mat", 1) == items


def test_reorder_unknown_key_noop():
    items = ["output", "masked"]
    assert reorder(items, "origin", 1) == items


def test_can_move_bounds():
    items = ["output", "masked", "mat"]
    assert can_move(items, "output", -1) is False
    assert can_move(items, "output", 1) is True
    assert can_move(items, "mat", 1) is False
    assert can_move(items, "masked", -1) is True
    assert can_move(items, "absent", 1) is False


def test_selected_keys_canonical_order():
    cmp = {"origin": True, "masked": True, "mat": False, "coarse": True}
    assert selected_keys(cmp) == ["masked", "coarse", "origin"]


def test_labels_cover_all_keys():
    for k in ALL_KEYS:
        assert k in COMPARE_LABELS and len(COMPARE_LABELS[k]) == 2


def test_output_present_even_when_order_omits_it():
    items = compare_items(ALL_ON, ["masked", "mat"], set())
    assert OUTPUT_KEY in items
    assert items[0] == OUTPUT_KEY  # defaults to the front when order omits it
    assert set(items) == set(ALL_KEYS)


def test_empty_order_falls_back_to_canonical():
    items = compare_items(ALL_ON, [], set())
    assert items[0] == OUTPUT_KEY
    assert set(items) == set(ALL_KEYS)


def test_duplicate_order_keys_deduped():
    items = compare_items(ALL_ON, ["output", "masked", "output", "mat"], set())
    assert items.count("output") == 1
    assert items.count("masked") == 1


def test_selected_keys_empty():
    assert selected_keys({}) == []
    assert selected_keys({k: False for k in REFERENCE_KEYS}) == []


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
