"""Pure compare-view logic for the PARF Output step.

No Streamlit/torch/PIL imports — the Streamlit layer (streamlit_app.py) owns the
session state and passes plain Python values in, so these rules stay unit-testable
in isolation.

Compare keys
------------
``output`` is the restored result and is ALWAYS present. The four references are
``masked`` (Damaged Portrait), ``mat`` (baseline), ``coarse`` (Stage 1), and
``origin`` (original portrait). ``order`` is a list of keys giving the user-defined
display order (shared by the grid and the zoom lightbox); ``removed`` is the set of
references hidden from the current comparison via a chip's x button.
"""

from typing import Dict, Iterable, List, Optional, Tuple

OUTPUT_KEY = "output"
REFERENCE_KEYS: Tuple[str, ...] = ("masked", "mat", "coarse", "origin")
ALL_KEYS: Tuple[str, ...] = (OUTPUT_KEY,) + REFERENCE_KEYS

# key -> (title, sub-label) for display.
COMPARE_LABELS: Dict[str, Tuple[str, str]] = {
    "output": ("Output", "Restored result"),
    "masked": ("Damaged Portrait Artwork", "Input with hole"),
    "mat": ("MAT", "Baseline"),
    "coarse": ("Coarse Restoration", "Stage 1"),
    "origin": ("Origin", "Original portrait"),
}


def selected_keys(cmp: Dict[str, bool]) -> List[str]:
    """Reference keys whose checkbox is on, in canonical order."""
    return [k for k in REFERENCE_KEYS if cmp.get(k)]


def compare_items(
    cmp: Dict[str, bool],
    order: Iterable[str],
    removed: Optional[Iterable[str]] = None,
) -> List[str]:
    """The keys to show in the compare view.

    Rules:
    - ``output`` is always present (never removable). It is reorderable like any
      other item, so it sits at whatever position ``order`` gives it; if ``order``
      omits it entirely, it defaults to the front.
    - A reference is included iff its checkbox is on AND it is not in ``removed``.
    - Items follow ``order`` (duplicates collapsed); any allowed reference missing
      from ``order`` is appended in canonical (REFERENCE_KEYS) order so newly-checked
      references show up predictably.
    """
    removed_set = set(removed or ())
    allowed = {OUTPUT_KEY}
    for k in REFERENCE_KEYS:
        if cmp.get(k) and k not in removed_set:
            allowed.add(k)

    result: List[str] = []
    for k in order:
        if k in allowed and k not in result:
            result.append(k)
    for k in REFERENCE_KEYS:
        if k in allowed and k not in result:
            result.append(k)
    if OUTPUT_KEY not in result:
        result.insert(0, OUTPUT_KEY)
    return result


def can_move(items: List[str], key: str, direction: int) -> bool:
    """Whether ``key`` can move by ``direction`` (-1 left/up, +1 right/down) within ``items``."""
    if key not in items:
        return False
    target = items.index(key) + direction
    return 0 <= target < len(items)


def reorder(items: Iterable[str], key: str, direction: int) -> List[str]:
    """Swap ``key`` with its neighbor in ``direction``; clamp at the ends. Returns a new list."""
    new = list(items)
    if key not in new:
        return new
    i = new.index(key)
    j = i + direction
    if 0 <= j < len(new):
        new[i], new[j] = new[j], new[i]
    return new
