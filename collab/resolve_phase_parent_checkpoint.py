#!/usr/bin/env python3
"""Resolve the best parent checkpoint for Phase 3 server training."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Iterable, List, Optional, Sequence


DEFAULT_RUN_GLOBS = ("faceart_phase2*", "phase2*")
DEFAULT_SEARCH_CONTAINERS = ("runs", "training-runs")
SNAPSHOT_PATTERN = re.compile(r"network-snapshot-(\d+)\.pkl$")


def normalize_run_globs(run_globs: Optional[Sequence[str]]) -> List[str]:
    if not run_globs:
        return list(DEFAULT_RUN_GLOBS)

    normalized: List[str] = []
    for item in run_globs:
        for token in str(item).split(","):
            token = token.strip()
            if token:
                normalized.append(token)
    return normalized or list(DEFAULT_RUN_GLOBS)


def unique_existing_dirs(paths: Iterable[Path]) -> List[Path]:
    seen = set()
    result: List[Path] = []
    for path in paths:
        try:
            key = str(path.resolve())
        except FileNotFoundError:
            key = str(path)
        if key in seen or not path.is_dir():
            continue
        seen.add(key)
        result.append(path)
    return result


def collect_candidate_run_dirs(
    root_dir: Path,
    preferred_run_dir: Optional[Path] = None,
    run_globs: Optional[Sequence[str]] = None,
) -> List[Path]:
    candidates: List[Path] = []
    if preferred_run_dir is not None:
        candidates.append(preferred_run_dir)

    for container in DEFAULT_SEARCH_CONTAINERS:
        container_dir = root_dir / container
        if not container_dir.is_dir():
            continue
        for pattern in normalize_run_globs(run_globs):
            candidates.extend(sorted(container_dir.glob(pattern)))

    return unique_existing_dirs(candidates)


def snapshot_sort_key(snapshot_path: Path):
    match = SNAPSHOT_PATTERN.fullmatch(snapshot_path.name)
    step = int(match.group(1)) if match else -1
    return (step, snapshot_path.stat().st_mtime, str(snapshot_path))


def discover_latest_snapshot(run_dirs: Iterable[Path]) -> Optional[Path]:
    latest_snapshot: Optional[Path] = None
    latest_key = None

    for run_dir in run_dirs:
        for snapshot_path in run_dir.rglob("network-snapshot-*.pkl"):
            if not snapshot_path.is_file():
                continue
            sort_key = snapshot_sort_key(snapshot_path)
            if latest_key is None or sort_key > latest_key:
                latest_snapshot = snapshot_path
                latest_key = sort_key

    return latest_snapshot


def resolve_parent_checkpoint(
    root_dir: Path,
    explicit_checkpoint_path: Optional[Path] = None,
    preferred_run_dir: Optional[Path] = None,
    run_globs: Optional[Sequence[str]] = None,
) -> Optional[Path]:
    if explicit_checkpoint_path is not None:
        if not explicit_checkpoint_path.is_file():
            raise FileNotFoundError(f"Explicit checkpoint path does not exist: {explicit_checkpoint_path}")
        return explicit_checkpoint_path

    candidate_dirs = collect_candidate_run_dirs(
        root_dir=root_dir,
        preferred_run_dir=preferred_run_dir,
        run_globs=run_globs,
    )
    return discover_latest_snapshot(candidate_dirs)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Resolve the most recent Phase 2 parent checkpoint for Phase 3 resume runs."
    )
    parser.add_argument("--root-dir", required=True, type=Path, help="Server working root that contains runs/")
    parser.add_argument("--checkpoint-path", type=Path, help="Explicit parent checkpoint path")
    parser.add_argument("--preferred-run-dir", type=Path, help="Preferred Phase 2 run root to search first")
    parser.add_argument(
        "--run-globs",
        nargs="*",
        default=None,
        help="Run directory glob(s), for example faceart_phase2* phase2*",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    resolved = resolve_parent_checkpoint(
        root_dir=args.root_dir,
        explicit_checkpoint_path=args.checkpoint_path,
        preferred_run_dir=args.preferred_run_dir,
        run_globs=args.run_globs,
    )

    if resolved is None:
        return 1

    print(resolved)
    return 0


if __name__ == "__main__":
    sys.exit(main())
