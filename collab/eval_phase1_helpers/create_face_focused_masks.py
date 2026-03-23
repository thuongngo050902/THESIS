#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

import sys
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
from datasets.mask_generator_512 import RandomMask  # noqa: E402


def save_mask(mask01: np.ndarray, out_path: Path):
    mask255 = (mask01 * 255.0).round().clip(0, 255).astype(np.uint8)
    Image.fromarray(mask255, mode="L").save(out_path)


def make_overlay(image_path: Path, mask_path: Path, out_path: Path):
    img = np.array(Image.open(image_path).convert("RGB"), dtype=np.uint8)
    mask = np.array(Image.open(mask_path).convert("L"), dtype=np.uint8)
    hole = mask == 0
    img[hole] = 0
    Image.fromarray(img, mode="RGB").save(out_path)


def centered_shift(mask: np.ndarray, target_cy: int, target_cx: int) -> np.ndarray:
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
    out[ny[valid], nx[valid]] = 0
    return out


def sample_focused_mask(target_center, ratio_range, seed_base, max_tries=64):
    lo, hi = ratio_range
    target = 0.5 * (lo + hi)
    best = None
    best_dist = 1e9

    for t in range(max_tries):
        np.random.seed(seed_base + t)
        base = RandomMask(512, hole_range=[0, 1])[0].astype(np.float32)
        focused = centered_shift(base, target_center[0], target_center[1])
        ratio = float((focused == 0).mean())
        if lo <= ratio <= hi:
            return focused, ratio, t + 1
        dist = abs(ratio - target)
        if dist < best_dist:
            best_dist = dist
            best = (focused, ratio, t + 1)

    # Fallback to the closest ratio candidate to guarantee completion.
    return best


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-dir", required=True)
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--overlay-root", required=True)
    parser.add_argument("--manifest-json", required=True)
    parser.add_argument("--seed", type=int, default=20260322)
    args = parser.parse_args()

    image_dir = Path(args.image_dir)
    out_root = Path(args.out_root)
    overlay_root = Path(args.overlay_root)
    manifest_json = Path(args.manifest_json)

    image_names = sorted([p.name for p in image_dir.glob("*.png")])
    if len(image_names) != 10:
        raise RuntimeError(f"Expected exactly 10 PNG images in {image_dir}, got {len(image_names)}")

    specs = [
        {
            "name": "center_large",
            "hole_range": [0.28, 0.36],
            "target_center": [256, 256],
            "test_group": "test1",
            "label": "Mask trung tam lon",
        },
        {
            "name": "center_medium",
            "hole_range": [0.20, 0.27],
            "target_center": [256, 256],
            "test_group": "test1",
            "label": "Mask trung tam vua",
        },
        {
            "name": "center_small",
            "hole_range": [0.12, 0.19],
            "target_center": [256, 256],
            "test_group": "test1",
            "label": "Mask trung tam nho",
        },
        {
            "name": "left_large",
            "hole_range": [0.28, 0.36],
            "target_center": [256, 196],
            "test_group": "test2",
            "label": "Mask lon lech trai",
        },
        {
            "name": "right_large",
            "hole_range": [0.28, 0.36],
            "target_center": [256, 316],
            "test_group": "test2",
            "label": "Mask lon lech phai",
        },
    ]

    out_root.mkdir(parents=True, exist_ok=True)
    overlay_root.mkdir(parents=True, exist_ok=True)
    manifest_json.parent.mkdir(parents=True, exist_ok=True)

    summary = {
        "seed": args.seed,
        "specs": [],
        "images": image_names,
        "hole_ratio": {},
    }

    for si, spec in enumerate(specs):
        mask_dir = out_root / spec["name"]
        ov_dir = overlay_root / spec["name"]
        mask_dir.mkdir(parents=True, exist_ok=True)
        ov_dir.mkdir(parents=True, exist_ok=True)

        summary["specs"].append(spec)
        summary["hole_ratio"][spec["name"]] = {}

        attempts_used = []

        for ii, name in enumerate(image_names):
            focused, ratio, attempts = sample_focused_mask(
                target_center=spec["target_center"],
                ratio_range=spec["hole_range"],
                seed_base=args.seed + si * 1000 + ii * 100,
            )
            attempts_used.append(attempts)

            mask_path = mask_dir / name
            save_mask(focused, mask_path)
            make_overlay(image_dir / name, mask_path, ov_dir / name)

            summary["hole_ratio"][spec["name"]][name] = ratio

        spec["sampling_max_attempts"] = max(attempts_used) if attempts_used else 0

    with open(manifest_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps({
        "spec_names": [s["name"] for s in specs],
        "counts": {s["name"]: len(image_names) for s in specs},
        "manifest": str(manifest_json),
    }, indent=2))


if __name__ == "__main__":
    main()
