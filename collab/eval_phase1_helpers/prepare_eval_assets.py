#!/usr/bin/env python3
import argparse
import json
import os
import re
import shutil
from pathlib import Path

import numpy as np
from PIL import Image

# Allow importing repo-local mask generator without modifying repo code.
REPO_ROOT = Path(__file__).resolve().parents[2]
import sys
sys.path.insert(0, str(REPO_ROOT))
from datasets.mask_generator_512 import RandomMask  # noqa: E402


def list_image_files(root: Path):
    exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
    files = [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in exts]
    files.sort(key=lambda p: p.name.lower())
    return files


def ensure_rgb_512(src: Path, dst: Path):
    img = Image.open(src).convert("RGB")
    img = img.resize((512, 512), Image.LANCZOS)
    img.save(dst, format="PNG")


def to_png_name(name: str) -> str:
    stem = Path(name).stem
    return f"{stem}.png"


def save_mask(mask01: np.ndarray, out_path: Path):
    mask255 = (mask01 * 255.0).round().clip(0, 255).astype(np.uint8)
    Image.fromarray(mask255, mode="L").save(out_path)


def make_overlay(image_path: Path, mask_path: Path, out_path: Path):
    img = np.array(Image.open(image_path).convert("RGB"), dtype=np.uint8)
    mask = np.array(Image.open(mask_path).convert("L"), dtype=np.uint8)
    hole = mask == 0
    img[hole] = 0
    Image.fromarray(img, mode="RGB").save(out_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", required=True)
    parser.add_argument("--test10-original", required=True)
    parser.add_argument("--test10-512", required=True)
    parser.add_argument("--mask-large", required=True)
    parser.add_argument("--mask-small", required=True)
    parser.add_argument("--overlay-large", required=True)
    parser.add_argument("--overlay-small", required=True)
    parser.add_argument("--manifest-json", required=True)
    parser.add_argument("--seed", type=int, default=20260322)
    parser.add_argument("--large-min", type=float, default=0.22)
    parser.add_argument("--large-max", type=float, default=0.30)
    parser.add_argument("--small-min", type=float, default=0.15)
    parser.add_argument("--small-max", type=float, default=0.20)
    args = parser.parse_args()

    raw_root = Path(args.raw_root)
    test10_original = Path(args.test10_original)
    test10_512 = Path(args.test10_512)
    mask_large_dir = Path(args.mask_large)
    mask_small_dir = Path(args.mask_small)
    overlay_large_dir = Path(args.overlay_large)
    overlay_small_dir = Path(args.overlay_small)
    manifest_json = Path(args.manifest_json)

    for p in [
        test10_original,
        test10_512,
        mask_large_dir,
        mask_small_dir,
        overlay_large_dir,
        overlay_small_dir,
        manifest_json.parent,
    ]:
        p.mkdir(parents=True, exist_ok=True)

    files = list_image_files(raw_root)
    if len(files) < 10:
        raise RuntimeError(f"Need at least 10 images in {raw_root}, found {len(files)}")

    selected = files[:10]

    # Keep base names stable and avoid collisions by appending index when needed.
    used = set()
    stable_names = []
    for i, src in enumerate(selected, start=1):
        name = to_png_name(src.name)
        if name in used:
            name = f"{Path(name).stem}_{i:02d}.png"
        used.add(name)
        stable_names.append(name)

    large_ratios = {}
    small_ratios = {}

    for idx, (src, out_name) in enumerate(zip(selected, stable_names)):
        dst_orig = test10_original / out_name
        dst_512 = test10_512 / out_name

        shutil.copy2(src, dst_orig)
        ensure_rgb_512(src, dst_512)

        np.random.seed(args.seed + idx)
        mask_large = RandomMask(512, hole_range=[args.large_min, args.large_max])[0]
        np.random.seed(args.seed + 1000 + idx)
        mask_small = RandomMask(512, hole_range=[args.small_min, args.small_max])[0]

        large_mask_path = mask_large_dir / out_name
        small_mask_path = mask_small_dir / out_name
        save_mask(mask_large, large_mask_path)
        save_mask(mask_small, small_mask_path)

        make_overlay(dst_512, large_mask_path, overlay_large_dir / out_name)
        make_overlay(dst_512, small_mask_path, overlay_small_dir / out_name)

        large_ratios[out_name] = float((mask_large == 0).mean())
        small_ratios[out_name] = float((mask_small == 0).mean())

    info = {
        "seed": args.seed,
        "large_hole_range": [args.large_min, args.large_max],
        "small_hole_range": [args.small_min, args.small_max],
        "selected_images": stable_names,
        "large_hole_ratio_per_image": large_ratios,
        "small_hole_ratio_per_image": small_ratios,
    }
    with open(manifest_json, "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)

    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
