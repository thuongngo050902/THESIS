#!/usr/bin/env python3
import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def hstack(images):
    widths, heights = zip(*(im.size for im in images))
    out = Image.new("RGB", (sum(widths), max(heights)), (255, 255, 255))
    x = 0
    for im in images:
        out.paste(im, (x, 0))
        x += im.width
    return out


def with_titles(images, titles):
    if len(images) != len(titles):
        raise ValueError("images and titles must have the same length")

    widths = [im.width for im in images]
    heights = [im.height for im in images]
    title_h = 44
    out = Image.new("RGB", (sum(widths), max(heights) + title_h), (255, 255, 255))

    font = ImageFont.load_default()
    draw = ImageDraw.Draw(out)

    x = 0
    for im, title in zip(images, titles):
        out.paste(im, (x, title_h))
        # Draw a subtle separator line under title row.
        draw.line([(x, title_h - 1), (x + im.width, title_h - 1)], fill=(210, 210, 210), width=1)
        bbox = draw.textbbox((0, 0), title, font=font)
        tw = bbox[2] - bbox[0]
        tx = x + max(6, (im.width - tw) // 2)
        draw.text((tx, 14), title, fill=(20, 20, 20), font=font)
        x += im.width

    return out


def load_rgb(path: Path):
    return Image.open(path).convert("RGB")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--original-dir", required=True)
    parser.add_argument("--overlay-dir", required=True)
    parser.add_argument("--mat-dir", required=True)
    parser.add_argument("--finetune-dir", required=True)
    parser.add_argument("--phase1-dir", required=True)
    parser.add_argument("--out-3up", required=True)
    parser.add_argument("--out-4up", required=True)
    parser.add_argument("--out-5up", required=True)
    parser.add_argument("--t1", default="Original")
    parser.add_argument("--t2", default="Masked Image")
    parser.add_argument("--t3", default="MAT Pretrained (Original)")
    parser.add_argument("--t4", default="MAT Finetuned + Additional Loss")
    parser.add_argument("--t5", default="MAT Modified Transformer")
    args = parser.parse_args()

    original_dir = Path(args.original_dir)
    overlay_dir = Path(args.overlay_dir)
    mat_dir = Path(args.mat_dir)
    finetune_dir = Path(args.finetune_dir)
    phase1_dir = Path(args.phase1_dir)

    out_3 = Path(args.out_3up)
    out_4 = Path(args.out_4up)
    out_5 = Path(args.out_5up)
    for d in [out_3, out_4, out_5]:
        d.mkdir(parents=True, exist_ok=True)

    names = sorted([p.name for p in original_dir.glob("*.png")])
    if not names:
        raise RuntimeError(f"No PNG images found in {original_dir}")

    for name in names:
        original = load_rgb(original_dir / name)
        overlay = load_rgb(overlay_dir / name)
        mat = load_rgb(mat_dir / name)
        finetune = load_rgb(finetune_dir / name)
        phase1 = load_rgb(phase1_dir / name)

        with_titles([original, overlay, mat], [args.t1, args.t2, args.t3]).save(out_3 / name)
        with_titles([original, overlay, mat, finetune], [args.t1, args.t2, args.t3, args.t4]).save(out_4 / name)
        with_titles([original, overlay, mat, finetune, phase1], [args.t1, args.t2, args.t3, args.t4, args.t5]).save(out_5 / name)

    print(f"Wrote {len(names)} images")


if __name__ == "__main__":
    main()
