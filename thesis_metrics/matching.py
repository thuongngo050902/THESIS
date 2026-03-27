from pathlib import Path


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}


def _list_images(directory):
    directory = Path(directory)
    if not directory.exists():
        raise ValueError("Directory does not exist: %s" % directory)
    if not directory.is_dir():
        raise ValueError("Path is not a directory: %s" % directory)
    return sorted(path for path in directory.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES)


def build_stem_map(directory):
    stem_map = {}
    for path in _list_images(directory):
        if path.stem in stem_map:
            raise ValueError("Duplicate stem '%s' found in %s" % (path.stem, directory))
        stem_map[path.stem] = path
    return stem_map


def _compare_stems(reference_label, reference_map, candidate_label, candidate_map):
    missing = sorted(set(reference_map) - set(candidate_map))
    extra = sorted(set(candidate_map) - set(reference_map))
    messages = []
    if missing:
        messages.append("Missing %s files relative to %s: %s" % (candidate_label, reference_label, ", ".join(missing)))
    if extra:
        messages.append("Extra %s files not found in %s: %s" % (candidate_label, reference_label, ", ".join(extra)))
    if messages:
        raise ValueError("\n".join(messages))


def collect_dataset_pairs(gt_dir, pred_dir):
    gt_map = build_stem_map(gt_dir)
    pred_map = build_stem_map(pred_dir)
    _compare_stems("GT", gt_map, "prediction", pred_map)
    return [(gt_map[stem], pred_map[stem]) for stem in sorted(gt_map)]


def build_comparison_manifest(gt_dir, result_dirs, masked_input_dir=None):
    gt_map = build_stem_map(gt_dir)
    result_maps = {}
    for model_name, directory in result_dirs.items():
        result_map = build_stem_map(directory)
        _compare_stems("GT", gt_map, model_name, result_map)
        result_maps[model_name] = result_map

    masked_map = None
    if masked_input_dir is not None:
        masked_map = build_stem_map(masked_input_dir)
        _compare_stems("GT", gt_map, "masked input", masked_map)

    return {
        "count": len(gt_map),
        "stems": sorted(gt_map),
        "gt_map": gt_map,
        "result_maps": result_maps,
        "masked_map": masked_map,
    }


def describe_manifest(manifest):
    lines = ["GT images: %d" % manifest["count"]]
    for model_name, mapping in manifest["result_maps"].items():
        lines.append("%s images: %d" % (model_name, len(mapping)))
    if manifest["masked_map"] is None:
        lines.append("masked input: not provided")
    else:
        lines.append("masked input images: %d" % len(manifest["masked_map"]))
    return lines
