from pathlib import Path


def _try_import_cv2_stack():
    try:
        import cv2  # type: ignore
        import numpy as np  # type: ignore
    except ImportError as exc:
        return None, None, "Qualitative panel generation skipped: %s" % exc
    return cv2, np, None


def _read_image(cv2, path):
    image = cv2.imread(str(path))
    if image is None:
        raise RuntimeError("Failed to read image: %s" % path)
    return image


def _resize(image, width, height, cv2):
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)


def _label_tile(image, label, cv2, np):
    label_height = 42
    canvas = np.zeros((image.shape[0] + label_height, image.shape[1], 3), dtype=image.dtype)
    canvas[label_height:, :, :] = image
    cv2.putText(canvas, label, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    return canvas


def _center_face_crop(image):
    height, width = image.shape[:2]
    crop_width = int(width * 0.58)
    crop_height = int(height * 0.58)
    start_x = max((width - crop_width) // 2, 0)
    start_y = max(int(height * 0.16), 0)
    end_x = min(start_x + crop_width, width)
    end_y = min(start_y + crop_height, height)
    return image[start_y:end_y, start_x:end_x]


def generate_qualitative_panels(manifest, output_dir, model_order, limit=4):
    cv2, np, error = _try_import_cv2_stack()
    if error is not None:
        return [], [error]

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stems = manifest["stems"][: max(int(limit), 0)]
    outputs = []
    notes = []

    for stem in stems:
        columns = [("GT", manifest["gt_map"][stem])]
        if manifest["masked_map"] is not None:
            columns.append(("Masked", manifest["masked_map"][stem]))
        for model_name in model_order:
            columns.append((model_name, manifest["result_maps"][model_name][stem]))

        images = [_read_image(cv2, path) for _, path in columns]
        tile_height, tile_width = images[0].shape[:2]
        tile_height = min(tile_height, 320)
        tile_width = min(tile_width, 320)

        full_tiles = []
        crop_tiles = []
        for (label, _), image in zip(columns, images):
            full = _resize(image, tile_width, tile_height, cv2)
            crop = _resize(_center_face_crop(image), tile_width, tile_height, cv2)
            full_tiles.append(_label_tile(full, label, cv2, np))
            crop_tiles.append(_label_tile(crop, "%s crop" % label, cv2, np))

        panel = np.hstack(full_tiles)
        crop_panel = np.hstack(crop_tiles)
        panel_path = output_dir / (stem + "_panel.png")
        crop_path = output_dir / (stem + "_center_crop.png")
        cv2.imwrite(str(panel_path), panel)
        cv2.imwrite(str(crop_path), crop_panel)
        outputs.extend([str(panel_path), str(crop_path)])

    if outputs:
        notes.append("Qualitative crops use a center-face fallback instead of landmark-specific crops.")
    return outputs, notes
