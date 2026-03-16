import math


def compute_total_kimg_from_epochs(num_images, total_epochs):
    if total_epochs is None:
        raise ValueError("total_epochs must not be None")
    if total_epochs <= 0:
        raise ValueError("total_epochs must be positive")
    if num_images <= 0:
        raise ValueError("num_images must be positive")
    total_images = num_images * total_epochs
    return int(math.ceil(total_images / 1000.0))


def resolve_training_schedule(total_epochs=None, total_kimg=None, num_images=None):
    if total_epochs is None:
        if total_kimg is None or num_images is None:
            raise ValueError("Provide total_epochs or both total_kimg and num_images")
        total_epochs = (total_kimg * 1000.0) / num_images
    if total_epochs <= 10:
        return {
            "profile": "short",
            "ffl_ratio": 0.005,
            "lr": 1e-4,
            "aug": "noaug",
            "enable_ffl_warmup": False,
            "ffl_warmup_kimg": 0.0,
        }
    if total_epochs <= 30:
        return {
            "profile": "medium",
            "ffl_ratio": 0.01,
            "lr": 8e-5,
            "aug": "noaug",
            "enable_ffl_warmup": False,
            "ffl_warmup_kimg": 0.0,
        }
    return {
        "profile": "long",
        "ffl_ratio": 0.02,
        "lr": 5e-5,
        "aug": "noaug",
        "enable_ffl_warmup": True,
        "ffl_warmup_kimg": 10.0,
    }


def compute_warmup_ratio(current_kimg, warmup_kimg):
    if warmup_kimg <= 0:
        return 1.0
    if current_kimg <= 0:
        return 0.0
    return min(1.0, current_kimg / warmup_kimg)
