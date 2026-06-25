# -*- coding: utf-8 -*-
"""Google Colab entrypoint for fine-tuning MAT on FACEART.

This script uses the current repository checkout. It does not clone a second
repo, so the training run will pick up the latest loss/schedule code that is
already present here.
"""

import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import PIL.Image


REPO_ROOT = Path(__file__).resolve().parents[1]
TRAIN_PY = REPO_ROOT / "train.py"
REQUIREMENTS_TXT = REPO_ROOT / "requirements.txt"

# -----------------------------------------------------------------------------
# Colab/user configuration.
# Change these paths for your environment. The defaults work for the common
# Colab case where the dataset zip and pretrained checkpoint are uploaded under
# /content. If you prefer Google Drive, point any of them into /content/drive.

DATASET_ZIP_PATH = Path("/content/FACEART.zip")
DATASET_EXTRACT_PATH = Path("/content/datasets")

FACEART_ROOT = DATASET_EXTRACT_PATH / "FACEART"
FACEART_TRAIN_PATH = FACEART_ROOT / "train"
FACEART_VAL_PATH = FACEART_ROOT / "val"

MAT_PRETRAINED_CHECKPOINT_PATH = Path("/content/Places_512_FullData.pkl")

OUTPUT_ROOT = Path("/content/mat_faceart_runs")
RUN_NAME = "faceart_phase3_structure_guidance"

INSTALL_REQUIREMENTS = True
FORCE_REEXTRACT = False
DRY_RUN = False

NUM_GPUS = 1
BATCH_SIZE = 4
NUM_WORKERS = 2
SNAP = 5
METRICS = "none"
TRAIN_EPOCHS = 40
TOTAL_KIMG = None

MIRROR = True
PR = 0.1
PL = False
TRUNCATION = 0.5
STYLE_MIX = 0.5
EMA = 10

# Keep these as None to let train.py resolve the current long-run schedule.
LAMBDA_FFL = None
FFL_WARMUP_KIMG = None
LR = None
AUG = None
ENABLE_REL_POS_BIAS = False
ENABLE_MASK_BIAS = False
ENABLE_DETERMINISTIC_LATENT_GATE = False
ENABLE_TRAN_ADAPTER_32 = False
ENABLE_TRAN_ADAPTER_16 = False
ENABLE_STRUCTURE_GUIDANCE = True
ENABLE_STRUCTURE_FUSE_16 = True
ENABLE_STRUCTURE_FUSE_STAGE2 = True
ENABLE_STRUCTURE_FUSE_32 = False
ENABLE_ADAPTIVE_STRUCTURE_GATE = True

# -----------------------------------------------------------------------------


def run(command, cwd=None, env=None):
    printable = " ".join(str(part) for part in command)
    print(f"\n$ {printable}")
    subprocess.run([str(part) for part in command], cwd=cwd, env=env, check=True)


def path_uses_drive(path):
    try:
        return Path(path).resolve().as_posix().startswith("/content/drive")
    except FileNotFoundError:
        return str(path).replace("\\", "/").startswith("/content/drive")


def maybe_mount_drive():
    candidate_paths = [
        DATASET_ZIP_PATH,
        DATASET_EXTRACT_PATH,
        FACEART_ROOT,
        MAT_PRETRAINED_CHECKPOINT_PATH,
        OUTPUT_ROOT,
    ]
    if not any(path_uses_drive(path) for path in candidate_paths):
        print("Skipping Google Drive mount.")
        return

    try:
        from google.colab import drive
    except ImportError as exc:
        raise RuntimeError("Configured paths require Google Drive, but google.colab is unavailable.") from exc

    drive.mount("/content/drive")


def ensure_repo_context():
    if not TRAIN_PY.exists():
        raise FileNotFoundError(f"Could not find train.py under repo root: {TRAIN_PY}")
    if not REQUIREMENTS_TXT.exists():
        raise FileNotFoundError(f"Could not find requirements.txt under repo root: {REQUIREMENTS_TXT}")
    print("Using repo root:", REPO_ROOT)


def install_dependencies():
    run([sys.executable, "-m", "pip", "install", "-r", REQUIREMENTS_TXT], cwd=str(REPO_ROOT))


def configure_pythonpath():
    repo_str = str(REPO_ROOT)
    os.environ["PYTHONPATH"] = repo_str + os.pathsep + os.environ.get("PYTHONPATH", "")
    if repo_str not in sys.path:
        sys.path.insert(0, repo_str)


def has_required_faceart_splits(root):
    return root.exists() and (root / "train").is_dir() and (root / "val").is_dir()


def locate_faceart_root(base_path):
    if has_required_faceart_splits(base_path):
        return base_path

    direct_child = base_path / "FACEART"
    if has_required_faceart_splits(direct_child):
        return direct_child

    for candidate in sorted(base_path.rglob("*")):
        if candidate.is_dir() and has_required_faceart_splits(candidate):
            return candidate

    raise FileNotFoundError(
        f"Could not locate FACEART root with train/ and val/ under {base_path}."
    )


def refresh_faceart_paths(root):
    global FACEART_ROOT, FACEART_TRAIN_PATH, FACEART_VAL_PATH
    FACEART_ROOT = root
    FACEART_TRAIN_PATH = FACEART_ROOT / "train"
    FACEART_VAL_PATH = FACEART_ROOT / "val"


def extract_faceart_dataset():
    if has_required_faceart_splits(FACEART_ROOT) and not FORCE_REEXTRACT:
        print("Using existing extracted FACEART dataset:", FACEART_ROOT)
        return

    if DATASET_ZIP_PATH.is_dir():
        detected_root = locate_faceart_root(DATASET_ZIP_PATH)
        refresh_faceart_paths(detected_root)
        print("Using FACEART directory directly:", FACEART_ROOT)
        return

    if not DATASET_ZIP_PATH.exists():
        raise FileNotFoundError(f"Missing dataset zip: {DATASET_ZIP_PATH}")

    DATASET_EXTRACT_PATH.mkdir(parents=True, exist_ok=True)

    if FORCE_REEXTRACT and FACEART_ROOT.exists():
        shutil.rmtree(FACEART_ROOT)

    print("Extracting FACEART zip:", DATASET_ZIP_PATH)
    print("Extraction target:", DATASET_EXTRACT_PATH)
    with zipfile.ZipFile(DATASET_ZIP_PATH, "r") as archive:
        archive.extractall(DATASET_EXTRACT_PATH)

    detected_root = locate_faceart_root(DATASET_EXTRACT_PATH)
    refresh_faceart_paths(detected_root)


def image_extensions():
    PIL.Image.init()
    return {ext.lower() for ext in PIL.Image.EXTENSION.keys()}


def collect_image_files(folder):
    valid_exts = image_extensions()
    return sorted(
        path for path in folder.rglob("*") if path.is_file() and path.suffix.lower() in valid_exts
    )


def print_dataset_inspection():
    train_files = collect_image_files(FACEART_TRAIN_PATH)
    val_files = collect_image_files(FACEART_VAL_PATH)

    if not train_files:
        raise RuntimeError(f"No images found in training split: {FACEART_TRAIN_PATH}")
    if not val_files:
        raise RuntimeError(f"No images found in validation split: {FACEART_VAL_PATH}")
    if not MAT_PRETRAINED_CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"Missing pretrained MAT checkpoint: {MAT_PRETRAINED_CHECKPOINT_PATH}"
        )

    optional_dirs = [
        FACEART_ROOT / "test",
        FACEART_ROOT / "test_masks_small",
        FACEART_ROOT / "clean_512",
        FACEART_ROOT / "raw",
    ]

    print("\nFACEART dataset inspection")
    print("FACEART_ROOT:", FACEART_ROOT)
    print("FACEART_TRAIN_PATH:", FACEART_TRAIN_PATH)
    print("FACEART_VAL_PATH:", FACEART_VAL_PATH)
    print("Train image count:", len(train_files))
    print("Val image count:", len(val_files))
    print("Train sample filenames:", [path.relative_to(FACEART_TRAIN_PATH).as_posix() for path in train_files[:5]])
    print("Val sample filenames:", [path.relative_to(FACEART_VAL_PATH).as_posix() for path in val_files[:5]])
    print("Checkpoint path:", MAT_PRETRAINED_CHECKPOINT_PATH)
    for optional_dir in optional_dirs:
        print(f"Found optional split {optional_dir.name}:", optional_dir.exists())

    return len(train_files), len(val_files)


def compute_total_kimg_from_epochs(num_images, total_epochs):
    return (num_images * total_epochs + 999) // 1000


def resolve_schedule_preview(train_count):
    from training.schedule_utils import resolve_training_schedule

    if TRAIN_EPOCHS is not None:
        total_kimg = compute_total_kimg_from_epochs(train_count, TRAIN_EPOCHS)
        schedule = resolve_training_schedule(total_epochs=TRAIN_EPOCHS)
        approx_epochs = TRAIN_EPOCHS
    elif TOTAL_KIMG is not None:
        total_kimg = TOTAL_KIMG
        schedule = resolve_training_schedule(total_kimg=TOTAL_KIMG, num_images=train_count)
        approx_epochs = (TOTAL_KIMG * 1000.0) / train_count
    else:
        raise ValueError("Set TRAIN_EPOCHS or TOTAL_KIMG.")

    effective_lambda_ffl = LAMBDA_FFL if LAMBDA_FFL is not None else schedule["ffl_ratio"]
    effective_lr = LR if LR is not None else schedule["lr"]
    effective_aug = AUG if AUG is not None else schedule["aug"]
    effective_ffl_warmup = (
        FFL_WARMUP_KIMG
        if FFL_WARMUP_KIMG is not None
        else schedule["ffl_warmup_kimg"] if schedule["enable_ffl_warmup"] else 0.0
    )

    print("\nResolved training schedule")
    print("Approx epochs:", approx_epochs)
    print("Approx total kimg:", total_kimg)
    print("Schedule profile:", schedule["profile"])
    print("Effective lambda_ffl:", effective_lambda_ffl)
    print("Effective lr:", effective_lr)
    print("Effective aug:", effective_aug)
    print("Effective ffl warmup kimg:", effective_ffl_warmup)

    return total_kimg


def build_train_command():
    run_outdir = OUTPUT_ROOT / RUN_NAME
    run_outdir.mkdir(parents=True, exist_ok=True)

    command = [
        sys.executable,
        "train.py",
        "--outdir",
        str(run_outdir),
        "--gpus",
        str(NUM_GPUS),
        "--batch",
        str(BATCH_SIZE),
        "--metrics",
        METRICS,
        "--data",
        str(FACEART_TRAIN_PATH),
        "--data_val",
        str(FACEART_VAL_PATH),
        "--dataloader",
        "datasets.dataset_512.ImageFolderMaskDataset",
        "--mirror",
        str(MIRROR),
        "--cond",
        "False",
        "--cfg",
        "places512",
        "--generator",
        "networks.mat.Generator",
        "--discriminator",
        "networks.mat.Discriminator",
        "--loss",
        "losses.loss.TwoStageLoss",
        "--resume",
        str(MAT_PRETRAINED_CHECKPOINT_PATH),
        "--pr",
        str(PR),
        "--pl",
        str(PL),
        "--truncation",
        str(TRUNCATION),
        "--style_mix",
        str(STYLE_MIX),
        "--ema",
        str(EMA),
        "--workers",
        str(NUM_WORKERS),
        "--snap",
        str(SNAP),
    ]

    if TRAIN_EPOCHS is not None:
        command.extend(["--epochs", str(TRAIN_EPOCHS)])
    elif TOTAL_KIMG is not None:
        command.extend(["--kimg", str(TOTAL_KIMG)])
    else:
        raise ValueError("Set TRAIN_EPOCHS or TOTAL_KIMG.")

    if AUG is not None:
        command.extend(["--aug", str(AUG)])
    if LAMBDA_FFL is not None:
        command.extend(["--lambda-ffl", str(LAMBDA_FFL)])
    if FFL_WARMUP_KIMG is not None:
        command.extend(["--ffl-warmup-kimg", str(FFL_WARMUP_KIMG)])
    if LR is not None:
        command.extend(["--lr", str(LR)])
    if ENABLE_REL_POS_BIAS:
        command.extend(["--enable-rel-pos-bias", str(ENABLE_REL_POS_BIAS)])
    if ENABLE_MASK_BIAS:
        command.extend(["--enable-mask-bias", str(ENABLE_MASK_BIAS)])
    if ENABLE_DETERMINISTIC_LATENT_GATE:
        command.extend(["--enable-deterministic-latent-gate", str(ENABLE_DETERMINISTIC_LATENT_GATE)])
    if ENABLE_TRAN_ADAPTER_32:
        command.extend(["--enable-tran-adapter-32", str(ENABLE_TRAN_ADAPTER_32)])
    if ENABLE_TRAN_ADAPTER_16:
        command.extend(["--enable-tran-adapter-16", str(ENABLE_TRAN_ADAPTER_16)])
    if ENABLE_STRUCTURE_GUIDANCE:
        command.extend(["--enable-structure-guidance", str(ENABLE_STRUCTURE_GUIDANCE)])
    if ENABLE_STRUCTURE_FUSE_16:
        command.extend(["--enable-structure-fuse-16", str(ENABLE_STRUCTURE_FUSE_16)])
    if ENABLE_STRUCTURE_FUSE_STAGE2:
        command.extend(["--enable-structure-fuse-stage2", str(ENABLE_STRUCTURE_FUSE_STAGE2)])
    if ENABLE_STRUCTURE_FUSE_32:
        command.extend(["--enable-structure-fuse-32", str(ENABLE_STRUCTURE_FUSE_32)])
    if ENABLE_ADAPTIVE_STRUCTURE_GATE:
        command.extend(["--enable-adaptive-structure-gate", str(ENABLE_ADAPTIVE_STRUCTURE_GATE)])
    return command


def print_train_preview(command):
    preview = "python train.py " + " ".join(str(part) for part in command[2:])
    print("\nTraining command preview")
    print(preview)


def latest_run_dir():
    run_root = OUTPUT_ROOT / RUN_NAME
    run_dirs = sorted(path for path in run_root.glob("*") if path.is_dir())
    if not run_dirs:
        raise FileNotFoundError(f"No MAT run directory created under {run_root}")
    return run_dirs[-1]


def main():
    maybe_mount_drive()
    ensure_repo_context()
    if INSTALL_REQUIREMENTS:
        install_dependencies()
    configure_pythonpath()

    extract_faceart_dataset()
    train_count, _val_count = print_dataset_inspection()
    resolve_schedule_preview(train_count)

    command = build_train_command()
    print("\nFinal paths passed into train.py")
    print("Train path:", FACEART_TRAIN_PATH)
    print("Validation path:", FACEART_VAL_PATH)
    print("Resume checkpoint:", MAT_PRETRAINED_CHECKPOINT_PATH)
    print_train_preview(command)

    if DRY_RUN:
        print("\nDRY_RUN=True, skipping training launch.")
        return

    run(command, cwd=str(REPO_ROOT), env=os.environ.copy())

    run_dir = latest_run_dir()
    print("\nFinished FACEART fine-tuning run.")
    print("Run directory:", run_dir)


if __name__ == "__main__":
    main()
