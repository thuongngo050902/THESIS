#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-train}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
ROOT_DIR="${ROOT_DIR:-$(cd "$REPO_DIR/.." && pwd)}"
VENV_PATH="${VENV_PATH:-/home/subnh3/.venv}"

DATASET_FOLDER_URL="${DATASET_FOLDER_URL:-https://drive.google.com/drive/folders/1eu1ib_mSptmVELnF5fowK2XwvxKvWL0k?usp=drive_link}"
CHECKPOINT_URL="${CHECKPOINT_URL:-https://drive.google.com/file/d/11gOhlb-F_uJzBhIry7SG228jE6ZWYjPJ/view?usp=sharing}"

DATA_RAW_DIR="${DATA_RAW_DIR:-$ROOT_DIR/FACEART_raw}"
DATA_DIR="${DATA_DIR:-$ROOT_DIR/FACEART}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-$ROOT_DIR/checkpoints}"
RUN_DIR="${RUN_DIR:-$ROOT_DIR/runs/faceart_phase1_relbias_gate}"
LOG_DIR="${LOG_DIR:-$ROOT_DIR/logs}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-$CHECKPOINT_DIR/resume_phase1_from_finetune_plus_loss.pkl}"

EPOCHS="${EPOCHS:-40}"
BATCH="${BATCH:-4}"
WORKERS="${WORKERS:-2}"
SNAP="${SNAP:-5}"

activate_venv() {
    if [ ! -f "$VENV_PATH/bin/activate" ]; then
        echo "Missing virtualenv activate script: $VENV_PATH/bin/activate"
        exit 1
    fi
    # shellcheck disable=SC1090
    source "$VENV_PATH/bin/activate"
}

install_runtime_tools() {
    activate_venv
    python -m pip install -U pip gdown
}

download_dataset_folder() {
    mkdir -p "$DATA_RAW_DIR" "$DATA_DIR"
    gdown --folder "$DATASET_FOLDER_URL" -O "$DATA_RAW_DIR" --remaining-ok || true

    find "$DATA_RAW_DIR" -type f -name '*.zip' -print0 | while IFS= read -r -d '' zip_path; do
        unzip_dir="${zip_path%.zip}"
        mkdir -p "$unzip_dir"
        unzip -o "$zip_path" -d "$unzip_dir"
    done

    train_src="$(find "$DATA_RAW_DIR" -type d -name train | head -n 1 || true)"
    val_src="$(find "$DATA_RAW_DIR" -type d -name val | head -n 1 || true)"

    if [ -z "${train_src:-}" ] || [ -z "${val_src:-}" ]; then
        echo "Could not locate train/ and val/ under $DATA_RAW_DIR"
        echo "Run: find \"$DATA_RAW_DIR\" -maxdepth 4 -type d"
        echo "If Google Drive folder download is incomplete, upload a single ZIP file and extract it manually."
        exit 1
    fi

    ln -sfn "$train_src" "$DATA_DIR/train"
    ln -sfn "$val_src" "$DATA_DIR/val"
}

download_checkpoint_file() {
    mkdir -p "$CHECKPOINT_DIR"
    if [ ! -f "$CHECKPOINT_PATH" ]; then
        gdown --fuzzy "$CHECKPOINT_URL" -O "$CHECKPOINT_PATH"
    fi
}

print_layout() {
    echo "ROOT_DIR=$ROOT_DIR"
    echo "REPO_DIR=$REPO_DIR"
    echo "DATA_DIR=$DATA_DIR"
    echo "CHECKPOINT_PATH=$CHECKPOINT_PATH"
    echo "RUN_DIR=$RUN_DIR"
    echo "MODE=$MODE"
}

validate_inputs() {
    if [ ! -d "$DATA_DIR/train" ]; then
        echo "Missing dataset path: $DATA_DIR/train"
        exit 1
    fi
    if [ ! -d "$DATA_DIR/val" ]; then
        echo "Missing dataset path: $DATA_DIR/val"
        exit 1
    fi
    if [ ! -f "$CHECKPOINT_PATH" ]; then
        echo "Missing checkpoint file: $CHECKPOINT_PATH"
        exit 1
    fi
}

run_train() {
    activate_venv
    validate_inputs
    mkdir -p "$RUN_DIR" "$LOG_DIR"
    cd "$REPO_DIR"

    cmd=(
        python train.py
        --outdir "$RUN_DIR"
        --data "$DATA_DIR/train"
        --data_val "$DATA_DIR/val"
        --cfg places512
        --resume "$CHECKPOINT_PATH"
        --epochs "$EPOCHS"
        --batch "$BATCH"
        --workers "$WORKERS"
        --snap "$SNAP"
        --metrics none
        --pr 0.1
        --pl False
        --truncation 0.5
        --style_mix 0.5
        --ema 10
        --lr 5e-5
        --lrt 1e-4
        --lambda-ffl 0.02
        --aug noaug
        --enable-rel-pos-bias True
        --enable-mask-bias True
        --enable-deterministic-latent-gate True
    )

    if [ "$MODE" = "dry-run" ]; then
        cmd+=(--dry-run)
    fi

    printf '%q ' "${cmd[@]}"
    echo
    "${cmd[@]}" 2>&1 | tee "$RUN_DIR/train.log"
}

case "$MODE" in
    prepare)
        install_runtime_tools
        download_dataset_folder
        download_checkpoint_file
        print_layout
        ;;
    dry-run)
        install_runtime_tools
        download_checkpoint_file
        print_layout
        run_train
        ;;
    train)
        install_runtime_tools
        download_checkpoint_file
        print_layout
        run_train
        ;;
    *)
        echo "Usage: bash collab/run_phase1_server_from_drive.sh [prepare|dry-run|train]"
        exit 1
        ;;
esac
