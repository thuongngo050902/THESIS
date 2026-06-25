#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-train}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
ROOT_DIR="${ROOT_DIR:-/home/subnh3/projects/ThuongNgo}"
VENV_PATH="${VENV_PATH:-/home/subnh3/.venv}"
EXPECTED_BRANCH="${EXPECTED_BRANCH:-codex/phase3-thesis-final}"

DATASET_FOLDER_URL="${DATASET_FOLDER_URL:-https://drive.google.com/drive/folders/1eu1ib_mSptmVELnF5fowK2XwvxKvWL0k?usp=drive_link}"
DATA_RAW_DIR="${DATA_RAW_DIR:-$ROOT_DIR/FACEART_raw}"
DATA_DIR="${DATA_DIR:-$ROOT_DIR/FACEART}"

RUN_ROOT="${RUN_ROOT:-$ROOT_DIR/runs}"
RUN_DIR="${RUN_DIR:-$RUN_ROOT/faceart_phase3_adaptive_structure_guidance}"
LOG_DIR="${LOG_DIR:-$ROOT_DIR/logs}"

CHECKPOINT_DIR="${CHECKPOINT_DIR:-$ROOT_DIR/checkpoints}"
CHECKPOINT_CACHE_PATH="${CHECKPOINT_CACHE_PATH:-$CHECKPOINT_DIR/resume_phase3_from_phase2.pkl}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-}"
CHECKPOINT_URL="${CHECKPOINT_URL:-}"
PREFERRED_PHASE2_RUN_DIR="${PREFERRED_PHASE2_RUN_DIR:-$RUN_ROOT/faceart_phase2_ffl}"
PHASE2_RUN_GLOBS="${PHASE2_RUN_GLOBS:-faceart_phase2*,phase2*}"
RESOLVED_CHECKPOINT_PATH=""

NUM_GPUS="${NUM_GPUS:-1}"
BATCH="${BATCH:-4}"
WORKERS="${WORKERS:-2}"
SNAP="${SNAP:-2}"
EPOCHS="${EPOCHS:-10}"
METRICS="${METRICS:-none}"
MIRROR="${MIRROR:-True}"
PR="${PR:-0.1}"
PL="${PL:-False}"
TRUNCATION="${TRUNCATION:-0.5}"
STYLE_MIX="${STYLE_MIX:-0.5}"
EMA="${EMA:-10}"
LR="${LR:-5e-5}"
LRT="${LRT:-1e-4}"
LAMBDA_FFL="${LAMBDA_FFL:-0.02}"
AUG="${AUG:-noaug}"

DATALOADER="${DATALOADER:-datasets.dataset_512.ImageFolderMaskDataset}"
CFG="${CFG:-places512}"
GENERATOR="${GENERATOR:-networks.mat.Generator}"
DISCRIMINATOR="${DISCRIMINATOR:-networks.mat.Discriminator}"
LOSS="${LOSS:-losses.loss.TwoStageLoss}"

ENABLE_REL_POS_BIAS="${ENABLE_REL_POS_BIAS:-True}"
ENABLE_MASK_BIAS="${ENABLE_MASK_BIAS:-True}"
ENABLE_DETERMINISTIC_LATENT_GATE="${ENABLE_DETERMINISTIC_LATENT_GATE:-True}"
ENABLE_TRAN_ADAPTER_32="${ENABLE_TRAN_ADAPTER_32:-True}"
ENABLE_TRAN_ADAPTER_16="${ENABLE_TRAN_ADAPTER_16:-True}"
ENABLE_STRUCTURE_GUIDANCE="${ENABLE_STRUCTURE_GUIDANCE:-True}"
ENABLE_STRUCTURE_FUSE_16="${ENABLE_STRUCTURE_FUSE_16:-True}"
ENABLE_STRUCTURE_FUSE_STAGE2="${ENABLE_STRUCTURE_FUSE_STAGE2:-True}"
ENABLE_STRUCTURE_FUSE_32="${ENABLE_STRUCTURE_FUSE_32:-False}"
ENABLE_ADAPTIVE_STRUCTURE_GATE="${ENABLE_ADAPTIVE_STRUCTURE_GATE:-True}"

activate_venv() {
    if [ ! -f "$VENV_PATH/bin/activate" ]; then
        echo "Missing virtualenv activate script: $VENV_PATH/bin/activate"
        exit 1
    fi
    # shellcheck disable=SC1090
    source "$VENV_PATH/bin/activate"
}

install_runtime_tools() {
    python -m pip install -U pip gdown
}

ensure_repo_context() {
    if [ ! -f "$REPO_DIR/train.py" ]; then
        echo "Missing train.py under repo root: $REPO_DIR"
        exit 1
    fi
    if [ ! -f "$REPO_DIR/collab/resolve_phase_parent_checkpoint.py" ]; then
        echo "Missing resolve helper under $REPO_DIR/collab"
        exit 1
    fi
}

require_phase3_branch() {
    if ! git -C "$REPO_DIR" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
        echo "Repo directory is not a git checkout: $REPO_DIR"
        exit 1
    fi

    current_branch="$(git -C "$REPO_DIR" branch --show-current)"
    if [ "$current_branch" != "$EXPECTED_BRANCH" ]; then
        echo "Expected branch $EXPECTED_BRANCH but found $current_branch"
        exit 1
    fi
}

link_dataset_from_raw_dir() {
    if [ ! -d "$DATA_RAW_DIR" ]; then
        return 1
    fi

    local train_src
    local val_src
    train_src="$(find "$DATA_RAW_DIR" -type d -name train | head -n 1 || true)"
    val_src="$(find "$DATA_RAW_DIR" -type d -name val | head -n 1 || true)"

    if [ -z "$train_src" ] || [ -z "$val_src" ]; then
        return 1
    fi

    mkdir -p "$DATA_DIR"
    ln -sfn "$train_src" "$DATA_DIR/train"
    ln -sfn "$val_src" "$DATA_DIR/val"
    return 0
}

download_dataset_folder() {
    if [ -z "$DATASET_FOLDER_URL" ]; then
        echo "DATASET_FOLDER_URL is empty and the dataset is not already available on the server."
        exit 1
    fi

    mkdir -p "$DATA_RAW_DIR" "$DATA_DIR"
    gdown --folder "$DATASET_FOLDER_URL" -O "$DATA_RAW_DIR" --remaining-ok || true

    find "$DATA_RAW_DIR" -type f -name '*.zip' -print0 | while IFS= read -r -d '' zip_path; do
        unzip_dir="${zip_path%.zip}"
        mkdir -p "$unzip_dir"
        unzip -o "$zip_path" -d "$unzip_dir"
    done

    if ! link_dataset_from_raw_dir; then
        echo "Could not locate train/ and val/ under $DATA_RAW_DIR"
        echo "Run: find \"$DATA_RAW_DIR\" -maxdepth 4 -type d"
        exit 1
    fi
}

prepare_dataset() {
    if [ -d "$DATA_DIR/train" ] && [ -d "$DATA_DIR/val" ]; then
        return 0
    fi

    if link_dataset_from_raw_dir; then
        return 0
    fi

    download_dataset_folder
}

resolve_phase2_checkpoint() {
    if [ -n "$CHECKPOINT_PATH" ]; then
        if [ ! -f "$CHECKPOINT_PATH" ]; then
            echo "Explicit CHECKPOINT_PATH does not exist: $CHECKPOINT_PATH"
            exit 1
        fi
        RESOLVED_CHECKPOINT_PATH="$CHECKPOINT_PATH"
        return 0
    fi

    local run_glob_args=()
    local run_glob
    IFS=',' read -r -a parsed_run_globs <<< "$PHASE2_RUN_GLOBS"
    for run_glob in "${parsed_run_globs[@]}"; do
        run_glob="$(printf '%s' "$run_glob" | sed 's/^ *//; s/ *$//')"
        if [ -n "$run_glob" ]; then
            run_glob_args+=("$run_glob")
        fi
    done

    local resolved_checkpoint
    if resolved_checkpoint="$(python "$REPO_DIR/collab/resolve_phase_parent_checkpoint.py" --root-dir "$ROOT_DIR" --preferred-run-dir "$PREFERRED_PHASE2_RUN_DIR" --run-globs "${run_glob_args[@]}" 2>/dev/null)"; then
        RESOLVED_CHECKPOINT_PATH="$resolved_checkpoint"
        return 0
    fi

    if [ -n "$CHECKPOINT_URL" ]; then
        mkdir -p "$CHECKPOINT_DIR"
        gdown --fuzzy "$CHECKPOINT_URL" -O "$CHECKPOINT_CACHE_PATH"
        RESOLVED_CHECKPOINT_PATH="$CHECKPOINT_CACHE_PATH"
        return 0
    fi

    echo "Could not locate a Phase 2 checkpoint under $RUN_ROOT"
    echo "Set CHECKPOINT_PATH=/absolute/path/to/network-snapshot-XXXXXX.pkl or CHECKPOINT_URL=<drive-link>"
    exit 1
}

print_layout() {
    echo "ROOT_DIR=$ROOT_DIR"
    echo "REPO_DIR=$REPO_DIR"
    echo "EXPECTED_BRANCH=$EXPECTED_BRANCH"
    echo "DATA_DIR=$DATA_DIR"
    echo "RUN_DIR=$RUN_DIR"
    echo "PREFERRED_PHASE2_RUN_DIR=$PREFERRED_PHASE2_RUN_DIR"
    echo "PHASE2_RUN_GLOBS=$PHASE2_RUN_GLOBS"
    echo "RESOLVED_CHECKPOINT_PATH=$RESOLVED_CHECKPOINT_PATH"
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
    if [ ! -f "$RESOLVED_CHECKPOINT_PATH" ]; then
        echo "Missing checkpoint file: $RESOLVED_CHECKPOINT_PATH"
        exit 1
    fi
}

run_train() {
    validate_inputs
    mkdir -p "$RUN_DIR" "$LOG_DIR"
    cd "$REPO_DIR"

    cmd=(
        python train.py
        --outdir "$RUN_DIR"
        --gpus "$NUM_GPUS"
        --batch "$BATCH"
        --metrics "$METRICS"
        --data "$DATA_DIR/train"
        --data_val "$DATA_DIR/val"
        --dataloader "$DATALOADER"
        --mirror "$MIRROR"
        --cond False
        --cfg "$CFG"
        --generator "$GENERATOR"
        --discriminator "$DISCRIMINATOR"
        --loss "$LOSS"
        --resume "$RESOLVED_CHECKPOINT_PATH"
        --epochs "$EPOCHS"
        --pr "$PR"
        --pl "$PL"
        --truncation "$TRUNCATION"
        --style_mix "$STYLE_MIX"
        --ema "$EMA"
        --workers "$WORKERS"
        --snap "$SNAP"
        --lr "$LR"
        --lrt "$LRT"
        --lambda-ffl "$LAMBDA_FFL"
        --aug "$AUG"
        --enable-rel-pos-bias "$ENABLE_REL_POS_BIAS"
        --enable-mask-bias "$ENABLE_MASK_BIAS"
        --enable-deterministic-latent-gate "$ENABLE_DETERMINISTIC_LATENT_GATE"
        --enable-tran-adapter-32 "$ENABLE_TRAN_ADAPTER_32"
        --enable-tran-adapter-16 "$ENABLE_TRAN_ADAPTER_16"
        --enable-structure-guidance "$ENABLE_STRUCTURE_GUIDANCE"
        --enable-structure-fuse-16 "$ENABLE_STRUCTURE_FUSE_16"
        --enable-structure-fuse-stage2 "$ENABLE_STRUCTURE_FUSE_STAGE2"
        --enable-structure-fuse-32 "$ENABLE_STRUCTURE_FUSE_32"
        --enable-adaptive-structure-gate "$ENABLE_ADAPTIVE_STRUCTURE_GATE"
    )

    if [ "$MODE" = "dry-run" ]; then
        cmd+=(--dry-run)
    fi

    printf '%q ' "${cmd[@]}"
    echo
    "${cmd[@]}" 2>&1 | tee "$RUN_DIR/server-launch.log"
}

main() {
    activate_venv
    ensure_repo_context
    require_phase3_branch

    case "$MODE" in
        prepare)
            install_runtime_tools
            prepare_dataset
            resolve_phase2_checkpoint
            print_layout
            ;;
        resolve-parent)
            install_runtime_tools
            resolve_phase2_checkpoint
            print_layout
            ;;
        dry-run)
            install_runtime_tools
            prepare_dataset
            resolve_phase2_checkpoint
            print_layout
            run_train
            ;;
        train)
            install_runtime_tools
            prepare_dataset
            resolve_phase2_checkpoint
            print_layout
            run_train
            ;;
        *)
            echo "Usage: bash collab/run_phase3_server_from_phase2.sh [prepare|resolve-parent|dry-run|train]"
            exit 1
            ;;
    esac
}

main
