# Phase 1 Server Handoff

This document packages the current Phase 1 code status and the exact server workflow for continuing training on the GPU server with the latest fine-tune checkpoint from Google Drive.

## Phase 1 code already implemented

The current branch for Phase 1 is:

```text
codex/phase1-faceart-train-ready
```

The main code changes already in this branch are:

- `train.py`
  - adds `--enable-rel-pos-bias`
  - adds `--enable-mask-bias`
  - adds `--enable-deterministic-latent-gate`
  - passes those flags into `synthesis_kwargs`
- `collab/train_mat_real_(2).py`
  - adds matching config toggles so Colab and server runs can use the same Phase 1 architecture flags
- `networks/mat.py`
  - adds relative position bias in `WindowAttention`
  - adds mask-aware additive bias in attention
  - adds deterministic latent gate for latent blending instead of the old stochastic dropout-style mixing
- `test_1/test_train_architecture_flags.py`
  - verifies CLI/config flags are threaded correctly
- `test_1/test_mat_transformer_architecture_contract.py`
  - verifies the new transformer architecture contract

## Verification already completed locally

The implementation has already passed these checks in the local development workspace:

```bash
python -m unittest discover -s test_1 -v
python -m py_compile train.py generate_image.py torch_utils/misc.py "collab/train_mat_real_(2).py" networks/mat.py test_1/test_colab_entrypoint_contract.py test_1/test_train_optimizer_betas.py test_1/test_schedule_utils.py test_1/test_training_loop_batch_contract.py test_1/test_train_architecture_flags.py test_1/test_mat_transformer_architecture_contract.py
```

Important note:

- This is verified at unit/static level.
- A full GPU runtime training smoke test has not been executed in the current local environment.

## Server assumptions

These instructions assume:

- server user: `subnh3`
- server venv: `/home/subnh3/.venv`
- working folder root: `/home/subnh3/projects/ThuongNgo`
- repo clone path: `/home/subnh3/projects/ThuongNgo/THESIS-phase1`
- dataset folder link:
  - `https://drive.google.com/drive/folders/1eu1ib_mSptmVELnF5fowK2XwvxKvWL0k?usp=drive_link`
- checkpoint file link:
  - `https://drive.google.com/file/d/11gOhlb-F_uJzBhIry7SG228jE6ZWYjPJ/view?usp=sharing`

## Important runtime note

Using `tmux` will keep the training job alive if:

- the SSH session closes
- VS Code disconnects
- the local laptop is turned off

Using `tmux` does **not** keep the training process alive if:

- the actual GPU server reboots
- the server loses power
- the server process is killed by an administrator

In those cases, resume from the latest training snapshot.

## One-time server setup

SSH into the server and start a persistent terminal session:

```bash
ssh subnh3@10.8.102.81
tmux new -s phase1_setup
```

Create the working root and clone the Phase 1 branch into a folder that does not touch the existing `main` training workspace:

```bash
mkdir -p /home/subnh3/projects/ThuongNgo
cd /home/subnh3/projects/ThuongNgo

git clone -b codex/phase1-faceart-train-ready https://github.com/thuongngo050902/THESIS.git THESIS-phase1
cd /home/subnh3/projects/ThuongNgo/THESIS-phase1
git branch --show-current
```

The branch output should be:

```text
codex/phase1-faceart-train-ready
```

## Recommended script to use on server

This repo now includes a server helper script:

```text
collab/run_phase1_server_from_drive.sh
```

Its default behavior is already configured for:

- root directory: `/home/subnh3/projects/ThuongNgo`
- venv: `/home/subnh3/.venv`
- dataset Drive folder link
- checkpoint Drive file link
- Phase 1 architecture flags

## Step 1: prepare dataset and checkpoint

From the repo root on the server:

```bash
cd /home/subnh3/projects/ThuongNgo/THESIS-phase1
bash collab/run_phase1_server_from_drive.sh prepare
```

This does the following:

- activates `/home/subnh3/.venv`
- installs `gdown`
- downloads the dataset folder into:
  - `/home/subnh3/projects/ThuongNgo/FACEART_raw`
- looks for `train/` and `val/`
- creates stable symlinks:
  - `/home/subnh3/projects/ThuongNgo/FACEART/train`
  - `/home/subnh3/projects/ThuongNgo/FACEART/val`
- downloads the checkpoint into:
  - `/home/subnh3/projects/ThuongNgo/checkpoints/resume_phase1_from_finetune_plus_loss.pkl`

## Step 2: verify paths before launching training

```bash
find /home/subnh3/projects/ThuongNgo/FACEART -maxdepth 2 \( -type l -o -type d \)
ls -lh /home/subnh3/projects/ThuongNgo/checkpoints/resume_phase1_from_finetune_plus_loss.pkl
```

You should see:

- `FACEART/train`
- `FACEART/val`
- the checkpoint `.pkl` file with a non-zero size

## Step 3: run dry-run first

```bash
cd /home/subnh3/projects/ThuongNgo/THESIS-phase1
bash collab/run_phase1_server_from_drive.sh dry-run
```

## Step 4: launch the real training inside tmux

Detached launch:

```bash
tmux new -d -s phase1_job "cd /home/subnh3/projects/ThuongNgo/THESIS-phase1 && bash collab/run_phase1_server_from_drive.sh train"
tmux ls
```

Monitor logs:

```bash
tail -f /home/subnh3/projects/ThuongNgo/runs/faceart_phase1_relbias_gate/train.log
```

Attach back to the running session:

```bash
tmux attach -t phase1_job
```

Detach without stopping the job:

```text
Ctrl+B, then D
```

## The training command embedded in the script

The helper script runs this effective command:

```bash
python train.py \
  --outdir /home/subnh3/projects/ThuongNgo/runs/faceart_phase1_relbias_gate \
  --data /home/subnh3/projects/ThuongNgo/FACEART/train \
  --data_val /home/subnh3/projects/ThuongNgo/FACEART/val \
  --cfg places512 \
  --resume /home/subnh3/projects/ThuongNgo/checkpoints/resume_phase1_from_finetune_plus_loss.pkl \
  --epochs 40 \
  --batch 4 \
  --workers 2 \
  --snap 5 \
  --metrics none \
  --pr 0.1 \
  --pl False \
  --truncation 0.5 \
  --style_mix 0.5 \
  --ema 10 \
  --lr 5e-5 \
  --lrt 1e-4 \
  --lambda-ffl 0.02 \
  --aug noaug \
  --enable-rel-pos-bias True \
  --enable-mask-bias True \
  --enable-deterministic-latent-gate True
```

## If the Drive dataset folder download is incomplete

`gdown --folder` is convenient, but large image folders can be unreliable depending on the Drive layout and file count.

If `prepare` fails to locate `train/` and `val/`, use this fallback instead:

1. Zip the dataset into a single archive on Google Drive.
2. Upload that single zip file.
3. Download it on the server with:

```bash
source /home/subnh3/.venv/bin/activate
python -m pip install -U gdown
gdown --fuzzy "YOUR_DATASET_ZIP_DRIVE_LINK" -O /home/subnh3/projects/ThuongNgo/FACEART.zip
unzip -o /home/subnh3/projects/ThuongNgo/FACEART.zip -d /home/subnh3/projects/ThuongNgo/
```

Then make sure these exist:

```text
/home/subnh3/projects/ThuongNgo/FACEART/train
/home/subnh3/projects/ThuongNgo/FACEART/val
```

## Short handoff prompt for another VS Code agent

You can give another agent this exact prompt:

```text
Read /collab/PHASE1_SERVER_HANDOFF.md and use /collab/run_phase1_server_from_drive.sh as the authoritative server launch script. The Phase 1 code is already implemented on branch codex/phase1-faceart-train-ready. I need the server workflow only: verify the branch clone path is /home/subnh3/projects/ThuongNgo/THESIS-phase1, verify dataset symlinks under /home/subnh3/projects/ThuongNgo/FACEART, verify checkpoint download from the Drive link, run the dry-run first, then launch training in tmux and tell me how to monitor logs.
```
