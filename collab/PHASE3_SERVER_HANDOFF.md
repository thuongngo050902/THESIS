# Phase 3 Server Handoff

This document is the authoritative server workflow for continuing Phase 3 training on a dedicated server clone that stays isolated from the Phase 1 and Phase 2 repos while resuming from the latest Phase 2 checkpoint.

## Branch and server layout

The Phase 3 branch is:

```text
codex/phase3-thesis-final
```

Recommended server layout:

- server user: `subnh3`
- server venv: `/home/subnh3/.venv`
- working root: `/home/subnh3/projects/ThuongNgo`
- Phase 1 repo stays where it is now
- Phase 2 repo stays where it is now
- Phase 3 repo clone path: `/home/subnh3/projects/ThuongNgo/THESIS-phase3`
- shared dataset root: `/home/subnh3/projects/ThuongNgo/FACEART`
- Phase 3 run root: `/home/subnh3/projects/ThuongNgo/runs/faceart_phase3_adaptive_structure_guidance`
- tmux session suggestion: `phase3_job`

## What this branch already includes

This branch now contains these server-side helpers:

- `collab/run_phase3_server_from_phase2.sh`
  - validates the repo is on `codex/phase3-thesis-final`
  - reuses `FACEART/train` and `FACEART/val` if they already exist
  - otherwise links them from `FACEART_raw` or downloads the dataset folder from Drive
  - auto-discovers the latest `network-snapshot-*.pkl` under Phase 2 run folders
  - falls back to `CHECKPOINT_PATH` or `CHECKPOINT_URL` when auto-discovery is not enough
  - launches the Phase 3 thesis pilot config with Phase 2 adapters plus Phase 3 structure guidance enabled
- `collab/resolve_phase_parent_checkpoint.py`
  - prints the latest Phase 2 checkpoint path the server script will resume from
- `collab/PHASE3_SERVER_AGENT_PROMPT.txt`
  - ready-to-use prompt for another coding agent

## Default Phase 3 pilot config embedded in the script

The script defaults to this thesis-oriented Phase 3 setup:

- `--enable-rel-pos-bias True`
- `--enable-mask-bias True`
- `--enable-deterministic-latent-gate True`
- `--enable-tran-adapter-32 True`
- `--enable-tran-adapter-16 True`
- `--enable-structure-guidance True`
- `--enable-structure-fuse-16 True`
- `--enable-structure-fuse-stage2 True`
- `--enable-structure-fuse-32 False`
- `--enable-adaptive-structure-gate True`

Other training defaults:

- `epochs=10`
- `batch=4`
- `workers=2`
- `snap=2`
- `lr=5e-5`
- `lrt=1e-4`
- `lambda_ffl=0.02`
- `aug=noaug`

You can override any of these with environment variables before calling the script.

## One-time Phase 3 clone on the server

SSH into the server and create a separate clone for Phase 3:

```bash
ssh subnh3@10.8.102.81
mkdir -p /home/subnh3/projects/ThuongNgo
cd /home/subnh3/projects/ThuongNgo

git clone -b codex/phase3-thesis-final https://github.com/thuongngo050902/THESIS.git THESIS-phase3
cd /home/subnh3/projects/ThuongNgo/THESIS-phase3
git branch --show-current
```

The branch output must be:

```text
codex/phase3-thesis-final
```

## Step 1: prepare dataset links and resolve the Phase 2 parent checkpoint

From the Phase 3 repo root on the server:

```bash
cd /home/subnh3/projects/ThuongNgo/THESIS-phase3
bash collab/run_phase3_server_from_phase2.sh prepare
```

This will:

- activate `/home/subnh3/.venv`
- install `gdown` if needed
- reuse `FACEART/train` and `FACEART/val` when they already exist
- otherwise rebuild the dataset links from `FACEART_raw`
- auto-discover the newest Phase 2 snapshot under `/home/subnh3/projects/ThuongNgo/runs/faceart_phase2*`
- print the resolved layout, including `RESOLVED_CHECKPOINT_PATH`

If you only want to inspect which Phase 2 snapshot will be used:

```bash
bash collab/run_phase3_server_from_phase2.sh resolve-parent
```

## Step 2: override the parent checkpoint only if needed

If auto-discovery selects the wrong Phase 2 snapshot, pin the exact parent checkpoint manually:

```bash
CHECKPOINT_PATH=/home/subnh3/projects/ThuongNgo/runs/faceart_phase2_ffl/00000-your-run/network-snapshot-000080.pkl \
  bash collab/run_phase3_server_from_phase2.sh resolve-parent
```

If the Phase 2 checkpoint is not on the server yet, provide a Drive link instead:

```bash
CHECKPOINT_URL="YOUR_PHASE2_CHECKPOINT_DRIVE_LINK" \
  bash collab/run_phase3_server_from_phase2.sh prepare
```

## Step 3: run a dry-run before the real launch

```bash
cd /home/subnh3/projects/ThuongNgo/THESIS-phase3
bash collab/run_phase3_server_from_phase2.sh dry-run
```

The dry-run should:

- print the final train command
- confirm the resolved Phase 2 parent checkpoint
- confirm the Phase 3 flags
- exit without starting the actual GPU training loop

## Step 4: launch the real training inside tmux

Detached launch:

```bash
tmux new -d -s phase3_job "cd /home/subnh3/projects/ThuongNgo/THESIS-phase3 && bash collab/run_phase3_server_from_phase2.sh train"
tmux ls
```

Attach later:

```bash
tmux attach -t phase3_job
```

Detach without stopping the run:

```text
Ctrl+B, then D
```

## Log monitoring

Server-side launch log:

```bash
tail -f /home/subnh3/projects/ThuongNgo/runs/faceart_phase3_adaptive_structure_guidance/server-launch.log
```

Latest numbered training run directory:

```bash
ls -dt /home/subnh3/projects/ThuongNgo/runs/faceart_phase3_adaptive_structure_guidance/* | head -n 1
```

Training log written by `train.py` inside the latest numbered run:

```bash
tail -f "$(ls -dt /home/subnh3/projects/ThuongNgo/runs/faceart_phase3_adaptive_structure_guidance/* | head -n 1)/log.txt"
```

## Manual checkpoint inspection helper

You can also inspect the parent checkpoint directly through the Python helper:

```bash
source /home/subnh3/.venv/bin/activate
python collab/resolve_phase_parent_checkpoint.py \
  --root-dir /home/subnh3/projects/ThuongNgo \
  --preferred-run-dir /home/subnh3/projects/ThuongNgo/runs/faceart_phase2_ffl \
  --run-globs faceart_phase2* phase2*
```

## Short prompt for another coding agent

Use the prompt in:

```text
collab/PHASE3_SERVER_AGENT_PROMPT.txt
```

That prompt tells another agent to clone the new Phase 3 branch into its own server folder, verify the resolved Phase 2 parent checkpoint, run the dry-run first, then launch training in tmux and report back the exact checkpoint path and log commands.