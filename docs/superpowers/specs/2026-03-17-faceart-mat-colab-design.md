# FACEART MAT Colab Fine-Tuning Design

## Goal

Update the existing MAT project so Google Colab can fine-tune the original MAT checkpoint on FACEART at 512x512 while keeping the original MAT loss active and adding a configurable Focal Frequency Loss term.

## Constraints

- Keep the original MAT generator and discriminator.
- Keep `losses.loss.TwoStageLoss` as the primary loss path.
- Add `lambda_ffl * ffl_loss` without removing the original loss terms.
- Preserve `train.py` and `generate_image.py` compatibility.
- Keep Google Drive paths and Colab usage practical.
- Modify only files that are necessary.

## Current Issues

- `datasets/`, `losses/`, and `networks/` are missing committed `__init__.py` files, so Colab notebooks currently patch package imports at runtime.
- `torch_utils/misc.py` still uses the older `Sampler` superclass call that the uploaded notebook patches manually for newer PyTorch.
- `generate_image.py` hard-imports `pyspng`, which is brittle on newer Colab images.
- The uploaded notebook mixes data collection, runtime patching, secrets, and training into one file instead of acting as a clean fine-tuning entrypoint.

## Design

### Repo changes

- Keep FFL implementation in `losses/focal_frequency_loss.py`.
- Keep `TwoStageLoss` in `losses/loss.py` and retain the current integration point for FFL.
- Add a user-facing `--lambda-ffl` option in `train.py` while preserving `--ffl-ratio` for backward compatibility.
- Commit missing package marker files in `datasets/`, `losses/`, and `networks/`.
- Update `torch_utils/misc.py` so `InfiniteSampler` does not require notebook-side patching.
- Make `generate_image.py` tolerate missing `pyspng` and fall back to PIL decoding.

### Colab notebook changes

- Rewrite the notebook entrypoint around FACEART fine-tuning only.
- Remove the hardcoded GitHub token and dataset scraping flow.
- Add explicit configuration cells for dataset paths, checkpoint paths, output paths, run name, batch size, duration, and `LAMBDA_FFL`.
- Ensure the notebook sets `PYTHONPATH`, installs dependencies, validates Drive paths, and launches `train.py` with the original MAT checkpoint as the starting point.
- Keep optional post-training inference cells for checkpoint comparison.

## Verification

- Add contract tests for source-level compatibility changes that can run in the current local environment.
- Run the new contract tests after patching.
- Run the existing schedule utility tests.
- If the local environment still lacks full ML dependencies, document that full runtime training verification is deferred to Colab.
