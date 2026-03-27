# Thesis Metrics Pipeline

Standalone thesis evaluation pipeline for MAT-based face-art inpainting.

## Locked Comparison Set

Every model output is compared directly against the same GT/original image folder.

The default comparison targets are:
- MAT original
- finetune_plus_loss
- architecture_phase1
- architecture_phase2

The GT/original image folder is always the numeric reference.
If masked input images are available, they are only used for qualitative panels.

## Locked Metric Stack

Main metrics:
- FID
- LPIPS
- P-IDS
- U-IDS

Supplementary metrics:
- PSNR
- SSIM
- L1

Optional metric:
- ArcFace cosine similarity

## Folder Layout

`thesis_metrics/`

- `config.example.yaml`
- `run_thesis_eval.py`
- `metric_wrappers/`
- `utils/`
- `results/`
- `qualitative/`

## Usage

Dry-run validation:

```bash
python thesis_metrics/run_thesis_eval.py --config thesis_metrics/config.example.yaml --dry-run
```

Full evaluation:

```bash
python thesis_metrics/run_thesis_eval.py --config thesis_metrics/config.example.yaml
```

## What The Pipeline Produces

Each run creates a result folder under `thesis_metrics/results/` containing:
- `summary.csv`
- `summary.json`
- `summary.md`
- `per_model_metrics/`
- `qualitative_panels/`
- `run_log.txt`
- `mismatch_report.txt` when folder stems do not align

## Practical Notes

- Run the full metric suite on the server for consistency.
- The pipeline reuses the existing scripts in `evaluatoin/` for FID, LPIPS, P-IDS, U-IDS, PSNR, SSIM, and L1.
- ArcFace is optional and will be skipped with a clear note if its dependencies are unavailable.
- Qualitative panels use a center-face crop fallback when precise landmark crops are not available.

## Required Config Inputs

- `gt_dir`
- `masked_input_dir` (optional)
- `results.mat_original`
- `results.finetune_plus_loss`
- `results.architecture_phase1`
- `results.architecture_phase2`
- `enable_arcface`
