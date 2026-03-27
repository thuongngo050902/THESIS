import sys
from pathlib import Path


def _ensure_repo_on_path():
    repo_root = Path(__file__).resolve().parents[2]
    repo_root_str = str(repo_root)
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)


def _run_main_metrics(pred_dir, gt_dir):
    _ensure_repo_on_path()
    from evaluatoin.cal_fid_pids_uids import calculate_metrics as calculate_fid_ids

    fid, pids, uids = calculate_fid_ids(str(pred_dir), str(gt_dir))
    return {
        "FID": float(fid),
        "P-IDS": float(pids),
        "U-IDS": float(uids),
    }


def _run_lpips(pred_dir, gt_dir):
    _ensure_repo_on_path()
    from evaluatoin.cal_lpips import calculate_metrics as calculate_lpips

    return {"LPIPS": float(calculate_lpips(str(pred_dir), str(gt_dir)))}


def _run_psnr_ssim_l1(pred_dir, gt_dir):
    _ensure_repo_on_path()
    from evaluatoin.cal_psnr_ssim_l1 import calculate_metrics as calculate_psnr_ssim_l1

    psnr, ssim, l1_value = calculate_psnr_ssim_l1(str(pred_dir), str(gt_dir))
    return {
        "PSNR": float(psnr),
        "SSIM": float(ssim),
        "L1": float(l1_value),
    }


def _run_arcface(pred_dir, gt_dir):
    try:
        import cv2  # type: ignore
        import numpy as np  # type: ignore
        from insightface.app import FaceAnalysis  # type: ignore
    except ImportError as exc:
        raise RuntimeError("ArcFace requires cv2, numpy, and insightface to be installed.") from exc

    app = FaceAnalysis(providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    app.prepare(ctx_id=0, det_size=(640, 640))

    _ensure_repo_on_path()
    from thesis_metrics.matching import collect_dataset_pairs

    scores = []
    for gt_path, pred_path in collect_dataset_pairs(gt_dir=gt_dir, pred_dir=pred_dir):
        gt_img = cv2.imread(str(gt_path))
        pred_img = cv2.imread(str(pred_path))
        gt_faces = app.get(gt_img)
        pred_faces = app.get(pred_img)
        if not gt_faces or not pred_faces:
            continue
        gt_embedding = gt_faces[0].normed_embedding
        pred_embedding = pred_faces[0].normed_embedding
        score = float(np.dot(gt_embedding, pred_embedding))
        scores.append(score)

    if not scores:
        raise RuntimeError("ArcFace could not find valid faces in any paired images.")
    return {"ArcFace": float(sum(scores) / len(scores))}


def run_metric_suite(pred_dir, gt_dir, enable_arcface=False):
    metrics = {}
    notes = []
    try:
        metrics.update(_run_main_metrics(pred_dir=pred_dir, gt_dir=gt_dir))
        metrics.update(_run_lpips(pred_dir=pred_dir, gt_dir=gt_dir))
        metrics.update(_run_psnr_ssim_l1(pred_dir=pred_dir, gt_dir=gt_dir))
    except Exception as exc:
        raise RuntimeError(
            "Failed to compute required metrics for %s against %s: %s" % (pred_dir, gt_dir, exc)
        ) from exc

    if enable_arcface:
        try:
            metrics.update(_run_arcface(pred_dir=pred_dir, gt_dir=gt_dir))
        except Exception as exc:
            notes.append("ArcFace skipped: %s" % exc)

    return metrics, notes
