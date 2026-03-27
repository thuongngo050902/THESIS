import csv
import json
from datetime import datetime
from pathlib import Path


METRIC_SPECS = [
    ("FID", "down", "FID↓"),
    ("LPIPS", "down", "LPIPS↓"),
    ("P-IDS", "up", "P-IDS↑"),
    ("U-IDS", "up", "U-IDS↑"),
    ("PSNR", "up", "PSNR↑"),
    ("SSIM", "up", "SSIM↑"),
    ("L1", "down", "L1↓"),
]


def _all_metric_specs(enabled_optional_metrics):
    specs = list(METRIC_SPECS)
    for metric_name in enabled_optional_metrics:
        if metric_name == "ArcFace":
            specs.append(("ArcFace", "up", "ArcFace↑"))
    return specs


def _format_metric_value(value):
    if value is None:
        return ""
    return "%.4f" % float(value)


def _best_by_metric(summary, enabled_optional_metrics):
    best = {}
    for metric_name, direction, _ in _all_metric_specs(enabled_optional_metrics):
        candidates = [(model_name, metrics[metric_name]) for model_name, metrics in summary.items() if metric_name in metrics]
        if not candidates:
            continue
        if direction == "down":
            model_name, metric_value = min(candidates, key=lambda item: item[1])
        else:
            model_name, metric_value = max(candidates, key=lambda item: item[1])
        best[metric_name] = {"model": model_name, "value": float(metric_value)}
    return best


def _write_csv(output_dir, summary, enabled_optional_metrics):
    columns = ["model"] + [metric_name for metric_name, _, _ in _all_metric_specs(enabled_optional_metrics)]
    with (output_dir / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for model_name, metrics in summary.items():
            row = {"model": model_name}
            row.update(metrics)
            writer.writerow(row)


def _write_per_model_json(output_dir, summary):
    per_model_dir = output_dir / "per_model_metrics"
    per_model_dir.mkdir(parents=True, exist_ok=True)
    for model_name, metrics in summary.items():
        payload = {"model": model_name, "metrics": metrics}
        (per_model_dir / (model_name + ".json")).write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _build_markdown_table(summary, enabled_optional_metrics, best_by_metric):
    headers = ["Model"] + [metric_label for _, _, metric_label in _all_metric_specs(enabled_optional_metrics)]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for model_name, metrics in summary.items():
        row = [model_name]
        for metric_name, _, _ in _all_metric_specs(enabled_optional_metrics):
            value = metrics.get(metric_name)
            rendered = _format_metric_value(value)
            if metric_name in best_by_metric and best_by_metric[metric_name]["model"] == model_name and rendered:
                rendered = "**%s**" % rendered
            row.append(rendered)
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def _build_interpretation(best_by_metric, enabled_optional_metrics):
    lines = []
    if "FID" in best_by_metric:
        lines.append("- Strongest realism by FID: `%s`" % best_by_metric["FID"]["model"])
    if "LPIPS" in best_by_metric:
        lines.append("- Strongest perceptual similarity by LPIPS: `%s`" % best_by_metric["LPIPS"]["model"])
    if "U-IDS" in best_by_metric:
        lines.append("- Strongest real-vs-fake indistinguishability by U-IDS: `%s`" % best_by_metric["U-IDS"]["model"])
    if "ArcFace" in best_by_metric and "ArcFace" in enabled_optional_metrics:
        lines.append("- Strongest identity consistency by ArcFace: `%s`" % best_by_metric["ArcFace"]["model"])
    return "\n".join(lines)


def _write_markdown(output_dir, experiment_name, summary, enabled_optional_metrics, qualitative_outputs, notes, best_by_metric):
    lines = [
        "# Thesis Metric Summary",
        "",
        "Experiment: `%s`" % experiment_name,
        "",
        _build_markdown_table(summary, enabled_optional_metrics, best_by_metric),
        "",
        "## Interpretation",
        _build_interpretation(best_by_metric, enabled_optional_metrics) or "- No interpretation available.",
        "",
        "## Qualitative Outputs",
    ]
    if qualitative_outputs:
        lines.extend("- `%s`" % path for path in qualitative_outputs)
    else:
        lines.append("- No qualitative outputs were generated.")
    lines.extend(["", "## Notes"])
    if notes:
        lines.extend("- %s" % note for note in notes)
    else:
        lines.append("- No additional notes.")
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_summary_outputs(output_dir, experiment_name, summary, enabled_optional_metrics, qualitative_outputs, notes):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    best_by_metric = _best_by_metric(summary, enabled_optional_metrics)

    payload = {
        "experiment_name": experiment_name,
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "summary": summary,
        "best_by_metric": best_by_metric,
        "qualitative_outputs": qualitative_outputs,
        "notes": notes,
    }
    (output_dir / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _write_csv(output_dir=output_dir, summary=summary, enabled_optional_metrics=enabled_optional_metrics)
    _write_per_model_json(output_dir=output_dir, summary=summary)
    _write_markdown(
        output_dir=output_dir,
        experiment_name=experiment_name,
        summary=summary,
        enabled_optional_metrics=enabled_optional_metrics,
        qualitative_outputs=qualitative_outputs,
        notes=notes,
        best_by_metric=best_by_metric,
    )
