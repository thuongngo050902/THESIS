import argparse
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from thesis_metrics.config import load_config, resolve_config_path
from thesis_metrics.matching import build_comparison_manifest, describe_manifest
from thesis_metrics.metric_wrappers import run_metric_suite
from thesis_metrics.qualitative import generate_qualitative_panels
from thesis_metrics.reporting import write_summary_outputs
from thesis_metrics.utils.io_utils import prepare_run_directory, write_lines


def _parse_args():
    parser = argparse.ArgumentParser(description="Run thesis metrics for MAT face-art comparisons.")
    parser.add_argument("--config", required=True, help="Path to the YAML or JSON config file.")
    parser.add_argument("--dry-run", action="store_true", help="Only validate folders and file matching.")
    parser.add_argument("--skip-qualitative", action="store_true", help="Skip qualitative panel generation.")
    return parser.parse_args()


def _resolve_result_dirs(config_path, config):
    result_dirs = {}
    for model_name, raw_path in config["results"].items():
        result_dirs[model_name] = resolve_config_path(config_path, raw_path)
    return result_dirs


def _enabled_optional_metrics(config):
    return ["ArcFace"] if config.get("enable_arcface", False) else []


def main():
    args = _parse_args()
    config_path = Path(args.config).resolve()
    config = load_config(config_path)

    experiment_name = config.get("experiment_name", "thesis_metrics_run")
    output_root = resolve_config_path(config_path, config.get("output_root", "results"))
    run_dir = prepare_run_directory(output_root=output_root, experiment_name=experiment_name)
    run_log_path = run_dir / "run_log.txt"
    mismatch_report_path = run_dir / "mismatch_report.txt"

    gt_dir = resolve_config_path(config_path, config.get("gt_dir"))
    masked_input_dir = resolve_config_path(config_path, config.get("masked_input_dir"))
    result_dirs = _resolve_result_dirs(config_path, config)

    log_lines = [
        "Experiment: %s" % experiment_name,
        "Config: %s" % config_path,
        "GT dir: %s" % gt_dir,
        "Masked input dir: %s" % (masked_input_dir if masked_input_dir is not None else "None"),
        "Enabled optional metrics: %s" % (", ".join(_enabled_optional_metrics(config)) or "none"),
    ]
    log_lines.extend("%s dir: %s" % (model_name, directory) for model_name, directory in result_dirs.items())

    try:
        manifest = build_comparison_manifest(
            gt_dir=gt_dir,
            result_dirs=result_dirs,
            masked_input_dir=masked_input_dir,
        )
    except ValueError as exc:
        mismatch_report_path.write_text(str(exc) + "\n", encoding="utf-8")
        log_lines.append("Mismatch detected: %s" % exc)
        write_lines(run_log_path, log_lines)
        raise SystemExit(str(exc))

    log_lines.extend(describe_manifest(manifest))

    if args.dry_run:
        log_lines.append("Dry-run complete. No metrics were executed.")
        write_lines(run_log_path, log_lines)
        print("\n".join(log_lines))
        return

    summary = {}
    notes = []
    for model_name, model_dir in result_dirs.items():
        log_lines.append("Running metrics for %s" % model_name)
        metrics, model_notes = run_metric_suite(
            pred_dir=model_dir,
            gt_dir=gt_dir,
            enable_arcface=config.get("enable_arcface", False),
        )
        summary[model_name] = metrics
        notes.extend("%s: %s" % (model_name, note) for note in model_notes)

    qualitative_outputs = []
    if args.skip_qualitative:
        notes.append("Qualitative panel generation skipped by flag.")
    else:
        qualitative_dir = run_dir / "qualitative_panels"
        generated_outputs, qualitative_notes = generate_qualitative_panels(
            manifest=manifest,
            output_dir=qualitative_dir,
            model_order=list(result_dirs.keys()),
            limit=int(config.get("qualitative_count", 4)),
        )
        qualitative_outputs.extend(generated_outputs)
        notes.extend(qualitative_notes)

    write_summary_outputs(
        output_dir=run_dir,
        experiment_name=experiment_name,
        summary=summary,
        enabled_optional_metrics=_enabled_optional_metrics(config),
        qualitative_outputs=qualitative_outputs,
        notes=notes,
    )
    write_lines(run_log_path, log_lines + ["Evaluation complete."])
    print("Results saved to %s" % run_dir)


if __name__ == "__main__":
    main()
