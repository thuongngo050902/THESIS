from datetime import datetime
from pathlib import Path


def prepare_run_directory(output_root, experiment_name):
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    run_dir = output_root / experiment_name
    if run_dir.exists():
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        run_dir = output_root / (experiment_name + "_" + timestamp)
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def write_lines(path, lines):
    text = "\n".join(str(line) for line in lines) + "\n"
    Path(path).write_text(text, encoding="utf-8")
