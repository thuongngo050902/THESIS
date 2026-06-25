import json
from pathlib import Path


def _strip_inline_comment(text):
    quote = None
    out = []
    for char in text:
        if char in ('"', "'"):
            if quote == char:
                quote = None
            elif quote is None:
                quote = char
        if char == "#" and quote is None:
            break
        out.append(char)
    return "".join(out).rstrip()


def _parse_scalar(raw_value):
    value = raw_value.strip()
    if value in {"null", "Null", "NULL", "~"}:
        return None
    if value in {"true", "True", "TRUE"}:
        return True
    if value in {"false", "False", "FALSE"}:
        return False
    if value.startswith(("'", '"')) and value.endswith(("'", '"')) and len(value) >= 2:
        return value[1:-1]
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    return value


def _simple_yaml_load(text):
    root = {}
    stack = [(-1, root)]

    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        if not raw_line.strip() or raw_line.lstrip().startswith("#"):
            continue
        stripped = _strip_inline_comment(raw_line)
        if not stripped.strip():
            continue
        indent = len(stripped) - len(stripped.lstrip(" "))
        content = stripped.strip()
        if ":" not in content:
            raise ValueError("Unsupported YAML syntax on line %d: %s" % (line_number, raw_line))
        key, raw_value = content.split(":", 1)
        key = key.strip()
        value = raw_value.strip()

        while len(stack) > 1 and indent <= stack[-1][0]:
            stack.pop()
        current = stack[-1][1]
        if value == "":
            child = {}
            current[key] = child
            stack.append((indent, child))
        else:
            current[key] = _parse_scalar(value)

    return root


def load_config(path):
    config_path = Path(path)
    text = config_path.read_text(encoding="utf-8").lstrip("\ufeff")
    suffix = config_path.suffix.lower()
    if suffix == ".json":
        config = json.loads(text)
    else:
        try:
            import yaml  # type: ignore
        except ImportError:
            config = _simple_yaml_load(text)
        else:
            config = yaml.safe_load(text)

    if not isinstance(config, dict):
        raise ValueError("Configuration must be a mapping at the top level.")
    if "results" not in config or not isinstance(config["results"], dict):
        raise ValueError("Configuration must contain a 'results' mapping.")
    return config


def resolve_config_path(config_path, raw_value):
    if raw_value in (None, ""):
        return None
    path = Path(raw_value)
    if path.is_absolute():
        return path
    return (Path(config_path).resolve().parent / path).resolve()
