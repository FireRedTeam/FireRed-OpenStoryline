from __future__ import annotations

import argparse
import re
import sys
import tomllib
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = ROOT_DIR / "config.toml"

class ConfigUpdateError(Exception):
    pass

SECTION_RE = re.compile(r"^\s*\[(.+?)\]\s*(?:#.*)?$")


def parse_assignment(raw: str) -> tuple[str, str]:
    if "=" not in raw:
        raise ConfigUpdateError("--set must be in KEY=VALUE format.")
    key, value = raw.split("=", 1)
    key = key.strip()
    if not key:
        raise ConfigUpdateError("KEY cannot be empty.")
    return key, value


def split_path(path: str) -> list[str]:
    parts = [p.strip() for p in path.split(".")]
    if not parts or any(not p for p in parts):
        raise ConfigUpdateError(f"Invalid path: {path}")
    return parts


def parse_section_header(line: str) -> list[str] | None:
    m = SECTION_RE.match(line)
    if not m:
        return None
    return split_path(m.group(1))


def get_existing_value(data: dict[str, Any], parts: list[str]) -> Any:
    cur: Any = data
    for part in parts:
        if not isinstance(cur, dict) or part not in cur:
            raise ConfigUpdateError(f"Configuration item does not exist: {'.'.join(parts)}")
        cur = cur[part]
    return cur


def coerce_value(raw: str, old_value: Any) -> Any:
    if isinstance(old_value, bool):
        v = raw.strip().lower()
        if v in {"true", "1", "yes", "on"}:
            return True
        if v in {"false", "0", "no", "off"}:
            return False
        raise ConfigUpdateError("Boolean values must be one of: true, false, 1, 0, yes, no, on, or off.")

    if isinstance(old_value, int) and not isinstance(old_value, bool):
        try:
            return int(raw.strip())
        except ValueError as e:
            raise ConfigUpdateError(f"Expected an integer, got: {raw!r}") from e

    if isinstance(old_value, float):
        try:
            return float(raw.strip())
        except ValueError as e:
            raise ConfigUpdateError(f"Expected an float, got: {raw!r}") from e

    if isinstance(old_value, str):
        return raw

    raise ConfigUpdateError(
        f"Only scalar values can be modified (str, int, float, bool); current type: {type(old_value).__name__}"
    )


def toml_escape_string(value: str) -> str:
    value = (
        value.replace("\\", "\\\\")
        .replace('"', '\\"')
        .replace("\b", "\\b")
        .replace("\t", "\\t")
        .replace("\n", "\\n")
        .replace("\f", "\\f")
        .replace("\r", "\\r")
    )
    return f'"{value}"'


def render_toml_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int) and not isinstance(value, bool):
        return str(value)
    if isinstance(value, float):
        return str(value)
    if isinstance(value, str):
        return toml_escape_string(value)
    raise ConfigUpdateError(f"Unsupported value type: {type(value).__name__}")


def split_value_and_comment(text: str) -> tuple[str, str]:
    quote: str | None = None
    escape = False

    for i, ch in enumerate(text):
        if quote is not None:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == quote:
                quote = None
            continue

        if ch in {'"', "'"}:
            quote = ch
        elif ch == "#":
            return text[:i].rstrip(), text[i:]

    return text.rstrip(), ""


def update_text(text: str, parts: list[str], new_value: Any) -> str:
    section_parts = parts[:-1]
    leaf = parts[-1]
    key_re = re.compile(
        rf"^(?P<prefix>\s*{re.escape(leaf)}\s*=\s*)(?P<body>.*?)(?P<newline>\r?\n?)$"
    )

    lines = text.splitlines(keepends=True)
    in_target_section = len(section_parts) == 0
    found_section = in_target_section

    for i, line in enumerate(lines):
        header = parse_section_header(line)
        if header is not None:
            in_target_section = header == section_parts
            if in_target_section:
                found_section = True
            continue

        if not in_target_section:
            continue
        if line.lstrip().startswith("#") or not line.strip():
            continue

        m = key_re.match(line)
        if not m:
            continue

        _, comment = split_value_and_comment(m.group("body"))
        new_line = m.group("prefix") + render_toml_value(new_value)
        if comment:
            new_line += " " + comment.lstrip()
        new_line += m.group("newline")
        lines[i] = new_line
        return "".join(lines)

    if not found_section:
        raise ConfigUpdateError(f"Configuration section not found: {'.'.join(section_parts)}")
    raise ConfigUpdateError(f"Configuration item not found: {'.'.join(parts)}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Update one config item in config.toml")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--set", required=True, metavar="KEY=VALUE")
    args = parser.parse_args()

    config_path = args.config.resolve()
    if not config_path.exists():
        print(f"Configuration file does not exist: {config_path}", file=sys.stderr)
        return 2

    try:
        key_path, raw_value = parse_assignment(args.set)
        parts = split_path(key_path)

        with config_path.open("rb") as f:
            data = tomllib.load(f)

        old_value = get_existing_value(data, parts)
        new_value = coerce_value(raw_value, old_value)

        text = config_path.read_text(encoding="utf-8")
        new_text = update_text(text, parts, new_value)
        config_path.write_text(new_text, encoding="utf-8")

        print(f"Updated: {key_path} = {new_value!r}")
        return 0

    except (ConfigUpdateError, tomllib.TOMLDecodeError) as e:
        print(f"Update failed: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())