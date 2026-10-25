#!/usr/bin/env python3
"""Check placeholders in zh-Hant localization entries changed since a revision."""

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
OVERRIDES_PATH = Path("Voco/Localization/overrides.json")
CATALOG_PATH = REPO_ROOT / "Voco/Localization/Localizable.xcstrings"
DEFAULT_BASE_REVISION = "8f4f9caa03850c1d12a1f1522a135281caf0b977"
FORMAT_SPECIFIER = re.compile(
    r"%(?:(\d+)\$)?[-+0# ]*(?:\d+|\*)?(?:\.(?:\d+|\*))?"
    r"(?:hh|h|ll|l|q|L|z|Z|t|j)?[@dDiuUxXoOfFeEgGaAcCsSp]"
)


def load_json(data: str):
    return json.loads(data)


def read_git_json(revision: str):
    result = subprocess.run(
        ["git", "show", f"{revision}:{OVERRIDES_PATH.as_posix()}"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return load_json(result.stdout)


def zh_hant_value(entry):
    return (
        entry.get("localizations", {})
        .get("zh-Hant", {})
        .get("stringUnit", {})
        .get("value")
    )


def english_source(catalog, key):
    entry = catalog.get("strings", {}).get(key, {})
    return (
        entry.get("localizations", {})
        .get("en", {})
        .get("stringUnit", {})
        .get("value", key)
    )


def placeholders(value: str):
    found = []
    index = 0
    while index < len(value):
        if value.startswith("%%", index):
            index += 2
            continue
        match = FORMAT_SPECIFIER.match(value, index)
        if not match:
            index += 1
            continue
        token = match.group(0)
        # In the app's speed labels, x is the literal multiplier suffix of %.1fx.
        end = match.end()
        if token.endswith("f") and end < len(value) and value[end] == "x":
            token += "x"
            end += 1
        token = re.sub(r"%\d+\$", "%", token)
        found.append(token)
        index = end
    return found


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-revision", default=DEFAULT_BASE_REVISION)
    args = parser.parse_args(argv)

    try:
        baseline = read_git_json(args.base_revision).get("strings", {})
        current = load_json(OVERRIDES_PATH.read_text(encoding="utf-8")).get("strings", {})
        catalog = load_json(CATALOG_PATH.read_text(encoding="utf-8"))
    except (OSError, subprocess.CalledProcessError, json.JSONDecodeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    changed = []
    for key, entry in current.items():
        translation = zh_hant_value(entry)
        if translation is None:
            continue
        if translation != zh_hant_value(baseline.get(key, {})):
            changed.append((key, translation))

    errors = []
    for key, translation in changed:
        source = english_source(catalog, key)
        expected = placeholders(source)
        actual = placeholders(translation)
        if expected != actual:
            errors.append((key, expected, actual))
        if "VoiceInk" in translation:
            errors.append((key, "zh-Hant must use Voco", "contains VoiceInk"))

    print(f"new zh-Hant entries checked={len(changed)}")
    print(f"placeholder/brand mismatches={len(errors)}")
    for key, expected, actual in errors:
        print(f"{key!r}: expected {expected!r}, got {actual!r}")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
