#!/usr/bin/env python3
import argparse
import copy
import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_UPSTREAM = REPO_ROOT / "VoiceInk" / "Localizable.xcstrings"
DEFAULT_OVERRIDES = REPO_ROOT / "Voco" / "Localization" / "overrides.json"
DEFAULT_OUTPUT = REPO_ROOT / "Voco" / "Localization" / "Localizable.xcstrings"
ALLOWED_OVERRIDE_LANGUAGES = {"en", "zh-Hant"}
FORMAT_PATTERN = re.compile(
    r"%(?:\d+\$)?[-+0#]*(?:\d+|\*)?(?:\.(?:\d+|\*))?"
    r"(?:hh|h|ll|l|q|L|z|Z|t|j)?[@dDiuUxXoOfFeEgGaAcCsSp%]"
)


def reject_duplicate_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def load_json(path):
    with path.open(encoding="utf-8") as handle:
        return json.load(handle, object_pairs_hook=reject_duplicate_keys)


def encoded_json(document):
    return json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def write_json(path, document):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(encoded_json(document), encoding="utf-8")


def localization_value(entry, language):
    return (
        entry.get("localizations", {})
        .get(language, {})
        .get("stringUnit", {})
        .get("value")
    )


def string_localization(value):
    return {"stringUnit": {"state": "translated", "value": value}}


def brand_override_key(key, upstream_strings):
    candidate = key.replace("Voco", "VoiceInk")
    return candidate if candidate != key and candidate in upstream_strings else key


def merge_override_entry(overrides, key, entry):
    strings = overrides["strings"]
    target = strings.setdefault(key, {"localizations": {}})
    for language, localization in entry.get("localizations", {}).items():
        if language not in ALLOWED_OVERRIDE_LANGUAGES:
            raise ValueError(f"unsupported override language {language!r} for {key!r}")
        existing = target["localizations"].get(language)
        if existing is not None and existing != localization:
            raise ValueError(f"conflicting {language} overrides for {key!r}")
        target["localizations"][language] = copy.deepcopy(localization)


def bootstrap_overrides(legacy, upstream):
    overrides = {"sourceLanguage": "en", "strings": {}, "version": "1.0"}
    upstream_strings = upstream["strings"]
    for legacy_key in sorted(legacy["strings"]):
        entry = legacy["strings"][legacy_key]
        key = brand_override_key(legacy_key, upstream_strings)
        translated = entry.get("localizations", {}).get("zh-Hant")
        localizations = {}
        if translated is not None:
            localizations["zh-Hant"] = copy.deepcopy(translated)
        merge_override_entry(overrides, key, {"localizations": localizations})
        candidate = legacy_key.replace("Voco", "VoiceInk")
        if candidate != legacy_key and candidate not in upstream_strings:
            alias = copy.deepcopy(localizations)
            alias["en"] = branded_localization(None, candidate)
            merge_override_entry(overrides, candidate, {"localizations": alias})

    for key in sorted(upstream_strings):
        if "VoiceInk" not in key:
            continue
        source = upstream_strings[key].get("localizations", {}).get("en")
        merge_override_entry(
            overrides,
            key,
            {"localizations": {"en": branded_localization(source, key)}},
        )
    return overrides


def format_specifiers(value):
    specifiers = []
    for match in FORMAT_PATTERN.findall(value):
        if match == "%%":
            continue
        specifiers.append(re.sub(r"^%\d+\$", "%", match))
    return sorted(specifiers)


def localization_values(entry, language):
    localization = entry.get("localizations", {}).get(language, {})
    values = []
    direct = localization.get("stringUnit", {}).get("value")
    if isinstance(direct, str):
        values.append(direct)
    for variation in localization.get("variations", {}).values():
        for case in variation.values():
            value = case.get("stringUnit", {}).get("value")
            if isinstance(value, str):
                values.append(value)
    return values


def branded_localization(source, fallback):
    if source is None:
        return string_localization(fallback.replace("VoiceInk", "Voco"))
    result = copy.deepcopy(source)

    def transform(value):
        if isinstance(value, dict):
            return {key: transform(item) for key, item in value.items()}
        if isinstance(value, list):
            return [transform(item) for item in value]
        if isinstance(value, str):
            return value.replace("VoiceInk", "Voco")
        return value

    return transform(result)


def formats_are_localized(source, translated):
    remaining = list(source)
    for specifier in translated:
        if specifier not in remaining:
            return False
        remaining.remove(specifier)
    return all(specifier == "%@" for specifier in remaining)


def validate_catalog(catalog):
    if catalog.get("sourceLanguage") != "en":
        raise ValueError("catalog sourceLanguage must be en")
    strings = catalog.get("strings")
    if not isinstance(strings, dict):
        raise ValueError("catalog strings must be an object")
    errors = []
    for key, entry in strings.items():
        source_values = localization_values(entry, "en") or [key]
        source_formats = [format_specifiers(value) for value in source_values]
        for language in entry.get("localizations", {}):
            values = localization_values(entry, language)
            if not values:
                errors.append(f"{key!r}: {language} has no stringUnit values")
                continue
            for value in values:
                translated = format_specifiers(value)
                if not any(formats_are_localized(source, translated) for source in source_formats):
                    errors.append(f"{key!r}: {language} format specifiers differ from en")
    if errors:
        raise ValueError("catalog validation failed:\n" + "\n".join(errors[:50]))


def generate_catalog(upstream, overrides):
    catalog = copy.deepcopy(upstream)
    strings = catalog["strings"]
    for key, entry in overrides.get("strings", {}).items():
        target = strings.setdefault(key, {})
        localizations = target.setdefault("localizations", {})
        for language, localization in entry.get("localizations", {}).items():
            if language not in ALLOWED_OVERRIDE_LANGUAGES:
                raise ValueError(f"unsupported override language {language!r} for {key!r}")
            localizations[language] = copy.deepcopy(localization)
    validate_catalog(catalog)
    return catalog


def language_counts(catalog):
    counts = {}
    for entry in catalog["strings"].values():
        for language in entry.get("localizations", {}):
            counts[language] = counts.get(language, 0) + 1
    return dict(sorted(counts.items()))


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream", type=Path, default=DEFAULT_UPSTREAM)
    parser.add_argument("--overrides", type=Path, default=DEFAULT_OVERRIDES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--bootstrap-legacy", type=Path)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    try:
        upstream = load_json(args.upstream)
        validate_catalog(upstream)
        if args.bootstrap_legacy:
            legacy = load_json(args.bootstrap_legacy)
            overrides = bootstrap_overrides(legacy, upstream)
            write_json(args.overrides, overrides)
        else:
            overrides = load_json(args.overrides)
        catalog = generate_catalog(upstream, overrides)
        rendered = encoded_json(catalog)
        if args.check:
            if not args.output.exists() or args.output.read_text(encoding="utf-8") != rendered:
                raise ValueError(f"generated catalog is stale: {args.output}")
        else:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(rendered, encoding="utf-8")
        print(
            f"catalog keys={len(catalog['strings'])} "
            f"override keys={len(overrides['strings'])} languages={language_counts(catalog)}"
        )
        print(f"output={args.output}{' (up to date)' if args.check else ''}")
        return 0
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
