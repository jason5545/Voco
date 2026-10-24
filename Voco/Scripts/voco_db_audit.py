#!/usr/bin/env python3
"""
Read-only Voco SwiftData/SQLite audit helper.

The script never opens a production store directly for queries. It first copies
the store plus WAL/SHM sidecars to a temporary directory, then opens the copy in
SQLite read-only mode.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sqlite3
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any


STORE_NAMES = ("default.store", "dictionary.store", "stats.store")
SEARCH_ROOTS = (
    Path.home() / "Library/Application Support",
    Path.home() / "Library/Containers",
    Path.home() / "Library/Group Containers",
)


@dataclass(frozen=True)
class StoreCopy:
    source: Path
    copied: Path


def main() -> int:
    args = parse_args()
    store_paths = locate_store_paths(args)

    if args.dry_run:
        print("Voco DB audit dry run")
        if not store_paths:
            print("No likely Voco stores found.")
            return 0
        for path in store_paths:
            print(path)
        return 0

    if not store_paths:
        print("No likely Voco SQLite/SwiftData stores found.", file=sys.stderr)
        print("Searched roots:", file=sys.stderr)
        for root in SEARCH_ROOTS:
            print(f"  {root}", file=sys.stderr)
        return 2

    with tempfile.TemporaryDirectory(prefix="voco-db-audit-", dir=args.tmp_dir) as tmp:
        tmp_dir = Path(tmp)
        copies = [copy_store(path, tmp_dir) for path in store_paths]
        report = {
            "copiedStores": [
                {"source": str(item.source), "copy": str(item.copied)}
                for item in copies
            ],
        "stores": [audit_store(item, sample_limit=args.limit) for item in copies],
        }

        if args.json:
            print(json.dumps(report, ensure_ascii=False, indent=2))
        else:
            print_human_report(report)

    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit Voco SwiftData stores through read-only copies.")
    parser.add_argument("--db", action="append", type=Path, help="Explicit store path. Can be passed more than once.")
    parser.add_argument("--store-dir", type=Path, help="Directory containing default.store/dictionary.store/stats.store.")
    parser.add_argument("--tmp-dir", type=Path, default=None, help="Temporary directory for copied stores.")
    parser.add_argument("--limit", type=int, default=8, help="Maximum recent/sample rows to print.")
    parser.add_argument("--json", action="store_true", help="Print JSON report.")
    parser.add_argument("--dry-run", action="store_true", help="Only print stores that would be audited.")
    return parser.parse_args()


def locate_store_paths(args: argparse.Namespace) -> list[Path]:
    explicit: list[Path] = []
    if args.db:
        explicit.extend(path.expanduser() for path in args.db)
    if args.store_dir:
        store_dir = args.store_dir.expanduser()
        explicit.extend(store_dir / name for name in STORE_NAMES)

    if explicit:
        return dedupe_existing(explicit)

    candidates: list[Path] = []
    default_app_support = Path.home() / "Library/Application Support/com.jasonchien.Voco"
    for name in STORE_NAMES:
        candidates.append(default_app_support / name)
    default_matches = dedupe_existing(candidates)
    if default_matches:
        return default_matches

    for root in SEARCH_ROOTS:
        if not root.exists():
            continue
        try:
            for path in root.rglob("*"):
                if not path.is_file():
                    continue
                lower = str(path).lower()
                if "voco" not in lower:
                    continue
                if path.name in STORE_NAMES or path.suffix.lower() in {".sqlite", ".sqlite3", ".store"}:
                    candidates.append(path)
        except (OSError, PermissionError):
            continue

    return dedupe_existing(candidates)


def dedupe_existing(paths: list[Path]) -> list[Path]:
    seen: set[Path] = set()
    result: list[Path] = []
    for path in paths:
        expanded = path.expanduser()
        if not expanded.exists() or not expanded.is_file():
            continue
        resolved = expanded.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        result.append(resolved)
    return result


def copy_store(source: Path, tmp_dir: Path) -> StoreCopy:
    copied = tmp_dir / source.name
    shutil.copy2(source, copied)
    for suffix in ("-wal", "-shm"):
        sidecar = source.with_name(source.name + suffix)
        if sidecar.exists():
            shutil.copy2(sidecar, copied.with_name(copied.name + suffix))
    return StoreCopy(source=source, copied=copied)


def audit_store(store: StoreCopy, sample_limit: int) -> dict[str, Any]:
    uri = f"file:{store.copied}?mode=ro"
    con = sqlite3.connect(uri, uri=True)
    con.row_factory = sqlite3.Row
    try:
        tables = list_tables(con)
        schemas = {table: schema_summary(con, table) for table in tables}
        return {
            "source": str(store.source),
            "copy": str(store.copied),
            "tables": tables,
            "schema": schemas,
            "wordReplacementCount": count_word_replacements(con, tables),
            "transcriptionCount": count_transcriptions(con, tables),
            "correctionFeedbackJSONCount": count_correction_feedback(con, tables),
            "ztextEnhancedDifferenceCount": count_text_enhanced_differences(con, tables),
            "routeDistribution": route_distribution(con, tables),
            "recentSamples": recent_samples(con, tables, limit=sample_limit),
        }
    finally:
        con.close()


def list_tables(con: sqlite3.Connection) -> list[str]:
    rows = con.execute(
        """
        SELECT name
        FROM sqlite_master
        WHERE type = 'table'
          AND name NOT LIKE 'sqlite_%'
        ORDER BY name
        """
    ).fetchall()
    return [row["name"] for row in rows]


def schema_summary(con: sqlite3.Connection, table: str) -> list[dict[str, Any]]:
    rows = con.execute(f"PRAGMA table_info({quote_ident(table)})").fetchall()
    return [
        {
            "name": row["name"],
            "type": row["type"],
            "notNull": bool(row["notnull"]),
            "primaryKey": bool(row["pk"]),
        }
        for row in rows
    ]


def count_word_replacements(con: sqlite3.Connection, tables: list[str]) -> int | None:
    table = find_table(tables, "ZWORDREPLACEMENT", "WORDREPLACEMENT")
    if not table:
        return None
    return scalar_int(con, f"SELECT COUNT(*) FROM {quote_ident(table)}")


def count_transcriptions(con: sqlite3.Connection, tables: list[str]) -> int | None:
    table = find_table(tables, "ZTRANSCRIPTION", "TRANSCRIPTION")
    if not table:
        return None
    return scalar_int(con, f"SELECT COUNT(*) FROM {quote_ident(table)}")


def count_correction_feedback(con: sqlite3.Connection, tables: list[str]) -> int | None:
    table = find_table(tables, "ZTRANSCRIPTION", "TRANSCRIPTION")
    if not table:
        return None
    columns = column_names(con, table)
    col = find_column(columns, "ZCORRECTIONFEEDBACKJSON", "CORRECTIONFEEDBACKJSON")
    if not col:
        return None
    sql = f"""
        SELECT COUNT(*)
        FROM {quote_ident(table)}
        WHERE {quote_ident(col)} IS NOT NULL
          AND TRIM({quote_ident(col)}) != ''
          AND TRIM({quote_ident(col)}) != '[]'
    """
    return scalar_int(con, sql)


def count_text_enhanced_differences(con: sqlite3.Connection, tables: list[str]) -> int | None:
    table = find_table(tables, "ZTRANSCRIPTION", "TRANSCRIPTION")
    if not table:
        return None
    columns = column_names(con, table)
    text_col = find_column(columns, "ZTEXT", "TEXT")
    enhanced_col = find_column(columns, "ZENHANCEDTEXT", "ENHANCEDTEXT")
    if not text_col or not enhanced_col:
        return None
    sql = f"""
        SELECT COUNT(*)
        FROM {quote_ident(table)}
        WHERE COALESCE(TRIM({quote_ident(text_col)}), '') != ''
          AND COALESCE(TRIM({quote_ident(enhanced_col)}), '') != ''
          AND TRIM({quote_ident(text_col)}) != TRIM({quote_ident(enhanced_col)})
    """
    return scalar_int(con, sql)


def route_distribution(con: sqlite3.Connection, tables: list[str]) -> dict[str, int] | None:
    table = find_table(tables, "ZTRANSCRIPTION", "TRANSCRIPTION")
    if not table:
        return None
    columns = column_names(con, table)
    route_col = find_column(columns, "ZCONFIDENCEROUTE", "CONFIDENCEROUTE")
    if not route_col:
        return None
    rows = con.execute(
        f"""
        SELECT COALESCE(NULLIF(TRIM({quote_ident(route_col)}), ''), 'unknown') AS route,
               COUNT(*) AS count
        FROM {quote_ident(table)}
        GROUP BY route
        ORDER BY count DESC, route
        """
    ).fetchall()
    return {row["route"]: int(row["count"]) for row in rows}


def recent_samples(con: sqlite3.Connection, tables: list[str], limit: int) -> list[dict[str, Any]] | None:
    table = find_table(tables, "ZTRANSCRIPTION", "TRANSCRIPTION")
    if not table:
        return None
    columns = column_names(con, table)
    text_col = find_column(columns, "ZTEXT", "TEXT")
    enhanced_col = find_column(columns, "ZENHANCEDTEXT", "ENHANCEDTEXT")
    raw_col = find_column(columns, "ZRAWTRANSCRIPT", "RAWTRANSCRIPT")
    route_col = find_column(columns, "ZCONFIDENCEROUTE", "CONFIDENCEROUTE")
    timestamp_col = find_column(columns, "ZTIMESTAMP", "TIMESTAMP")

    select_parts = ["rowid AS rowid"]
    for label, col in (
        ("timestamp", timestamp_col),
        ("route", route_col),
        ("text", text_col),
        ("enhancedText", enhanced_col),
        ("rawTranscript", raw_col),
    ):
        if col:
            select_parts.append(f"{quote_ident(col)} AS {quote_ident(label)}")

    order = f"ORDER BY {quote_ident(timestamp_col)} DESC" if timestamp_col else "ORDER BY rowid DESC"
    rows = con.execute(
        f"""
        SELECT {", ".join(select_parts)}
        FROM {quote_ident(table)}
        {order}
        LIMIT ?
        """
        ,
        (max(0, limit),),
    ).fetchall()

    samples: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        for key in ("text", "enhancedText", "rawTranscript"):
            if key in item:
                item[f"{key}Length"] = len(item[key] or "")
                item[f"{key}Preview"] = preview(item[key])
                del item[key]
        samples.append(item)
    return samples


def find_table(tables: list[str], *candidates: str) -> str | None:
    upper_map = {table.upper(): table for table in tables}
    for candidate in candidates:
        if candidate.upper() in upper_map:
            return upper_map[candidate.upper()]
    for table in tables:
        table_upper = table.upper()
        if any(candidate.upper() in table_upper for candidate in candidates):
            return table
    return None


def column_names(con: sqlite3.Connection, table: str) -> list[str]:
    rows = con.execute(f"PRAGMA table_info({quote_ident(table)})").fetchall()
    return [row["name"] for row in rows]


def find_column(columns: list[str], *candidates: str) -> str | None:
    upper_map = {column.upper(): column for column in columns}
    for candidate in candidates:
        if candidate.upper() in upper_map:
            return upper_map[candidate.upper()]
    for column in columns:
        column_upper = column.upper()
        if any(candidate.upper() in column_upper for candidate in candidates):
            return column
    return None


def scalar_int(con: sqlite3.Connection, sql: str) -> int:
    row = con.execute(sql).fetchone()
    return int(row[0]) if row else 0


def quote_ident(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


def preview(value: Any, limit: int = 48) -> str | None:
    if value is None:
        return None
    text = str(value).replace("\n", " ").strip()
    if len(text) <= limit:
        return text
    return text[: limit - 1] + "..."


def print_human_report(report: dict[str, Any]) -> None:
    print("Voco DB audit report")
    print("Copied stores:")
    for item in report["copiedStores"]:
        print(f"  {item['source']} -> {item['copy']}")

    for store in report["stores"]:
        print()
        print(f"Store: {store['source']}")
        print("Tables:")
        for table in store["tables"]:
            print(f"  - {table}")
        print("Schema summary:")
        for table, columns in store["schema"].items():
            names = ", ".join(f"{col['name']}:{col['type']}" for col in columns)
            print(f"  {table}: {names}")
        print(f"Word replacements: {store['wordReplacementCount']}")
        print(f"Transcriptions: {store['transcriptionCount']}")
        print(f"Correction feedback JSON records: {store['correctionFeedbackJSONCount']}")
        print(f"ZTEXT vs ZENHANCEDTEXT differences: {store['ztextEnhancedDifferenceCount']}")
        print(f"Route distribution: {store['routeDistribution']}")
        if store["recentSamples"]:
            print("Recent samples:")
            for sample in store["recentSamples"]:
                print(f"  - {sample}")


if __name__ == "__main__":
    raise SystemExit(main())
