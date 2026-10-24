#!/usr/bin/env python3
"""
Toggle Voco Phase 1 phonetic shadow collection through macOS UserDefaults.

This helper intentionally does not touch SwiftData stores, word replacements,
or transcription output. Phase 1 candidate application remains disabled in the
app, and this script also keeps its defaults key false for audit clarity.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


DOMAIN = "com.jasonchien.Voco"
SHADOW_LOGGING_KEY = "VocoPhoneticShadowLoggingEnabled"
CANDIDATE_APPLICATION_KEY = "VocoPhoneticCandidateApplicationEnabled"
APP_SUPPORT = Path.home() / "Library/Application Support/com.jasonchien.Voco"
SHADOW_LOG_DIR = APP_SUPPORT / "ShadowLogs"


def main() -> int:
    args = parse_args()

    if args.command == "enable":
        write_bool(args.domain, SHADOW_LOGGING_KEY, True)
        write_bool(args.domain, CANDIDATE_APPLICATION_KEY, False)
    elif args.command == "disable":
        write_bool(args.domain, SHADOW_LOGGING_KEY, False)
        write_bool(args.domain, CANDIDATE_APPLICATION_KEY, False)
    elif args.command == "status":
        pass
    else:
        raise AssertionError(f"Unhandled command: {args.command}")

    status = read_status(args.domain)
    if args.json:
        print(json.dumps(status, ensure_ascii=False, indent=2, sort_keys=True))
    else:
        print(f"domain: {status['domain']}")
        print(f"{SHADOW_LOGGING_KEY}: {str(status[SHADOW_LOGGING_KEY]).lower()}")
        print(f"{CANDIDATE_APPLICATION_KEY}: {str(status[CANDIDATE_APPLICATION_KEY]).lower()}")
        print(f"shadowLogDirectory: {status['shadowLogDirectory']}")
        print(f"shadowLogDirectoryExists: {str(status['shadowLogDirectoryExists']).lower()}")

    if status[CANDIDATE_APPLICATION_KEY] is not False:
        print("Candidate application defaults key is not false.", file=sys.stderr)
        return 1

    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Enable, disable, or inspect Voco Phase 1 shadow logging defaults."
    )
    parser.add_argument("command", choices=("enable", "disable", "status"))
    parser.add_argument("--domain", default=DOMAIN, help=f"UserDefaults domain. Default: {DOMAIN}")
    parser.add_argument("--json", action="store_true", help="Print status as JSON.")
    return parser.parse_args()


def write_bool(domain: str, key: str, value: bool) -> None:
    raw = "true" if value else "false"
    subprocess.run(
        ["/usr/bin/defaults", "write", domain, key, "-bool", raw],
        check=True,
    )


def read_status(domain: str) -> dict[str, Any]:
    return {
        "domain": domain,
        SHADOW_LOGGING_KEY: read_bool(domain, SHADOW_LOGGING_KEY, default=False),
        CANDIDATE_APPLICATION_KEY: read_bool(domain, CANDIDATE_APPLICATION_KEY, default=False),
        "shadowLogDirectory": str(SHADOW_LOG_DIR),
        "shadowLogDirectoryExists": SHADOW_LOG_DIR.exists(),
    }


def read_bool(domain: str, key: str, default: bool) -> bool:
    result = subprocess.run(
        ["/usr/bin/defaults", "read", domain, key],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    if result.returncode != 0:
        return default

    value = result.stdout.strip().lower()
    if value in {"1", "true", "yes"}:
        return True
    if value in {"0", "false", "no"}:
        return False
    return default


if __name__ == "__main__":
    raise SystemExit(main())
