#!/usr/bin/env python3
"""
Test refined strategies based on optimization analysis findings.

Key insight from analysis:
- Current left_run <= 12 misses word splits like 原因, 層面, 強大 (at run 15)
- But raising threshold blindly adds false positives (clause breaks)
- Observation: word splits near sentence end have small right_run (R <= 3)
- Hypothesis: "left <= 12 OR right <= N" catches word splits without false positives
"""

import sqlite3
import os
from collections import Counter

from validate_comma_cleanup import (
    clean_de_comma, clean_le_comma, clean_verb_pronoun_comma,
    clean_fixed_phrases, is_cjk, BOUNDARY_PUNCTUATION,
)

DB_PATH = os.path.expanduser(
    "~/Library/Application Support/com.jasonchien.Voco/default.store"
)
CORE_DATA_EPOCH = 978307200

RED = "\033[91m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
DIM = "\033[2m"
BOLD = "\033[1m"
RESET = "\033[0m"


def read_records(limit=500):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.execute(
        """
        SELECT
            datetime(ZTIMESTAMP + ?, 'unixepoch', 'localtime') as dt,
            ZTEXT,
            ZENHANCEDTEXT
        FROM ZTRANSCRIPTION
        WHERE ZENHANCEDTEXT IS NOT NULL
            AND length(ZENHANCEDTEXT) > 0
        ORDER BY ZTIMESTAMP DESC
        LIMIT ?
        """,
        (CORE_DATA_EPOCH, limit),
    )
    rows = cursor.fetchall()
    conn.close()
    return rows


def apply_rule_based(text):
    result = clean_de_comma(text)
    result = clean_le_comma(result)
    result = clean_verb_pronoun_comma(result)
    result = clean_fixed_phrases(result)
    return result


def find_candidates(text, ztext):
    """Find commas where pair is in ztext, with L/R run info."""
    chars = list(text)
    candidates = []
    cjk_run = 0

    for i, ch in enumerate(chars):
        if (ch == "，" and i >= 1 and i + 1 < len(chars)
                and is_cjk(chars[i - 1]) and is_cjk(chars[i + 1])):

            pair = chars[i - 1] + chars[i + 1]
            if pair not in ztext:
                if is_cjk(ch):
                    cjk_run += 1
                elif ch in BOUNDARY_PUNCTUATION:
                    cjk_run = 0
                continue

            # Forward run
            fwd_run = 0
            for j in range(i + 1, len(chars)):
                if is_cjk(chars[j]):
                    fwd_run += 1
                elif chars[j] in BOUNDARY_PUNCTUATION:
                    break

            candidates.append({
                "pos": i,
                "pair": pair,
                "L": cjk_run,
                "R": fwd_run,
                "context": text[max(0, i - 8):i + 9],
            })

        if is_cjk(ch):
            cjk_run += 1
        elif ch in BOUNDARY_PUNCTUATION:
            cjk_run = 0

    return candidates


def apply_strategy(candidates, predicate):
    """Return candidates that would be removed by the given predicate."""
    return [c for c in candidates if predicate(c)]


def main():
    rows = read_records(500)
    print(f"\n{BOLD}=== Refined Strategy Testing (500 records) ==={RESET}\n")

    # Collect all candidates
    all_candidates = []
    for dt, ztext, zenhanced in rows:
        after_rules = apply_rule_based(zenhanced)
        candidates = find_candidates(after_rules, ztext)
        for c in candidates:
            c["dt"] = dt
            c["ztext"] = ztext
            c["enhanced"] = zenhanced
        all_candidates.extend(candidates)

    print(f"Total ZTEXT-matched comma candidates: {len(all_candidates)}\n")

    # ── Strategy comparison ──
    strategies = [
        ("A: left <= 12 (current)",           lambda c: c["L"] <= 12),
        ("B: left <= 12 OR right <= 2",       lambda c: c["L"] <= 12 or c["R"] <= 2),
        ("C: left <= 12 OR right <= 3",       lambda c: c["L"] <= 12 or c["R"] <= 3),
        ("D: left <= 12 OR right <= 4",       lambda c: c["L"] <= 12 or c["R"] <= 4),
        ("E: left <= 12 OR right <= 5",       lambda c: c["L"] <= 12 or c["R"] <= 5),
        ("F: left <= 15",                     lambda c: c["L"] <= 15),
        ("G: no limit",                       lambda c: True),
    ]

    print(f"{BOLD}Strategy comparison:{RESET}")
    baseline = set(i for i, c in enumerate(all_candidates) if c["L"] <= 12)

    for name, pred in strategies:
        result_set = set(i for i, c in enumerate(all_candidates) if pred(c))
        added = result_set - baseline
        lost = baseline - result_set
        print(f"  {name:35s}  {len(result_set):3d} removals  "
              f"(+{len(added):2d} / -{len(lost):2d} vs current)")

    # ── Deep dive: what does each strategy add? ──
    for name, pred in strategies[1:6]:  # B through F
        result_set = set(i for i, c in enumerate(all_candidates) if pred(c))
        added = result_set - baseline
        lost = baseline - result_set

        if not added and not lost:
            continue

        print(f"\n{BOLD}{'─' * 60}{RESET}")
        print(f"{BOLD}{name}{RESET}")

        if added:
            print(f"\n  {GREEN}Would ADD:{RESET}")
            for i in sorted(added):
                c = all_candidates[i]
                print(f"    {DIM}{c['dt']}{RESET}  {c['context']}")
                print(f"      pair={YELLOW}{c['pair']}{RESET}  L={c['L']} R={c['R']}")

        if lost:
            print(f"\n  {RED}Would LOSE:{RESET}")
            for i in sorted(lost):
                c = all_candidates[i]
                print(f"    {DIM}{c['dt']}{RESET}  {c['context']}")
                print(f"      pair={YELLOW}{c['pair']}{RESET}  L={c['L']} R={c['R']}")

    # ── Manual quality assessment of valve-kept commas ──
    print(f"\n{BOLD}{'─' * 60}{RESET}")
    print(f"{BOLD}Manual Assessment: All 17 valve-kept commas{RESET}")
    print(f"{DIM}Mark each as: ✓ (should keep) or ✗ (should remove){RESET}\n")

    valve_kept = [c for c in all_candidates if c["L"] > 12]
    for c in valve_kept:
        pair = c["pair"]
        # Heuristic assessment
        # Word splits (compound words being broken) → should remove
        # Clause breaks (new subject/verb) → should keep
        r = c["R"]
        context = c["context"]
        assessment = ""

        # Very short right side → almost always a word split or sentence-final
        if r <= 1:
            assessment = f"{RED}✗ remove{RESET} (R={r}, near sentence end)"
        elif r <= 3:
            assessment = f"{RED}✗ remove{RESET} (R={r}, very short right segment)"
        elif r >= 10:
            assessment = f"{GREEN}✓ keep{RESET} (R={r}, provides structure in long run)"
        else:
            assessment = f"{YELLOW}? ambiguous{RESET} (R={r})"

        print(f"  {DIM}{c['dt']}{RESET}  {context}")
        print(f"    pair={c['pair']}  L={c['L']} R={r}  → {assessment}")
        print()


if __name__ == "__main__":
    main()
