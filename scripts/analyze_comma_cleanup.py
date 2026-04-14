#!/usr/bin/env python3
"""
Deep analysis of ZTEXT-guided comma cleanup results.

Categorizes each comma removal and checks for false positives.
"""

import sqlite3
import os
import re
from collections import Counter

# Import rules from validate script
from validate_comma_cleanup import (
    clean, clean_de_comma, clean_le_comma, clean_verb_pronoun_comma,
    clean_fixed_phrases, clean_by_original_comparison,
    is_cjk, BOUNDARY_PUNCTUATION,
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


def find_removed_commas(original_enhanced: str, cleaned: str):
    """Find positions and context of removed commas."""
    removals = []
    oi, ci = 0, 0
    while oi < len(original_enhanced):
        if ci < len(cleaned) and original_enhanced[oi] == cleaned[ci]:
            oi += 1
            ci += 1
        else:
            # This char was removed (should be a comma)
            before = original_enhanced[max(0, oi-3):oi]
            after = original_enhanced[oi+1:oi+4]
            removals.append({
                "pos": oi,
                "before": before,
                "after": after,
                "context": original_enhanced[max(0, oi-6):oi+7],
            })
            oi += 1
    return removals


def classify_removal(before: str, after: str, ztext: str):
    """Classify why a comma was removed."""
    if not before or not after:
        return "edge"

    char_before = before[-1]
    char_after = after[0] if after else ""

    # Check which rule removed it
    if char_before == "的":
        return "的-rule"
    if char_before == "了":
        return "了-rule"

    # Check if it's a verb+pronoun pattern
    pronouns = ["我們", "你們", "他們", "她們", "我", "你", "您", "他", "她", "它"]
    for p in pronouns:
        if after.startswith(p):
            return "verb+pronoun"

    # Fixed phrases
    fixed = [("另外", "一"), ("其中", "一"), ("其他", "一"), ("另外", "也"), ("另外", "還")]
    for b, a in fixed:
        if before.endswith(b) and after.startswith(a):
            return "fixed-phrase"

    # Must be ZTEXT comparison
    return "ztext-compare"


def analyze_ztext_removals(enhanced: str, ztext: str):
    """Isolate only the ZTEXT-comparison removals by running rules step by step."""
    # Run all rules EXCEPT ztext comparison
    after_rules = clean_de_comma(enhanced)
    after_rules = clean_le_comma(after_rules)
    after_rules = clean_verb_pronoun_comma(after_rules)
    after_rules = clean_fixed_phrases(after_rules)

    # Now run ONLY ztext comparison
    after_ztext = clean_by_original_comparison(after_rules, ztext)

    if after_rules == after_ztext:
        return []  # No ztext-specific removals

    # Find what ztext comparison removed
    removals = []
    ri, zi = 0, 0
    chars_rules = list(after_rules)
    chars_ztext = list(after_ztext)

    while ri < len(chars_rules):
        if zi < len(chars_ztext) and chars_rules[ri] == chars_ztext[zi]:
            ri += 1
            zi += 1
        else:
            # Comma removed by ztext comparison
            before_ctx = after_rules[max(0, ri-5):ri]
            after_ctx = after_rules[ri+1:ri+6]
            char_before = chars_rules[ri-1] if ri > 0 else ""
            char_after = chars_rules[ri+1] if ri+1 < len(chars_rules) else ""
            pair = f"{char_before}{char_after}"

            # Check if pair actually in ztext
            in_ztext = pair in ztext

            # Measure CJK run
            cjk_run = 0
            for j in range(ri-1, -1, -1):
                if is_cjk(chars_rules[j]):
                    cjk_run += 1
                elif chars_rules[j] in BOUNDARY_PUNCTUATION:
                    break
                # non-CJK non-punct (like letters/digits) — keep counting

            removals.append({
                "context": f"...{before_ctx}[✕，]{after_ctx}...",
                "pair": pair,
                "in_ztext": in_ztext,
                "cjk_run": cjk_run,
                "char_before": char_before,
                "char_after": char_after,
            })
            ri += 1

    return removals


def main():
    rows = read_records(500)
    print(f"\n{BOLD}=== Deep Analysis of Comma Cleanup (500 records) ==={RESET}\n")

    # Collect all ZTEXT-comparison removals
    all_ztext_removals = []
    removal_pairs = Counter()
    records_with_ztext_removal = 0

    for dt, ztext, zenhanced in rows:
        removals = analyze_ztext_removals(zenhanced, ztext)
        if removals:
            records_with_ztext_removal += 1
            for r in removals:
                all_ztext_removals.append({"dt": dt, **r})
                removal_pairs[r["pair"]] += 1

    print(f"{BOLD}ZTEXT-comparison removals:{RESET}")
    print(f"  Total removals: {len(all_ztext_removals)}")
    print(f"  Records affected: {records_with_ztext_removal}")
    print()

    # Show all removals grouped by quality assessment
    print(f"{BOLD}── All ZTEXT-comparison removals (review for false positives) ──{RESET}\n")

    for r in all_ztext_removals:
        pair_display = f"{r['pair']}"
        run_display = f"run={r['cjk_run']}"
        print(f"  {DIM}{r['dt']}{RESET}  {r['context']}  pair={YELLOW}{pair_display}{RESET}  {run_display}")

    # Show most common pairs
    print(f"\n{BOLD}── Most common removed pairs ──{RESET}\n")
    for pair, count in removal_pairs.most_common(30):
        print(f"  {pair}: {count}x")

    # Potential false positive analysis
    print(f"\n{BOLD}── Potential false positives (common bigrams that might be coincidental) ──{RESET}\n")

    # A false positive happens when pair exists in ztext but at a DIFFERENT location
    # (coincidental match). Check by looking at whether the pair appears multiple times
    # in the ztext or if the surrounding context differs.
    suspicious = []
    for r in all_ztext_removals:
        dt, pair = r["dt"], r["pair"]
        # Find the corresponding record
        for _, ztext, zenhanced in rows:
            removals_check = analyze_ztext_removals(zenhanced, ztext)
            for rc in removals_check:
                if rc["pair"] == pair and rc["context"] == r["context"]:
                    # Check how many times this pair appears in ztext
                    count_in_ztext = ztext.count(pair)
                    if count_in_ztext > 1:
                        suspicious.append({
                            **r,
                            "pair_count_in_ztext": count_in_ztext,
                        })
                    break
            break

    if suspicious:
        print(f"  Found {len(suspicious)} removals where pair appears >1x in ZTEXT (possible coincidental match):")
        for s in suspicious:
            print(f"    {DIM}{s['dt']}{RESET}  {s['context']}  pair={YELLOW}{s['pair']}{RESET} (appears {s['pair_count_in_ztext']}x in ZTEXT)")
    else:
        print(f"  {GREEN}No obvious false positives found.{RESET}")

    # Safety valve analysis
    print(f"\n{BOLD}── Commas KEPT by safety valve (CJK run > 12) ──{RESET}\n")

    kept_by_valve = []
    for dt, ztext, zenhanced in rows:
        after_rules = clean_de_comma(zenhanced)
        after_rules = clean_le_comma(after_rules)
        after_rules = clean_verb_pronoun_comma(after_rules)
        after_rules = clean_fixed_phrases(after_rules)

        chars = list(after_rules)
        cjk_run = 0
        for i, ch in enumerate(chars):
            if ch == "，" and i >= 1 and i + 1 < len(chars) and is_cjk(chars[i-1]) and is_cjk(chars[i+1]):
                pair = chars[i-1] + chars[i+1]
                if pair in ztext and cjk_run > 12:
                    before_ctx = after_rules[max(0, i-5):i]
                    after_ctx = after_rules[i+1:i+6]
                    kept_by_valve.append({
                        "dt": dt,
                        "context": f"...{before_ctx}[，]{after_ctx}...",
                        "pair": pair,
                        "cjk_run": cjk_run,
                    })
            if is_cjk(ch):
                cjk_run += 1
            elif ch in BOUNDARY_PUNCTUATION:
                cjk_run = 0

    if kept_by_valve:
        for k in kept_by_valve:
            print(f"  {DIM}{k['dt']}{RESET}  {k['context']}  pair={k['pair']}  run={k['cjk_run']}")
    else:
        print(f"  {DIM}None — safety valve never triggered.{RESET}")

    # Test different thresholds
    print(f"\n{BOLD}── Threshold sensitivity analysis ──{RESET}\n")
    for threshold in [8, 10, 12, 15, 20, 999]:
        count = 0
        for dt, ztext, zenhanced in rows:
            after_rules = clean_de_comma(zenhanced)
            after_rules = clean_le_comma(after_rules)
            after_rules = clean_verb_pronoun_comma(after_rules)
            after_rules = clean_fixed_phrases(after_rules)

            chars = list(after_rules)
            removals = 0
            cjk_run = 0
            for i, ch in enumerate(chars):
                if ch == "，" and i >= 1 and i + 1 < len(chars) and is_cjk(chars[i-1]) and is_cjk(chars[i+1]):
                    pair = chars[i-1] + chars[i+1]
                    if pair in ztext and cjk_run <= threshold:
                        removals += 1
                if is_cjk(ch):
                    cjk_run += 1
                elif ch in BOUNDARY_PUNCTUATION:
                    cjk_run = 0
            count += removals
        label = "no limit" if threshold == 999 else f"<= {threshold}"
        print(f"  Threshold {label:>10}: {count} removals")


if __name__ == "__main__":
    main()
