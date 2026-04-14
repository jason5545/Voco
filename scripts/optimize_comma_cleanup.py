#!/usr/bin/env python3
"""
Optimize ZTEXT-guided comma cleanup: find false positives, test thresholds,
and explore alternative strategies.
"""

import sqlite3
import os
from collections import Counter, defaultdict

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
MAGENTA = "\033[95m"
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
    """Apply all rule-based cleanups (NOT ztext comparison)."""
    result = clean_de_comma(text)
    result = clean_le_comma(result)
    result = clean_verb_pronoun_comma(result)
    result = clean_fixed_phrases(result)
    return result


def find_commas_for_ztext_check(text):
    """Find all commas that could potentially be removed by ZTEXT comparison."""
    chars = list(text)
    candidates = []
    cjk_run = 0

    for i, ch in enumerate(chars):
        if (ch == "，" and i >= 1 and i + 1 < len(chars)
                and is_cjk(chars[i - 1]) and is_cjk(chars[i + 1])):
            # Also compute forward run to next punctuation
            fwd_run = 0
            for j in range(i + 1, len(chars)):
                if is_cjk(chars[j]):
                    fwd_run += 1
                elif chars[j] in BOUNDARY_PUNCTUATION:
                    break

            candidates.append({
                "pos": i,
                "pair": chars[i - 1] + chars[i + 1],
                "char_before": chars[i - 1],
                "char_after": chars[i + 1],
                "cjk_run_left": cjk_run,
                "cjk_run_right": fwd_run,
                "context": text[max(0, i - 6):i + 7],
            })

        if is_cjk(ch):
            cjk_run += 1
        elif ch in BOUNDARY_PUNCTUATION:
            cjk_run = 0

    return candidates


def check_coincidental_match(pair, enhanced_text, ztext, comma_pos):
    """Check if a pair match in ztext is coincidental (at a different location).

    Strategy: find all occurrences of the pair in both texts and see
    if the matching position in ztext is plausibly the same phrase.
    """
    # Find all occurrences in ztext
    ztext_positions = []
    start = 0
    while True:
        idx = ztext.find(pair, start)
        if idx == -1:
            break
        ztext_positions.append(idx)
        start = idx + 1

    if not ztext_positions:
        return {"match": False}

    # The pair should appear at roughly the same relative position
    # in both texts. Use a wide window since LLM can change text length.
    enhanced_ratio = comma_pos / max(len(enhanced_text), 1)

    best_dist = float("inf")
    best_pos = -1
    for zpos in ztext_positions:
        ztext_ratio = zpos / max(len(ztext), 1)
        dist = abs(enhanced_ratio - ztext_ratio)
        if dist < best_dist:
            best_dist = dist
            best_pos = zpos

    # If closest match is within 30% of relative position, it's plausible
    is_coincidental = best_dist > 0.3
    return {
        "match": True,
        "ztext_positions": ztext_positions,
        "count_in_ztext": len(ztext_positions),
        "best_position_distance": best_dist,
        "is_coincidental": is_coincidental,
    }


def main():
    rows = read_records(500)
    print(f"\n{BOLD}=== Comma Cleanup Optimization Analysis (500 records) ==={RESET}\n")

    # ── Analysis 1: Coincidental bigram matches ──
    print(f"{BOLD}{'─' * 60}{RESET}")
    print(f"{BOLD}1. Coincidental Bigram Match Analysis{RESET}")
    print(f"{DIM}   Check if pairs match at wrong positions in ZTEXT{RESET}\n")

    coincidental_matches = []
    all_candidates = []

    for dt, ztext, zenhanced in rows:
        after_rules = apply_rule_based(zenhanced)
        candidates = find_commas_for_ztext_check(after_rules)

        for c in candidates:
            if c["pair"] in ztext:
                match_info = check_coincidental_match(
                    c["pair"], after_rules, ztext, c["pos"]
                )
                c["dt"] = dt
                c["ztext"] = ztext
                c["match_info"] = match_info
                all_candidates.append(c)

                if match_info["is_coincidental"]:
                    coincidental_matches.append(c)

    if coincidental_matches:
        print(f"  {RED}Found {len(coincidental_matches)} potential coincidental matches:{RESET}")
        for c in coincidental_matches:
            mi = c["match_info"]
            print(f"    {DIM}{c['dt']}{RESET}  {c['context']}")
            print(f"      pair={YELLOW}{c['pair']}{RESET}  "
                  f"position_dist={mi['best_position_distance']:.2f}  "
                  f"count_in_ztext={mi['count_in_ztext']}")
    else:
        print(f"  {GREEN}No coincidental matches found in {len(all_candidates)} candidates.{RESET}")

    # ── Analysis 2: Both-direction distance analysis ──
    print(f"\n{BOLD}{'─' * 60}{RESET}")
    print(f"{BOLD}2. Bi-directional Distance Analysis{RESET}")
    print(f"{DIM}   Check left AND right CJK run for each removal candidate{RESET}\n")

    short_both_sides = []  # Low run on both sides → definitely remove
    long_one_side = []     # One side is long → might be needed

    for c in all_candidates:
        left, right = c["cjk_run_left"], c["cjk_run_right"]
        min_side = min(left, right)
        max_side = max(left, right)
        c["min_run"] = min_side
        c["max_run"] = max_side

    # Distribution of min(left, right)
    min_run_dist = Counter(c["min_run"] for c in all_candidates)
    print(f"  Distribution of min(left_run, right_run):")
    for run in sorted(min_run_dist.keys()):
        bar = "█" * min_run_dist[run]
        print(f"    {run:3d}: {min_run_dist[run]:3d} {bar}")

    # ── Analysis 3: Threshold comparison with different strategies ──
    print(f"\n{BOLD}{'─' * 60}{RESET}")
    print(f"{BOLD}3. Strategy Comparison{RESET}")
    print(f"{DIM}   Compare current vs alternative approaches{RESET}\n")

    strategies = {
        "current (left_run <= 12)": lambda c: c["cjk_run_left"] <= 12,
        "left_run <= 15": lambda c: c["cjk_run_left"] <= 15,
        "left_run <= 20": lambda c: c["cjk_run_left"] <= 20,
        "min(L,R) <= 8": lambda c: c["min_run"] <= 8,
        "min(L,R) <= 10": lambda c: c["min_run"] <= 10,
        "min(L,R) <= 12": lambda c: c["min_run"] <= 12,
        "max(L,R) <= 15": lambda c: c["max_run"] <= 15,
        "no limit": lambda c: True,
    }

    for name, predicate in strategies.items():
        removals = [c for c in all_candidates if predicate(c)]
        print(f"  {name:25s}: {len(removals):3d} removals")

    # ── Analysis 4: Cases where valve WRONGLY keeps a comma ──
    print(f"\n{BOLD}{'─' * 60}{RESET}")
    print(f"{BOLD}4. Safety Valve Analysis: Kept Commas That Might Be Wrong{RESET}")
    print(f"{DIM}   Commas kept by current threshold (left_run > 12) but pair is in ZTEXT{RESET}\n")

    kept_by_valve = [c for c in all_candidates if c["cjk_run_left"] > 12]

    if kept_by_valve:
        for c in kept_by_valve:
            # Try to assess if this looks like a word split vs clause break
            pair = c["pair"]
            left, right = c["cjk_run_left"], c["cjk_run_right"]
            assessment = "word-split?" if right >= 3 else "clause-break?"
            print(f"  {DIM}{c['dt']}{RESET}  {c['context']}")
            print(f"    pair={YELLOW}{pair}{RESET}  L={left} R={right}  → {assessment}")
    else:
        print(f"  {DIM}None{RESET}")

    # ── Analysis 5: Would min(L,R) be better? ──
    print(f"\n{BOLD}{'─' * 60}{RESET}")
    print(f"{BOLD}5. min(L,R) Strategy Deep Dive{RESET}")
    print(f"{DIM}   Show what min(L,R) <= 10 would add/remove vs current{RESET}\n")

    current_set = set(i for i, c in enumerate(all_candidates) if c["cjk_run_left"] <= 12)
    minlr_set = set(i for i, c in enumerate(all_candidates) if c["min_run"] <= 10)

    added = minlr_set - current_set
    lost = current_set - minlr_set

    print(f"  Current (left <= 12): {len(current_set)} removals")
    print(f"  min(L,R) <= 10:      {len(minlr_set)} removals")
    print(f"  Would ADD: {len(added)}, Would LOSE: {len(lost)}")

    if added:
        print(f"\n  {GREEN}Would ADD these removals:{RESET}")
        for i in sorted(added):
            c = all_candidates[i]
            print(f"    {DIM}{c['dt']}{RESET}  {c['context']}  "
                  f"pair={YELLOW}{c['pair']}{RESET}  L={c['cjk_run_left']} R={c['cjk_run_right']}")

    if lost:
        print(f"\n  {RED}Would LOSE these removals:{RESET}")
        for i in sorted(lost):
            c = all_candidates[i]
            print(f"    {DIM}{c['dt']}{RESET}  {c['context']}  "
                  f"pair={YELLOW}{c['pair']}{RESET}  L={c['cjk_run_left']} R={c['cjk_run_right']}")

    # ── Analysis 6: Overall quality assessment ──
    print(f"\n{BOLD}{'─' * 60}{RESET}")
    print(f"{BOLD}6. Overall Quality Summary{RESET}\n")

    total_commas_in_enhanced = 0
    total_commas_after_rules = 0
    total_ztext_removals_current = len(current_set)

    for dt, ztext, zenhanced in rows:
        total_commas_in_enhanced += zenhanced.count("，")
        after_rules = apply_rule_based(zenhanced)
        total_commas_after_rules += after_rules.count("，")

    print(f"  Total commas in LLM output:    {total_commas_in_enhanced}")
    print(f"  After rule-based cleanup:       {total_commas_after_rules}")
    print(f"  ZTEXT comparison removals:      {total_ztext_removals_current}")
    print(f"  Final comma count:              {total_commas_after_rules - total_ztext_removals_current}")
    print(f"  Coincidental match risk:        {len(coincidental_matches)}/{len(all_candidates)} ({len(coincidental_matches)/max(len(all_candidates),1)*100:.1f}%)")
    print()


if __name__ == "__main__":
    main()
