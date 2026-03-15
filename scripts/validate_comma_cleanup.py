#!/usr/bin/env python3
"""
Validate PostLLMCommaCleanup rules against real Voco transcription data.

Reads the Voco SwiftData database and applies the same comma cleanup rules
as PostLLMCommaCleanup.swift, then shows a comparison to verify correctness.

Usage:
    python3 scripts/validate_comma_cleanup.py
    python3 scripts/validate_comma_cleanup.py --limit 50
    python3 scripts/validate_comma_cleanup.py --all   # show all records, not just changed ones
"""

import sqlite3
import argparse
import os
from datetime import datetime, timezone, timedelta

# ============================================================
# Rules (mirror of PostLLMCommaCleanup.swift)
# ============================================================

# After 的+comma, keep comma if text after comma starts with any of these
DE_COMMA_KEEP_PREFIXES = [
    # Multi-char first
    "我們", "你們", "他們", "她們",
    "大家", "自己",
    "但是", "不過", "可是", "所以", "因為",
    "而且", "然後", "如果", "雖然", "於是", "因此",
    "只要", "只有", "只是", "即使", "儘管",
    "除非", "否則", "既然", "無論", "不管",
    "不是", "不要", "不能", "不會", "不可",
    # Single-char
    "我", "你", "您", "他", "她", "它",
    "但", "卻", "再", "只",
]

# Response words that form interjections with 的 (好的/對的/是的/行的)
INTERJECTION_CHARS = set("好對是行")

# Punctuation that indicates a sentence/clause boundary
BOUNDARY_PUNCTUATION = set("，。？！、；：…,.?!:;\n")

# Verb markers in the segment after comma that indicate sentence-final 的
VERB_MARKERS = ["為", "成"]

# Prepositions/complements that always take pronoun objects
# 把他帶走, 被他發現, 替他想, 拖累到他
SAFE_PREPOSITIONS = set("把被替到")

# Pronouns that can be verb/preposition objects (multi-char first for prefix matching)
OBJECT_PRONOUNS = ["我們", "你們", "他們", "她們", "我", "你", "您", "他", "她", "它"]

# Sentence-final particles — comma after these + pronoun is a real clause break
SENTENCE_FINAL_PARTICLES = set("了的嗎呢吧啊呀哦嘛囉")

# Complement patterns after 了 that should never have a comma
LE_COMPLEMENT_PREFIXES = ["一下", "一些", "一點", "一番", "一會", "一陣"]

# Characters that form a word with 了 (了解, 了結, etc.)
# Note: 然/卻 excluded — too easily confused with 然後/卻是 (conjunctions)
LE_WORD_SECOND_CHARS = set("解結斷事得無")


def is_cjk(ch: str) -> bool:
    cp = ord(ch)
    return (0x4E00 <= cp <= 0x9FFF) or (0x3400 <= cp <= 0x4DBF)


def clean_de_comma(text: str) -> str:
    """Remove comma after 的 when it incorrectly splits a modifier-noun phrase."""
    chars = list(text)
    result = []
    i = 0
    while i < len(chars):
        if (
            chars[i] == "的"
            and i + 1 < len(chars) and chars[i + 1] == "，"
            and i + 2 < len(chars) and is_cjk(chars[i + 2])
        ):
            after_comma = "".join(chars[i + 2:])

            # Exception 1: text after comma starts with a clause-starting word
            starts_new_clause = any(after_comma.startswith(p) for p in DE_COMMA_KEEP_PREFIXES)

            # Exception 2: interjection pattern (好的，/對的，/是的，)
            is_interjection = False
            if i >= 1 and chars[i - 1] in INTERJECTION_CHARS:
                if i == 1:
                    is_interjection = True
                elif i >= 2 and (chars[i - 2] in BOUNDARY_PUNCTUATION or chars[i - 2] == " "):
                    is_interjection = True

            # Exception 3: sentence-final 的 after verb phrase
            # If segment after comma contains verb markers (為/成), 的 is likely nominalizing
            is_sentence_final = False
            next_punct_idx = None
            for j, ch in enumerate(after_comma):
                if ch in BOUNDARY_PUNCTUATION:
                    next_punct_idx = j
                    break
            segment = after_comma[:next_punct_idx] if next_punct_idx is not None else after_comma
            if len(segment) >= 2 and any(m in segment for m in VERB_MARKERS):
                is_sentence_final = True

            should_keep = starts_new_clause or is_interjection or is_sentence_final

            if not should_keep:
                result.append("的")
                i += 2  # skip 的 and ，
                continue
        result.append(chars[i])
        i += 1
    return "".join(result)


def clean_le_comma(text: str) -> str:
    """Remove comma after 了 for known complement/word-split patterns."""
    chars = list(text)
    result = []
    i = 0
    while i < len(chars):
        if (
            chars[i] == "了"
            and i + 1 < len(chars) and chars[i + 1] == "，"
            and i + 2 < len(chars)
        ):
            after_comma = "".join(chars[i + 2:])
            is_complement = any(after_comma.startswith(p) for p in LE_COMPLEMENT_PREFIXES)
            is_word_split = chars[i + 2] in LE_WORD_SECOND_CHARS
            if is_complement or is_word_split:
                result.append("了")
                i += 2
                continue
        result.append(chars[i])
        i += 1
    return "".join(result)


def clean_verb_pronoun_comma(text: str) -> str:
    """Remove comma between verb/preposition and its pronoun object.

    Rule 1: Safe preposition (把/被/替/到) + ，+ pronoun → always remove
    Rule 2: non-particle CJK + ，+ pronoun + 的 → remove (pronoun is possessive object)
    """
    chars = list(text)
    result = []
    i = 0
    while i < len(chars):
        if (
            chars[i] == "，"
            and i >= 1
            and i + 1 < len(chars)
        ):
            before = chars[i - 1]
            after = "".join(chars[i + 1:])

            matched_pronoun = None
            for p in OBJECT_PRONOUNS:
                if after.startswith(p):
                    matched_pronoun = p
                    break

            if matched_pronoun:
                after_pronoun = after[len(matched_pronoun):]

                # Rule 1: safe preposition before comma
                if before in SAFE_PREPOSITIONS:
                    i += 1  # skip comma
                    continue

                # Rule 2: pronoun+的, before is not a sentence-final particle
                if (
                    after_pronoun.startswith("的")
                    and before not in SENTENCE_FINAL_PARTICLES
                    and is_cjk(before)
                ):
                    i += 1  # skip comma
                    continue

        result.append(chars[i])
        i += 1
    return "".join(result)


FIXED_PHRASES = [
    ("另外", "一"), ("其中", "一"), ("其他", "一"),
    ("另外", "也"), ("另外", "還"),
]


def clean_fixed_phrases(text: str) -> str:
    """Remove commas that split known fixed phrases."""
    result = text
    for before, after in FIXED_PHRASES:
        result = result.replace(f"{before}，{after}", f"{before}{after}")
    return result


def clean_by_original_comparison(text: str, original_text: str) -> str:
    """Remove commas the LLM inserted between chars that were adjacent in ASR original.

    Checks the original text directly (with punctuation intact). If the pair
    charBefore+charAfter exists as a substring in the original, the LLM wrongly
    split adjacent chars → remove the comma.

    Safety valve: if the CJK run since last punctuation > 12 chars, keep the comma.
    """
    if not original_text:
        return text

    chars = list(text)
    result = []
    cjk_run = 0

    for i, ch in enumerate(chars):
        if (
            ch == "，"
            and i >= 1
            and i + 1 < len(chars)
            and is_cjk(chars[i - 1])
            and is_cjk(chars[i + 1])
        ):
            pair = chars[i - 1] + chars[i + 1]
            if pair in original_text and cjk_run <= 12:
                continue  # remove comma

        result.append(ch)
        if is_cjk(ch):
            cjk_run += 1
        elif ch in BOUNDARY_PUNCTUATION:
            cjk_run = 0

    return "".join(result)


def clean(text: str, original_text: str | None = None) -> str:
    """Apply all comma cleanup rules."""
    if len(text) < 3:
        return text
    result = text
    result = clean_de_comma(result)
    result = clean_le_comma(result)
    result = clean_verb_pronoun_comma(result)
    result = clean_fixed_phrases(result)
    if original_text:
        result = clean_by_original_comparison(result, original_text)
    return result


# ============================================================
# Highlighting
# ============================================================

RED = "\033[91m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
DIM = "\033[2m"
BOLD = "\033[1m"
RESET = "\033[0m"


def highlight_diff(original: str, cleaned: str) -> str:
    """Show the cleaned text with removed commas highlighted."""
    # Find positions where commas were removed
    result = []
    oi, ci = 0, 0
    while oi < len(original):
        if ci < len(cleaned) and original[oi] == cleaned[ci]:
            result.append(original[oi])
            oi += 1
            ci += 1
        else:
            # This character was removed (should be a comma)
            result.append(f"{RED}✕{original[oi]}{RESET}")
            oi += 1
    return "".join(result)


def highlight_de_le_commas(text: str) -> str:
    """Highlight all 的，and 了，patterns in the text for visibility."""
    return text.replace("的，", f"{YELLOW}的，{RESET}").replace("了，", f"{YELLOW}了，{RESET}")


# ============================================================
# Database
# ============================================================

DB_PATH = os.path.expanduser(
    "~/Library/Application Support/com.jasonchien.Voco/default.store"
)
CORE_DATA_EPOCH = 978307200  # 2001-01-01 00:00:00 UTC


def read_records(limit: int = 100):
    """Read transcription records that have enhanced text with 的，or 了，."""
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


# ============================================================
# Main
# ============================================================


def main():
    parser = argparse.ArgumentParser(description="Validate PostLLMCommaCleanup rules")
    parser.add_argument("--limit", type=int, default=200, help="Max records to read")
    parser.add_argument("--all", action="store_true", help="Show all records, not just changed ones")
    args = parser.parse_args()

    if not os.path.exists(DB_PATH):
        print(f"Database not found: {DB_PATH}")
        return

    rows = read_records(args.limit)
    print(f"\n{BOLD}Loaded {len(rows)} records from database{RESET}\n")

    changed_count = 0
    correct_removal = 0
    wrong_removal = 0
    missed_count = 0

    for dt, ztext, zenhanced in rows:
        cleaned = clean(zenhanced, original_text=ztext)
        has_change = cleaned != zenhanced

        # Check if original ZTEXT can tell us if the change is correct
        # If ZTEXT doesn't have the comma at that position, our removal is correct
        if has_change:
            changed_count += 1
            print(f"{CYAN}── {dt} ──{RESET}")
            print(f"  {DIM}原文 (ZTEXT):{RESET}  {ztext}")
            print(f"  {DIM}增強 (LLM): {RESET}  {highlight_de_le_commas(zenhanced)}")
            print(f"  {DIM}清理 (ours):{RESET}  {GREEN}{cleaned}{RESET}")
            print(f"  {DIM}差異:       {RESET}  {highlight_diff(zenhanced, cleaned)}")
            print()
        elif args.all:
            # Show records that weren't changed (for completeness)
            has_pattern = "的，" in zenhanced or "了，" in zenhanced
            if has_pattern:
                print(f"{DIM}── {dt} ──{RESET}")
                print(f"  {DIM}增強 (LLM): {RESET}  {highlight_de_le_commas(zenhanced)}")
                print(f"  {DIM}→ 保留 (exception matched){RESET}")
                print()

    # Also check for patterns we might be MISSING
    print(f"\n{BOLD}{'=' * 60}{RESET}")
    print(f"{BOLD}Summary{RESET}")
    print(f"  Records scanned:  {len(rows)}")
    print(f"  Commas removed:   {changed_count}")

    # Show records where 的，or 了，still exist after cleanup (kept by exceptions)
    kept_count = 0
    for dt, ztext, zenhanced in rows:
        cleaned = clean(zenhanced, original_text=ztext)
        if "的，" in cleaned or "了，" in cleaned:
            kept_count += 1

    print(f"  Commas kept (exception): {kept_count}")
    print()

    # Detailed analysis of kept commas
    if kept_count > 0:
        print(f"{BOLD}Kept commas (verify these are correct):{RESET}")
        for dt, ztext, zenhanced in rows:
            cleaned = clean(zenhanced, original_text=ztext)
            for particle in ["的，", "了，"]:
                idx = 0
                while True:
                    idx = cleaned.find(particle, idx)
                    if idx == -1:
                        break
                    # Extract context: 5 chars before and after
                    start = max(0, idx - 5)
                    end = min(len(cleaned), idx + len(particle) + 5)
                    context = cleaned[start:end]
                    print(f"  {DIM}{dt}{RESET}  ...{YELLOW}{context}{RESET}...")
                    idx += 1
        print()


if __name__ == "__main__":
    main()
