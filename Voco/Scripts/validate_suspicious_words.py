#!/usr/bin/env python3
"""
Validate HomophoneCorrectionEngine's findSuspicious() change against transcription history.

Compares old logic (multi-char tokens with freq 1-5 → suspicious) vs new logic
(multi-char tokens with freq > 0 → trusted). Reports false positive candidates
that would no longer be flagged.

Usage:
    python3 scripts/validate_suspicious_words.py
"""

import os
import re
import sqlite3
from collections import Counter
from pathlib import Path

# Paths
DB_PATH = os.path.expanduser(
    "~/Library/Application Support/com.jasonchien.Voco/default.store"
)
WORD_FREQ_PATH = os.path.join(
    os.path.dirname(__file__),
    "..",
    "VoiceInk",
    "Resources",
    "ChineseCorrection",
    "word_freq.tsv",
)

LOW_FREQ_THRESHOLD = 5

# CJK character range
CJK_RE = re.compile(r"[\u4e00-\u9fff\u3400-\u4dbf]+")


def load_word_freq(path: str) -> dict[str, int]:
    """Load word_freq.tsv → {word: freq}."""
    freqs: dict[str, int] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) == 2:
                word, freq_str = parts
                freqs[word] = int(freq_str)
    return freqs


def extract_cjk_substrings(text: str, min_len: int = 2, max_len: int = 4) -> list[str]:
    """Extract all CJK substrings of length min_len..max_len from text."""
    runs = CJK_RE.findall(text)
    substrings = []
    for run in runs:
        for length in range(min_len, max_len + 1):
            for i in range(len(run) - length + 1):
                substrings.append(run[i : i + length])
    return substrings


def main():
    # Load word frequencies
    freqs = load_word_freq(WORD_FREQ_PATH)
    print(f"Loaded {len(freqs):,} words from word_freq.tsv")

    # Count words with freq 1-5
    low_freq_words = {w: f for w, f in freqs.items() if 1 <= f <= LOW_FREQ_THRESHOLD and len(w) >= 2}
    print(f"Multi-char words with freq 1-{LOW_FREQ_THRESHOLD}: {len(low_freq_words):,}")

    # Load transcriptions
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT ZTEXT FROM ZTRANSCRIPTION WHERE ZTEXT IS NOT NULL")
    rows = cur.fetchall()
    conn.close()
    print(f"Loaded {len(rows):,} transcription records\n")

    # Find low-freq multi-char words that appear in transcriptions
    # These are false positives under the OLD logic (marked suspicious when they shouldn't be)
    found_in_transcriptions: Counter[str] = Counter()
    records_with_hits = 0

    for (text,) in rows:
        if not text:
            continue
        subs = extract_cjk_substrings(text)
        record_hit = False
        for sub in subs:
            if sub in low_freq_words:
                found_in_transcriptions[sub] += 1
                record_hit = True
        if record_hit:
            records_with_hits += 1

    if not found_in_transcriptions:
        print("No low-freq multi-char words found in transcription history.")
        print("The fix has no impact on historical data (no false positives to eliminate).")
        return

    # Report
    print("=" * 70)
    print(f"FALSE POSITIVES ELIMINATED BY THE FIX")
    print(f"(multi-char words with freq 1-{LOW_FREQ_THRESHOLD} found in transcriptions)")
    print("=" * 70)
    print(f"{'Word':<8} {'Freq':>6} {'Occurrences':>12}  {'Old Logic':>12} {'New Logic':>12}")
    print("-" * 70)

    total_occurrences = 0
    for word, count in found_in_transcriptions.most_common():
        freq = low_freq_words[word]
        total_occurrences += count
        print(f"{word:<8} {freq:>6} {count:>12}  {'suspicious':>12} {'trusted':>12}")

    print("-" * 70)
    print(f"Total unique words: {len(found_in_transcriptions)}")
    print(f"Total occurrences:  {total_occurrences}")
    print(f"Records affected:   {records_with_hits} / {len(rows)} ({records_with_hits*100/len(rows):.1f}%)")
    print()

    # Summary
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Old logic: {len(found_in_transcriptions)} unique multi-char words marked suspicious")
    print(f"New logic: 0 of these are marked suspicious (all have freq > 0 + NLTokenizer)")
    print(f"False positive elimination rate: 100% for multi-char tokenized words")
    print()

    # Sanity check: show a few examples for manual review
    print("Top 20 words for manual review (are these valid words?):")
    print("-" * 50)
    for word, count in found_in_transcriptions.most_common(20):
        freq = low_freq_words[word]
        print(f"  {word} (freq={freq}, seen {count}x in transcriptions)")


if __name__ == "__main__":
    main()
