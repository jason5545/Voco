#!/usr/bin/env python3
"""
Generate pinyin and word frequency data files for Voco's homophone correction engine.

Outputs (to VoiceInk/Resources/ChineseCorrection/):
  1. char_pinyin.json  — char → [pinyin with tone] mapping
  2. pinyin_chars.json  — toneless pinyin → [chars] mapping
  3. word_freq.tsv      — Traditional Chinese word frequency (tab-separated)

Dependencies: pypinyin, opencc-python-reimplemented
"""

import json
import os
import re
import sys
import urllib.request
from collections import defaultdict

try:
    from pypinyin import pinyin, Style
    import opencc
except ImportError as e:
    print(f"Missing dependency: {e}")
    print("Install with: pip3 install pypinyin opencc-python-reimplemented")
    sys.exit(1)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "VoiceInk", "Resources", "ChineseCorrection")
CACHE_DIR = os.path.join(SCRIPT_DIR, ".cache")

JIEBA_DICT_URL = "https://raw.githubusercontent.com/fxsjy/jieba/master/extra_dict/dict.txt.big"


def ensure_dirs():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(CACHE_DIR, exist_ok=True)


def download_jieba_dict():
    """Download jieba's big dictionary if not cached."""
    cached = os.path.join(CACHE_DIR, "dict.txt.big")
    if os.path.exists(cached):
        print(f"  Using cached jieba dict: {cached}")
        return cached
    print(f"  Downloading jieba dict.txt.big ...")
    urllib.request.urlretrieve(JIEBA_DICT_URL, cached)
    print(f"  Saved to {cached}")
    return cached


# ---------------------------------------------------------------------------
# Step 1: char_pinyin.json — character → pinyin list
# ---------------------------------------------------------------------------
def generate_char_pinyin():
    """Use pypinyin to build char→[pinyin] for CJK Unified Ideographs."""
    print("[1/3] Generating char_pinyin.json ...")
    char_pinyin = {}

    # CJK Unified Ideographs: U+4E00 to U+9FFF
    for code in range(0x4E00, 0x9FFF + 1):
        char = chr(code)
        readings = pinyin(char, style=Style.TONE3, heteronym=True)
        if readings and readings[0]:
            # Filter out empty strings and deduplicate
            pys = list(dict.fromkeys(r for r in readings[0] if r))
            if pys:
                char_pinyin[char] = pys

    # CJK Extension A: U+3400 to U+4DBF (fewer chars, but covers some used ones)
    for code in range(0x3400, 0x4DBF + 1):
        char = chr(code)
        readings = pinyin(char, style=Style.TONE3, heteronym=True)
        if readings and readings[0]:
            pys = list(dict.fromkeys(r for r in readings[0] if r))
            if pys:
                char_pinyin[char] = pys

    out_path = os.path.join(OUTPUT_DIR, "char_pinyin.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(char_pinyin, f, ensure_ascii=False, separators=(",", ":"))
    print(f"  {len(char_pinyin)} characters → {out_path}")
    return char_pinyin


# ---------------------------------------------------------------------------
# Step 2: pinyin_chars.json — toneless pinyin → [chars]
# ---------------------------------------------------------------------------
def strip_tone(py: str) -> str:
    """Remove trailing tone number: 'bian4' → 'bian'."""
    return re.sub(r"\d+$", "", py)


def generate_pinyin_chars(char_pinyin: dict):
    """Invert char_pinyin to toneless-pinyin → [chars]."""
    print("[2/3] Generating pinyin_chars.json ...")
    pinyin_chars = defaultdict(list)

    for char, readings in char_pinyin.items():
        seen_toneless = set()
        for r in readings:
            toneless = strip_tone(r)
            if toneless and toneless not in seen_toneless:
                seen_toneless.add(toneless)
                pinyin_chars[toneless].append(char)

    # Sort chars in each group for determinism
    result = {k: sorted(v) for k, v in sorted(pinyin_chars.items())}

    out_path = os.path.join(OUTPUT_DIR, "pinyin_chars.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, separators=(",", ":"))
    print(f"  {len(result)} pinyin groups → {out_path}")
    return result


# ---------------------------------------------------------------------------
# Step 3: word_freq.tsv — Traditional Chinese word frequency
# ---------------------------------------------------------------------------

# Taiwan-specific words to add/boost
TAIWAN_EXTRA_WORDS = {
    "捷運": 50000,
    "悠遊卡": 30000,
    "健保": 40000,
    "程式碼": 60000,
    "程式": 80000,
    "辨識": 70000,
    "語音辨識": 50000,
    "轉錄": 40000,
    "設定": 60000,
    "預設": 40000,
    "偵測": 40000,
    "存取": 30000,
    "網路": 60000,
    "品質": 40000,
    "伺服器": 50000,
    "連線": 40000,
    "檔案": 60000,
    "資料夾": 40000,
    "應用程式": 50000,
    "視窗": 40000,
    "螢幕": 40000,
    "韌體": 20000,
    "列印": 30000,
    "硬碟": 30000,
    "記憶體": 30000,
    "處理器": 30000,
    "點擊": 30000,
    "游標": 20000,
    "機器學習": 30000,
    "深度學習": 30000,
    "人工智慧": 40000,
    "演算法": 30000,
    "大語言模型": 30000,
    "逗號": 30000,
    "句號": 30000,
    "問號": 20000,
    "驚嘆號": 20000,
    "分號": 15000,
    "冒號": 15000,
    "引號": 15000,
    "括號": 15000,
    "區公所": 25000,
    "北投": 30000,
    "漸凍人": 20000,
    "配送": 30000,
    "運算": 40000,
    "雲端運算": 25000,
    "模擬飛行": 20000,
    "額度": 25000,
    "推送": 25000,
    "日誌": 25000,
    "專案": 40000,
    "單指": 15000,
}


def generate_word_freq(char_pinyin: dict):
    """
    Convert jieba dict.txt.big to Traditional Chinese word freq.
    Format: word<TAB>freq
    """
    print("[3/3] Generating word_freq.tsv ...")

    jieba_path = download_jieba_dict()
    converter = opencc.OpenCC("s2twp")

    # Read jieba dict: word freq pos
    word_freq = defaultdict(int)
    with open(jieba_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                word = parts[0]
                try:
                    freq = int(parts[1])
                except ValueError:
                    continue
                # Convert to Traditional Chinese (Taiwan phrases)
                tw_word = converter.convert(word)
                word_freq[tw_word] += freq

    # Add Taiwan-specific words (use max to not downgrade existing entries)
    for word, freq in TAIWAN_EXTRA_WORDS.items():
        word_freq[word] = max(word_freq[word], freq)

    # Also add individual characters from char_pinyin with a base frequency
    # This ensures single-char lookups always work
    for char in char_pinyin:
        if char not in word_freq:
            word_freq[char] = 1

    # Sort by frequency descending for human readability
    sorted_words = sorted(word_freq.items(), key=lambda x: -x[1])

    out_path = os.path.join(OUTPUT_DIR, "word_freq.tsv")
    with open(out_path, "w", encoding="utf-8") as f:
        for word, freq in sorted_words:
            f.write(f"{word}\t{freq}\n")

    print(f"  {len(sorted_words)} words → {out_path}")
    return word_freq


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
def validate(char_pinyin, pinyin_chars, word_freq):
    """Spot-check known confusion pairs."""
    print("\n[Validation]")
    test_cases = [
        ("城市", "程式", "cheng"),
        ("邊視", "辨識", "bian"),
        ("雨停", "語音", "yu"),
    ]
    all_ok = True
    for wrong, correct, expected_py in test_cases:
        # Check that wrong and correct chars share a pinyin group
        for i, (wc, cc) in enumerate(zip(wrong, correct)):
            if wc == cc:
                continue
            w_pys = [strip_tone(p) for p in char_pinyin.get(wc, [])]
            c_pys = [strip_tone(p) for p in char_pinyin.get(cc, [])]
            shared = set(w_pys) & set(c_pys)
            if shared:
                print(f"  OK: '{wc}'↔'{cc}' share pinyin {shared}")
            else:
                print(f"  WARN: '{wc}' pinyins={w_pys}, '{cc}' pinyins={c_pys}, no overlap!")
                all_ok = False

        # Check correct word exists in word_freq
        if correct in word_freq:
            print(f"  OK: '{correct}' in word_freq (freq={word_freq[correct]})")
        else:
            print(f"  WARN: '{correct}' NOT in word_freq!")
            all_ok = False

    if all_ok:
        print("  All validation checks passed!")
    else:
        print("  Some checks had warnings (may still work with broader matching).")


def main():
    print("=== Voco Pinyin Data Preparation ===\n")
    ensure_dirs()

    char_pinyin = generate_char_pinyin()
    pinyin_chars = generate_pinyin_chars(char_pinyin)
    word_freq = generate_word_freq(char_pinyin)

    validate(char_pinyin, pinyin_chars, word_freq)

    # Print file sizes
    print("\n[Output files]")
    for fname in ["char_pinyin.json", "pinyin_chars.json", "word_freq.tsv"]:
        path = os.path.join(OUTPUT_DIR, fname)
        size = os.path.getsize(path)
        if size > 1024 * 1024:
            print(f"  {fname}: {size / 1024 / 1024:.1f} MB")
        else:
            print(f"  {fname}: {size / 1024:.0f} KB")

    print("\nDone! Files are in VoiceInk/Resources/ChineseCorrection/")


if __name__ == "__main__":
    main()
