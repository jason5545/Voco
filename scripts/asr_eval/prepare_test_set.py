#!/usr/bin/env python3
"""
從 Voco SwiftData 資料庫匯出測試音訊集。
挑選 enhancedText != text 的紀錄（LLM 做了修正 → ASR 可能有錯）。

輸出：test_set.json — 每筆包含音訊路徑、原始文字、增強文字、時長、類別標籤。
"""

import json
import os
import re
import sqlite3
import sys
from pathlib import Path
from urllib.parse import unquote, urlparse

DB_PATH = os.path.expanduser(
    "~/Library/Application Support/com.jasonchien.Voco/default.store"
)
OUTPUT_DIR = Path(__file__).parent / "test_data"
OUTPUT_FILE = OUTPUT_DIR / "test_set.json"

# --- 分類規則 ---

# 常見英文技術術語（用於偵測 code-switching）
ENGLISH_TECH_TERMS = re.compile(
    r"\b(?:ChatGPT|GPT|GitHub|repo|Claude\s*Code|Claude|async|await|edge\s*case|"
    r"API|SDK|NPM|pip|Docker|Kubernetes|webhook|endpoint|deploy|push|pull|merge|"
    r"commit|branch|tag|debug|log|proxy|server|client|token|cache|queue|stack|"
    r"bug|fix|patch|release|build|test|CI|CD|PR|MCP|LLM|AI|ML|"
    r"whisper|qwen|model|prompt|context|embedding|inference|"
    r"Swift|Python|TypeScript|JavaScript|Kotlin|React|Next\.?js|Node\.?js|"
    r"Xcode|VSCode|terminal|shell|bash|zsh|"
    r"ONNX|MLX|CoreML|Metal|CUDA|GPU|CPU|RAM|SSD|USB|HDMI|WiFi|Bluetooth|"
    r"Mac|macOS|iOS|Android|Windows|Linux|Ubuntu|"
    r"OK|ID|URL|HTTP|HTTPS|SSH|FTP|DNS|IP|TCP|UDP|JSON|XML|YAML|CSV|"
    r"PDF|HTML|CSS|SQL|NoSQL|SQLite|PostgreSQL|MongoDB|Redis)\b",
    re.IGNORECASE,
)

# Latin 字母序列（含數字）
LATIN_PATTERN = re.compile(r"[A-Za-z][A-Za-z0-9_./-]*")

# CJK 字元
CJK_PATTERN = re.compile(r"[\u4e00-\u9fff\u3400-\u4dbf]")


def classify(text: str, enhanced_text: str, duration: float) -> list[str]:
    """根據內容和時長分類。一筆紀錄可以有多個標籤。"""
    tags = []

    has_cjk = bool(CJK_PATTERN.search(text))
    latin_spans = LATIN_PATTERN.findall(text)
    # 過濾掉純數字
    meaningful_latin = [s for s in latin_spans if re.search(r"[A-Za-z]", s)]
    has_latin = len(meaningful_latin) > 0
    tech_matches = ENGLISH_TECH_TERMS.findall(text + " " + enhanced_text)

    # 內容類別
    if has_cjk and not has_latin:
        tags.append("pure_chinese")
    elif has_cjk and has_latin:
        tags.append("code_switching")
    elif not has_cjk and has_latin:
        tags.append("english_dominant")

    if len(tech_matches) >= 3:
        tags.append("tech_term_heavy")

    # 時長類別
    if duration < 5:
        tags.append("short")
    elif duration < 15:
        tags.append("medium")
    elif duration < 30:
        tags.append("long")
    else:
        tags.append("very_long")

    # 修正類型分析
    if text != enhanced_text:
        # 找出差異（簡易 char-level diff）
        diff_chars = sum(1 for a, b in zip(text, enhanced_text) if a != b)
        diff_chars += abs(len(text) - len(enhanced_text))
        ratio = diff_chars / max(len(text), 1)
        if ratio > 0.3:
            tags.append("heavy_correction")
        elif ratio > 0.1:
            tags.append("moderate_correction")
        else:
            tags.append("light_correction")

    return tags


def file_url_to_path(url: str) -> str | None:
    """將 file:// URL 轉換為本機路徑，並驗證檔案存在。"""
    if not url.startswith("file://"):
        return None
    parsed = urlparse(url)
    path = unquote(parsed.path)
    if os.path.isfile(path):
        return path
    return None


def main():
    if not os.path.isfile(DB_PATH):
        print(f"資料庫不存在：{DB_PATH}", file=sys.stderr)
        sys.exit(1)

    conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row

    # 查詢有音訊 + LLM 做了修正的紀錄
    rows = conn.execute("""
        SELECT
            ZAUDIOFILEURL, ZDURATION, ZTEXT, ZENHANCEDTEXT,
            ZTRANSCRIPTIONMODELNAME, ZTIMESTAMP
        FROM ZTRANSCRIPTION
        WHERE ZAUDIOFILEURL IS NOT NULL AND LENGTH(ZAUDIOFILEURL) > 0
          AND ZENHANCEDTEXT IS NOT NULL AND ZENHANCEDTEXT <> ZTEXT
          AND ZTEXT IS NOT NULL AND LENGTH(ZTEXT) > 0
        ORDER BY ZTIMESTAMP DESC
    """).fetchall()

    conn.close()

    test_cases = []
    skipped = 0

    for row in rows:
        audio_path = file_url_to_path(row["ZAUDIOFILEURL"])
        if not audio_path:
            skipped += 1
            continue

        text = row["ZTEXT"]
        enhanced = row["ZENHANCEDTEXT"]
        duration = row["ZDURATION"] or 0.0
        model = row["ZTRANSCRIPTIONMODELNAME"] or "unknown"

        tags = classify(text, enhanced, duration)

        test_cases.append({
            "audio_path": audio_path,
            "duration": round(duration, 2),
            "asr_text": text,
            "enhanced_text": enhanced,
            "original_model": model,
            "tags": tags,
        })

    # 統計
    tag_counts: dict[str, int] = {}
    for tc in test_cases:
        for tag in tc["tags"]:
            tag_counts[tag] = tag_counts.get(tag, 0) + 1

    print(f"總紀錄數：{len(rows)}")
    print(f"有效（音訊檔存在）：{len(test_cases)}")
    print(f"跳過（音訊不存在）：{skipped}")
    print(f"\n標籤分佈：")
    for tag, count in sorted(tag_counts.items(), key=lambda x: -x[1]):
        print(f"  {tag}: {count}")

    # 從每個類別取樣，建立精簡測試集
    # 目標：~50 筆，涵蓋各類別
    sampled = sample_balanced(test_cases)

    print(f"\n精簡測試集：{len(sampled)} 筆")
    sampled_tags: dict[str, int] = {}
    for tc in sampled:
        for tag in tc["tags"]:
            sampled_tags[tag] = sampled_tags.get(tag, 0) + 1
    for tag, count in sorted(sampled_tags.items(), key=lambda x: -x[1]):
        print(f"  {tag}: {count}")

    # 輸出
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 完整集
    full_output = OUTPUT_DIR / "full_test_set.json"
    with open(full_output, "w", encoding="utf-8") as f:
        json.dump(test_cases, f, ensure_ascii=False, indent=2)
    print(f"\n完整測試集已寫入：{full_output}")

    # 精簡集（用於快速評測）
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(sampled, f, ensure_ascii=False, indent=2)
    print(f"精簡測試集已寫入：{OUTPUT_FILE}")


def sample_balanced(test_cases: list[dict], target: int = 50) -> list[dict]:
    """從各類別平衡取樣。優先取 code-switching 和 heavy_correction。"""

    # 優先類別（這些是我們最想測的）
    priority_tags = [
        "code_switching",
        "tech_term_heavy",
        "heavy_correction",
        "english_dominant",
    ]
    secondary_tags = [
        "pure_chinese",
        "moderate_correction",
    ]

    selected_indices: set[int] = set()

    # 先從優先類別各取 10 筆
    for tag in priority_tags:
        candidates = [
            i for i, tc in enumerate(test_cases)
            if tag in tc["tags"] and i not in selected_indices
        ]
        for idx in candidates[:10]:
            selected_indices.add(idx)

    # 再從次要類別補到 target
    remaining = target - len(selected_indices)
    if remaining > 0:
        for tag in secondary_tags:
            candidates = [
                i for i, tc in enumerate(test_cases)
                if tag in tc["tags"] and i not in selected_indices
            ]
            per_tag = max(remaining // len(secondary_tags), 3)
            for idx in candidates[:per_tag]:
                selected_indices.add(idx)
            remaining = target - len(selected_indices)
            if remaining <= 0:
                break

    # 如果還不夠，隨機補
    if len(selected_indices) < target:
        remaining_indices = [
            i for i in range(len(test_cases)) if i not in selected_indices
        ]
        import random
        random.seed(42)
        random.shuffle(remaining_indices)
        for idx in remaining_indices[: target - len(selected_indices)]:
            selected_indices.add(idx)

    return [test_cases[i] for i in sorted(selected_indices)]


if __name__ == "__main__":
    main()
