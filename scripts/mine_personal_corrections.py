#!/usr/bin/env python3
"""從 SwiftData 資料庫挖掘個人化校正 pattern

掃描 Transcription 記錄中 original_text vs enhanced_text 的差異，
找出重複出現的替換 pattern，特別是 1字→2字 的「音節擴展」類型。
支援拼音相似度 filter，分離語音錯誤 vs 語意改寫。

用法:
  python3 scripts/mine_personal_corrections.py [--db PATH] [--min-count N]
  python3 scripts/mine_personal_corrections.py --phonetic-only

預設 DB 路徑: ~/Library/Application Support/com.jasonchien.Voco/default.store
"""

import argparse
import difflib
import json
import re
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path

from pypinyin import pinyin, Style


DEFAULT_DB = Path.home() / "Library/Application Support/com.jasonchien.Voco/default.store"

# 拼音相似度門檻：≥ 此值視為語音類替換
PHONETIC_THRESHOLD = 0.5


def connect_db(db_path: Path) -> sqlite3.Connection:
    if not db_path.exists():
        raise FileNotFoundError(f"資料庫不存在: {db_path}")
    return sqlite3.connect(str(db_path))


def fetch_transcriptions(conn: sqlite3.Connection) -> list[dict]:
    """取得有 LLM 增強結果的轉錄記錄"""
    cursor = conn.execute("""
        SELECT
            ZTIMESTAMP,
            ZTEXT,
            ZENHANCEDTEXT
        FROM ZTRANSCRIPTION
        WHERE ZTEXT IS NOT NULL
          AND ZENHANCEDTEXT IS NOT NULL
          AND ZTEXT != ZENHANCEDTEXT
        ORDER BY ZTIMESTAMP DESC
    """)

    rows = []
    for ts, text, enhanced in cursor:
        if text and enhanced and text.strip() != enhanced.strip():
            rows.append({
                "timestamp": ts,
                "original": text.strip(),
                "enhanced": enhanced.strip(),
            })
    return rows


def is_cjk(ch: str) -> bool:
    v = ord(ch)
    return 0x4E00 <= v <= 0x9FFF or 0x3400 <= v <= 0x4DBF


def get_pinyin_str(text: str) -> str:
    """取得 CJK 文字的無聲調拼音字串（空格分隔）"""
    cjk_chars = [c for c in text if is_cjk(c)]
    if not cjk_chars:
        return ""
    py_list = pinyin("".join(cjk_chars), style=Style.NORMAL, errors="ignore")
    return " ".join(p[0] for p in py_list)


def pinyin_similarity(s1: str, s2: str) -> float:
    """計算兩段 CJK 文字的拼音相似度 (0.0–1.0)"""
    py1 = get_pinyin_str(s1)
    py2 = get_pinyin_str(s2)
    if not py1 or not py2:
        return 0.0
    return difflib.SequenceMatcher(None, py1, py2).ratio()


def extract_replacements(original: str, enhanced: str) -> list[tuple[str, str]]:
    """用 SequenceMatcher 找出 original → enhanced 的替換配對

    回傳: [(original_segment, enhanced_segment), ...]
    """
    replacements = []
    matcher = difflib.SequenceMatcher(None, original, enhanced)

    for op, i1, i2, j1, j2 in matcher.get_opcodes():
        if op == "replace":
            orig_seg = original[i1:i2]
            enh_seg = enhanced[j1:j2]
            # 只看 CJK 字元的替換
            if any(is_cjk(c) for c in orig_seg) and any(is_cjk(c) for c in enh_seg):
                replacements.append((orig_seg, enh_seg))

    return replacements


def analyze_patterns(rows: list[dict]) -> dict:
    """分析所有替換 pattern"""

    # 所有替換 pattern 的計數
    all_replacements: Counter = Counter()
    # 1字→2字 的替換（音節擴展類型）
    expand_1to2: Counter = Counter()
    # 2字→2字 的替換（同音替換類型）
    replace_2to2: Counter = Counter()
    # 1字→N字 (N>2)
    expand_1toN: Counter = Counter()
    # N字→M字 (其他)
    other: Counter = Counter()

    # 帶上下文的範例
    examples: dict[tuple[str, str], list[str]] = defaultdict(list)
    # 拼音相似度快取
    similarities: dict[tuple[str, str], float] = {}

    for row in rows:
        repls = extract_replacements(row["original"], row["enhanced"])
        for orig, enh in repls:
            pair = (orig, enh)
            all_replacements[pair] += 1

            # 計算拼音相似度（只算一次）
            if pair not in similarities:
                similarities[pair] = pinyin_similarity(orig, enh)

            # 記錄範例（最多 3 個）
            if len(examples[pair]) < 3:
                examples[pair].append(f"{row['original']} → {row['enhanced']}")

            # 分類
            orig_cjk = [c for c in orig if is_cjk(c)]
            enh_cjk = [c for c in enh if is_cjk(c)]

            if len(orig_cjk) == 1 and len(enh_cjk) == 2:
                expand_1to2[pair] += 1
            elif len(orig_cjk) == 2 and len(enh_cjk) == 2:
                replace_2to2[pair] += 1
            elif len(orig_cjk) == 1 and len(enh_cjk) > 2:
                expand_1toN[pair] += 1
            else:
                other[pair] += 1

    return {
        "all": all_replacements,
        "expand_1to2": expand_1to2,
        "replace_2to2": replace_2to2,
        "expand_1toN": expand_1toN,
        "other": other,
        "examples": examples,
        "similarities": similarities,
        "total_rows": len(rows),
    }


def print_section(title: str, counter: Counter, analysis: dict,
                  min_count: int, phonetic_only: bool, show_all: bool = False):
    """印出某類替換的報告區段"""
    sims = analysis["similarities"]
    examples = analysis["examples"]

    # 按語音/語意分類
    phonetic = [(p, c) for p, c in counter.most_common() if sims.get(p, 0) >= PHONETIC_THRESHOLD]
    semantic = [(p, c) for p, c in counter.most_common() if sims.get(p, 0) < PHONETIC_THRESHOLD]

    print(f"\n{'=' * 78}")
    print(title)
    print(f"{'=' * 78}")
    print(f"總計: {len(counter)} 種 pattern, 共 {sum(counter.values())} 次")
    print(f"  語音類 (拼音相似 ≥{PHONETIC_THRESHOLD}): {len(phonetic)} 種, {sum(c for _, c in phonetic)} 次")
    print(f"  語意類 (拼音不相似): {len(semantic)} 種, {sum(c for _, c in semantic)} 次")

    # 語音類
    if phonetic:
        freq_phonetic = [(p, c) for p, c in phonetic if c >= min_count]
        print(f"\n── 語音類 (出現 ≥{min_count} 次) ──")
        if freq_phonetic:
            print(f"{'原文':>8} → {'修正':<8} {'次數':>4}  {'拼音':30} {'sim':>5}")
            print("-" * 78)
            for (orig, enh), cnt in freq_phonetic:
                py_orig = get_pinyin_str(orig)
                py_enh = get_pinyin_str(enh)
                sim = sims.get((orig, enh), 0)
                print(f"{orig:>8} → {enh:<8} {cnt:>4}  {py_orig}→{py_enh:20} {sim:>5.2f}")
        else:
            print(f"  (沒有出現 ≥{min_count} 次的語音類 pattern)")

        if show_all and phonetic:
            print(f"\n── 語音類 (全部) ──")
            print(f"{'原文':>8} → {'修正':<8} {'次數':>4}  {'拼音':30} {'sim':>5}")
            print("-" * 78)
            for (orig, enh), cnt in phonetic[:50]:
                py_orig = get_pinyin_str(orig)
                py_enh = get_pinyin_str(enh)
                sim = sims.get((orig, enh), 0)
                print(f"{orig:>8} → {enh:<8} {cnt:>4}  {py_orig}→{py_enh:20} {sim:>5.2f}")
            if len(phonetic) > 50:
                print(f"  ... 還有 {len(phonetic) - 50} 個")

    # 語意類（除非 phonetic_only）
    if not phonetic_only and semantic:
        freq_semantic = [(p, c) for p, c in semantic if c >= min_count]
        if freq_semantic:
            print(f"\n── 語意類 (出現 ≥{min_count} 次) ──")
            print(f"{'原文':>8} → {'修正':<8} {'次數':>4}  {'拼音':30} {'sim':>5}")
            print("-" * 78)
            for (orig, enh), cnt in freq_semantic[:20]:
                py_orig = get_pinyin_str(orig)
                py_enh = get_pinyin_str(enh)
                sim = sims.get((orig, enh), 0)
                print(f"{orig:>8} → {enh:<8} {cnt:>4}  {py_orig}→{py_enh:20} {sim:>5.2f}")


def print_report(analysis: dict, min_count: int, phonetic_only: bool):
    """印出分析報告"""
    sims = analysis["similarities"]

    print("=" * 78)
    print("個人化校正 Pattern 挖掘報告")
    if phonetic_only:
        print(f"（僅顯示語音類：拼音相似度 ≥ {PHONETIC_THRESHOLD}）")
    print("=" * 78)
    print(f"\n總記錄數（有 LLM 修正的）: {analysis['total_rows']}")
    print(f"不同替換 pattern 總數: {len(analysis['all'])}")

    # 全局語音/語意統計
    all_phonetic = sum(1 for p in analysis["all"] if sims.get(p, 0) >= PHONETIC_THRESHOLD)
    all_semantic = len(analysis["all"]) - all_phonetic
    print(f"  語音類: {all_phonetic} 種 ({all_phonetic*100//max(len(analysis['all']),1)}%)")
    print(f"  語意類: {all_semantic} 種 ({all_semantic*100//max(len(analysis['all']),1)}%)")

    # 各類報告
    print_section(
        "1字→2字 替換 (音節擴展/壓縮類型)",
        analysis["expand_1to2"], analysis, min_count, phonetic_only, show_all=True,
    )

    print_section(
        "2字→2字 替換 (同音/近音替換類型)",
        analysis["replace_2to2"], analysis, min_count, phonetic_only, show_all=True,
    )

    print_section(
        "1字→N字 替換 (N>2)",
        analysis["expand_1toN"], analysis, min_count, phonetic_only,
    )

    print_section(
        "其他替換 (N字→M字)",
        analysis["other"], analysis, min_count, phonetic_only,
    )

    # 統計摘要
    print(f"\n{'=' * 78}")
    print("統計摘要")
    print(f"{'=' * 78}")
    for label, key in [("1字→2字", "expand_1to2"), ("2字→2字", "replace_2to2"),
                       ("1字→N字", "expand_1toN"), ("其他", "other")]:
        counter = analysis[key]
        ph = sum(1 for p in counter if sims.get(p, 0) >= PHONETIC_THRESHOLD)
        se = len(counter) - ph
        print(f"  {label}: {len(counter)} 種 ({ph} 語音 / {se} 語意), 共 {sum(counter.values())} 次")


def export_rules(analysis: dict, min_count: int, output_path: Path, phonetic_only: bool):
    """匯出高信心規則為 JSON"""
    sims = analysis["similarities"]
    rules = []

    for key in ["expand_1to2", "replace_2to2"]:
        for (orig, enh), cnt in analysis[key].most_common():
            if cnt < min_count:
                continue
            sim = sims.get((orig, enh), 0)
            if phonetic_only and sim < PHONETIC_THRESHOLD:
                continue
            rules.append({
                "original": orig,
                "corrected": enh,
                "count": cnt,
                "type": key,
                "pinyin_original": get_pinyin_str(orig),
                "pinyin_corrected": get_pinyin_str(enh),
                "pinyin_similarity": round(sim, 3),
                "examples": analysis["examples"][(orig, enh)],
            })

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(rules, f, ensure_ascii=False, indent=2)

    print(f"\n已匯出 {len(rules)} 條規則到: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="挖掘個人化校正 pattern")
    parser.add_argument("--db", type=Path, default=DEFAULT_DB,
                        help="SwiftData 資料庫路徑")
    parser.add_argument("--min-count", type=int, default=2,
                        help="最低出現次數 (預設: 2)")
    parser.add_argument("--export", type=Path, default=None,
                        help="匯出規則 JSON 路徑 (可選)")
    parser.add_argument("--phonetic-only", action="store_true",
                        help="只顯示語音類替換（拼音相似度 ≥ 0.5）")
    args = parser.parse_args()

    try:
        conn = connect_db(args.db)
    except FileNotFoundError as e:
        print(f"錯誤: {e}")
        print(f"\n這台機器沒有 Voco 的資料庫。")
        print(f"請在有資料的機器上執行此腳本。")
        print(f"\n用法:")
        print(f"  python3 scripts/mine_personal_corrections.py")
        print(f"  python3 scripts/mine_personal_corrections.py --phonetic-only")
        print(f"  python3 scripts/mine_personal_corrections.py --min-count 3 --export rules.json")
        return

    print(f"資料庫: {args.db}")
    rows = fetch_transcriptions(conn)
    conn.close()

    if not rows:
        print("沒有找到有 LLM 修正的記錄。")
        return

    analysis = analyze_patterns(rows)
    print_report(analysis, args.min_count, args.phonetic_only)

    if args.export:
        export_rules(analysis, args.min_count, args.export, args.phonetic_only)


if __name__ == "__main__":
    main()
