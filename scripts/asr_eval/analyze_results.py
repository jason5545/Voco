#!/usr/bin/env python3
"""
分析 ASR 模型評測結果。
比較各模型的轉錄品質，以 LLM 增強文字為參考基準。
"""

import json
import re
from collections import defaultdict
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
RESULTS_FILE = SCRIPT_DIR / "results" / "results.json"
TEST_SET_FILE = SCRIPT_DIR / "test_data" / "test_set.json"


def normalize_text(text: str) -> str:
    """正規化文字以便比較（移除空白和標點差異）。"""
    text = text.lower().strip()
    # 移除多餘空白
    text = re.sub(r"\s+", " ", text)
    # 移除常見標點
    text = re.sub(r"[，。！？、；：""''（）【】《》…—\-,\.!\?;:\"'\(\)\[\]]+", "", text)
    return text


def char_error_rate(ref: str, hyp: str) -> float:
    """計算字元錯誤率（CER），用 edit distance。"""
    ref_chars = list(normalize_text(ref))
    hyp_chars = list(normalize_text(hyp))

    if not ref_chars:
        return 0.0 if not hyp_chars else 1.0

    # Wagner-Fischer edit distance
    m, n = len(ref_chars), len(hyp_chars)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if ref_chars[i - 1] == hyp_chars[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,      # deletion
                dp[i][j - 1] + 1,      # insertion
                dp[i - 1][j - 1] + cost  # substitution
            )
    return dp[m][n] / m


def check_english_terms(ref: str, hyp: str) -> dict:
    """檢查英文技術術語是否被正確辨識。"""
    # 從參考文字中提取英文片段
    ref_terms = set(re.findall(r"[A-Za-z][A-Za-z0-9._-]+", ref))
    if not ref_terms:
        return {"total": 0, "correct": 0, "missed": []}

    hyp_lower = hyp.lower()
    correct = 0
    missed = []
    for term in ref_terms:
        if term.lower() in hyp_lower:
            correct += 1
        else:
            missed.append(term)
    return {"total": len(ref_terms), "correct": correct, "missed": missed}


def is_simplified_chinese(text: str) -> bool:
    """粗略檢查是否包含簡體字（非繁體）。"""
    simplified_markers = set("这个来说们对还点里没问题为发时着么过让的能")
    chars = set(text)
    return bool(chars & simplified_markers)


def main():
    with open(RESULTS_FILE, encoding="utf-8") as f:
        results = json.load(f)

    with open(TEST_SET_FILE, encoding="utf-8") as f:
        test_set = json.load(f)

    # 建立 audio → test_case 映射
    tc_map = {}
    for tc in test_set:
        name = Path(tc["audio_path"]).name
        tc_map[name] = tc

    # 整理成 model → [{result, test_case}]
    model_data = defaultdict(list)
    for r in results:
        tc = tc_map.get(r["audio"])
        if tc:
            model_data[r["model"]].append({"result": r, "tc": tc})

    # --- 全域指標 ---
    print("=" * 80)
    print("ASR 模型評測結果摘要")
    print("=" * 80)
    print(f"測試集：{len(test_set)} 筆音訊\n")

    # 表頭
    header = f"{'模型':<25} {'平均CER':>8} {'英文正確率':>10} {'avg RTF':>9} {'簡體率':>6}"
    print(header)
    print("-" * 65)

    model_stats = {}
    for model in sorted(model_data.keys()):
        items = model_data[model]
        cers = []
        term_correct = 0
        term_total = 0
        rtfs = []
        simplified_count = 0

        for item in items:
            r = item["result"]
            tc = item["tc"]
            # 以 enhanced_text 為參考（如果有的話，這是 LLM 修正後的版本）
            ref = tc.get("enhanced_text") or tc.get("asr_text", "")
            hyp = r["transcription"]

            cer = char_error_rate(ref, hyp)
            cers.append(cer)
            rtfs.append(r["rtf"])

            terms = check_english_terms(ref, hyp)
            term_correct += terms["correct"]
            term_total += terms["total"]

            if is_simplified_chinese(hyp):
                simplified_count += 1

        avg_cer = sum(cers) / len(cers) if cers else 0
        avg_rtf = sum(rtfs) / len(rtfs) if rtfs else 0
        term_rate = term_correct / term_total if term_total > 0 else 0
        simp_rate = simplified_count / len(items) if items else 0

        model_stats[model] = {
            "avg_cer": avg_cer,
            "term_rate": term_rate,
            "avg_rtf": avg_rtf,
            "simp_rate": simp_rate,
            "cers": cers,
            "items": items,
        }

        print(f"{model:<25} {avg_cer:>7.1%} {term_rate:>9.1%} {avg_rtf:>9.4f} {simp_rate:>5.0%}")

    # --- 按類別分析 ---
    print(f"\n{'=' * 80}")
    print("按類別分析（平均 CER）")
    print("=" * 80)

    categories = ["code_switching", "pure_chinese", "tech_term_heavy", "english_dominant"]
    cat_header = f"{'模型':<25}"
    for cat in categories:
        cat_header += f" {cat[:12]:>12}"
    print(cat_header)
    print("-" * (25 + 13 * len(categories)))

    for model in sorted(model_data.keys()):
        items = model_data[model]
        row = f"{model:<25}"
        for cat in categories:
            cat_items = [item for item in items if cat in item["tc"].get("tags", [])]
            if cat_items:
                cat_cers = []
                for item in cat_items:
                    ref = item["tc"].get("enhanced_text") or item["tc"].get("asr_text", "")
                    hyp = item["result"]["transcription"]
                    cat_cers.append(char_error_rate(ref, hyp))
                avg = sum(cat_cers) / len(cat_cers)
                row += f" {avg:>11.1%}"
            else:
                row += f" {'N/A':>11}"
        print(row)

    # --- 逐筆比較（挑最有代表性的 case） ---
    print(f"\n{'=' * 80}")
    print("代表性案例逐筆比較")
    print("=" * 80)

    # 挑 code-switching 和 tech_term_heavy 的案例
    interesting_audios = set()
    for tc in test_set:
        tags = tc.get("tags", [])
        if "code_switching" in tags or "tech_term_heavy" in tags:
            interesting_audios.add(Path(tc["audio_path"]).name)
    # 取前 10 個
    interesting_audios = sorted(interesting_audios)[:10]

    for audio_name in interesting_audios:
        tc = tc_map.get(audio_name)
        if not tc:
            continue
        ref = tc.get("enhanced_text") or tc.get("asr_text", "")
        print(f"\n--- {audio_name[:20]}... ({tc['duration']:.1f}s) ---")
        print(f"  參考: {ref[:80]}")

        for model in sorted(model_data.keys()):
            items = [i for i in model_data[model] if i["result"]["audio"] == audio_name]
            if items:
                hyp = items[0]["result"]["transcription"]
                cer = char_error_rate(ref, hyp)
                # 標記差異
                marker = "✅" if cer < 0.05 else "⚠️" if cer < 0.15 else "❌"
                print(f"  {marker} {model:<22}: {hyp[:80]}")

    # --- 英文術語細節 ---
    print(f"\n{'=' * 80}")
    print("英文術語辨識細節")
    print("=" * 80)

    for model in sorted(model_data.keys()):
        missed_terms = defaultdict(int)
        for item in model_data[model]:
            ref = item["tc"].get("enhanced_text") or item["tc"].get("asr_text", "")
            hyp = item["result"]["transcription"]
            terms = check_english_terms(ref, hyp)
            for t in terms["missed"]:
                missed_terms[t.lower()] += 1
        if missed_terms:
            top_missed = sorted(missed_terms.items(), key=lambda x: -x[1])[:10]
            print(f"\n{model}:")
            for term, count in top_missed:
                print(f"  {term}: 漏掉 {count} 次")

    # 存分析結果
    analysis_file = SCRIPT_DIR / "results" / "analysis.json"
    analysis = {}
    for model, stats in model_stats.items():
        analysis[model] = {
            "avg_cer": round(stats["avg_cer"], 4),
            "english_term_rate": round(stats["term_rate"], 4),
            "avg_rtf": round(stats["avg_rtf"], 4),
            "simplified_chinese_rate": round(stats["simp_rate"], 4),
        }
    with open(analysis_file, "w", encoding="utf-8") as f:
        json.dump(analysis, f, ensure_ascii=False, indent=2)
    print(f"\n分析結果已存入：{analysis_file}")


if __name__ == "__main__":
    main()
