#!/usr/bin/env python3
"""
驗證 BERT MLM scoring 效果。

模擬 Voco BERTScorer 的邏輯：
  1. 把 position 遮罩為 [MASK]
  2. 取該位置的 logits
  3. candidateLogit - originalLogit = score

用法：
    python3 scripts/test_bert_scoring.py
"""

import time
from pathlib import Path

import numpy as np
import torch
from transformers import BertForMaskedLM, BertTokenizer


def load_model():
    print("載入 bert-base-chinese...")
    tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
    model = BertForMaskedLM.from_pretrained("bert-base-chinese")
    model.eval()
    return tokenizer, model


def score_replacement(
    tokenizer, model, text: str, position: int,
    original_char: str, candidate_char: str
) -> float:
    """模擬 BERTScorer.score() — 遮罩 position，回傳 candLogit - origLogit"""
    chars = list(text)
    chars[position] = "[MASK]"
    masked_text = "".join(chars)

    # tokenize（BERT char-level：每字一 token）
    inputs = tokenizer(masked_text, return_tensors="pt")
    masked_index = position + 1  # +1 for [CLS]

    with torch.no_grad():
        logits = model(**inputs).logits[0, masked_index]  # [vocab_size]

    orig_id = tokenizer.convert_tokens_to_ids(original_char)
    cand_id = tokenizer.convert_tokens_to_ids(candidate_char)

    return (logits[cand_id] - logits[orig_id]).item()


def score_word_replacement(
    tokenizer, model, text: str, word_offset: int,
    original_word: str, candidate_word: str
) -> float:
    """模擬 BERTScorer.scoreWordReplacement() — 多字加總"""
    total = 0.0
    for i, (o, c) in enumerate(zip(original_word, candidate_word)):
        if o != c:
            total += score_replacement(
                tokenizer, model, text, word_offset + i, o, c
            )
    return total


def fmt(score: float) -> str:
    """格式化 score，附帶判斷"""
    prob_ratio = np.exp(score)
    return f"{score:+.2f} (≈{prob_ratio:.1f}x)"


def run_test(tokenizer, model, text, word_offset, original, candidate, description):
    score = score_word_replacement(
        tokenizer, model, text, word_offset, original, candidate
    )
    print(f"  {description}")
    print(f"    「{text}」")
    print(f"    {original} → {candidate}  score={fmt(score)}")

    # threshold 判斷
    homophone_threshold = 2.0
    nasal_threshold = 2.5
    threshold = nasal_threshold if is_nasal_pair(original, candidate) else homophone_threshold
    verdict = "✅ 會修正" if score > threshold else "❌ 不修正"
    print(f"    threshold={threshold}  → {verdict}")
    print()
    return score


def is_nasal_pair(a, b):
    """簡單判斷是否為鼻音對"""
    nasal_pairs = {
        ("銀", "螢"), ("螢", "銀"),
        ("幕", "幕"),
        ("民", "明"), ("明", "民"),
        ("品", "瓶"), ("瓶", "品"),
    }
    for ca, cb in zip(a, b):
        if (ca, cb) in nasal_pairs:
            return True
    return False


def main():
    tokenizer, model = load_model()
    print()

    # ══════════════════════════════════════════════════
    print("=" * 60)
    print("測試 1：上下文消歧義 — 銀幕 vs 螢幕")
    print("=" * 60)
    print()

    # 電影語境 → 銀幕 正確
    run_test(tokenizer, model,
             "我在銀幕上看電影", 2, "銀幕", "螢幕",
             "電影語境：銀幕 → 螢幕？（不該修正）")

    # 電腦語境 → 螢幕 正確，銀幕 不正確
    run_test(tokenizer, model,
             "電腦銀幕很亮", 2, "銀幕", "螢幕",
             "電腦語境：銀幕 → 螢幕？（應該修正）")

    # ══════════════════════════════════════════════════
    print("=" * 60)
    print("測試 2：同音字 — 頻率 vs 語境")
    print("=" * 60)
    print()

    run_test(tokenizer, model,
             "這個消息很振奮", 5, "振", "震",
             "振奮人心：振 → 震？（不該修正）")

    run_test(tokenizer, model,
             "地震震動了整座城市", 2, "震", "振",
             "地震：震 → 振？（不該修正）")

    # ══════════════════════════════════════════════════
    print("=" * 60)
    print("測試 3：常見語音辨識錯誤")
    print("=" * 60)
    print()

    run_test(tokenizer, model,
             "我們要注意安全", 5, "全", "泉",
             "安全 → 安泉？（不該修正）")

    run_test(tokenizer, model,
             "他的工作非常辛苦", 6, "辛", "新",
             "辛苦 → 新苦？（不該修正）")

    run_test(tokenizer, model,
             "他的成績非常優異", 6, "優", "幽",
             "優異 → 幽異？（不該修正）")

    # ══════════════════════════════════════════════════
    print("=" * 60)
    print("測試 4：真正的語音辨識錯誤（應該修正）")
    print("=" * 60)
    print()

    run_test(tokenizer, model,
             "今天氣溫很底", 5, "底", "低",
             "很底 → 很低？（應該修正）")

    run_test(tokenizer, model,
             "他在銀行存款", 2, "銀", "銀",
             "銀行：銀 → 銀（相同字，score 應為 0）")

    run_test(tokenizer, model,
             "請問一下園來是這樣", 4, "園", "原",
             "園來 → 原來？（應該修正）")

    run_test(tokenizer, model,
             "他非常刻褲", 4, "褲", "苦",
             "刻褲 → 刻苦？（應該修正）")

    # ══════════════════════════════════════════════════
    print("=" * 60)
    print("測試 5：效能基準")
    print("=" * 60)
    print()

    text = "我今天去看了一部很好看的電影然後回家吃飯"
    n_runs = 20
    start = time.perf_counter()
    for _ in range(n_runs):
        score_replacement(tokenizer, model, text, 5, "了", "瞭")
    elapsed = time.perf_counter() - start
    avg_ms = elapsed / n_runs * 1000
    print(f"  單次 forward pass 平均耗時：{avg_ms:.1f} ms（{n_runs} 次平均）")
    print(f"  （Core ML + Neural Engine 在實際裝置上會更快）")
    print()

    # ══════════════════════════════════════════════════
    print("=" * 60)
    print("測試 6：比較頻率 scoring vs BERT scoring")
    print("=" * 60)
    print()

    comparisons = [
        ("我在銀幕上看電影", 2, "銀幕", "螢幕",
         "銀幕→螢幕 (電影語境)", 10, 50000),
        ("電腦銀幕很亮", 2, "銀幕", "螢幕",
         "銀幕→螢幕 (電腦語境)", 10, 50000),
        ("今天氣溫很底", 5, "底", "低",
         "底→低", 5000, 30000),
        ("他非常刻褲", 4, "褲", "苦",
         "褲→苦", 200, 15000),
    ]

    print(f"  {'案例':<25} {'頻率score':>10} {'BERT score':>12} {'頻率判斷':>8} {'BERT判斷':>8}")
    print(f"  {'-'*25} {'-'*10} {'-'*12} {'-'*8} {'-'*8}")

    for text, offset, orig, cand, desc, orig_freq, cand_freq in comparisons:
        freq_score = np.log(cand_freq) - np.log(orig_freq + 1)
        bert_score = score_word_replacement(
            tokenizer, model, text, offset, orig, cand
        )
        freq_verdict = "修正" if freq_score > 2.5 else "保留"
        bert_verdict = "修正" if bert_score > 2.0 else "保留"
        print(f"  {desc:<25} {freq_score:>+10.2f} {bert_score:>+12.2f} {freq_verdict:>8} {bert_verdict:>8}")

    print()


if __name__ == "__main__":
    main()
