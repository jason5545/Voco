#!/usr/bin/env python3
"""SyllableExpansionEngine 邏輯模擬與測試

用 Python 重現 Swift 引擎的核心邏輯，分析：
1. 哪些 merge form 對應到真實的單字？（引擎的搜索空間）
2. 在什麼樣的句子中，引擎會正確觸發？
3. 為什麼在 3327 筆真實資料中 TP=0？

不需要音訊或 Whisper，純邏輯分析。
"""

import json
import math
import re
from collections import defaultdict
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "VoiceInk" / "Resources" / "ChineseCorrection"

# ── Load data ──────────────────────────────────────────────

def load_data():
    with open(DATA_DIR / "char_pinyin.json") as f:
        char_pinyin = json.load(f)  # char → [pinyin_with_tone, ...]

    word_freq = {}
    with open(DATA_DIR / "word_freq.tsv") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) == 2:
                word_freq[parts[0]] = int(parts[1])

    bigram_freq = {}
    with open(DATA_DIR / "bigram_freq.tsv") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) == 2:
                bigram_freq[parts[0]] = int(parts[1])

    return char_pinyin, word_freq, bigram_freq


char_pinyin, word_freq, bigram_freq = load_data()

# ── Pinyin helpers ─────────────────────────────────────────

INITIALS = [
    "zh", "ch", "sh",
    "b", "p", "m", "f",
    "d", "t", "n", "l",
    "g", "k", "h",
    "j", "q", "x",
    "r", "z", "c", "s",
    "y", "w",
]

TONE_MAP = str.maketrans(
    "āáǎàēéěèīíǐìōóǒòūúǔùǖǘǚǜü",
    "aaaaeeeeiiiioooouuuuvvvvv",
)

def strip_tone(py: str) -> str:
    # Handle both numbered tones (ying4) and diacritics (yīng)
    result = py.translate(TONE_MAP)
    # Remove trailing digit (numbered tone)
    if result and result[-1].isdigit():
        result = result[:-1]
    return result

def pinyin_initial(py: str) -> str:
    for ini in INITIALS:
        if py.startswith(ini):
            return ini
    return ""

def pinyin_final(py: str) -> str:
    ini = pinyin_initial(py)
    return py[len(ini):]

def merge_pinyin(p1: str, p2: str) -> str:
    """模擬快速語音的音節壓縮: initial(p1) + final(p2)"""
    return pinyin_initial(p1) + pinyin_final(p2)

def toneless_pinyin(char: str) -> list[str]:
    """取得字的無聲調拼音列表（只用主要讀音）"""
    readings = char_pinyin.get(char, [])
    if not readings:
        return []
    return [strip_tone(readings[0])]

def edit_distance(a: str, b: str) -> int:
    m, n = len(a), len(b)
    prev = list(range(n + 1))
    for i in range(1, m + 1):
        curr = [i] + [0] * n
        for j in range(1, n + 1):
            if a[i-1] == b[j-1]:
                curr[j] = prev[j-1]
            else:
                curr[j] = min(prev[j-1], prev[j], curr[j-1]) + 1
        prev = curr
    return prev[n]

# ── Engine constants (matching Swift) ──────────────────────

BIGRAM_WEIGHT = 0.5
INTERNAL_BIGRAM_WEIGHT = 0.3
DISTANCE_PENALTY = 2.0
MIN_CONTEXT_IMPROVEMENT = 3.0
MIN_TOTAL_SCORE = 7.0
SUSPICIOUS_BIGRAM_THRESHOLD = 50
HIGH_FREQ_CHAR_THRESHOLD = 5000
MAX_MERGE_EDIT_DISTANCE = 0
MIN_FREQ_RATIO = 10.0

SKIP_CHARS = set("的了嗎呢吧啊哦喔嗯呀是在有和也都就不我你他她它們這那個把被讓會能可要得地著過到從與及或而但因為所以如")

# ── Build merge index ──────────────────────────────────────

def build_merge_index():
    """Build: merge_pinyin → [(word, freq)]"""
    index = defaultdict(list)
    for word, freq in word_freq.items():
        chars = list(word)
        if len(chars) != 2:
            continue
        r1 = char_pinyin.get(chars[0], [])
        r2 = char_pinyin.get(chars[1], [])
        if not r1 or not r2:
            continue
        p1 = strip_tone(r1[0])
        p2 = strip_tone(r2[0])
        m = merge_pinyin(p1, p2)
        if m:
            index[m].append((word, freq))
    return dict(index)

print("Building merge index...")
merge_index = build_merge_index()
print(f"  {len(merge_index)} merge forms covering {sum(len(v) for v in merge_index.values())} words")

# ── Analysis 1: 哪些單字可以被展開？──────────────────────

def find_expandable_chars():
    """
    找出所有理論上可以被展開的單字。
    條件：
    1. 字的拼音在 merge_index 中有對應的 2-char word
    2. 字的詞頻 ≤ highFreqCharThreshold (5000)
    3. 候選詞的 freq / 字的 freq ≥ minFreqRatio (10)
    """
    expandable = []

    for char, readings in char_pinyin.items():
        if char in SKIP_CHARS:
            continue
        char_freq = word_freq.get(char, 0)
        if char_freq > HIGH_FREQ_CHAR_THRESHOLD:
            continue

        py = strip_tone(readings[0])
        candidates = merge_index.get(py, [])

        valid_candidates = []
        for word, wfreq in candidates:
            if char_freq > 0 and wfreq / char_freq < MIN_FREQ_RATIO:
                continue
            valid_candidates.append((word, wfreq))

        if valid_candidates:
            # Sort by freq descending
            valid_candidates.sort(key=lambda x: -x[1])
            expandable.append({
                "char": char,
                "char_freq": char_freq,
                "pinyin": py,
                "candidates": valid_candidates[:5],  # top 5
            })

    expandable.sort(key=lambda x: -x["candidates"][0][1])
    return expandable

print("\n" + "=" * 70)
print("分析 1: 理論上可展開的字 (merge index hit + freq ratio 通過)")
print("=" * 70)
expandable = find_expandable_chars()
print(f"共 {len(expandable)} 個字有展開候選")
print(f"\nTop 30 (按最佳候選詞頻排序):")
print(f"{'字':>4} {'freq':>6} {'拼音':>8}  →  {'候選詞':<8} {'候選freq':>8}  {'ratio':>6}")
print("-" * 60)
for item in expandable[:30]:
    c = item["candidates"][0]
    ratio = c[1] / item["char_freq"] if item["char_freq"] > 0 else float("inf")
    print(f"{item['char']:>4} {item['char_freq']:>6} {item['pinyin']:>8}  →  {c[0]:<8} {c[1]:>8}  {ratio:>6.1f}x")

# ── Analysis 2: 模擬引擎的完整評分流程 ────────────────────

def get_bigram_freq(bigram: str) -> int:
    return bigram_freq.get(bigram, 0)

def is_cjk(ch: str) -> bool:
    v = ord(ch)
    return 0x4E00 <= v <= 0x9FFF or 0x3400 <= v <= 0x4DBF

def detect_suspicious(text: str) -> list[tuple[str, int]]:
    """模擬 detectSuspicious — 簡化版（不用 NLTokenizer，逐字掃描）"""
    chars = list(text)
    suspects = []

    for i, ch in enumerate(chars):
        if not is_cjk(ch):
            continue
        if ch in SKIP_CHARS:
            continue

        ch_freq = word_freq.get(ch, 0)
        if ch_freq > HIGH_FREQ_CHAR_THRESHOLD:
            continue

        # Skip if adjacent to non-CJK
        if i > 0 and not is_cjk(chars[i-1]):
            continue
        if i + 1 < len(chars) and not is_cjk(chars[i+1]):
            continue

        # Neighbor word guard
        if i > 0 and is_cjk(chars[i-1]):
            left_word = chars[i-1] + ch
            if word_freq.get(left_word, 0) > 0:
                continue
        if i + 1 < len(chars) and is_cjk(chars[i+1]):
            right_word = ch + chars[i+1]
            if word_freq.get(right_word, 0) > 0:
                continue

        # Bigram check
        left_bf = get_bigram_freq(chars[i-1] + ch) if i > 0 else 0
        right_bf = get_bigram_freq(ch + chars[i+1]) if i + 1 < len(chars) else 0

        if left_bf <= SUSPICIOUS_BIGRAM_THRESHOLD and right_bf <= SUSPICIOUS_BIGRAM_THRESHOLD:
            suspects.append((ch, i))

    return suspects

def score_candidate(word: str, word_freq_val: int, merge_dist: int,
                    orig_char: str, left_ctx: str | None, right_ctx: str | None) -> float | None:
    """模擬 scoreCandidate"""
    chars = list(word)
    if len(chars) != 2:
        return None

    orig_freq = word_freq.get(orig_char, 0)
    if orig_freq > 0 and word_freq_val / orig_freq < MIN_FREQ_RATIO:
        return None

    # Original context
    orig_left = math.log(get_bigram_freq(left_ctx + orig_char) + 1) if left_ctx else 0
    orig_right = math.log(get_bigram_freq(orig_char + right_ctx) + 1) if right_ctx else 0

    # New context
    new_left = math.log(get_bigram_freq(left_ctx + chars[0]) + 1) if left_ctx else 0
    new_right = math.log(get_bigram_freq(chars[1] + right_ctx) + 1) if right_ctx else 0

    ctx_improvement = BIGRAM_WEIGHT * ((new_left - orig_left) + (new_right - orig_right))
    if ctx_improvement < MIN_CONTEXT_IMPROVEMENT:
        return None

    internal_bf = get_bigram_freq(chars[0] + chars[1])

    score = (math.log(word_freq_val + 1)
             + ctx_improvement
             + INTERNAL_BIGRAM_WEIGHT * math.log(internal_bf + 1)
             - DISTANCE_PENALTY * merge_dist)

    if score < MIN_TOTAL_SCORE:
        return None

    return score

def simulate_engine(text: str) -> list[dict]:
    """完整模擬引擎，回傳所有觸發的修正"""
    suspects = detect_suspicious(text)
    results = []
    chars = list(text)

    for ch, idx in suspects:
        pinyins = toneless_pinyin(ch)
        if not pinyins:
            continue

        left_ctx = chars[idx - 1] if idx > 0 and is_cjk(chars[idx - 1]) else None
        right_ctx = chars[idx + 1] if idx + 1 < len(chars) and is_cjk(chars[idx + 1]) else None

        best_word = None
        best_score = 0

        for py in pinyins:
            for merge_py, words in merge_index.items():
                dist = edit_distance(merge_py, py)
                if dist > MAX_MERGE_EDIT_DISTANCE:
                    continue
                for word, wf in words:
                    s = score_candidate(word, wf, dist, ch, left_ctx, right_ctx)
                    if s is not None and s > best_score:
                        best_score = s
                        best_word = word

        if best_word:
            results.append({
                "char": ch,
                "position": idx,
                "replacement": best_word,
                "score": best_score,
                "left_ctx": left_ctx,
                "right_ctx": right_ctx,
            })

    return results

# ── Analysis 3: 系統性測試 — 用已知壓縮配對構造句子 ───────

print("\n" + "=" * 70)
print("分析 2: 系統性構造測試句子")
print("=" * 70)

def find_compression_pairs():
    """
    找出所有理論上的壓縮配對:
    單字 X 的拼音 == merge(2-char word Y 的兩個字的拼音)

    也就是說，如果 Whisper 把 Y 壓縮成 X，引擎應該要能修回來。
    """
    pairs = []

    for char, readings in char_pinyin.items():
        if char in SKIP_CHARS:
            continue
        char_freq_val = word_freq.get(char, 0)
        if char_freq_val > HIGH_FREQ_CHAR_THRESHOLD:
            continue

        py = strip_tone(readings[0])
        candidates = merge_index.get(py, [])

        for word, wfreq in candidates:
            if char_freq_val > 0 and wfreq / char_freq_val < MIN_FREQ_RATIO:
                continue
            pairs.append((char, char_freq_val, word, wfreq, py))

    pairs.sort(key=lambda x: -x[3])
    return pairs

pairs = find_compression_pairs()
print(f"找到 {len(pairs)} 個理論壓縮配對 (X → Y)")

# 用高頻配對構造測試句子
print(f"\n用 top 50 配對構造測試句子，模擬引擎...")
print(f"{'壓縮字':>6} → {'原詞':<8} {'句子':<30} {'結果':<10} {'分數':>6}")
print("-" * 80)

# 我們需要找到 context 字，使得 bigram 改善足夠大
def find_good_context(target_word: str) -> tuple[str | None, str | None]:
    """為目標詞找到好的左右 context 字"""
    chars = list(target_word)

    # 找左 context: 需要 bigram(left, chars[0]) 高
    best_left = None
    best_left_freq = 0
    for bg, freq in bigram_freq.items():
        if len(bg) == 2 and bg[1] == chars[0] and freq > best_left_freq:
            if is_cjk(bg[0]) and bg[0] not in SKIP_CHARS:
                best_left = bg[0]
                best_left_freq = freq

    # 找右 context: 需要 bigram(chars[1], right) 高
    best_right = None
    best_right_freq = 0
    for bg, freq in bigram_freq.items():
        if len(bg) == 2 and bg[0] == chars[1] and freq > best_right_freq:
            if is_cjk(bg[1]) and bg[1] not in SKIP_CHARS:
                best_right = bg[1]
                best_right_freq = freq

    return best_left, best_right

test_results = {"triggered": [], "not_triggered": []}
seen_words = set()

for char, char_freq_val, word, wfreq, py in pairs[:100]:
    if word in seen_words:
        continue
    seen_words.add(word)

    left_ctx, right_ctx = find_good_context(word)

    # 構造句子：左context + 壓縮字 + 右context
    sentence = ""
    if left_ctx:
        sentence += left_ctx
    sentence += char  # 壓縮後的單字
    if right_ctx:
        sentence += right_ctx

    # 加一些 padding 讓句子更自然
    sentence = "今天" + sentence + "很好"

    results = simulate_engine(sentence)

    triggered = any(r["char"] == char and r["replacement"] == word for r in results)

    if triggered:
        match = [r for r in results if r["char"] == char][0]
        print(f"{char:>6} → {word:<8} {sentence:<30} {'✓ 觸發':>10} {match['score']:>6.1f}")
        test_results["triggered"].append((char, word, sentence, match["score"]))
    else:
        # 分析為什麼沒觸發
        reason = "?"
        suspects = detect_suspicious(sentence)
        suspect_chars = [s[0] for s in suspects]
        if char not in suspect_chars:
            reason = "未被偵測為可疑"
        else:
            reason = "評分未通過"

        if len(test_results["not_triggered"]) < 20:  # 只印前 20 個
            print(f"{char:>6} → {word:<8} {sentence:<30} {'✗ ' + reason:>10}")
        test_results["not_triggered"].append((char, word, sentence, reason))

print(f"\n統計: {len(test_results['triggered'])} 觸發 / {len(test_results['not_triggered'])} 未觸發")

# ── Analysis 4: 深度分析觸發條件 ──────────────────────────

print("\n" + "=" * 70)
print("分析 3: 引擎觸發的必要條件深度分析")
print("=" * 70)

print("\n各個 gate 的過濾效果:")

# Gate 1: highFreqCharThreshold
high_freq_count = sum(1 for c in char_pinyin if word_freq.get(c, 0) > HIGH_FREQ_CHAR_THRESHOLD)
print(f"  1. highFreqCharThreshold (>{HIGH_FREQ_CHAR_THRESHOLD}): 過濾 {high_freq_count}/{len(char_pinyin)} 字")

# Gate 2: merge index hit (distance=0)
has_merge = 0
for char in char_pinyin:
    if char in SKIP_CHARS:
        continue
    readings = char_pinyin[char]
    py = strip_tone(readings[0])
    if py in merge_index:
        has_merge += 1
print(f"  2. merge index 命中 (distance=0): {has_merge} 字有候選")

# Gate 3: freq ratio
passes_ratio = 0
for char in char_pinyin:
    if char in SKIP_CHARS:
        continue
    char_f = word_freq.get(char, 0)
    if char_f > HIGH_FREQ_CHAR_THRESHOLD:
        continue
    readings = char_pinyin[char]
    py = strip_tone(readings[0])
    for word, wf in merge_index.get(py, []):
        if char_f == 0 or wf / char_f >= MIN_FREQ_RATIO:
            passes_ratio += 1
            break
print(f"  3. freq ratio (≥{MIN_FREQ_RATIO}x): {passes_ratio} 字通過")

# Gate 4: neighbor word guard — 這個要看具體句子
print(f"  4. neighbor word guard: 取決於上下文（相鄰字是否構成已知詞）")
print(f"  5. bigram threshold (≤{SUSPICIOUS_BIGRAM_THRESHOLD}): 取決於上下文")
print(f"  6. context improvement (≥{MIN_CONTEXT_IMPROVEMENT}): 取決於上下文")
print(f"  7. total score (≥{MIN_TOTAL_SCORE}): 取決於上下文")

# ── Analysis 5: 真實世界場景分析 ──────────────────────────

print("\n" + "=" * 70)
print("分析 4: 真實世界 Whisper 壓縮錯誤分析")
print("=" * 70)

# 手動列出已知的 Whisper 壓縮案例（來自 xvoice 和經驗）
known_compressions = [
    ("硬", "語音", "yǔ yīn → yìng"),
    ("雲", "語音", "yǔ yīn → yún (xvoice 記錄)"),
    ("盈", "語音", "yǔ yīn → yíng"),
    ("令", "命令", "mìng lìng → lìng (不是壓縮，是丟音節)"),
]

print("\n已知壓縮案例測試:")
for comp_char, orig_word, note in known_compressions:
    char_f = word_freq.get(comp_char, 0)
    word_f = word_freq.get(orig_word, 0)
    readings = char_pinyin.get(comp_char, [])
    py = strip_tone(readings[0]) if readings else "?"

    # 檢查 merge 是否匹配
    word_chars = list(orig_word)
    if len(word_chars) == 2:
        r1 = char_pinyin.get(word_chars[0], [])
        r2 = char_pinyin.get(word_chars[1], [])
        if r1 and r2:
            p1 = strip_tone(r1[0])
            p2 = strip_tone(r2[0])
            merged = merge_pinyin(p1, p2)
            dist = edit_distance(merged, py)
        else:
            merged = "?"
            dist = -1
    else:
        merged = "N/A"
        dist = -1

    print(f"\n  {comp_char}({py}, freq={char_f}) ← {orig_word}(freq={word_f})")
    print(f"  merge({orig_word}) = {merged}, 與 {comp_char}({py}) 的 edit distance = {dist}")
    print(f"  distance ≤ {MAX_MERGE_EDIT_DISTANCE}? {'✓ YES' if dist <= MAX_MERGE_EDIT_DISTANCE else '✗ NO'}")
    if char_f > 0 and word_f > 0:
        ratio = word_f / char_f
        print(f"  freq ratio: {word_f}/{char_f} = {ratio:.1f}x (需要 ≥{MIN_FREQ_RATIO}x: {'✓' if ratio >= MIN_FREQ_RATIO else '✗'})")
    if char_f > HIGH_FREQ_CHAR_THRESHOLD:
        print(f"  ✗ 字頻 {char_f} > {HIGH_FREQ_CHAR_THRESHOLD} — 被 highFreqCharThreshold 過濾")
    print(f"  注: {note}")

# ── Analysis 6: 如果放寬 maxMergeEditDistance=1 會怎樣？──

print("\n" + "=" * 70)
print("分析 5: 放寬 maxMergeEditDistance=1 的影響")
print("=" * 70)

# 重新計算 expandable chars with dist=1
expandable_dist1 = 0
expandable_dist0 = 0
new_pairs = []

for char, readings in char_pinyin.items():
    if char in SKIP_CHARS:
        continue
    char_f = word_freq.get(char, 0)
    if char_f > HIGH_FREQ_CHAR_THRESHOLD:
        continue

    py = strip_tone(readings[0])

    has_dist0 = py in merge_index
    has_dist1 = False

    for merge_py in merge_index:
        if edit_distance(merge_py, py) == 1:
            has_dist1 = True
            # 找高頻候選
            for word, wf in merge_index[merge_py]:
                if char_f == 0 or wf / char_f >= MIN_FREQ_RATIO:
                    if wf > 10000:
                        new_pairs.append((char, char_f, word, wf, py, merge_py))
            break

    if has_dist0:
        expandable_dist0 += 1
    if has_dist1 and not has_dist0:
        expandable_dist1 += 1

print(f"  distance=0: {expandable_dist0} 字有候選")
print(f"  distance=1 (新增): +{expandable_dist1} 字")
print(f"  distance=1 新增的高頻配對 (候選 freq > 10000):")
new_pairs.sort(key=lambda x: -x[3])
for char, cf, word, wf, py, mpy in new_pairs[:20]:
    print(f"    {char}({py}) → {word}(freq={wf})  [merge={mpy}]")

# ── Summary ───────────────────────────────────────────────

print("\n" + "=" * 70)
print("總結")
print("=" * 70)
print(f"""
SyllableExpansionEngine 的核心假設：
  Whisper 會把 2 音節壓成 1 字，壓縮方式 = initial(音1) + final(音2)

現實問題：
  1. 觸發條件非常苛刻 — 字必須同時滿足:
     - 不在 skipChars 中
     - 詞頻 ≤ {HIGH_FREQ_CHAR_THRESHOLD}
     - 左右 bigram 都 ≤ {SUSPICIOUS_BIGRAM_THRESHOLD}
     - 不與相鄰字構成已知詞
     - 候選詞頻/字頻 ≥ {MIN_FREQ_RATIO}x
     - context improvement ≥ {MIN_CONTEXT_IMPROVEMENT}
     - total score ≥ {MIN_TOTAL_SCORE}

  2. 壓縮模型(initial+final)覆蓋率有限 — 很多真實 Whisper 錯誤
     不是音節壓縮，而是同音替換（已由 HomophoneEngine 處理）

  3. maxMergeEditDistance=0 很安全但漏掉很多案例
     (如 語音→硬：merge=yin, 硬=ying, distance=1)

觸發測試: {len(test_results['triggered'])} 觸發 / {len(test_results['not_triggered'])} 未觸發
""")
