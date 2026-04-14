#!/usr/bin/env python3
"""SyllableExpansionEngine 深度分析

聚焦：
1. 分析 2 中 100 個配對的失敗原因分類
2. merge 模型 vs 真實 Whisper 錯誤類型比對
3. 引擎在什麼「假想場景」下才有用？
"""

import json
import math
from collections import defaultdict, Counter
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "VoiceInk" / "Resources" / "ChineseCorrection"

# ── Load data (same as before) ─────────────────────────────

with open(DATA_DIR / "char_pinyin.json") as f:
    char_pinyin = json.load(f)

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

INITIALS = [
    "zh", "ch", "sh", "b", "p", "m", "f", "d", "t", "n", "l",
    "g", "k", "h", "j", "q", "x", "r", "z", "c", "s", "y", "w",
]

def strip_tone(py):
    tone_map = str.maketrans("āáǎàēéěèīíǐìōóǒòūúǔùǖǘǚǜü", "aaaaeeeeiiiioooouuuuvvvvv")
    result = py.translate(tone_map)
    if result and result[-1].isdigit():
        result = result[:-1]
    return result

def pinyin_initial(py):
    for ini in INITIALS:
        if py.startswith(ini):
            return ini
    return ""

def pinyin_final(py):
    return py[len(pinyin_initial(py)):]

def merge_pinyin(p1, p2):
    return pinyin_initial(p1) + pinyin_final(p2)

def is_cjk(ch):
    v = ord(ch)
    return 0x4E00 <= v <= 0x9FFF or 0x3400 <= v <= 0x4DBF

SKIP_CHARS = set("的了嗎呢吧啊哦喔嗯呀是在有和也都就不我你他她它們這那個把被讓會能可要得地著過到從與及或而但因為所以如")
HIGH_FREQ = 5000
MIN_RATIO = 10.0
BIGRAM_THRESH = 50
MIN_CTX_IMPROVE = 3.0
MIN_SCORE = 7.0

# ── 分析 1: Merge 模型的根本問題 ──────────────────────────

print("=" * 70)
print("深度分析 1: merge 模型 vs Whisper 真實錯誤模式")
print("=" * 70)

print("""
merge 模型假設: initial(字1) + final(字2) → 壓縮成一個字
例如: 語(yu) + 音(yin) → merge = y + in = "yin"

但 Whisper 的錯誤模式不是「音節壓縮」，而是:
  a) 同音替換 (homophone): 語音→雨音→魚音
  b) 相似音替換 (near-homophone): 辨識→變色→邊視
  c) 直接丟字: 語音→音 (丟了前面的音節)
  d) 幻覺/重複: 語音→語音語音語音

xvoice 的 SYLLABLE_EXPANSIONS 只有一條: yun → yuyin (雲→語音)
讓我們驗證這條規則是否符合 merge 模型:
""")

# 驗證 xvoice 的唯一規則
p1 = strip_tone(char_pinyin["語"][0])  # yu
p2 = strip_tone(char_pinyin["音"][0])  # yin
merged = merge_pinyin(p1, p2)
yun_py = strip_tone(char_pinyin["雲"][0])
print(f"  語({p1}) + 音({p2}) → merge = {pinyin_initial(p1)} + {pinyin_final(p2)} = \"{merged}\"")
print(f"  雲 = \"{yun_py}\"")
print(f"  \"{merged}\" == \"{yun_py}\"? {'YES ✓' if merged == yun_py else 'NO ✗'}")
print(f"  edit distance = {sum(1 for a, b in zip(merged, yun_py) if a != b) + abs(len(merged)-len(yun_py))}")

print(f"""
結果: 即使是 xvoice 自己的案例(雲→語音)，merge 模型也不匹配！
  merge(語音) = "yin" ≠ "yun"

這代表引擎的核心假設（音節壓縮 = initial+final）與真實錯誤模式不吻合。
Whisper 把「語音」錯聽成「雲」，不是因為音節壓縮，而是因為語音相似度。
""")

# ── 分析 2: 那些「能觸發」的配對，是有意義的修正嗎？────

print("=" * 70)
print("深度分析 2: 能通過 merge 匹配的配對品質分析")
print("=" * 70)

# 找所有 distance=0 且 freq ratio 通過的配對，看有多少是「合理的壓縮」
print("\n隨機抽樣 50 個 merge 匹配配對，看是否有意義:\n")
print(f"{'壓縮字':>6} {'拼音':>8} {'字freq':>8} → {'展開詞':<8} {'詞freq':>8} {'merge公式':<30} {'合理?'}")
print("-" * 95)

# Build merge index
merge_index = defaultdict(list)
for word, freq in word_freq.items():
    if len(word) != 2:
        continue
    r1 = char_pinyin.get(word[0], [])
    r2 = char_pinyin.get(word[1], [])
    if not r1 or not r2:
        continue
    m = merge_pinyin(strip_tone(r1[0]), strip_tone(r2[0]))
    if m:
        merge_index[m].append((word, freq))

# 找常見字（freq 100-5000）的配對
sample_pairs = []
for char in sorted(char_pinyin.keys()):
    if char in SKIP_CHARS:
        continue
    cf = word_freq.get(char, 0)
    if cf < 100 or cf > HIGH_FREQ:
        continue
    readings = char_pinyin[char]
    py = strip_tone(readings[0])
    candidates = merge_index.get(py, [])
    for word, wf in candidates:
        if wf / cf >= MIN_RATIO and wf > 5000:
            sample_pairs.append((char, cf, py, word, wf))
            break  # 每個字只取最佳候選

    if len(sample_pairs) >= 50:
        break

for char, cf, py, word, wf in sample_pairs:
    r1 = char_pinyin.get(word[0], ["?"])
    r2 = char_pinyin.get(word[1], ["?"])
    p1 = strip_tone(r1[0])
    p2 = strip_tone(r2[0])
    formula = f"i({p1})+f({p2})={pinyin_initial(p1)}+{pinyin_final(p2)}"

    # 判斷是否合理: 字和詞在語義上有沒有可能被混淆
    reasonable = "?"  # 需要人工判斷，但可以用一些啟發式
    # 如果壓縮字本身有獨立意義且常用，Whisper 不太會搞錯
    if cf > 1000:
        reasonable = "▲ 字太常用"
    elif wf < 5000:
        reasonable = "▲ 詞不夠常用"
    else:
        reasonable = "◯ 可能"

    print(f"{char:>6} {py:>8} {cf:>8} → {word:<8} {wf:>8} {formula:<30} {reasonable}")

# ── 分析 3: 如果不用 merge 模型，改用什麼？──────────────

print("\n" + "=" * 70)
print("深度分析 3: Whisper 真實的「2字→1字」錯誤分析")
print("=" * 70)

print("""
已知的真實案例（來自 xvoice DIRECT_CORRECTIONS + 經驗）:

  雲 → 語音     yun ← yuyin    (丟了中間音節, 非 merge)
  雨停 → 語音   yuting ← yuyin (近音替換)
  耳度 → 額度   erdu ← edu     (同音替換)
  變色 → 辨識   bianse ← bianshi (韻母替換)
  大宇 → 大語言 dayu ← dayuyan  (丟了後面音節)

這些錯誤的共同特徵:
  1. 不是「音節壓縮」(initial+final)，而是「近音替換」或「丟音節」
  2. 大多數情況是 2 字保持 2 字（同音替換），已由 HomophoneEngine 處理
  3. 真正的 2字→1字 非常罕見，且不遵循 merge 規則

所以 SyllableExpansionEngine 的 merge 模型與現實錯誤模式不匹配。
""")

# ── 分析 4: 引擎唯一能有用的場景 ──────────────────────────

print("=" * 70)
print("深度分析 4: 引擎可能有用的假想場景")
print("=" * 70)

print("""
理論上引擎能正確觸發的場景需要同時滿足:

  1. Whisper 真的把 2 音節壓成 1 字
  2. 壓縮遵循 initial(字1)+final(字2) 的規則
  3. 壓縮後的字是罕見字（freq ≤ 5000）
  4. 壓縮後的字與上下文 bigram 極低（≤ 50）
  5. 壓縮後的字不與相鄰字形成已知詞
  6. 目標詞遠比壓縮字常見（≥ 10x）
  7. 替換後 bigram context 顯著改善

現在讓我們找幾個「理論上完美」的場景:
""")

# 找一些能同時滿足所有條件的配對+上下文
perfect_cases = []

for char, cf, py, word, wf in sample_pairs[:30]:
    if cf > 500:
        continue  # 要罕見的字

    word_chars = list(word)

    # 找左 context: bigram(left, word[0]) 要高，bigram(left, char) 要低
    for bg, freq in sorted(bigram_freq.items(), key=lambda x: -x[1])[:5000]:
        if len(bg) != 2 or not is_cjk(bg[0]) or not is_cjk(bg[1]):
            continue
        if bg[1] != word_chars[0]:
            continue
        left = bg[0]
        # 驗證 left+char 的 bigram 很低
        left_orig_bf = bigram_freq.get(left + char, 0)
        if left_orig_bf > BIGRAM_THRESH:
            continue
        # 驗證 left+char 不是已知詞
        if word_freq.get(left + char, 0) > 0:
            continue

        # 找右 context
        for bg2, freq2 in sorted(bigram_freq.items(), key=lambda x: -x[1])[:5000]:
            if len(bg2) != 2 or not is_cjk(bg2[0]) or not is_cjk(bg2[1]):
                continue
            if bg2[0] != word_chars[1]:
                continue
            right = bg2[1]
            # 驗證 char+right 的 bigram 很低
            right_orig_bf = bigram_freq.get(char + right, 0)
            if right_orig_bf > BIGRAM_THRESH:
                continue
            if word_freq.get(char + right, 0) > 0:
                continue

            # 計算 score
            orig_left = math.log(left_orig_bf + 1)
            orig_right = math.log(right_orig_bf + 1)
            new_left = math.log(freq + 1)
            new_right = math.log(freq2 + 1)
            ctx_improve = 0.5 * ((new_left - orig_left) + (new_right - orig_right))

            if ctx_improve < MIN_CTX_IMPROVE:
                continue

            internal_bf = bigram_freq.get(word, 0)
            score = (math.log(wf + 1) + ctx_improve
                     + 0.3 * math.log(internal_bf + 1))

            if score >= MIN_SCORE:
                sentence = f"{left}{char}{right}"
                expected = f"{left}{word}{right}"
                perfect_cases.append({
                    "sentence": sentence,
                    "expected": expected,
                    "char": char,
                    "word": word,
                    "score": score,
                    "char_freq": cf,
                    "word_freq": wf,
                })
                break
        if len(perfect_cases) >= 10:
            break
    if len(perfect_cases) >= 10:
        break

if perfect_cases:
    print(f"找到 {len(perfect_cases)} 個「理論上完美」的案例:\n")
    for case in perfect_cases:
        print(f"  輸入: {case['sentence']}")
        print(f"  期望: {case['expected']}")
        print(f"  修正: {case['char']}(freq={case['char_freq']}) → {case['word']}(freq={case['word_freq']})")
        print(f"  分數: {case['score']:.1f}")
        print()
else:
    print("  找不到完美案例（這就是問題所在）\n")

# ── 分析 5: 與 xvoice 的根本差異 ──────────────────────────

print("=" * 70)
print("深度分析 5: xvoice vs Voco 音節擴展的根本差異")
print("=" * 70)

print("""
┌─────────────────┬────────────────────────┬────────────────────────┐
│                 │ xvoice (Python)        │ Voco (Swift)           │
├─────────────────┼────────────────────────┼────────────────────────┤
│ 方式            │ 硬編碼規則             │ 動態統計偵測           │
│ 錯誤模型        │ 不假設機制，人工觀察   │ initial+final 壓縮     │
│ 規則數          │ 1 條 (yun→yuyin)       │ ~3000 個 merge form    │
│ False Positive  │ 幾乎為零              │ 11/3327 (0.33%)        │
│ True Positive   │ 偶爾命中              │ 0/3327 (0%)            │
│ 維護方式        │ 從錯誤日誌手動新增     │ 自動（但不準確）       │
├─────────────────┼────────────────────────┼────────────────────────┤
│ 關鍵差異        │ xvoice 的「音節擴展」實│ Voco 試圖自動化，但    │
│                 │ 際上就是人工觀察到的直  │ merge 模型不符合真實   │
│                 │ 接替換規則，和拼音計算  │ Whisper 錯誤模式       │
│                 │ 無關                    │                        │
└─────────────────┴────────────────────────┴────────────────────────┘

結論:
  xvoice 的「音節擴展」有效是因為它本質上不是音節擴展——它只是
  DIRECT_CORRECTIONS 的一部分（手動規則）。名字叫「音節擴展」但
  實際做的是「直接替換」。

  Voco 嘗試把這個概念泛化為自動偵測引擎，但:
  1. merge 模型假設錯誤：Whisper 不按 initial+final 壓縮
  2. 過濾條件太多：即使偶爾有真壓縮，也被 7 層 gate 擋掉
  3. 真實 2字→1字 極其罕見：大多數 Whisper 錯誤是 2字→2字 替換

建議:
  ✗ 不建議重新啟用 SyllableExpansionEngine
  ✓ xvoice 的 yun→yuyin 這類案例，繼續由 PinyinCorrector 的
    alwaysApply 規則處理即可
  ✓ 如果有新的壓縮案例，加到 PinyinCorrector 的規則表中
""")
