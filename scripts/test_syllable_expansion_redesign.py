#!/usr/bin/env python3
"""SyllableExpansionEngine 重新設計分析

針對構音障礙語音特性，分析引擎應該怎麼改。

構音障礙的 ASR 錯誤模式（與一般人不同）：
1. 音節合併：兩個音節糊成一個（但不遵循 initial+final 規則）
2. 音節丟失：直接少了一個音節
3. 子音模糊：聲母不清楚，如 b/p/m 混淆
4. 母音偏移：韻母不標準
5. 整體模糊：ASR 用最接近的常見字替代

所以需要的不是 "merge model"，而是 "phonetic similarity + context" 模型。
"""

import json
import math
from collections import defaultdict
from pathlib import Path
from itertools import product

DATA_DIR = Path(__file__).parent.parent / "VoiceInk" / "Resources" / "ChineseCorrection"

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

# ── 方案比較 ──────────────────────────────────────────────

print("=" * 70)
print("SyllableExpansionEngine 重新設計方案分析")
print("=" * 70)

print("""
┌─────────────────────────────────────────────────────────┐
│ 使用者語音特性 (腦性麻痺構音障礙)                        │
│                                                         │
│ • 語音辨識率 60-70%（對機器說話，激動時較好）            │
│ • 說話需全身力氣 → 語音極不清楚                         │
│ • 肌肉控制不精確 → 音節可能真的糊在一起或丟失           │
│ • 對機器說話「激動就會好很多」→ 品質波動大               │
└─────────────────────────────────────────────────────────┘
""")

# ── 方案 A: 放寬現有 merge 模型 ──────────────────────────

print("=" * 70)
print("方案 A: 放寬現有 merge 模型 (maxMergeEditDistance=1 或 2)")
print("=" * 70)

# 統計 distance 0/1/2 各能覆蓋多少
merge_index = defaultdict(list)
for word, freq in word_freq.items():
    if len(word) != 2:
        continue
    r1, r2 = char_pinyin.get(word[0], []), char_pinyin.get(word[1], [])
    if not r1 or not r2:
        continue
    p1, p2 = strip_tone(r1[0]), strip_tone(r2[0])
    m = pinyin_initial(p1) + pinyin_final(p2)
    if m:
        merge_index[m].append((word, freq))

# 對一組常見字，看 distance 0/1/2 各能找到什麼
test_chars = "硬雲盈令陰引印因銀寅"
print(f"\n以 {test_chars} 為例，不同 distance 能匹配到的候選詞:\n")

def edit_distance(a, b):
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

for char in test_chars:
    readings = char_pinyin.get(char, [])
    if not readings:
        continue
    py = strip_tone(readings[0])
    cf = word_freq.get(char, 0)

    hits = {0: [], 1: [], 2: []}
    for merge_py, words in merge_index.items():
        d = edit_distance(merge_py, py)
        if d <= 2:
            for w, wf in words:
                if wf > 5000:  # 只看高頻詞
                    hits[d].append((w, wf))

    for d in hits:
        hits[d].sort(key=lambda x: -x[1])
        hits[d] = hits[d][:3]  # top 3

    print(f"  {char}({py}, freq={cf}):")
    for d in [0, 1, 2]:
        if hits[d]:
            words_str = ", ".join(f"{w}({wf})" for w, wf in hits[d])
            print(f"    dist={d}: {words_str}")
        else:
            print(f"    dist={d}: (無)")

print("""
問題: 即使放寬到 distance=2，merge 模型本身的「initial+final」假設
     仍然不正確。構音障礙的音節合併模式千變萬化，不可能用一個
     固定公式覆蓋。而且放寬 distance 會大幅增加 false positive。
""")

# ── 方案 B: 拼音相似度模型（取代 merge）───────────────────

print("=" * 70)
print("方案 B: 拼音相似度模型 (取代 merge)")
print("=" * 70)

print("""
核心思路: 不假設壓縮機制，改用「拼音 edit distance」。
  - 2字詞的完整拼音 vs 1字的拼音，只要 edit distance 夠小就考慮
  - 例如: 語音 = "yuyin" (5字母), 雲 = "yun" (3字母), dist = 2

這個方式更符合構音障礙的錯誤模式：
  - 音節糊在一起 → 2字拼音被壓短
  - 子音模糊 → 部分字母被替換
  - 音節丟失 → 直接少了一段拼音
""")

# 建立 2-char word 的完整拼音索引
word_pinyin_index = {}
for word, freq in word_freq.items():
    if len(word) != 2 or freq < 5000:
        continue
    r1, r2 = char_pinyin.get(word[0], []), char_pinyin.get(word[1], [])
    if not r1 or not r2:
        continue
    full_py = strip_tone(r1[0]) + strip_tone(r2[0])
    word_pinyin_index[word] = (full_py, freq)

print(f"2字高頻詞 (freq≥5000): {len(word_pinyin_index)} 個\n")

# 測試: 用拼音 edit distance 找候選
print("範例: 用全拼音 edit distance 找 1字→2字 候選\n")
test_cases = [
    ("雲", "語音辨識很重要"),     # 已知案例
    ("硬", "硬體設備"),            # 可能的壓縮
    ("盈", "螢幕很亮"),            # 可能的壓縮
    ("陰", "語音輸入系統"),        # 同拼音
    ("伍", "網路很慢"),            # merge 能匹配的
    ("乩", "經濟發展"),            # merge 能匹配的
]

for char, context in test_cases:
    readings = char_pinyin.get(char, [])
    if not readings:
        continue
    char_py = strip_tone(readings[0])
    cf = word_freq.get(char, 0)

    candidates = []
    for word, (full_py, wf) in word_pinyin_index.items():
        # 全拼音 edit distance
        d = edit_distance(full_py, char_py)
        # 正規化: distance / max(len), 越小越相似
        norm_d = d / max(len(full_py), len(char_py))
        if norm_d <= 0.5 and wf / max(cf, 1) >= 5:  # 寬鬆門檻
            candidates.append((word, full_py, wf, d, norm_d))

    candidates.sort(key=lambda x: (x[3], -x[2]))

    print(f"  {char}({char_py}, freq={cf}) 在「{context}」中:")
    if candidates:
        for w, wpy, wf, d, nd in candidates[:5]:
            print(f"    → {w}({wpy}, freq={wf}) dist={d} norm={nd:.2f}")
    else:
        print(f"    (無候選)")
    print()

print("""
問題: 拼音 edit distance 的搜索空間太大，會產生很多不相關的候選。
     例如 "雲(yun)" 和 "語音(yuyin)" 的 distance=2，但
     "雲(yun)" 和 "預言(yuyan)" 的 distance 也是 2。
     需要很強的 context 信號才能區分。
""")

# ── 方案 C: 基於歷史錯誤的學習式引擎 ─────────────────────

print("=" * 70)
print("方案 C: 基於個人歷史錯誤的學習式引擎 (推薦)")
print("=" * 70)

print("""
核心思路: 不預設壓縮模型，直接從使用者的歷史錯誤中學習。

架構:
  1. 資料來源: SwiftData 中的 Transcription 記錄
     - original_text (ASR 原始輸出)
     - enhanced_text (LLM 修正後)
     - 兩者的差異就是「錯誤→正確」的配對

  2. 自動挖掘 1字→2字 的替換模式:
     - diff(original, enhanced) 找出所有替換
     - 篩選: original 是 1 字, enhanced 是 2 字
     - 累計同一個 pattern 出現的次數
     - 出現 N 次以上的 pattern → 自動加入規則

  3. 應用方式:
     - 類似 PinyinCorrector 的 alwaysApply 規則
     - 但規則是自動從歷史中挖掘的
     - 可以設定最低觀察次數（如 3 次）避免偶發錯誤

優點:
  ✓ 不需要假設壓縮機制 — 直接從真實錯誤學習
  ✓ 完全個人化 — 學到的是這個使用者的特定語音模式
  ✓ 隨時間改善 — 用越多，規則越準確
  ✓ False positive 極低 — 因為每條規則都是多次驗證過的

缺點:
  △ 冷啟動問題 — 前期沒有足夠資料
  △ 需要 LLM 修正結果作為 ground truth
  △ 如果 LLM 修正本身也有錯誤，會學到錯誤的規則

實作估計:
  - 新增一個 PersonalCorrectionLearner service
  - 定期掃描 Transcription DB 挖掘 pattern
  - 學到的規則存入 UserDefaults 或獨立 JSON
  - 在 ChinesePostProcessingService 管線的最前面應用
""")

# ── 方案 D: 結合 B+C 的混合方案 ──────────────────────────

print("=" * 70)
print("方案 D: 混合方案 — 拼音相似度 + 歷史學習 (最推薦)")
print("=" * 70)

print("""
核心思路: 結合方案 B 的拼音相似度和方案 C 的歷史學習。

設計:
  1. 偵測層 (Detection):
     - 保留「可疑字偵測」但放寬條件
     - 不看 merge，改看:
       a) NLTokenizer 分出的單字 token
       b) 字頻低 (< threshold)
       c) bigram context 弱 (< threshold)

  2. 候選層 (Candidate Generation):
     - 主要來源: 個人歷史錯誤表 (方案 C)
     - 補充來源: 拼音相似度搜索 (方案 B)
       - full pinyin edit distance ≤ 2
       - 候選詞頻 >> 原字頻

  3. 評分層 (Scoring):
     - 歷史出現次數 (最強信號)
     - bigram context 改善
     - BERT MLM score (如果啟用)
     - 拼音相似度

  4. 個人化規則 (Personal Rules):
     - 從歷史中學到的高信心規則 (出現 ≥ 3 次)
     - 可以直接套用，不需要每次重新計算

管線位置:
  Whisper → OpenCC → 個人規則(方案C) → 同音字校正 → 鼻音校正 → LLM

這樣的好處:
  ✓ 個人規則在 LLM 之前，減少 LLM 負擔
  ✓ 不依賴固定的壓縮模型
  ✓ 隨使用時間自動改善
  ✓ 對一般使用者也有用（學習個人口音特徵）
""")

# ── 量化分析: 歷史學習的可行性 ─────────────────────────────

print("=" * 70)
print("量化分析: 歷史學習的資料需求")
print("=" * 70)

print("""
假設:
  - 使用者每天語音輸入 100 次
  - ASR 錯誤率 30-40% (基於 60-70% 辨識率)
  - 其中 1字→2字 的錯誤佔 5% (保守估計)
  - 每天約 1.5-2 個新的 1字→2字 錯誤

學習速度:
  - 第 1 週: ~10 個 pattern，尚未達到 3 次門檻
  - 第 2 週: ~20 個 pattern，最常見的 5-8 個達到 3 次門檻
  - 第 1 月: ~50 個 pattern，20-30 個達到門檻
  - 第 3 月: 大部分常見錯誤都已學到

冷啟動加速:
  如果已有歷史資料庫，可以一次性挖掘 → 立即有效
  SwiftData 中如果有幾千筆記錄，應該能立即得到上百條規則
""")

# ── 最終建議 ──────────────────────────────────────────────

print("=" * 70)
print("最終建議")
print("=" * 70)

print("""
┌─────────────────────────────────────────────────────────────┐
│ 不建議: 修復現有 SyllableExpansionEngine                     │
│   原因: merge 模型(initial+final)的基本假設不符合現實        │
│                                                              │
│ 建議: 用方案 D 取代整個引擎                                  │
│                                                              │
│ 第一步 (最小可行):                                           │
│   - 寫一個腳本掃描現有 SwiftData 資料庫                      │
│   - 挖掘 original_text vs enhanced_text 的 1字→2字 替換      │
│   - 看有沒有足夠的 pattern → 決定是否值得做成引擎            │
│                                                              │
│ 第二步 (如果有足夠 pattern):                                 │
│   - 把 SyllableExpansionEngine 改造為 PersonalCorrectionEngine│
│   - 從 DB 自動學習，定期更新規則                              │
│   - 用 bigram context + 拼音相似度做評分                     │
│                                                              │
│ 第三步 (進階):                                               │
│   - 加入 BERT MLM scoring 做二次驗證                         │
│   - 擴展到 2字→2字 的個人化錯誤模式                          │
│   - 設定頁面讓使用者看到/管理學到的規則                       │
└─────────────────────────────────────────────────────────────┘
""")
