# ASR 模型評測計畫

**建立日期**：2026-02-27
**目的**：用實際錄音檔測試不同 ASR 模型，找出對使用者語音（中文為主、夾雜英文技術術語）效果最好的模型

---

## 現有問題摘要

### 1. Code-switching 不穩定（中英夾雜）

**現象**：同一句話重複說，ASR 結果時好時壞。

| 嘗試 | ASR 輸出 | 正確？ |
|------|---------|--------|
| 我用 ChatGPT 在 GitHub 部署了很多 repo。 | ✅ 全對 | ✅ |
| 我用 ChatGPT 來給他補充了很多 report。 | GitHub→給他補充, repo→report | ❌ |
| 我用 ChatGPT 在 GitHub 播出了很多 Ripple。 | 部署→播出, repo→Ripple | ❌ |
| 我用叉子飛機組織雞他部署了很多 ripple。 | 全部亂掉 | ❌ |
| 我用 ChatGPT 在 GitHub 部署了很多 repo。 | ✅ 全對 | ✅ |

**根因**：Qwen3-ASR 的 code-switch remap（中文偵測 → English 二次推論）有隨機性。同一段音訊每次推論結果可能不同。

### 2. 英文術語被音譯為中文

**範例**：
- `github` → `吉他`（偶發，ASR 信心度可能不低）
- `ChatGPT` → `叉J P G` / `超 G P`（偶發）
- `repo` → `report` / `Ripple`（常見）
- `部署` → `不服` / `複述` / `播出`（同音字，已有 LLM 修正但 validator 有時擋掉）

### 3. LLM Validator dropped-term 誤殺

**已修復（138e4f3）**：`extractTechnicalTerms` 把 CJK 字元也當成 term 的一部分，導致整句話被判為 protected term。修正後 CJK 字元作為分界符，只提取 Latin 段。

### 4. LLM 修正正確但被 Validator 拒絕

**已緩解（138e4f3）**：新增保守重試機制。當 validator 以 content-drift / short-edit-budget / dropped-term 拒絕時，用最小修正 prompt 重試一次。但保守 prompt 傾向原樣保留，不一定能修正 ASR 錯誤。

---

## 目前的 ASR 模型

| 模型 | 引擎 | 大小 | 特點 |
|------|------|------|------|
| Qwen3-ASR 0.6B (4-bit) | MLX | ~400 MB | 目前主力，中文 WER 不錯但 code-switching 不穩定 |
| Qwen3-ASR 1.7B (8-bit) | MLX | ~1.8 GB | 理論上更準，但更慢 |
| Whisper Large v3 Turbo | whisper.cpp | 1.5 GB | 多語言，code-switching 可能更好 |
| Whisper Large v3 Turbo (Q5_0) | whisper.cpp | 547 MB | 量化版 |
| Whisper Large v3 | whisper.cpp | 2.9 GB | 最準的 Whisper |
| Whisper Large v2 | whisper.cpp | 2.9 GB | 舊版 |
| Whisper Base | whisper.cpp | 142 MB | 快但不準 |
| Parakeet V2/V3 | NVIDIA | ~490 MB | 英文極快，中文不支援 |
| Groq Whisper | Cloud API | — | 雲端，低延遲 |

## 候選新模型

根據調查，以下模型值得納入評測（依優先順序排列）：

### 第一梯隊（最有潛力）

| 模型 | 大小 | 推論引擎 | 中文 CER | Code-switching | 備註 |
|------|------|---------|----------|----------------|------|
| **FireRedASR2-AED** | ~1.1B | sherpa-onnx | 3.05%（AISHELL-1 SOTA） | 支援（中英混合訓練） | 2025-06 發布，小紅書團隊，sherpa-onnx 已整合 |
| **SenseVoice-Small** | 228 MB (int8) | sherpa-onnx / FunASR | 4.96-5.14% | 支援（中日韓英粵） | 阿里，超輕量，70ms/10s 延遲，sherpa-onnx 有 C API |
| **Whisper Large v3 Turbo (MLX)** | ~800 MB | lightning-whisper-mlx | 同 Whisper v3 Turbo | 同 Whisper | M2 Ultra 上 50x realtime，Apple Silicon 原生加速 |
| **Fun-ASR-Nano-2512** | 未公開 | sherpa-onnx（進行中） | 未公開 | 支援 31 語言 | 阿里 2025-12 新模型，Paraformer 架構，非自回歸 |

### 第二梯隊（值得試但有限制）

| 模型 | 大小 | 推論引擎 | 中文 CER | Code-switching | 備註 |
|------|------|---------|----------|----------------|------|
| **GLM-ASR-Nano-2512** | ~1.5B | sherpa-onnx | 7.17%（WenetSpeech） | 有限 | 智譜 AI，CER 不如 FireRedASR2 |
| **Moonshine-tiny-zh** | 27M | ONNX | 中文微調版 | 不支援 | 超小但無 code-switching，僅供純中文場景 |
| **NVIDIA Canary-1B** | ~1B | NeMo | 良好 | 支援（4 語言） | 需要 NeMo 環境，整合成本高 |

### 不適用（排除）

| 模型 | 原因 |
|------|------|
| Meta MMS | 主打低資源語言，中文 ASR 表現一般 |
| wav2vec2-BERT | 需要外接 LM decoder，不適合即時轉錄 |
| Distil-Whisper | 只有英文版本 |

### 推論引擎摘要

| 引擎 | Apple Silicon 支援 | 說明 |
|------|-------------------|------|
| **MLX** | 原生 GPU | 目前 Qwen3-ASR 用的，效能最好 |
| **sherpa-onnx** | CoreML / CPU | 跨平台 C API，已整合 FireRedASR2、SenseVoice 等，可編譯為 macOS framework |
| **lightning-whisper-mlx** | 原生 GPU | Whisper 專用 MLX 包裝，Python API |
| **whisper.cpp** | Metal GPU | 目前 Whisper 用的 |

---

## 評測計畫

### 第一步：準備測試音訊集

從 `~/Library/Application Support/com.jasonchien.Voco/Recordings/` 中挑選有代表性的錄音檔（目前共 1233 筆）。

需要的測試類別：
1. **純中文**：一般對話、技術討論
2. **中英夾雜**：提到 ChatGPT、GitHub、repo、Claude Code 等
3. **英文術語密集**：程式開發相關（async/await、edge case 等）
4. **短句**：一句話（< 5 秒）
5. **長句**：連續說話（> 30 秒）

**挑選方式**：從 SwiftData 的 Transcription 紀錄中，找出有 `enhancedText != text`（LLM 做了修正）的錄音，這些最可能包含 ASR 錯誤。

### 第二步：離線批次測試

寫一個 Python 腳本，用 `mlx_audio`（Qwen3）和 `whisper`（OpenAI）分別轉錄同一批音訊檔。

```python
# 要測試的模型：
models = {
    # --- 現有 (MLX) ---
    "qwen3-0.6b-4bit": "mlx-community/Qwen3-ASR-0.6B-4bit",
    "qwen3-1.7b-8bit": "mlx-community/Qwen3-ASR-1.7B-8bit",

    # --- 現有 (whisper.cpp / Whisper Python) ---
    "whisper-large-v3-turbo": "openai/whisper-large-v3-turbo",
    "whisper-large-v3": "openai/whisper-large-v3",

    # --- MLX Whisper (lightning-whisper-mlx) ---
    "whisper-large-v3-turbo-mlx": "mlx-community/whisper-large-v3-turbo",

    # --- sherpa-onnx 模型 ---
    "firered2-aed": "sherpa-onnx/FireRedASR2-AED",       # 中文 SOTA
    "sensevoice-small": "sherpa-onnx/SenseVoice-Small",   # 超輕量 228MB

    # --- 待確認可用性 ---
    # "fun-asr-nano-2512": "...",   # sherpa-onnx 整合進行中
    # "glm-asr-nano": "...",        # sherpa-onnx 有支援
}
```

### 第三步：評測指標

| 指標 | 說明 |
|------|------|
| 英文術語正確率 | ChatGPT、GitHub、repo 等是否被正確辨識 |
| 中文同音字錯誤率 | 部署/不服、程式/成事 等 |
| Code-switching 穩定性 | 同一段音訊跑 3 次，結果是否一致 |
| 延遲 | 每段音訊的推論時間 |
| avgLogProb 分佈 | 各模型的信心度分佈差異 |

### 第四步：推論引擎整合可行性

目前 Voco 支援兩種引擎：MLX（Qwen3-ASR）和 whisper.cpp（Whisper 系列）。

需要評估的整合路徑：

| 路徑 | 模型 | 工作量 | 備註 |
|------|------|--------|------|
| **lightning-whisper-mlx** | Whisper v3 Turbo MLX | 中 | Python 包裝，需 bridge 或改寫為 Swift；`mlx-community` 已有模型權重 |
| **sherpa-onnx** | FireRedASR2、SenseVoice | 高 | C API → Swift bridge；需編譯 sherpa-onnx 為 macOS framework；支援 CoreML backend |
| **直接 MLX Swift** | SenseVoice | 高 | 目前無現成 MLX Swift 實作，需自行移植 |

**建議策略**：
1. 先用 Python 跑離線批次測試，確認哪個模型效果最好
2. 效果確認後，再投入整合引擎的工程量
3. 如果 sherpa-onnx 模型勝出，考慮用 sherpa-onnx 的 C API 做 Swift binding

### 第五步：整合回 Voco

根據評測結果，決定：
1. 是否切換預設模型
2. 是否需要新增模型支援（如 MLX Whisper）
3. code-switch remap 策略是否需要調整
4. 信心度路由的閾值是否需要依模型調整

---

## 已實作的改善措施（2026-02-27）

| 功能 | 說明 | commit |
|------|------|--------|
| Token-level logprob 提示 | 低信心度詞彙注入 LLM prompt `<UNCERTAIN_WORDS>` | `138e4f3` |
| WordReplacement 反哺 LLM | 已知 ASR 錯誤對照表注入 LLM prompt `<KNOWN_ASR_ERRORS>` | `138e4f3` |
| 保守重試 | Validator 拒絕時用最小修正 prompt 重試 | `138e4f3` |
| dropped-term 修正 | CJK 字元作為 term 分界符，避免整句被當 protected term | `138e4f3` |

---

## 錄音檔位置

```
~/Library/Application Support/com.jasonchien.Voco/Recordings/
```

1233 筆 WAV 檔（16-bit PCM, 16 kHz mono）。

## 相關檔案

| 檔案 | 用途 |
|------|------|
| `VoiceInk/Qwen3ASR/Qwen3ASRModel.swift` | MLX 推論 + logprob 收集 |
| `VoiceInk/Qwen3ASR/Qwen3ASREngine.swift` | 模型載入 + warmup + 分段轉錄 |
| `VoiceInk/Services/Qwen3TranscriptionService.swift` | Qwen3 轉錄入口 |
| `VoiceInk/Services/ChinesePostProcessing/` | 中文後處理管線（OpenCC、拼音、信心度路由、LLM 驗證） |
| `VoiceInk/Services/AIEnhancement/AIEnhancementService.swift` | LLM 增強（含 UNCERTAIN_WORDS / KNOWN_ASR_ERRORS 注入） |
| `VoiceInk/Models/AIPrompts.swift` | LLM 提示詞（含保守重試 prompt） |
| `VoiceInk/Models/PredefinedModels.swift` | 所有預定義模型清單 |
| `scripts/` | 現有測試/資料準備腳本 |
