# Whisper MLX 遷移可行性評估

**調查日期**：2026-02-27
**目的**：評估把 Voco 的 Whisper 從 whisper.cpp 切換到 MLX 的可行性

---

## 調查結論（先說結論）

**目前不存在成熟的「MLX Swift Whisper」方案。** 但有三條可行路線，各有取捨：

| 路線 | 方案 | 成熟度 | 工作量 | 風險 |
|------|------|--------|--------|------|
| **A** | **WhisperKit (CoreML)** | ★★★★★ | 低 | 低 |
| **B** | **mlx-swift-audio** | ★★☆☆☆ | 中 | 高 |
| **C** | 自行移植（基於 Qwen3-ASR 架構） | — | 極高 | 極高 |
| **D** | 維持 whisper.cpp，等生態成熟 | — | 零 | 零 |

**建議**：**路線 A（WhisperKit）最務實**。它不是 MLX 而是 CoreML，但效果等同或更好。

---

## 一、現有方案調查

### 1. WhisperKit (argmaxinc) — 最成熟的 Swift Whisper

- **GitHub**：[argmaxinc/WhisperKit](https://github.com/argmaxinc/WhisperKit)（5.7k stars）
- **最後更新**：2026-02-19
- **最新版本**：v0.15.0 (2024-11-07)
- **技術**：**CoreML**（不是 MLX），透過 CoreML 部署到 ANE / GPU / CPU
- **模型格式**：CoreML `.mlmodelc`，託管在 HuggingFace `argmaxinc/whisperkit-coreml`
- **SPM 整合**：完整支援，最低 macOS 14.0
- **ICML 2025 論文**：在 Whisper Large v3 Turbo 上達到 0.46s 延遲、2.2% WER
- **MLX 支援**：[Issue #33](https://github.com/argmaxinc/WhisperKit/issues/33) 有討論但未實作，主要原因是 MLX 不支援 ANE

**優點**：
- Argmax 公司維護，ICML 論文背書，社區最大
- SPM 直接加依賴即可
- 支援 ANE 硬體加速（Apple Silicon 的專用推論晶片）
- 支援串流轉錄

**缺點**：
- 底層是 CoreML 不是 MLX
- 模型需要 CoreML 格式（不能直接用 HuggingFace 的 safetensors）
- 最近的正式 release 是 2024-11（但 repo 仍活躍）

### 2. mlx-swift-audio (DePasqualeOrg) — 唯一的 MLX Swift Whisper

- **GitHub**：[DePasqualeOrg/mlx-swift-audio](https://github.com/DePasqualeOrg/mlx-swift-audio)（107 stars）
- **最後活動**：2025-12-14 仍在開發
- **支援模型**：Whisper（含 large-v3-turbo）、Fun-ASR
- **技術**：**純 MLX Swift**
- **SPM 整合**：有（指向 main branch，無正式 release）
- **授權**：MIT

**優點**：
- 真正用 MLX Swift 跑 Whisper
- 可直接用 mlx-community 的 safetensors 模型
- MIT 開源

**缺點**：
- 僅 107 stars，早期開發，文件明確寫「Expect breaking changes」
- 無正式 release，需追蹤 main branch
- 穩定性未知

### 3. mlx-audio-swift (Blaizzy) — 不含 Whisper

- **GitHub**：[Blaizzy/mlx-audio-swift](https://github.com/Blaizzy/mlx-audio-swift)（383 stars）
- **技術**：MLX Swift
- **STT 支援**：Qwen3-ASR、Voxtral — **不含 Whisper**
- Python 版 mlx-audio 有 Whisper，但 Swift 版目前還沒移植

### 4. Apple 官方 mlx-swift-examples — 無 Whisper

- [Issue #146](https://github.com/ml-explore/mlx-swift-examples/issues/146) 有人要求加 Whisper 範例
- 官方未採納，僅指向第三方專案
- mlx-swift (v0.30.6) 本身只是底層陣列框架，無音訊模組

### 5. 其他

| 專案 | 說明 | Whisper 支援 |
|------|------|-------------|
| FluidAudio | CoreML + ANE，Parakeet 模型 | ❌ 且不支援中文 |
| lightning-whisper-mlx | Python MLX，號稱最快 | ❌ 純 Python |
| qwen3-asr-swift | MLX Swift | ❌ 只有 Qwen3-ASR |

---

## 二、效能基準比較

### Python 端基準（M4 Pro 24GB）

| 排名 | 方案 | 時間 | 技術 |
|------|------|------|------|
| 1 | FluidAudio CoreML | 0.19s | CoreML + ANE (Parakeet) |
| 2 | MLX-Whisper (Python) | 1.02s | MLX |
| 3 | **whisper.cpp** | **1.23s** | C++ + Metal/CoreML |
| 4 | WhisperKit | 2.22s* | CoreML |

> \* WhisperKit benchmark 可能包含模型載入時間。WhisperKit ICML 論文聲稱串流延遲 0.46s。

### 另一基準（2026-01, large-v3-turbo）

- **whisper.cpp**：26.7s
- **mlx-whisper (Python)**：13.1s
- **結論**：MLX 比 whisper.cpp **快約 2 倍**

### 我們的評測結果（50 筆測試集）

| 模型 | 引擎 | 平均 RTF | CER |
|------|------|---------|-----|
| Whisper v3 Turbo | mlx_whisper (Python) | 0.228 | 50.4% |
| Qwen3-ASR 1.7B | MLX Swift | 0.153 | 61.0% |

> Whisper v3 Turbo MLX 的 CER 和 FireRedASR2 並列第一，且**唯一輸出繁體中文**。

---

## 三、Voco 現有架構分析

### whisper.cpp 整合邊界

```
┌─────────────────────────────────────┐
│         WhisperState (主控)          │
│  - 狀態機（idle→recording→transcribing）│
│  - 模型生命週期                       │
│  - 流程協調                          │
└────────────┬────────────────────────┘
             │
    ┌────────┼────────────────┐
    ▼        ▼                ▼
┌───────────┐ ┌──────────────────────┐
│LibWhisper │ │ TranscriptionService │
│(C API)    │ │ Registry              │
│- whisper_ │ │ ├─ Local (whisper.cpp)│
│  full()   │ │ ├─ Qwen3 (MLX)       │
│- VAD      │ │ ├─ Parakeet           │
│- logprob  │ │ ├─ Cloud              │
└───────────┘ │ └─ Apple Native       │
              └──────────────────────┘
```

### 需替換的範圍

| 檔案 | 改動程度 | 說明 |
|------|---------|------|
| `LibWhisper.swift` | **全部替換** | whisper.cpp C API 呼叫 |
| `WhisperState+LocalModelManager.swift` | **部分替換** | 模型下載格式改變 |
| `LocalTranscriptionService.swift` | **部分替換** | 呼叫新後端 API |
| `PredefinedModels.swift` | **擴展** | 新增模型定義 |
| SPM 依賴 | **替換** | 移除 whisper.cpp，加新依賴 |

### 不需改的（解耦良好）

- `TranscriptionService` 協定 — 統一介面
- WhisperState 狀態機邏輯
- 所有後處理管線（中文校正、LLM 增強等）
- UI 層、Qwen3-ASR、雲端服務

### 關鍵設計限制

| 功能 | whisper.cpp 提供 | 新後端需支援 |
|------|-----------------|-------------|
| Token logprob | ✅ `whisper_full_get_token_p()` | 信心度路由依賴這個 |
| VAD | ✅ 內建 | 路由省 LLM 費用 |
| 語言偵測 | ✅ | 中文後處理需要 |
| 16kHz mono WAV | ✅ | 音訊格式統一 |
| 離線推論 | ✅ | 核心需求 |

---

## 四、Qwen3-ASR MLX 架構（可複用性）

Voco 已有完整的 MLX Swift 推論管線（2,553 行 Swift）：

| 元件 | 行數 | 可複用於 Whisper |
|------|------|-----------------|
| 音訊前處理（Mel spectrogram） | 257 | ⚠️ 部分可用（hop length 不同：Qwen=160, Whisper=160 ✅） |
| Audio Encoder（Transformer） | 341 | ❌ 架構不同（Qwen 用 CNN+Transformer，Whisper 不同） |
| Text Decoder（Quantized） | 243 | ❌ Whisper 用 cross-attention，Qwen 用 concatenation |
| Weight Loading（safetensors） | 311 | ✅ 格式相同 |
| Tokenizer（BPE） | 290 | ❌ Whisper 用 tiktoken |
| Inference Loop（KV cache） | 452 | ⚠️ 模式可參考，但細節不同 |
| HuggingFace 下載器 | 199 | ✅ 直接可用 |

**結論**：自行移植需要重寫 Whisper encoder、decoder、tokenizer（約 1,500+ 行新程式碼），只有 weight loading 和下載器可直接複用。**不建議自行移植**。

---

## 五、各路線詳細評估

### 路線 A：WhisperKit (CoreML) ⭐ 推薦

**工作量估算**：
1. 加 SPM 依賴：`https://github.com/argmaxinc/whisperkit`
2. 新增 `WhisperKitTranscriptionService: TranscriptionService`（~200 行）
3. 新增 `ModelProvider.whisperKit` case
4. 模型下載改為 WhisperKit 的 CoreML 格式
5. 調整信心度取得方式（WhisperKit 提供 token-level probabilities）

**預估**：3-5 天可完成基本整合

**優點**：
- 穩定、成熟、社區大
- ANE 加速（Apple Silicon 專用推論硬體，比 Metal GPU 更省電更快）
- 支援 Whisper Large v3 Turbo
- 維持離線能力
- Token logprob 可取得（信心度路由不用改）

**缺點**：
- 不是 MLX（和 Qwen3-ASR 用不同的推論引擎）
- CoreML 模型較大（需要轉換後的格式）
- WhisperKit 正式 release 更新較慢（但 repo 活躍）

**和 whisper.cpp 的共存**：可以並存。加一個新的 provider，舊的 whisper.cpp 保留作為 fallback。

### 路線 B：mlx-swift-audio

**工作量估算**：
1. 加 SPM 依賴（追蹤 main branch）
2. 新增 `MLXWhisperTranscriptionService`（~300 行）
3. 模型用 HuggingFace mlx-community 的 safetensors（和 Qwen3-ASR 同格式）

**預估**：5-7 天（需要除錯不穩定的早期專案）

**優點**：
- 和 Qwen3-ASR 同屬 MLX 生態（統一推論引擎）
- 模型格式統一（safetensors）
- 理論上 Metal GPU 效能好

**缺點**：
- 107 stars，早期開發，breaking changes 風險
- 無正式 release
- 如果上游棄坑，需要自己維護
- 不支援 ANE（MLX 目前只走 Metal GPU）

### 路線 C：自行移植

**不建議**。需 1,500+ 行新 Swift 程式碼實作 Whisper 的 encoder、decoder、tiktoken tokenizer。風險極高、工作量極大，且上面兩個方案都是更好的選擇。

### 路線 D：維持現狀

保留 whisper.cpp，等待：
- mlx-swift-audio 成熟
- WhisperKit 出 MLX 後端
- Apple 官方出 Whisper MLX Swift 範例

**缺點**：放棄 Whisper v3 Turbo MLX 帶來的 CER 改善和繁體中文輸出。

> **注意**：whisper.cpp 已經支援 Whisper Large v3 Turbo（`ggml-large-v3-turbo`），Voco 的 PredefinedModels 也已定義。所以「升級到 v3 Turbo」不一定要換引擎，用 whisper.cpp 也可以。差別在 MLX 版理論上更快（但我們的評測顯示 RTF 差距不大：0.228 vs whisper.cpp 未測）。

---

## 六、HuggingFace MLX Whisper 模型

[mlx-community Whisper Collection](https://huggingface.co/collections/mlx-community/whisper) 有 **48 個 MLX 格式模型**：

| 模型 | 量化選項 | 格式 |
|------|---------|------|
| whisper-tiny ~ whisper-large-v3 | fp32, fp16, 8-bit, 4-bit | safetensors |
| whisper-large-v3-turbo | fp16, 量化 | safetensors |

這些模型的 safetensors 格式和 Qwen3-ASR 相同，理論上 Swift MLX 可讀取。但**模型架構**（encoder/decoder 層的 Python 程式碼）需要有對應的 Swift 實作才能使用。

---

## 七、建議策略

### 最佳方案：WhisperKit + 保留 whisper.cpp

```
階段一（立即可做）：
  └─ 用 whisper.cpp 下載 ggml-large-v3-turbo
     （Voco 已有定義，只是沒下載）
     → 零成本升級，看看 v3 Turbo 在 whisper.cpp 上的表現

階段二（1-2 週）：
  └─ 整合 WhisperKit 作為新的 Whisper 後端
     → provider: .whisperKit
     → 和 .local (whisper.cpp) 並存
     → 使用者可在設定中選擇

階段三（觀察）：
  └─ 追蹤 mlx-swift-audio 的成熟度
     → 如果穩定到 v1.0，考慮替換 WhisperKit
     → 統一為 MLX 生態（和 Qwen3-ASR 一致）
```

### 為什麼不完全移除 whisper.cpp？

1. **低階 Mac 相容性**：whisper.cpp 的小模型（tiny/base）在 8GB RAM Mac 上仍有用
2. **穩定性保障**：whisper.cpp 是最久經考驗的引擎
3. **使用者已下載的模型**：刪除會導致使用者需要重新下載
4. **維護成本低**：保留不需要額外維護

---

## 八、相關資源

| 資源 | 連結 |
|------|------|
| WhisperKit | https://github.com/argmaxinc/WhisperKit |
| WhisperKit CoreML 模型 | https://huggingface.co/argmaxinc/whisperkit-coreml |
| mlx-swift-audio | https://github.com/DePasqualeOrg/mlx-swift-audio |
| mlx-community Whisper | https://huggingface.co/collections/mlx-community/whisper |
| WhisperKit ICML 論文 | https://arxiv.org/abs/2507.10860 |
| mac-whisper-speedtest | https://github.com/anvanvan/mac-whisper-speedtest |
| WhisperKit MLX Issue | https://github.com/argmaxinc/WhisperKit/issues/33 |

---

## 九、和評測結果的對照

根據 `asr-model-evaluation-results.md` 的評測：

| 模型 | CER | 英文術語 | RTF | 輸出語體 |
|------|-----|---------|-----|---------|
| Whisper v3 Turbo (mlx_whisper) | **50.4%** | 49.2% | 0.228 | **繁體中文** |
| Qwen3-ASR 1.7B (MLX Swift) | 61.0% | **81.4%** | **0.153** | 簡體中文 |

切換到 Whisper v3 Turbo 的預期效益：
- **CER 降低**：50.4% vs 61.0%（比 Qwen3 好）
- **繁體輸出**：省去 OpenCC 簡→繁轉換
- **英文術語**：比 Qwen3 差（49.2% vs 81.4%），但 LLM 後處理可彌補

最均衡策略仍是 **Qwen3-1.7B（主力）+ LLM 同音字修正**，因為英文術語辨識遙遙領先。Whisper v3 Turbo 作為備選方案提供給不需要英文術語辨識的使用者。
