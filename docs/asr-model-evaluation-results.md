# ASR 模型評測結果

**測試日期**：2026-02-27
**測試平台**：Apple Silicon Mac (arm64), macOS 26.2
**測試集**：50 筆音訊（從 578 筆有 LLM 修正紀錄的錄音中平衡取樣）
**參考基準**：LLM 增強文字（enhancedText，已修正版本）

---

## 測試模型

| 模型 | 推論引擎 | 模型大小 | 備註 |
|------|---------|---------|------|
| Qwen3-ASR 0.6B (4-bit) | mlx_audio (MLX) | ~400 MB | 目前 Voco 選項之一 |
| Qwen3-ASR 1.7B (8-bit) | mlx_audio (MLX) | ~1.8 GB | 目前 Voco 主力 |
| Whisper Large v3 Turbo | mlx_whisper (MLX) | ~800 MB | MLX 原生加速 |
| FireRedASR v1 AED-L | sherpa-onnx (ONNX) | 1.6 GB | 小紅書團隊，中文 SOTA |
| FireRedASR2-AED | PyTorch (CPU) | 4.4 GB | 小紅書最新版，2026-02 發布 |
| SenseVoice-Small | sherpa-onnx (ONNX) | 1.1 GB | 阿里，超輕量 |

---

## 總體排名

| 排名 | 模型 | 平均 CER | 英文術語正確率 | 平均 RTF | 輸出語體 |
|------|------|---------|-------------|---------|---------|
| 1 | **Whisper v3 Turbo MLX** | **50.4%** | 49.2% | 0.228 | 繁體中文 |
| 1 | **FireRedASR2-AED** | **50.4%** | **55.9%** | 0.985 | 簡體中文 |
| 3 | FireRedASR v1 AED-L | 58.3% | 33.9% | 0.340 | 簡體中文 |
| 4 | Qwen3-ASR 1.7B | 61.0% | **81.4%** | **0.153** | 簡體中文 |
| 5 | SenseVoice-Small | 62.4% | 39.0% | **0.020** | 簡體中文 |
| 6 | Qwen3-ASR 0.6B | 746%* | 44.1% | 0.208 | 簡體中文 |

> \* Qwen3-ASR 0.6B 有嚴重的重複/幻覺問題（如「英雄狼」←音訊檔），CER 飆高。
> CER 以 LLM 增強文字為參考基準，50% 左右表示 ASR 原始輸出和 LLM 修正版有約一半差異（含簡繁差異、標點差異）。

---

## 按類別分析（平均 CER）

| 模型 | Code-switching | 純中文 | 英文術語密集 | 英文為主 |
|------|---------------|--------|------------|---------|
| Qwen3-ASR 1.7B | **30.7%** | 41.3% | **31.0%** | 142% |
| Whisper v3 Turbo MLX | 31.2% | 42.1% | 38.6% | 96.0% |
| FireRedASR2-AED | 38.1% | 45.8% | 39.0% | **78.6%** |
| Qwen3-ASR 0.6B | 40.8% | * | 44.8% | 186% |
| SenseVoice-Small | 42.3% | 56.7% | 44.8% | 106% |
| FireRedASR v1 AED-L | 44.6% | 50.4% | 49.1% | 93.4% |

> Code-switching 場景：**Qwen3-1.7B 最佳**，Whisper v3 Turbo 緊隨其後。

---

## 英文術語辨識（常見漏失）

| 術語 | Qwen3 1.7B | Whisper v3T | FireRed2 | FireRed v1 | SenseVoice |
|------|-----------|-------------|----------|-----------|------------|
| ChatGPT | ✅ | ❌ (XGPT) | ✅ | ❌ (差距毕竟) | ❌ (GBT) |
| GitHub | ⚠️ (偶爾 Guitar) | ❌ | ✅ | ❌ (吉他) | ❌ |
| repo | ✅ | ✅ | ✅ | ✅ | ❌ (report) |
| commit | ✅ | ✅ | ❌ (卡密特) | ❌ (卡米特) | ✅ |
| tag | ⚠️ (泰格) | ✅ | ❌ (泰格) | ❌ (泰格) | ❌ |
| endpoint | ✅ | ✅ | ❌ (m point) | ❌ (M POINT) | ❌ (mpoint) |
| release | ✅ | ✅ | ✅ | ✅ | ❌ (rele) |
| Claude | ❌ | ❌ | ❌ | ❌ | ❌ |
| MLX | ✅ | ✅ | ✅ | ⚠️ (M L X) | ✅ |
| user preference | ✅ | ✅ | ✅ | ✅ | ❌ (preferencefer) |

> **Qwen3-1.7B 英文術語辨識明顯領先**（81.4%），但 "部署"→"不服" 這類中文同音字問題仍存在。
> 所有模型都無法正確辨識 "Claude"。

---

## 代表性案例逐筆比較

### 案例 1：「我在 ChatGPT 部署了很多 repo」(5.8s)

| 模型 | 輸出 | 評價 |
|------|------|------|
| **FireRedASR2** | 我在 chatgpt部署了很多 repo | ✅ 完美 |
| Qwen3-1.7B | 我在 ChatGPT **不服**了很多 repo | ⚠️ 部署→不服 |
| Whisper v3T | 我在**XGPT**部署了很多repo | ⚠️ ChatGPT→XGPT |
| Qwen3-0.6B | 我在**查GPT**，部署了很多repo | ⚠️ |
| SenseVoice | 我在**GBT**部署了很多 **report** | ❌ 兩個錯 |
| FireRed v1 | 我在**差距毕竟**部署了很多 REPO | ❌ ChatGPT 全壞 |

### 案例 2：「語言切換的 tag 是 auto」(12.4s)

| 模型 | 輸出 | 評價 |
|------|------|------|
| **Qwen3-1.7B** | 语言切换的tag是auto | ✅ 完美 |
| **Whisper v3T** | 語言接換的tag是AUTO | ⚠️ 切→接 |
| Qwen3-0.6B | 泰格是auto | ⚠️ tag→泰格 |
| FireRed2 | 泰戈斯奧特 | ❌ 全壞 |
| FireRed v1 | 泰格斯奧頭 | ❌ 全壞 |
| SenseVoice | t是 autoto | ❌ |

### 案例 3：「驗證一下 OpenAI 的 API 位置...endpoint 存不存在」(17.6s)

| 模型 | OpenAI | API | token | endpoint | 整體 |
|------|--------|-----|-------|----------|------|
| **Qwen3-1.7B** | ✅ Open AI | ✅ | ✅ | ✅ | 最佳 |
| Whisper v3T | ✅ OpenAI | ✅ | ✅ | ✅ | ✅ |
| FireRed2 | ✅ openai | ✅ | ✅ | ❌ m point | 差一個 |
| FireRed v1 | ✅ OPEN AI | ✅ | ✅ | ❌ M POINT | 差一個 |
| Qwen3-0.6B | ❌ Oppo AI | ✅ | ❌ 偷坑 | ✅ | 差兩個 |
| SenseVoice | ❌ op本I | ❌ | ❌ taken | ❌ mpoint | 全壞 |

---

## 速度比較

| 模型 | 平均 RTF | 50 筆總耗時 | 體感 |
|------|---------|-----------|------|
| SenseVoice-Small | 0.020 | ~10s | 極快，幾乎即時 |
| Qwen3-ASR 1.7B | 0.153 | ~54s | 快 |
| Qwen3-ASR 0.6B | 0.208 | ~120s | 普通（有重複問題拖慢） |
| Whisper v3 Turbo MLX | 0.228 | ~62s | 快 |
| FireRedASR v1 AED-L | 0.340 | ~180s | 可接受 |
| FireRedASR2-AED (CPU) | 0.985 | ~607s | 慢（無 GPU 加速） |

> RTF < 1 表示比即時快。FireRedASR2 在 CPU 上接近即時，若有 MPS/MLX 加速應會快很多。

---

## 結論與建議

### 各模型定位

1. **Qwen3-ASR 1.7B** — **英文術語辨識最強**，code-switching 場景 CER 最低。缺點是中文同音字（部署→不服）和輸出簡體。適合技術對話場景。

2. **Whisper v3 Turbo MLX** — **整體 CER 最低（並列第一）**，速度快，且**唯一輸出繁體中文**。英文術語辨識中等（tag ✅ 但 ChatGPT ❌）。性價比最高。

3. **FireRedASR2-AED** — **CER 並列第一**，英文術語辨識中上（ChatGPT ✅、repo ✅）。但 CPU 上太慢（RTF 接近 1），且目前無 MLX/ONNX 加速方案。

4. **FireRedASR v1 AED-L** — 表現中規中矩，但英文術語辨識最差（ChatGPT、commit 全壞），不推薦。

5. **SenseVoice-Small** — 速度驚人但準確度差。適合需要極低延遲且不在意準確度的場景。

6. **Qwen3-ASR 0.6B** — 有嚴重的重複/幻覺問題，**不推薦使用**。

### 建議策略

| 策略 | 方案 | 預期效果 |
|------|------|---------|
| **最簡單** | 切換到 Whisper v3 Turbo MLX | CER 降低、繁體輸出省去 OpenCC、英文術語中等 |
| **最均衡** | Qwen3-1.7B（主力）+ LLM 同音字修正加強 | 英文最好、中文靠 LLM 修正補救 |
| **未來觀察** | 等 FireRedASR2 出 ONNX/MLX 版本 | 目前 PyTorch CPU 太慢，不適合即時轉錄 |

### 下一步

1. **Whisper v3 Turbo MLX 值得在 Voco 中試用** — 目前只有 whisper.cpp 的 Whisper Large v2，升級到 MLX 版 v3 Turbo 應該全面提升
2. **Qwen3-1.7B 的 LLM 修正管線可以加強** — 針對 "部署→不服" 這類高頻同音字，加入更多 WordReplacement 規則
3. **FireRedASR2** — 追蹤 sherpa-onnx 的整合進度，一旦有 ONNX 版本立即重測

---

## 檔案位置

| 檔案 | 說明 |
|------|------|
| `scripts/asr_eval/prepare_test_set.py` | 從 SwiftData 匯出測試集 |
| `scripts/asr_eval/run_eval.py` | 批次評測腳本 |
| `scripts/asr_eval/analyze_results.py` | 結果分析 |
| `scripts/asr_eval/test_data/test_set.json` | 精簡測試集（50 筆） |
| `scripts/asr_eval/test_data/full_test_set.json` | 完整測試集（578 筆） |
| `scripts/asr_eval/results/results.json` | 原始評測結果（300 筆） |
| `scripts/asr_eval/results/analysis.json` | 分析摘要 |
