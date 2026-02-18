# Confidence Routing 程式碼地圖

## 相關檔案與程式碼片段

### 1. ChinesePostProcessingService.swift

`VoiceInk/Services/ChinesePostProcessing/ChinesePostProcessingService.swift`

#### 判斷入口 — `shouldSkipLLMEnhancement()` (L165-224)

依序檢查：

1. Confidence routing 是否啟用 (L166-169)
2. 純英文 → 跳過 (L172-175)
3. 簡單短回應 → 跳過 (L178-181)
4. 標點密度不足 → 強制 LLM (L184-190)
5. **Provider 分流** (L192-210)：
   - `.local` → avgLogProb 判斷
   - `.qwen3` → `qwen3TextQualityCheck()` 啟發式
   - 其他 → fall through
6. 歧義標點 → 強制 LLM (L212-215)
7. 重複偵測 → 強制 LLM (L216-219)
8. 預設 → 走 LLM (L221-222)

#### Qwen3 啟發式 — `qwen3TextQualityCheck()` (L280-318)

- 語助詞過多 → false（強制 LLM）
- 條列式內容 → false（強制 LLM）
- CJK 字數 ≤ 閾值 → true（跳過 LLM）
- 語速異常 → false（強制 LLM）
- 長文 → false（走 LLM）

#### 主管線 — `process()` (L96-160)

- 依序跑 OpenCC、標點、拼音、口語標點、重複偵測
- L152 呼叫 `shouldSkipLLMEnhancement()` 得出 `needsLLMCorrection`

---

### 2. WhisperState.swift

`VoiceInk/Whisper/WhisperState.swift`

#### `transcribeAudio(on:)` (L301-574) 中的路由決策

設定 provider 資訊 (L385-392)：

```swift
postProcessor.lastModelProvider = model.provider
let preAudioAsset = AVURLAsset(url: url)
let preAudioDuration = (try? CMTimeGetSeconds(await preAudioAsset.load(.duration))) ?? 0.0
postProcessor.lastAudioDuration = preAudioDuration
```

呼叫 post-processing 並取得結果 (L393-403)：

```swift
let ppResult = postProcessor.process(text)
ppNeedsLLM = ppResult.needsLLMCorrection
```

路由決策 (L437-438)：

```swift
let shouldSkipEnhancement = postProcessor.isEnabled && !ppNeedsLLM
```

三條路徑 (L440-511)：

- `!shouldSkipEnhancement` → 送 LLM 增強 (L440-478)
- `shouldSkipEnhancement` → 跳過，但有 **safety net** (L479-511)：檢查標點密度，不足時強制補送 LLM

---

### 3. ChinesePostProcessingSettingsView.swift

`VoiceInk/Views/Settings/ChinesePostProcessingSettingsView.swift`

UI 設定 (L34-52)：

- Confidence Routing 開關
- Log-Prob Threshold 滑桿（Whisper 用）
- Qwen3 Skip Threshold 滑桿（Qwen3 用）

---

### 4. LocalTranscriptionService.swift（未修改）

`VoiceInk/Services/LocalTranscriptionService.swift`

設定 `lastAvgLogProb` 的地方，Whisper 轉錄完成後寫入：

```swift
postProcessor.lastAvgLogProb = avgLogProb
```

---

## 日誌位置

`~/Library/Logs/Voco/confidence-routing.log`
