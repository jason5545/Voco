# WhisperKit Language Detection Issue

## 環境

- macOS 26.2, Apple Silicon (arm64)
- WhisperKit 0.9+ via SPM
- 模型：`openai_whisper-large-v3_turbo_954MB`（壓縮版 CoreML）
- DecodingOptions：`task: .transcribe`, `language: nil`, `detectLanguage: true`, `usePrefillPrompt: true`, `temperature: 0.0`
- 語言設定：Auto-detect

## 問題描述

WhisperKit 的 auto-detect 語言偵測**對中文完全不可靠**。使用者說中文時，模型頻繁將中文語音錯誤偵測為其他語言，導致輸出變成「翻譯」而非「轉錄」。

同樣的音訊用 whisper.cpp（同 large-v3-turbo 模型的 GGML 版本）語言偵測正常。

## 實際 Log 記錄

### 案例 1：中文 →  俄語
```
語音內容：「今天天氣不錯」
detectedLang=ru, avgLogProb=-0.714
輸出："Сегодня солнце не плачет"（俄語翻譯）
```

### 案例 2：中文 → 葡萄牙語
```
語音內容：「我會再試試看」
detectedLang=pt, avgLogProb=-0.606
輸出："Eu vou experimentar de novo"（葡萄牙語翻譯）
```

### 案例 3：中文 → 印尼語（高信心錯誤）
```
語音內容：某句中文（具體內容不確定）
detectedLang=id, avgLogProb=-0.335（信心相對高）
輸出："Saya di GitHub tersebut juga terjual banyak Repo"（印尼語翻譯）
```

### 案例 4：中文 → 義大利語
```
語音內容：「今天天氣不錯」
detectedLang=it, avgLogProb=-0.740
輸出："Gin天天气不错!"（部分 CJK + Latin 混合，「今」變 Gin）
```

### 案例 5：中文 → 德語（完全拼音化）
```
語音內容：「今天天氣不錯」
detectedLang=de, avgLogProb=-0.501
輸出："Gin-tian-tian-ti-bu-tuo!"（完全拼音化，像 pinyin）
```

### 案例 6：中文 → 西班牙語
```
語音內容：「又來了，看起來他又嘗試翻譯了」
detectedLang=es, avgLogProb=-0.513
輸出："¡Ahora! ¡Parece que él ha intentado traducir! Gracias."
```

### 案例 7：中文 → 斯洛維尼亞語
```
語音內容：「今天天氣不錯」
detectedLang=sl, avgLogProb=-0.570
輸出："Včetek je zelo dobro."
```

### 案例 8：中文 → 波斯語
```
語音內容：「我再多試幾次」
detectedLang=fa, avgLogProb=-1.789
輸出："امید وقت از این دیرسی هیچ سی دیسی"
```

### 對照：英文語音正常
```
語音內容："It's coming out again!"
detectedLang=en, avgLogProb=-0.725
輸出："It's coming out again!"（正確）
```

### 對照：葡萄牙語語音正常
```
語音內容："Vamos testar"
detectedLang=pt, avgLogProb=-0.062
輸出："Vamos testar"（正確）
```

## 觀察到的模式

1. **中文語音 → 各種語言翻譯**：模型不是轉錄，而是翻譯成偵測到的錯誤語言
2. **偵測的語言極度不穩定**：同一句「今天天氣不錯」在不同嘗試中被偵測為 ru、it、de、sl 等完全不同的語言
3. **信心度不一致**：有時錯誤偵測的信心度很低（-1.789），有時卻相對高（-0.335）
4. **強制指定 `language: "zh"` 時一切正常**：用 zh retry 後，avgLogProb 大幅提升（如 -0.714 → -0.217），輸出正確
5. **whisper.cpp 不受影響**：同一台機器、同一個 Whisper large-v3-turbo 模型的 GGML 版本，auto-detect 對中文運作正常
6. **英文和葡萄牙語語音偵測正常**：問題似乎只影響中文

## 目前的 Workaround

在 `WhisperKitTranscriptionService` 中加了 retry 邏輯：
1. 第一次用 auto-detect 轉錄
2. 如果輸出不含 CJK 字元，用 `language: "zh"` retry
3. 如果 retry 結果信心更高或包含 CJK，使用 retry 結果

這有效但不理想：每次非 CJK 輸出都會多一次推論，增加延遲。

## 想釐清的問題

1. 這是 CoreML 模型轉換時的已知問題嗎？（壓縮版 `_954MB` vs 完整版）
2. WhisperKit 的語言偵測實作和 whisper.cpp 有什麼不同，為什麼行為差異這麼大？
3. 有沒有 WhisperKit 的設定或參數可以改善中文的語言偵測準確度？
4. `openai_whisper-large-v3_turbo`（完整版）是否有相同問題？

## 相關程式碼

```swift
let options = DecodingOptions(
    task: .transcribe,
    language: nil,        // auto-detect
    temperature: 0.0,
    usePrefillPrompt: true,
    detectLanguage: true,
    wordTimestamps: false,
    promptTokens: promptTokens
)

let results = try await kit.transcribe(
    audioPath: audioURL.path,
    decodeOptions: options
)
```

## 版本資訊

- WhisperKit: 0.9+（SPM, from argmaxinc/WhisperKit）
- 模型 repo: argmaxinc/whisperkit-coreml
- 模型 variant: `openai_whisper-large-v3_turbo_954MB`
