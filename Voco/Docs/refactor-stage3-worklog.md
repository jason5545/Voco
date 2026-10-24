# 第 2 階段剩餘與第 3 階段工作紀錄

基準：fork `6f908999`、pinned upstream `22e5ec5479b712797c3b338fcb875961e58800b6`。第 0 步試合併為 99 條（`UU 62`、`UD 37`）；完整 `VoiceInkTests` 為 508 個測試、497 通過、10 失敗、1 略過，10 個唯一失敗與計畫第 6 節相同；Python 為 `Ran 74 tests ... OK`。

## 舊檔 → upstream 新檔 → 掛點

以下新檔已在動手前直接讀過 pinned upstream。掛點是目前 fork 要先集中的位置；第 4 階段才採上游完整新殼。

| 目前 fork 舊檔 | upstream `22e5ec54` 新檔與 API 形狀 | Voco 去處／保留掛點 |
|---|---|---|
| `VoiceInk/Models/TranscriptionModel.swift` | `Features/ModelLibrary/Models/TranscriptionModel.swift`；上游只有通用 model structs | 四個 Voco model structs 到 `ASR/Models.swift`；上游只留四個 provider enum cases |
| `VoiceInk/Models/TranscriptionModelRegistry.swift` | 同目錄 `TranscriptionModelRegistry.swift`；`predefinedModels + cloudModels + customModels` | Qwen／MLX／CoreML catalog 到 `ASR/VocoModelCatalog.swift`，原 registry 只 augmentation 一次 |
| `VoiceInk/Transcription/Engine/TranscriptionModelManager.swift` | `Features/ModelLibrary/State/TranscriptionModelManager.swift`；provider switch | Voco 下載狀態到 `ASR/VocoModelDownloadStatus.swift`；provider case 留單行 |
| `VoiceInk/Models/Transcription.swift` | `Infrastructure/Persistence/Models/Transcription.swift`；stored fields＋base init＋cancel | stored fields 與 init 留原檔；JSON、detailDisplayTexts、metadata、cancel 到 `Persistence/Transcription+Voco.swift` |
| `VoiceInk/Models/SessionMetric.swift` | `Infrastructure/Persistence/Models/SessionMetric.swift` | stored fields 與 init 留原檔；JSON、統計與記錄到 `Persistence/SessionMetric+Voco.swift` |
| `VoiceInk/Models/WordReplacement.swift` | `Infrastructure/Persistence/Models/WordReplacement.swift` | source／hitCount／lastSeen 與 init 留原檔；候選計算與審核到 `Persistence/WordReplacement+Voco.swift`；不套上游全部啟用語意 |
| `VoiceInk/Services/BackupTypes.swift` | `Features/Settings/Backup/BackupTypes.swift`；optional DTO fields＋CodingKeys | 必要 optional fields 留原檔；Voco 專用 DTO／計算到 `Persistence/VocoBackupTypes.swift` |
| `VoiceInk/Views/Settings/SettingsView.swift` | `Features/Settings/Views/SettingsView.swift`；Form sections | Voco sections 到 `UI/Settings/VocoSettingsSections.swift`，只留組裝與個別行為掛點 |
| `VoiceInk/EmailSupport.swift` | `Features/Settings/Support/EmailSupport.swift`；static support entry | Voco support 實作到 `UI/VocoSupport.swift`，原 entry 薄轉呼 |
| `VoiceInk/Views/Dictionary/DictionarySettingsView.swift`、`VocabularyView.swift`、`WordReplacementView.swift` | `Features/Dictionary/Views/` 同名新檔；section、vocabulary、replacement rows | Rime／學習候選到 `Personalization/` fork views；不在第 4 階段前復活或強套缺依賴的新殼 |
| `VoiceInk/Views/History/TranscriptionListItem.swift` | `Features/History/Components/TranscriptionListItem.swift`；cell 單一文字與背景 | metadata 到 `UI/VocoTranscriptionMetadataSection.swift`，cell 只留插入點 |
| `VoiceInk/Views/AI Models/ModelCardView.swift`、`ModelManagementView.swift` | `Features/ModelLibrary/Components/ModelCardView.swift`、`Views/ModelManagementView.swift`；provider dispatch 與 local filter | card factory、catalog filter／actions 到 `UI/VocoModelCardFactory.swift`、`ASR/VocoModelCatalog.swift` |
| `VoiceInk/Views/Recorder/AudioVisualizerView.swift` | `Features/Recording/Components/AudioVisualizerView.swift`；只有 transcribing／enhancing | editing presentation 到 `UI/VocoProcessingStatusDisplay.swift`，不改上游 enum |
| `VoiceInk/Views/TranscriptionResultView.swift` | `Features/AudioImport/State/TranscriptionResultView.swift`；raw-value enum | 保留 raw values，display 到 `UI/TranscriptionTab+Voco.swift` |
| `VoiceInk/Services/LastTranscriptionService.swift` | `Features/History/Workflows/LastTranscriptionService.swift`；copy／paste／retry entries | 最終文字選擇到 `Persistence/VocoTranscriptionOutput.swift`，原 entries 只改呼叫 |
| `VoiceInk/Transcription/Engine/TranscriptionServiceRegistry.swift` | `Features/Recording/Workflows/TranscriptionServiceRegistry.swift`；非 final class、`service(for:)`、`cleanup()` | `ASR/VocoTranscriptionServiceRegistry.swift` subclass 加四個 service；engine 工廠注入同一實例，prewarm 共用，不另建 registry |
| `VoiceInk/Transcription/Engine/TranscriptionService.swift` | 同目錄；`TranscriptionRequestContext(language:prompt:)` 與 `scoped(to:)` | Qwen adapter flag 到 `ASR/VocoASRRequestContext.swift`，不改上游 struct 形狀以外的通用邏輯 |
| `VoiceInk/Transcription/Engine/TranscriptionPipeline.swift` | 同目錄；ASR → enhance → delivery → save | 邏輯到 `Correction/VocoTranscriptionProcessor.swift`；只留 after-ASR、post-enhancement、pre-delivery、record 四個薄掛點 |
| `VoiceInk/Transcription/Engine/VoiceInkEngine.swift` | 同目錄；registry init、start flow、cleanup；upstream 已有 `RealtimeAudioChunkGate` | fork 狀態到 `EditMode/VocoEngineCoordinator.swift`；start flow 與 bounded gate 到獨立 Voco 檔；移除 associated objects |
| `VoiceInk/CoreAudioRecorder.swift` | `Infrastructure/Audio/CoreAudioRecorder.swift`；atomics、ring、invalidate 已存在 | 只抽仍缺的 RNNoise 到 `Audio/VocoAudioRenderPipeline.swift`；不重搬上游已有 ring／atomics |
| `VoiceInk/Paste/CursorPaster.swift` | `Infrastructure/SystemIntegration/Paste/CursorPaster.swift`；`startPasteAtCursor` 回傳 `PasteOutcome`、queued restore、directTyping | transaction、target guard、landing retry、restore chain 到 `Paste/VocoPasteService.swift`；原 public entry 薄轉呼，不提前移植 directTyping／AutoLearn |
| `VoiceInk/Paste/ClipboardManager.swift` | 同目錄 `ClipboardManager.swift`；set／copy／get | transaction coordinator、restore chain、context policy 到 `Paste/ClipboardCoordination.swift`；通用 ClipboardManager 留原檔 |
| `VoiceInk/Services/SelectedTextService.swift` | `Infrastructure/SystemIntegration/Accessibility/SelectedTextService.swift`；35 行通用 SelectedTextKit entry | AX 選取證據、Electron markers、clipboard lock 到 `Context/VocoSelectionService.swift`；原 entry 薄轉呼 |
| `VoiceInk/Services/AIEnhancement/AIEnhancementService.swift` | `Features/Enhancement/Workflows/AIEnhancementService.swift`；raw request＋`AIEnhancementResult` | context／custom request／edit／merge／disabled 到 `Correction/VocoEnhancementContext.swift`、`VocoEnhancementRequester.swift`；保留 raw request 薄掛點 |
| `VoiceInk/Transcription/Engine/TranscriptionDelivery.swift` | `Features/Recording/Workflows/TranscriptionDelivery.swift`；`Request`／`Actions` | final paste outcome 到 `Paste/VocoDelivery.swift`，只接交付結果 |

## 固定邊界

- 不在這輪實際合併 upstream，不做第 4、5 階段，不復活舊 Dashboard／History，不啟用上游 AutoLearn。
- upstream 已有 realtime gate、CoreAudio ring／atomics／invalidate，這些不作為 Voco 新功能重搬。
- `engine.serviceRegistry` 是唯一 registry；prewarm 必接同一實例。
- 所有仍在上游檔案的 hook 最後登錄到 `PATCHES.md`，含位置、用途與不能搬的原因。
- 不 push；Release 部署後停在第 3 階段，等 Jason 手動驗收。
