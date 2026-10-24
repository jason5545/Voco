# Voco 重構計畫（第 2～5 階段執行版）

本文件是自包含工作說明。沒有參與先前討論的人，應能依本文完成第 2～5 階段，而不必重查對話。第 1 階段只做機械集中，已於 2026-10-24 完成並部署；本文件不授權省略後續驗收或直接 push。每個階段仍依 Jason 的放行範圍執行。

## 1. 基線與目前已知數字

- Upstream：`22e5ec5479b712797c3b338fcb875961e58800b6`（簡寫 `22e5ec54`）。
- Merge-base：`8e31ffaacfc64d4f90805a0412e34de714d31fc9`（簡寫 `8e31ffaa`）。
- 第 1 階段起點：`7f3e8e2476aaab4205cc20fbac77effa0f235fd1`。
- 第 0 步試合併：**182 unmerged paths**：`UU 82`、`AU 58`、`UD 41`、`AA 1`。
- 第 1 階段後試合併：**121 unmerged paths**：`UU 79`、`AU 0`、`UD 41`、`AA 1`。
- 減少 61 的路徑由兩部分組成：58 個原 `AU` 的 file-location 衝突因 fork 新檔移到獨立 `Voco/` 而消失；另外 `.gitignore`、`Makefile`、`README.md` 恢復上游，再少 3 個 `UU`。沒有新增衝突路徑。
- 指揮官在 `59937eb9` 重跑也得 121；最初誤算的 124 已正式修正為 121。
- 第 1 階段最終 commit：`3d7e23db7f39c27ed56b8771b3f9b63eddaf864c`，已 push，部署版為 `2.0 (179)`。

## 2. Jason 在 2026-10-24 的固定決定

1. **先重構再合併上游，分 5 階段。** 先在目前 fork 上集中及抽離，再對 pinned upstream 合併，避免 Git 把 fork 新檔誤判到 `Features/Settings/Backup`。
2. **合併初期關閉上游 AutoLearn。** 詞庫只由 Voco correction learning 流程寫入，避免雙管線重複學習及 `isEnabled` 語意衝突；另開評估，不在重構中整併。
3. **舊 Dashboard／History 不保留。** 採上游新畫面，只移植 Voco metadata、規則助手與 navigation policy，不復活上游已刪的舊 view。
4. **Fork 程式碼集中 `Voco/` 根目錄、同一個 app target。** 不為 app-coupled code 硬拆 Swift package，避免 public access 與循環依賴；RNNoise 這類本來就獨立的 C package 是例外。
5. **Appcast URL 不變，授權集中單一 `VocoLicensePolicy`。** 不在各 Pro view 分別寫死 licensed 或改品牌。
6. **VocoiOS／VocoKeyboard 自 2026-10-24 起不驗收。** 兩者在搬移前就可能建不起來（`PredefinedPrompts` 在 HEAD 已不存在）；這輪不修，也不為它們新增 shared membership。平台檔案已在 `Voco/Platforms/`。
7. **其他分支已放棄，本機已刪。** 不考慮舊分支之後能否合回；本輪不刪除、不修改它們。

## 3. 處置縮寫與完整去處表

- **D**：刪掉過時 patch，直接採上游。
- **L**：集中到品牌／在地化，不保留邏輯 patch。
- **E**：搬成 fork extension、純函式或獨立型別。
- **H**：邏輯搬到 Voco service，上游只留薄掛點。
- **S**：只保留必要 schema、enum case、stored property 或 DTO field，其他搬出。
- **R**：在 composition root 換成 Voco-owned view／service。
- **I**：只保留最小 identity patch（bundle、logger、storage、keychain 等）。
- **U**：通用 bug fix；上游已有就刪，仍缺才保留最小修正。
- **M**：舊檔不復活，功能移植到上游新殼。
- **X**：Xcode、簽章、設定檔或 release 工具處理。

行數是 merge-base 到第 1 階段前 fork 的 `+新增/-刪除`。具體去處是第 2～5 階段的預定位置，不是說那些抽離已全部完成；只有標 **第 1 階段已完成** 的項目已經落定。

### 3A. 上游仍在同一路徑的 9 檔

| 舊路徑 | 改動行數 | 上游路徑 | 功能 | 處置 | 具體去處 |
|---|---:|---|---|---|---|

| `.gitignore` | +36/-1 | `.gitignore` | 私有資料、cache、模型 artifact、agent 筆記 | **X** | `Voco/.gitignore` 與 `.git/info/exclude`；根檔恢復上游；**第 1 階段已完成** |
| `Makefile` | +1/-1 | `Makefile` | provisioning flags | **X** | `Voco/Scripts/build.sh`；根 Makefile 恢復上游；**第 1 階段已完成** |
| `README.md` | +124/-70 | `README.md` | Voco 品牌與使用說明 | **X** | `Voco/Docs/README.md`；根 README 恢復上游；**第 1 階段已完成** |
| `VoiceInk.xcodeproj/project.pbxproj` | +514/-61 | `VoiceInk.xcodeproj/project.pbxproj` | targets、套件、簽章、bundle IDs、同步群組 | **X** | `Voco/Config/` 與唯一 Xcode 接線點；必要 target membership 保留 |
| `VoiceInk.xcodeproj/project.xcworkspace/xcshareddata/swiftpm/Package.resolved` | +36/-9 | `VoiceInk.xcodeproj/project.xcworkspace/xcshareddata/swiftpm/Package.resolved` | MLX、OpenCC、Numerics 與套件版本 | **X** | 合併後由 Xcode 重新解析，不手工保留舊 pin |
| `VoiceInk.xcodeproj/xcshareddata/xcschemes/VoiceInk.xcscheme` | +3/-3 | `VoiceInk.xcodeproj/xcshareddata/xcschemes/VoiceInk.xcscheme` | Debug tests、停用平行測試 | **X** | `Voco/Config/` 的 fork 專用 test scheme；主 scheme 還原上游 |
| `VoiceInk/Info.plist` | +5/-5 | `VoiceInk/Info.plist` | Sparkle feed／公鑰、品牌權限文案 | **X** | 文案移到 `Voco/Localization/`；feed 與公鑰保留最小明示 patch，URL 不變 |
| `VoiceInk/VoiceInk.entitlements` | +6/-2 | `VoiceInk/VoiceInk.entitlements` | App Group、iCloud、keychain group | **X** | `Voco/Config/` 的完整 fork entitlements，pbxproj 只指向它 |
| `appcast.xml` | +6/-3 | `appcast.xml` | Voco 更新 feed | **X** | `Voco/Scripts/` release 工具管理；現有 URL 不動，placeholder 不得發佈 |

### 3B. 上游已改名的 106 檔

| 舊路徑 | 改動行數 | 上游新路徑／替代位置 | 功能 | 處置 | 具體去處 |
|---|---:|---|---|---|---|
| `VoiceInk/AppDelegate.swift` | +13/-0 | `VoiceInk/App/Lifecycle/AppDelegate.swift` | VoiceInk→Voco／XVoice migration | **H** | `Voco/Sources/Bootstrap/VocoBootstrap.swift`；上游 AppDelegate 只留一次啟動呼叫 |
| `VoiceInk/AppIntents/DismissMiniRecorderIntent.swift` | +3/-3 | `VoiceInk/App/AppIntents/DismissMiniRecorderIntent.swift` | App Intent 品牌 | **L** | `Voco/Localization/`：集中英文與 zh-Hant overrides，產生合併 catalog，不改上游邏輯 |
| `VoiceInk/AppIntents/ToggleMiniRecorderIntent.swift` | +5/-5 | `VoiceInk/App/AppIntents/ToggleMiniRecorderIntent.swift` | App Intent 品牌與錯誤文案 | **L** | `Voco/Localization/`：集中英文與 zh-Hant overrides，產生合併 catalog，不改上游邏輯 |
| `VoiceInk/CoreAudioRecorder.swift` | +363/-100 | `VoiceInk/Infrastructure/Audio/CoreAudioRecorder.swift` | RNNoise、ring buffer、非阻塞 render、wake | **H** | `Voco/Sources/Audio/VocoAudioRenderPipeline.swift`；只接仍缺的 RNNoise／ring，保留上游 atomics／invalidate |
| `VoiceInk/EmailSupport.swift` | +12/-17 | `VoiceInk/Features/Settings/Support/EmailSupport.swift` | Voco support 地址與 mailto | **R** | `Voco/Sources/UI/VocoSupport.swift`；上游只替換入口 |
| `VoiceInk/Models/AudioFileQueueItem.swift` | +14/-4 | `VoiceInk/Features/AudioImport/State/AudioFileQueueItem.swift` | processing phase 在地化 | **E/L** | `Voco/Sources/UI/QueueItemStatus+Voco.swift`；保留原 raw values，extension 加 display |
| `VoiceInk/Models/LanguageDictionary.swift` | +8/-1 | `VoiceInk/Features/ModelLibrary/Models/LanguageDictionary.swift` | 預設語言 auto | **E** | `Voco/Sources/ASR/VocoLanguageDefaults.swift`；computed defaults，不改上游語言表 |
| `VoiceInk/Models/SessionMetric.swift` | +324/-1 | `VoiceInk/Infrastructure/Persistence/Models/SessionMetric.swift` | correction／paste metadata | **S** | stored fields 留上游；JSON／計算／記錄邏輯到 `Voco/Sources/Persistence/SessionMetric+Voco.swift` |
| `VoiceInk/Models/Transcription.swift` | +558/-6 | `VoiceInk/Infrastructure/Persistence/Models/Transcription.swift` | ASR、candidate、feedback、adapter metadata | **S** | stored fields 留上游；其他到 `Voco/Sources/Persistence/Transcription+Voco.swift` |
| `VoiceInk/Models/TranscriptionModel.swift` | +81/-2 | `VoiceInk/Features/ModelLibrary/Models/TranscriptionModel.swift` | 4 個 provider cases 與 model structs | **S** | enum cases 留上游；structs 到 `Voco/Sources/ASR/Models.swift` |
| `VoiceInk/Models/TranscriptionModelRegistry.swift` | +141/-3 | `VoiceInk/Features/ModelLibrary/Models/TranscriptionModelRegistry.swift` | Qwen／MLX catalog | **E/H** | `Voco/Sources/ASR/VocoModelCatalog.swift`；上游只 append 一次 |
| `VoiceInk/Models/WordReplacement.swift` | +44/-1 | `VoiceInk/Infrastructure/Persistence/Models/WordReplacement.swift` | learning source／hitCount／lastSeen | **S/E** | stored fields 留上游；computed 到 `Voco/Sources/Persistence/WordReplacement+Voco.swift`；AutoLearn 初期關閉 |
| `VoiceInk/Modes/BrowserURLService.swift` | +18/-18 | `VoiceInk/Infrastructure/SystemIntegration/Context/BrowserURLService.swift` | 只有空白差異 | **D** | 直接採上游新版，刪除 fork patch，不另建 wrapper |
| `VoiceInk/Modes/ModeConfig.swift` | +7/-7 | `VoiceInk/Features/Modes/State/ModeConfig.swift` | 在地化、預設 auto | **L** | `Voco/Localization/`；auto 預設集中到 `Voco/Sources/ASR/VocoLanguageDefaults.swift` |
| `VoiceInk/Modes/ModeRuntimeConfiguration.swift` | +1/-1 | `VoiceInk/Features/Modes/State/ModeRuntimeConfiguration.swift` | `mode.icon.value`；採上游新 icon API | **D** | 直接採上游新版，刪除 fork patch，不另建 wrapper |
| `VoiceInk/Modes/ModeSettingsPanelView.swift` | +27/-6 | `VoiceInk/Features/Modes/Views/ModeSettingsPanelView.swift` | mode count 文案 | **L** | `Voco/Localization/`：集中英文與 zh-Hant overrides，產生合併 catalog，不改上游邏輯 |
| `VoiceInk/Modes/ModeTriggerSection.swift` | +1/-1 | `VoiceInk/Features/Modes/Views/ModeTriggerSection.swift` | trigger 說明品牌 | **L** | `Voco/Localization/`：集中英文與 zh-Hant overrides，產生合併 catalog，不改上游邏輯 |
| `VoiceInk/Modes/StarterModeTemplate.swift` | +10/-10 | `VoiceInk/Features/Modes/Templates/StarterModeTemplate.swift` | starter mode 在地化與品牌 | **L** | `Voco/Localization/`：集中英文與 zh-Hant overrides，產生合併 catalog，不改上游邏輯 |
| `VoiceInk/Notifications/AppNotificationView.swift` | +10/-1 | `VoiceInk/App/Notifications/Views/AppNotificationView.swift` | action button init | **R** | `Voco/Sources/UI/VocoActionNotificationView.swift`，不改上游 view |
| `VoiceInk/Notifications/NotificationManager.swift` | +64/-0 | `VoiceInk/App/Notifications/NotificationManager.swift` | action notification panel | **R/H** | `Voco/Sources/UI/VocoActionNotificationPresenter.swift`；上游只留可選入口 |
| `VoiceInk/Paste/ClipboardManager.swift` | +124/-0 | `VoiceInk/Infrastructure/SystemIntegration/Paste/ClipboardManager.swift` | pasteboard transaction、restore chain、context policy | **E** | `Voco/Sources/Paste/ClipboardManager+Voco.swift` 與 `ClipboardCoordination.swift` |
| `VoiceInk/Paste/PasteMethod.swift` | +2/-2 | `VoiceInk/Infrastructure/SystemIntegration/Paste/PasteMethod.swift` | paste method 在地化 | **L** | `Voco/Localization/`：集中英文與 zh-Hant overrides，產生合併 catalog，不改上游邏輯 |
| `VoiceInk/Services/AIEnhancement/AIEnhancementService.swift` | +116/-20 | `VoiceInk/Features/Enhancement/Workflows/AIEnhancementService.swift` | context、custom request、edit／merge、disabled | **H** | `Voco/Sources/Correction/VocoEnhancementContext.swift` 與 `VocoEnhancementRequester.swift`；上游只留 raw request 掛點 |
| `VoiceInk/Services/AIEnhancement/AIService.swift` | +2/-2 | `VoiceInk/Features/Enhancement/State/AIService.swift` | Soniox v5；上游已具備 | **D** | 直接採上游新版，刪除 fork patch，不另建 wrapper |
| `VoiceInk/Services/APIKeyManager.swift` | +1/-1 | `VoiceInk/Infrastructure/Credentials/APIKeyManager.swift` | logger subsystem | **I** | 只留一行 `AppIdentifiers.subsystem` identity patch |
| `VoiceInk/Services/AudioFileTranscriptionManager.swift` | +60/-18 | `VoiceInk/Features/AudioImport/Workflows/AudioFileTranscriptionManager.swift` | transcode、canonicalization、style guard | **H** | `Voco/Sources/Correction/VocoTranscriptionWorkflow.swift`，與 recorder 共用 |
| `VoiceInk/Services/AudioFileTranscriptionService.swift` | +84/-12 | `VoiceInk/Features/History/Workflows/AudioFileTranscriptionService.swift` | retranscribe、feedback、style guard | **H** | `Voco/Sources/Correction/VocoTranscriptionWorkflow.swift` |
| `VoiceInk/Services/BackupImporter.swift` | +55/-4 | `VoiceInk/Features/Settings/Backup/BackupImporter.swift` | Voco 設定與詞庫匯入 | **H** | `Voco/Sources/Persistence/VocoBackupImporter.swift`；上游只接匯入前後 hook |
| `VoiceInk/Services/BackupTypes.swift` | +69/-7 | `VoiceInk/Features/Settings/Backup/BackupTypes.swift` | backup DTO、learning metadata | **S/E** | optional fields／CodingKeys 留上游；其他型別到 `Voco/Sources/Persistence/VocoBackupTypes.swift` |
| `VoiceInk/Services/CustomVocabularyService.swift` | +44/-5 | `VoiceInk/Features/Dictionary/Workflows/CustomVocabularyService.swift` | 預設詞彙與去重 | **E/H** | `Voco/Sources/Personalization/VocoVocabularyProvider.swift`＋`Voco/Sources/Resources/` 預設詞表 |
| `VoiceInk/Services/ImportExportService.swift` | +66/-23 | `VoiceInk/Features/Settings/Backup/ImportExportService.swift` | Voco 匯出／匯入 | **H** | `Voco/Sources/Persistence/VocoBackupService.swift`；上游前後各一個 augmentation hook |
| `VoiceInk/Services/LastTranscriptionService.swift` | +86/-22 | `VoiceInk/Features/History/Workflows/LastTranscriptionService.swift` | 最終輸出選擇與重用 | **E/H** | `Voco/Sources/Persistence/VocoTranscriptionOutput.swift` 純函式，上游只改呼叫 |
| `VoiceInk/Services/ModelPrewarmService.swift` | +182/-33 | `VoiceInk/App/Lifecycle/ModelPrewarmService.swift` | shared registry、keep-alive、battery | **R** | `Voco/Sources/ASR/VocoModelResidencyService.swift`，必接 `engine.serviceRegistry` |
| `VoiceInk/Services/RecordingContextSnapshot.swift` | +8/-3 | `VoiceInk/Features/Recording/Context/RecordingContextSnapshot.swift` | clipboard／selected-text 注入 | **H** | `Voco/Sources/Context/VocoRecordingContextCapture.swift` |
| `VoiceInk/Services/SelectedTextService.swift` | +410/-10 | `VoiceInk/Infrastructure/SystemIntegration/Accessibility/SelectedTextService.swift` | AX 選取證據、Electron markers、clipboard lock | **H** | `Voco/Sources/Context/VocoSelectionService.swift`；原 entry 薄轉呼 |
| `VoiceInk/Services/SessionMetricRecorder.swift` | +94/-1 | `VoiceInk/Infrastructure/Persistence/Statistics/SessionMetricRecorder.swift` | correction metrics | **E/H** | `Voco/Sources/Persistence/VocoSessionMetricRecorder.swift`；上游記錄後補 Voco metadata |
| `VoiceInk/Services/StreamingKeysMigration.swift` | +60/-37 | `VoiceInk/App/Migrations/StreamingKeysMigration.swift` | Soniox v5 與 mode JSON remap | **H** | `Voco/Sources/Bootstrap/VocoStreamingKeysMigration.swift`；上游 migration 後呼叫 |
| `VoiceInk/Services/VoiceInkCSVExportService.swift` | +258/-22 | `VoiceInk/Features/History/Export/VoiceInkCSVExportService.swift` | correction CSV | **R** | `Voco/Sources/Persistence/VocoCSVExportService.swift`；上游 export 入口改用它 |
| `VoiceInk/Shortcuts/ModeShortcutManager.swift` | +1/-1 | `VoiceInk/Features/Shortcuts/Coordination/ModeShortcutManager.swift` | 只有空白差異 | **D** | 直接採上游新版，刪除 fork patch，不另建 wrapper |
| `VoiceInk/Shortcuts/RecorderPanelShortcutManager.swift` | +46/-9 | `VoiceInk/Features/Shortcuts/Coordination/RecorderPanelShortcutManager.swift` | double-Escape、提示狀態 | **H** | `Voco/Sources/EditMode/VocoEscapeShortcutController.swift`；上游只接事件 |
| `VoiceInk/Shortcuts/RecordingShortcutManager.swift` | +28/-23 | `VoiceInk/Features/Shortcuts/Coordination/RecordingShortcutManager.swift` | cooldown、觸發防護、在地化 | **H/L** | `Voco/Sources/EditMode/VocoShortcutTiming.swift`；文案到 `Voco/Localization/` |
| `VoiceInk/Shortcuts/ShortcutRecorder.swift` | +7/-1 | `VoiceInk/Features/Shortcuts/Views/ShortcutRecorder.swift` | cancel action 可用 Escape | **H** | `Voco/Sources/EditMode/VocoShortcutCapturePolicy.swift`；上游只留一個 branch |
| `VoiceInk/Shortcuts/ShortcutValidator.swift` | +17/-8 | `VoiceInk/Features/Shortcuts/State/ShortcutValidator.swift` | action-aware validation | **H/L** | `Voco/Sources/EditMode/VocoShortcutValidator.swift`；錯誤文案到 `Voco/Localization/` |
| `VoiceInk/SoundPlaybackEngine.swift` | +33/-4 | `VoiceInk/Infrastructure/Audio/SoundPlaybackEngine.swift` | start sound completion | **E/H** | `Voco/Sources/Audio/VocoSoundPlayback.swift`＋上游單一 completion hook |
| `VoiceInk/Transcription/Cloud/CustomCloudModelManager.swift` | +1/-1 | `VoiceInk/Features/ModelLibrary/State/CustomCloudModelManager.swift` | logger subsystem | **I** | 只留一行 identity patch |
| `VoiceInk/Transcription/Cloud/SonioxProvider.swift` | +3/-3 | `VoiceInk/Infrastructure/Providers/Transcription/Cloud/SonioxProvider.swift` | Soniox v5；上游已具備 | **D** | 直接採上游新版，刪除 fork patch，不另建 wrapper |
| `VoiceInk/Transcription/Engine/AudioFileProcessor.swift` | +503/-3 | `VoiceInk/Infrastructure/Audio/AudioFileProcessor.swift` | bounded transcode、asset reader、metadata policy | **R** | `Voco/Sources/Audio/VocoAudioFileProcessor.swift`；由 import manager 注入 |
| `VoiceInk/Transcription/Engine/RecorderUIManager.swift` | +348/-7 | `VoiceInk/Features/Recording/Presentation/RecorderUIManager.swift` | Edit Mode、panel lifecycle、double-stop | **H** | `Voco/Sources/EditMode/VocoRecorderUICoordinator.swift`；上游只留 start／dismiss／cancel hooks |
| `VoiceInk/Transcription/Engine/TranscriptionModelManager.swift` | +21/-0 | `VoiceInk/Features/ModelLibrary/State/TranscriptionModelManager.swift` | 新引擎下載狀態 | **E/H** | `Voco/Sources/ASR/VocoModelDownloadStatus.swift` |
| `VoiceInk/Transcription/Engine/TranscriptionPipeline.swift` | +303/-16 | `VoiceInk/Features/Recording/Workflows/TranscriptionPipeline.swift` | 中文後處理、canonicalization、Edit Mode、shadow、delivery | **H** | `Voco/Sources/Correction/VocoTranscriptionProcessor.swift`；上游只留 after-ASR／post-enhancement／pre-delivery／record 四個薄掛點 |
| `VoiceInk/Transcription/Engine/TranscriptionService.swift` | +17/-1 | `VoiceInk/Features/Recording/Workflows/TranscriptionService.swift` | Qwen adapter flag | **H** | `Voco/Sources/ASR/VocoASRRequestContext.swift` 包上游 context，不改上游 struct |
| `VoiceInk/Transcription/Engine/TranscriptionServiceRegistry.swift` | +16/-0 | `VoiceInk/Features/Recording/Workflows/TranscriptionServiceRegistry.swift` | 新引擎 registry、cleanup | **R** | `Voco/Sources/ASR/VocoTranscriptionServiceRegistry.swift` subclass，一次工廠注入並共享 prewarm |
| `VoiceInk/Transcription/Engine/TranscriptionSession.swift` | +1/-1 | `VoiceInk/Features/Recording/Workflows/TranscriptionSession.swift` | logger subsystem | **I** | 只留一行 identity patch |
| `VoiceInk/Transcription/Engine/VoiceInkEngine.swift` | +368/-143 | `VoiceInk/Features/Recording/Workflows/VoiceInkEngine.swift` | recording start flow、realtime gate、Edit Mode、cleanup | **H** | `Voco/Sources/EditMode/VocoEngineCoordinator.swift`、`VocoRecordingStartFlow.swift`、`RealtimeAudioChunkGate.swift`；去掉 associated-object 狀態 |
| `VoiceInk/Transcription/Streaming/FluidAudioStreamingProvider.swift` | +1/-1 | `VoiceInk/Infrastructure/Providers/Transcription/FluidAudio/Streaming/FluidAudioStreamingProvider.swift` | Int16 endian／clipping 通用修正 | **U** | 保留最小通用 bug fix，不進 Voco 架構；上游已有則刪，否則保留最小通用修正 |
| `VoiceInk/Transcription/Streaming/SonioxStreamingProvider.swift` | +1/-1 | `VoiceInk/Infrastructure/Providers/Transcription/Cloud/Streaming/SonioxStreamingProvider.swift` | Soniox rt-v5；上游已具備 | **D** | 直接採上游新版，刪除 fork patch，不另建 wrapper |
| `VoiceInk/Transcription/Streaming/StreamingTranscriptionService.swift` | +42/-9 | `VoiceInk/Features/Recording/Streaming/StreamingTranscriptionService.swift` | bounded chunks／drop metrics | **H** | `Voco/Sources/Audio/VocoBoundedAudioChunkSource.swift` |
| `VoiceInk/Transcription/Whisper/LibWhisper.swift` | +51/-10 | `VoiceInk/Infrastructure/Providers/Transcription/Whisper/LibWhisper.swift` | VAD option、confidence | **H/E** | `Voco/Sources/ASR/WhisperContext+Voco.swift`；上游只留 context 存取掛點 |
| `VoiceInk/Transcription/Whisper/WhisperPrompt.swift` | +1/-1 | `VoiceInk/Infrastructure/Providers/Transcription/Whisper/WhisperPrompt.swift` | 預設語言 | **E** | 改用 `Voco/Sources/ASR/VocoLanguageDefaults.swift` |
| `VoiceInk/Transcription/Whisper/WhisperTranscriptionService.swift` | +1/-8 | `VoiceInk/Infrastructure/Providers/Transcription/Whisper/WhisperTranscriptionService.swift` | 共用 WAV reader | **U** | `Voco/Sources/Audio/VocoWAVSampleReader.swift`；上游已有同等修正則刪；上游已有則刪，否則保留最小通用修正 |
| `VoiceInk/Views/AI Models/APIKeyManagementView.swift` | +3/-3 | `VoiceInk/Features/ModelLibrary/Views/APIKeyManagementView.swift` | 品牌／驗證錯誤文案 | **L** | `Voco/Localization/`：集中英文與 zh-Hant overrides，產生合併 catalog，不改上游邏輯 |
| `VoiceInk/Views/AI Models/CustomModelCardView.swift` | +2/-2 | `VoiceInk/Features/ModelLibrary/Components/CustomModelCardView.swift` | model card 在地化 | **L** | `Voco/Localization/`：集中英文與 zh-Hant overrides，產生合併 catalog，不改上游邏輯 |
| `VoiceInk/Views/AI Models/CustomProviderManagementView.swift` | +42/-36 | `VoiceInk/Features/ModelLibrary/Views/CustomProviderManagementView.swift` | provider 文案與驗證訊息 | **L** | `Voco/Localization/`：集中英文與 zh-Hant overrides，產生合併 catalog，不改上游邏輯 |
| `VoiceInk/Views/AI Models/FluidAudioModelCardView.swift` | +2/-2 | `VoiceInk/Features/ModelLibrary/Components/FluidAudioModelCardView.swift` | model card 在地化 | **L** | `Voco/Localization/`：集中英文與 zh-Hant overrides，產生合併 catalog，不改上游邏輯 |
| `VoiceInk/Views/AI Models/LanguageSelectionView.swift` | +2/-2 | `VoiceInk/Features/ModelLibrary/Components/LanguageSelectionView.swift` | 預設 auto | **E/L** | `Voco/Sources/ASR/VocoLanguageDefaults.swift`；文案到 `Voco/Localization/` |
| `VoiceInk/Views/AI Models/ModelCardView.swift` | +37/-0 | `VoiceInk/Features/ModelLibrary/Components/ModelCardView.swift` | 新引擎 card dispatch | **H** | `Voco/Sources/UI/VocoModelCardFactory.swift`；上游一個 default hook |
| `VoiceInk/Views/AI Models/ModelManagementView.swift` | +22/-10 | `VoiceInk/Features/ModelLibrary/Views/ModelManagementView.swift` | 新引擎過濾、傳參、在地化 | **H/L** | `Voco/Sources/ASR/VocoModelCatalog.swift`＋`Voco/Sources/UI/VocoModelManagement.swift`；文案到 `Voco/Localization/` |
| `VoiceInk/Views/AI Models/NativeModelCardView.swift` | +2/-2 | `VoiceInk/Features/ModelLibrary/Components/NativeModelCardView.swift` | model card 在地化 | **L** | `Voco/Localization/`：集中英文與 zh-Hant overrides，產生合併 catalog，不改上游邏輯 |
| `VoiceInk/Views/AI Models/ProviderCloudManagementView.swift` | +8/-6 | `VoiceInk/Features/ModelLibrary/Views/ProviderCloudManagementView.swift` | cloud provider 文案 | **L** | `Voco/Localization/`：集中英文與 zh-Hant overrides，產生合併 catalog，不改上游邏輯 |
| `VoiceInk/Views/AI Models/ProviderDetailPanel.swift` | +9/-7 | `VoiceInk/Features/ModelLibrary/Views/ProviderDetailPanel.swift` | 模型數量／驗證文案 | **L** | `Voco/Localization/`：集中英文與 zh-Hant overrides，產生合併 catalog，不改上游邏輯 |
| `VoiceInk/Views/AI Models/ProviderLocalManagementView.swift` | +20/-14 | `VoiceInk/Features/ModelLibrary/Views/ProviderLocalManagementView.swift` | local provider 文案 | **L** | `Voco/Localization/`：集中英文與 zh-Hant overrides，產生合併 catalog，不改上游邏輯 |
| `VoiceInk/Views/AI Models/WhisperModelCardView.swift` | +3/-3 | `VoiceInk/Features/ModelLibrary/Components/WhisperModelCardView.swift` | model card 在地化 | **L** | `Voco/Localization/`：集中英文與 zh-Hant overrides，產生合併 catalog，不改上游邏輯 |
| `VoiceInk/Views/AudioFileRow.swift` | +2/-2 | `VoiceInk/Features/AudioImport/Views/AudioFileRow.swift` | phase／tab 在地化 | **L** | `Voco/Localization/`：集中英文與 zh-Hant overrides，產生合併 catalog，不改上游邏輯 |
| `VoiceInk/Views/AudioPlayerView.swift` | +7/-0 | `VoiceInk/Features/History/Components/AudioPlayerView.swift` | Fix with AI 按鈕 | **H** | `Voco/Sources/RuleAssistant/UI/VocoRuleAssistantButton.swift`；上游單一插入點 |
| `VoiceInk/Views/Common/AppControls.swift` | +5/-5 | `VoiceInk/DesignSystem/Components/AppControls.swift` | LocalizedStringKey；上游多已具備 | **L** | `Voco/Localization/`：集中英文與 zh-Hant overrides，產生合併 catalog，不改上游邏輯 |
| `VoiceInk/Views/Common/CompactHeroSection.swift` | +2/-2 | `VoiceInk/DesignSystem/Components/CompactHeroSection.swift` | LocalizedStringKey | **L** | `Voco/Localization/`：集中英文與 zh-Hant overrides，產生合併 catalog，不改上游邏輯 |
| `VoiceInk/Views/Components/ExpandableSettingsRow.swift` | +5/-5 | `VoiceInk/DesignSystem/Components/ExpandableSettingsRow.swift` | LocalizedStringKey | **L** | `Voco/Localization/`：集中英文與 zh-Hant overrides，產生合併 catalog，不改上游邏輯 |
| `VoiceInk/Views/Components/InfoTip.swift` | +4/-4 | `VoiceInk/DesignSystem/Components/InfoTip.swift` | LocalizedStringKey | **L** | `Voco/Localization/`：集中英文與 zh-Hant overrides，產生合併 catalog，不改上游邏輯 |
| `VoiceInk/Views/ContentView.swift` | +0/-3 | `VoiceInk/App/Navigation/ContentView.swift` | 隱藏 license destination | **R** | `Voco/Sources/UI/VocoNavigationPolicy.swift`；不刪上游 enum case |
| `VoiceInk/Views/Dictionary/DictionaryQuickAddPanel.swift` | +1/-1 | `VoiceInk/Features/Dictionary/QuickAdd/DictionaryQuickAddPanel.swift` | 品牌 | **L** | `Voco/Localization/`：集中英文與 zh-Hant overrides，產生合併 catalog，不改上游邏輯 |
| `VoiceInk/Views/Dictionary/DictionarySettingsView.swift` | +37/-10 | `VoiceInk/Features/Dictionary/Views/DictionarySettingsView.swift` | Rime／Voco sections | **H** | `Voco/Sources/Personalization/VocoDictionarySettingsSection.swift`；上游只插入一次 |
| `VoiceInk/Views/Dictionary/VocabularyView.swift` | +273/-0 | `VoiceInk/Features/Dictionary/Views/VocabularyView.swift` | Rime import UI | **H** | `Voco/Sources/Personalization/VocoRimeImportSection.swift`；上游只插入一次 |
| `VoiceInk/Views/Dictionary/WordReplacementView.swift` | +104/-17 | `VoiceInk/Features/Dictionary/Views/WordReplacementView.swift` | learning candidates、審核、排序 | **H** | `Voco/Sources/Personalization/VocoLearningCandidatesSection.swift` |
| `VoiceInk/Views/History/TranscriptionListItem.swift` | +343/-1 | `VoiceInk/Features/History/Components/TranscriptionListItem.swift` | correction metadata | **H** | `Voco/Sources/UI/VocoTranscriptionMetadataSection.swift`；上游 cell 只插入一次 |
| `VoiceInk/Views/LicenseManagementView.swift` | +4/-4 | `VoiceInk/Features/Licensing/Views/LicenseManagementView.swift` | 品牌；銷售 UI 由 license policy 隱藏 | **L** | `Voco/Localization/`：集中英文與 zh-Hant overrides，產生合併 catalog，不改上游邏輯 |
| `VoiceInk/Views/MenuBarView.swift` | +128/-102 | `VoiceInk/App/MenuBar/MenuBarView.swift` | 直接選模型／provider／語言、async launch-at-login | **R** | `Voco/Sources/UI/VocoMenuBarView.swift`；composition root 替換 |
| `VoiceInk/Views/Onboarding/AIProviderVerificationCard.swift` | +1/-1 | `VoiceInk/Features/Onboarding/Components/AIProviderVerificationCard.swift` | 品牌 | **L** | `Voco/Localization/`：集中英文與 zh-Hant overrides，產生合併 catalog，不改上游邏輯 |
| `VoiceInk/Views/Onboarding/OnboardingContextAwarenessScreen.swift` | +6/-6 | `VoiceInk/Features/Onboarding/Views/OnboardingContextAwarenessScreen.swift` | 品牌 | **L** | `Voco/Localization/`：集中英文與 zh-Hant overrides，產生合併 catalog，不改上游邏輯 |
| `VoiceInk/Views/Onboarding/OnboardingExperienceModels.swift` | +4/-4 | `VoiceInk/Features/Onboarding/Models/OnboardingExperienceModels.swift` | 品牌 | **L** | `Voco/Localization/`：集中英文與 zh-Hant overrides，產生合併 catalog，不改上游邏輯 |
| `VoiceInk/Views/Onboarding/OnboardingLicenseCards.swift` | +2/-2 | `VoiceInk/Features/Onboarding/Components/OnboardingLicenseCards.swift` | 品牌 | **L** | `Voco/Localization/`：集中英文與 zh-Hant overrides，產生合併 catalog，不改上游邏輯 |
| `VoiceInk/Views/Onboarding/OnboardingLicenseScreen.swift` | +1/-1 | `VoiceInk/Features/Onboarding/Views/OnboardingLicenseScreen.swift` | 品牌 | **L** | `Voco/Localization/`：集中英文與 zh-Hant overrides，產生合併 catalog，不改上游邏輯 |
| `VoiceInk/Views/Onboarding/OnboardingPermissionModels.swift` | +14/-14 | `VoiceInk/Features/Onboarding/State/OnboardingPermissionModels.swift` | 在地化 | **L** | `Voco/Localization/`：集中英文與 zh-Hant overrides，產生合併 catalog，不改上游邏輯 |
| `VoiceInk/Views/Onboarding/OnboardingTrustScreen.swift` | +3/-3 | `VoiceInk/Features/Onboarding/Views/OnboardingTrustScreen.swift` | 品牌 | **L** | `Voco/Localization/`：集中英文與 zh-Hant overrides，產生合併 catalog，不改上游邏輯 |
| `VoiceInk/Views/Onboarding/PermissionStepRow.swift` | +1/-1 | `VoiceInk/Features/Onboarding/Components/PermissionStepRow.swift` | 品牌 | **L** | `Voco/Localization/`：集中英文與 zh-Hant overrides，產生合併 catalog，不改上游邏輯 |
| `VoiceInk/Views/Onboarding/TranscriptionModelDownloadCard.swift` | +2/-2 | `VoiceInk/Features/Onboarding/Components/TranscriptionModelDownloadCard.swift` | 在地化 | **L** | `Voco/Localization/`：集中英文與 zh-Hant overrides，產生合併 catalog，不改上游邏輯 |
| `VoiceInk/Views/Recorder/AudioVisualizerView.swift` | +4/-1 | `VoiceInk/Features/Recording/Components/AudioVisualizerView.swift` | editing 狀態 | **R** | `Voco/Sources/UI/VocoProcessingStatusDisplay.swift`；不改上游 enum |
| `VoiceInk/Views/Recorder/MiniRecorderPanel.swift` | +10/-7 | `VoiceInk/Features/Recording/Presentation/MiniRecorderPanel.swift` | screen resolver、show result | **U** | 優先採上游 `RecorderScreenResolver`；只保留仍缺的通用修正；上游已有則刪，否則保留最小通用修正 |
| `VoiceInk/Views/Recorder/MiniRecorderView.swift` | +22/-4 | `VoiceInk/Features/Recording/Views/MiniRecorderView.swift` | dictionary confirmation、Edit Mode 樣式 | **H/R** | `Voco/Sources/EditMode/VocoMiniRecorderView.swift` 或上游單一 supplementary slot |
| `VoiceInk/Views/Recorder/NotchRecorderPanel.swift` | +14/-2 | `VoiceInk/Features/Recording/Presentation/NotchRecorderPanel.swift` | screen resolver | **U** | 優先採上游，只保留仍缺的通用修正；上游已有則刪，否則保留最小通用修正 |
| `VoiceInk/Views/Recorder/NotchRecorderView.swift` | +29/-11 | `VoiceInk/Features/Recording/Views/NotchRecorderView.swift` | confirmation panel | **H/R** | `Voco/Sources/EditMode/VocoNotchRecorderView.swift` 或單一 supplementary slot |
| `VoiceInk/Views/Recorder/RecorderComponents.swift` | +98/-6 | `VoiceInk/Features/Recording/Components/RecorderComponents.swift` | Edit Mode 樣式、copy、dictionary view | **E/R** | `Voco/Sources/EditMode/DictionaryConfirmationView.swift`、`RecorderSupplementaryPresentation.swift`；上游只留 supplementary slot |
| `VoiceInk/Views/Recorder/RecorderStateProvider.swift` | +5/-0 | `VoiceInk/Features/Recording/State/RecorderStateProvider.swift` | pending entry／Edit Mode protocol | **R** | `Voco/Sources/EditMode/VocoRecorderStateProviding.swift`；不改上游 protocol |
| `VoiceInk/Views/Settings/AudioSetupView.swift` | +2/-2 | `VoiceInk/Features/AudioSettings/Views/AudioSetupView.swift` | 品牌、modelUID；上游已有 modelUID | **D** | 採上游實作；品牌文案到 `Voco/Localization/` |
| `VoiceInk/Views/Settings/SettingsView.swift` | +21/-15 | `VoiceInk/Features/Settings/Views/SettingsView.swift` | Voco settings、shortcut、clipboard、launch-at-login | **H** | `Voco/Sources/UI/Settings/VocoSettingsSections.swift`＋個別行為小掛點 |
| `VoiceInk/Views/Sidebar/AppSidebar.swift` | +4/-17 | `VoiceInk/App/Navigation/AppSidebar.swift` | 隱藏 license、在地化 | **R/L** | `Voco/Sources/UI/VocoNavigationPolicy.swift`；刪掉空 `.onAppear` patch，文案到 `Voco/Localization/` |
| `VoiceInk/Views/TranscriptionResultView.swift` | +10/-4 | `VoiceInk/Features/AudioImport/State/TranscriptionResultView.swift` | tab 在地化 | **E/L** | `Voco/Sources/UI/TranscriptionTab+Voco.swift`；保留 raw values |

### 3C. 上游已刪除的 41 檔

| 舊路徑 | 改動行數 | 上游新路徑／替代位置 | 功能 | 處置 | 具體去處 |
|---|---:|---|---|---|---|
| `VoiceInk/AppDefaults.swift` | +89/-53 | `VoiceInk/App/Configuration/AppDefaults.swift` | Voco／上游預設值合併 | **H** | `Voco/Sources/Bootstrap/VocoDefaults.swift`：`VocoBootstrap.registerDefaults()` 包上游＋fork defaults，原檔恢復 |
| `VoiceInk/Models/AIPrompts.swift` | +140/-3 | `VoiceInk/Core/Enhancement/AIPrompts.swift` | 品牌、中文、merge prompts | **R** | `Voco/Sources/Correction/VocoPrompts.swift`；全部 prompt 集中，不改上游 AIPrompts |
| `VoiceInk/Models/LicenseViewModel.swift` | +2/-6 | `VoiceInk/Features/Licensing/State/LicenseViewModel.swift` | 個人版可用、隱藏銷售 | **I/R** | `Voco/Sources/Bootstrap/VocoLicensePolicy.swift` 單一政策，不多點寫死 licensed |
| `VoiceInk/Modes/ModeTriggerHelpers.swift` | +8/-6 | `VoiceInk/Features/Modes/Models/ModeTriggerHelpers.swift` | trigger count 文案 | **L** | `Voco/Localization/`；採上游新檔 `VoiceInk/Features/Modes/Models/ModeTriggerHelpers.swift` |
| `VoiceInk/Modes/ModeView.swift` | +2/-2 | `VoiceInk/Features/Modes/Views/ModeView.swift` | 品牌 | **L** | `Voco/Localization/`；採上游新檔 `VoiceInk/Features/Modes/Views/ModeView.swift` |
| `VoiceInk/Modes/TriggerInstalledApps.swift` | +52/-33 | `VoiceInk/Infrastructure/SystemIntegration/Context/TriggerInstalledApps.swift` | symlink／去重掃描 | **U** | `Voco/Sources/Context/VocoInstalledAppScanner.swift` 純函式；上游已有則刪；上游已有則刪，否則保留最小通用修正 |
| `VoiceInk/Paste/CursorPaster.swift` | +239/-21 | `VoiceInk/Infrastructure/SystemIntegration/Paste/CursorPaster.swift` | transaction、target guard、landing retry、restore | **H** | `Voco/Sources/Paste/VocoPasteService.swift`；上游 public entry 薄轉呼 |
| `VoiceInk/PlaybackController.swift` | +1/-1 | `VoiceInk/Infrastructure/SystemIntegration/Media/PlaybackController.swift` | Task cancellation 通用修正 | **U** | 保留一行通用 bug fix；上游已有則刪；上游已有則刪，否則保留最小通用修正 |
| `VoiceInk/Recorder.swift` | +52/-5 | `VoiceInk/Features/Recording/Capture/Recorder.swift` | audio lifecycle、pause task | **H** | `Voco/Sources/Audio/VocoAudioLifecycleController.swift`＋錄音工廠注入；上游 invalidate 優先 |
| `VoiceInk/Services/AudioDeviceConfiguration.swift` | +1/-1 | `VoiceInk/Infrastructure/Audio/Devices/AudioDeviceConfiguration.swift` | logger identity | **I** | 只留一行 identity patch |
| `VoiceInk/Services/AudioDeviceManager.swift` | +88/-37 | `VoiceInk/Infrastructure/Audio/Devices/AudioDeviceManager.swift` | modelUID／rebind；上游已有 | **D** | 採上游實作；只在仍有 delta 時留 identity patch |
| `VoiceInk/Services/DictionaryService.swift` | +34/-0 | `VoiceInk/Features/Dictionary/Workflows/DictionaryService.swift` | 批次加詞與去重 | **E** | `Voco/Sources/Personalization/DictionaryService+Voco.swift` extension；原檔恢復 |
| `VoiceInk/Services/KeychainService.swift` | +2/-2 | `VoiceInk/Infrastructure/Credentials/KeychainService.swift` | logger 與 service bundle ID | **I** | 保留最小 identity patch；不改舊 keychain namespace |
| `VoiceInk/Services/LogExporter.swift` | +2/-2 | `VoiceInk/Infrastructure/Diagnostics/LogExporter.swift` | logger subsystem | **I** | 只留 identity patch |
| `VoiceInk/Services/SessionMetricMigrationService.swift` | +66/-10 | `VoiceInk/Infrastructure/Persistence/Migrations/SessionMetricMigrationService.swift` | versioned stats backfill | **H** | `Voco/Sources/Persistence/VocoSessionMetricBackfill.swift`；上游 migration 後執行 |
| `VoiceInk/Services/SystemInfoService.swift` | +2/-11 | `VoiceInk/Features/Settings/Diagnostics/SystemInfoService.swift` | license 顯示與 selectedLanguage helper | **I/E** | `Voco/Sources/Bootstrap/VocoSystemInfo.swift`＋`Voco/Sources/ASR/VocoLanguageDefaults.swift`；保留最小 identity patch |
| `VoiceInk/Services/TranscriptionAutoCleanupService.swift` | +2/-2 | `VoiceInk/Infrastructure/Persistence/Cleanup/TranscriptionAutoCleanupService.swift` | logger 與 Application Support ID | **I** | 只留 identity patch |
| `VoiceInk/Services/UserDefaultsManager.swift` | +6/-0 | `VoiceInk/Infrastructure/Persistence/Preferences/UserDefaultsManager.swift` | modelUID key；上游已有則刪 | **D/E** | 若仍缺則移到 `Voco/Sources/Audio/UserDefaults+VocoAudio.swift` extension |
| `VoiceInk/Transcription/Engine/TranscriptionDelivery.swift` | +21/-5 | `VoiceInk/Features/Recording/Workflows/TranscriptionDelivery.swift` | final paste／result、trailing space | **H** | `Voco/Sources/Paste/VocoDelivery.swift`；上游只接交付結果 |
| `VoiceInk/Transcription/FluidAudio/FluidAudioTranscriptionService.swift` | +2/-14 | `VoiceInk/Infrastructure/Providers/Transcription/FluidAudio/FluidAudioTranscriptionService.swift` | 共用 WAV reader、identity | **U** | `Voco/Sources/Audio/VocoWAVSampleReader.swift`；identity 最小 patch；上游已有則刪，否則保留最小通用修正 |
| `VoiceInk/Transcription/Native/NativeAppleTranscriptionService.swift` | +20/-2 | `VoiceInk/Infrastructure/Providers/Transcription/AppleSpeech/NativeAppleTranscriptionService.swift` | Apple locale mapping、logger | **E/I** | `Voco/Sources/ASR/VocoAppleLocaleMapper.swift`；identity 最小 patch |
| `VoiceInk/Transcription/Processing/TranscriptionOutputFilter.swift` | +12/-3 | `VoiceInk/Features/Recording/Processing/TranscriptionOutputFilter.swift` | 輸出結果記錄、字元集合 | **H** | `Voco/Sources/Correction/VocoOutputFilter.swift` 包結果記錄，保持字元集合不變 |
| `VoiceInk/Transcription/Processing/WordReplacementService.swift` | +57/-2 | `VoiceInk/Features/Dictionary/Workflows/WordReplacementService.swift` | enabled-rule／casing pipeline | **H** | `Voco/Sources/Personalization/VocoWordReplacementService.swift`；不可套上游全部啟用語意 |
| `VoiceInk/Views/AI Models/CloudModelCardView.swift` | +6/-6 | `上游新共用 model cards` | model card 在地化 | **L** | `Voco/Localization/`；採上游新共用 card |
| `VoiceInk/Views/AI Models/ModelSettingsPanel.swift` | +2/-1 | `VoiceInk/Features/ModelLibrary/Views/ModelSettingsPanel.swift` | settings tab 在地化 | **L** | `Voco/Localization/`；採上游新檔 `VoiceInk/Features/ModelLibrary/Views/ModelSettingsPanel.swift` |
| `VoiceInk/Views/Common/TranscriptionInfoPanel.swift` | +33/-0 | `VoiceInk/Features/History/Views/TranscriptionInfoPanel.swift` | correction coverage／events metadata | **H** | `Voco/Sources/RuleAssistant/UI/VocoCorrectionMetadataRows.swift`；掛上游新 History InfoPanel |
| `VoiceInk/Views/Dashboard/DashboardContent.swift` | +29/-61 | `VoiceInk/Features/Dashboard/Views/DashboardContent.swift` | 品牌、license、dashboard metadata | **M** | 採上游新 dashboard＋`Voco/Localization/`、`VocoLicensePolicy`，不復活舊 layout |
| `VoiceInk/Views/Dashboard/DashboardPromotionsSection.swift` | +11/-11 | `上游新 Dashboard components` | 隱藏 Pro／affiliate 銷售 | **M** | 不保留舊檔；以 `VocoLicensePolicy` 隱藏，不搬舊 promo view |
| `VoiceInk/Views/Dashboard/DashboardStatCard.swift` | +3/-3 | `上游新 dashboard cards` | dashboard card 在地化 | **L** | `Voco/Localization/`；採上游新 cards |
| `VoiceInk/Views/Dashboard/ModelPerformancePanel.swift` | +19/-15 | `上游新 ModelPerformance 系列` | performance metrics、在地化 | **M/L** | 採上游新 ModelPerformance 系列；Voco metrics 另掛 `Voco/Sources/UI/`，不保留舊配色 patch |
| `VoiceInk/Views/Dashboard/PerformanceAnalysisPanelView.swift` | +1/-1 | `VoiceInk/Features/History/Views/HistoryAnalysisPanelView.swift` | section header 在地化 | **L** | `Voco/Localization/`；採上游 `VoiceInk/Features/History/Views/HistoryAnalysisPanelView.swift` |
| `VoiceInk/Views/Dashboard/PerformanceAnalysisView.swift` | +4/-3 | `上游新 History analysis` | ruleAssistant panel mode | **M** | 改由 `Voco/Sources/RuleAssistant/UI/VocoHistoryPanel.swift` 管理，不加上游 enum case |
| `VoiceInk/Views/DashboardView.swift` | +3/-12 | `VoiceInk/Features/Dashboard/Views/DashboardView.swift` | license wrapper | **M** | 採上游新 DashboardView＋`VocoLicensePolicy`，不保留舊 wrapper |
| `VoiceInk/Views/History/InlineHistoryView.swift` | +91/-6 | `VoiceInk/Features/History/Views/HistoryView.swift` | correction refresh、panel、row fallback | **M/H** | 掛上游新 HistoryView＋`Voco/Sources/RuleAssistant/UI/VocoHistoryPanel.swift`；不復活舊檔 |
| `VoiceInk/Views/History/TranscriptionDetailView.swift` | +4/-10 | `上游新 detail view` | detailDisplayTexts | **E/M** | `Voco/Sources/Persistence/Transcription+Voco.swift`＋新 detail 單一渲染掛點 |
| `VoiceInk/Views/History/TranscriptionHistoryView.swift` | +61/-0 | `上游新 HistoryView` | rule assistant panel／refresh | **M/H** | 掛上游新 HistoryView＋`Voco/Sources/RuleAssistant/UI/VocoHistoryPanel.swift` |
| `VoiceInk/Views/Recorder/MiniWindowManager.swift` | +13/-2 | `VoiceInk/Features/Recording/Presentation/MiniWindowManager.swift` | StartupTracer／panel lifecycle | **H** | `Voco/Sources/EditMode/VocoPanelLifecycle.swift`；上游 show 行為優先 |
| `VoiceInk/Views/Recorder/NotchWindowManager.swift` | +13/-2 | `VoiceInk/Features/Recording/Presentation/NotchWindowManager.swift` | StartupTracer／panel lifecycle | **H** | `Voco/Sources/EditMode/VocoPanelLifecycle.swift`；上游 show 行為優先 |
| `VoiceInk/VoiceInk.swift` | +95/-67 | `VoiceInk/App/VoiceInk.swift` | app composition、storage、shared registry、prewarm／migration | **H** | `Voco/Sources/Bootstrap/VocoBootstrap.swift`、`VocoStorageConfiguration.swift`；上游已有 error handling 不重複 |
| `VoiceInk/WindowManager.swift` | +1/-1 | `VoiceInk/App/Windows/WindowManager.swift` | window title | **L** | `Voco/Localization/`；採上游 `VoiceInk/App/Windows/WindowManager.swift` |
| `VoiceInkTests/VoiceInkTests.swift` | +4867/-3 | `上游刪除；test target 移到 Tests/` | 4867 行 fork 測試與 harness | **R** | 拆到 `Voco/Tests/**/*ForkTests.swift`＋`Voco/Tests/Support/`；不改上游測試 |



## 4. 第 2～5 階段執行範圍與驗收

### 第 2 階段：低風險刪 patch、在地化、extensions、UI 與 schema

**先做：** 對每個預定掛點，先讀 upstream `22e5ec54` 中對應的新檔案（即上表的新路徑），把「舊檔 → 新檔 → 可一對一放入的掛點」寫進當階段工作紀錄。不要先在舊路徑設計完才發現上游已改變結構。

**範圍：**
- 套用 D／U：Soniox v5、空白、上游已收回的 modelUID、atomics、screen resolver 等，不搬過時碼。
- 品牌與 zh-Hant 用 `Voco/Localization/` overrides＋generator 產生合併 catalog；保留上游 de／fr／zh-Hans，檢查缺 key、重複 key、format specifiers。
- 搬 E／R：model structs／catalog、純計算 extension、fork views、settings、history metadata、CSV、menu bar。
- SwiftData 只留必要 stored fields；enum provider cases、backup optional DTO fields 留上游並逐項登錄到 `PATCHES.md`，其他抽離。
- 完成上表標為第 2 階段的項目，不提前做核心 pipeline。

**驗收：**
- macOS Debug build 通過；完整 `VoiceInkTests` 與第 6 節的 10 個既有失敗清單完全一致，不能新增失敗。
- Backup 新舊格式 round-trip、SwiftData 舊 store 開啟、zh-Hant／英文 UI、規則助手、詞庫審核都通過。
- 用相同 upstream `22e5ec54` 在 `/tmp` 試合併：必須低於 121，且每個減少的路徑能對應實際移除的 patch；abort 並 `git worktree remove`。
- 依第 5 節完成 Release 備份、部署、30 秒啟動與檔案 smoke。

### 第 3 階段：核心 pipeline、registry、audio、paste 薄掛點

**先做：** 同樣先讀 upstream `22e5ec54` 的 `Features/Recording/Workflows`、audio、paste、enhancement 新檔，確認每個掛點在合併後的位置與 API 形狀。

**範圍：**
- `VocoTranscriptionServiceRegistry` 用 subclass／工廠注入，共享 `engine.serviceRegistry`；不可另建 registry。
- Pipeline 只留 after-ASR、post-enhancement、pre-delivery、record 四個薄掛點；Edit Mode、canonicalization、shadow、style guard 全在 Voco。
- `VocoEngineCoordinator` 擁有 fork 狀態，移除 associated-object 方案；start flow 與 bounded realtime gate 獨立。
- CoreAudio 只補上游仍缺的 RNNoise／ring pipeline；paste／selection／enhancement 都改成上游薄 entry＋Voco service。
- 每個仍留在上游檔案的 hook 都登錄到 `PATCHES.md`，包括位置、用途與不能搬的原因。

**驗收：**
- 同第 2 階段的 build、測試清單、部署與檔案 smoke。
- 另做錄音停止、連按、sleep／wake、裝置切換、長音檔轉碼、Chromium／Gemini／Electron 貼上、clipboard restore、Edit Mode 選取改變時中止。
- 第三次試合併不得高於前一階段；剩餘每個衝突都必須是 `PATCHES.md` 已登錄的必要上游 patch，不預設未實測的個位數目標。

### 第 4 階段：實際合併上游

**範圍：**
- 在試合併證明下降後，才對同一 pinned upstream `22e5ec54` 實際 merge。
- 採上游 `App／Features／Infrastructure／DesignSystem` 新結構；上游已刪或已改名的舊檔直接跟進，舊 Dashboard／History 不復活。
- 合併初期關閉上游 AutoLearn，詞庫只由 Voco 學習流程寫入。
- `Package.resolved` 用合併後依賴重新解析；catalog 合併成多語 superset。
- 記錄實際衝突數並與第 3 階段試算比較；任何新增的上游修改都要重新分類，不硬套舊表。

**驗收：**
- 實際衝突數與試算一致；若不一致，停下逐個解釋，不部署。
- build、測試清單、部署與檔案 smoke 全過。此階段允許上游新功能與 UI 改變，但不允許已決定保留的 Voco 行為退化。

### 第 5 階段：定版與建立可重複的 sync gate

**範圍：**
- 完成 `PATCHES.md` 的 upstream baseline、殘留 patch、部署 commit 與衝突紀錄。
- 檢查 `git diff upstream/main...main`：應只剩 `Voco/`、AGENTS 最後段、PATCHES，以及 allowlist 中的上游薄掛點／schema／config。
- 將每次 sync 的 build、test、trial merge、deployment 寫成可重複流程，避免只靠對話記憶。

**驗收：**
- `VoiceInkTests` full suite 的失敗清單仍與第 6 節相同；若清單改變，先查明原因，不把重構報成完成。
- Release build、`ditto` 安裝、啟動與完整日常 smoke。
- iOS 不在本輪驗收；mac 單測通過不被誤當成 keyboard extension 正常。

## 5. 部署與檔案驗收（照 `AGENTS.md`）

每個需要部署的階段都按以下順序，失敗即用 `/tmp/Voco.app.bak-20261024` 還原：

1. Release build 成功。
2. 部署前先 `ditto /Applications/Voco.app /tmp/Voco.app.bak-20261024`，確認備份存在；不要刪除備份。
3. 結束正在跑的 Voco，Release 產物用 `ditto` 覆蓋 `/Applications/Voco.app`，再啟動。
4. `codesign --verify --deep`（**不帶 `--strict`**）的結果與備份相同；`codesign -d --entitlements :-` 內容與備份逐字相同，貼出 `diff`。
5. Bundle ID、`CFBundleShortVersionString`、`CFBundleVersion` 與備份相同（本次為 `com.jasonchien.Voco`、`2.0 (179)`；若上游本來就改版本，寫明，不強改回來）。
6. `pgrep -fl Voco` 有 PID，等 30 秒還活著；檢查 log 沒有 crash 或啟動錯誤。既有噪音與已證實的既有錯誤要分開記錄，不隱瞞。
7. 能用檔案做的 smoke 要與搬前比對；Qwen 轉錄與 runtime replay 必須語意相同。麥克風錄音、TextEdit 貼上、Chromium／Gemini 貼上留給 Jason 實際操作並記錄結果。

`codesign --verify --deep --strict` 的 upstream symlink 失敗**不算閘門**；不要為此 clean build 或手改 bundle。App Group 資料夾不存在也**不算閘門**；非 sandbox app 不要手建。

## 6. 既有測試失敗基線（10 個唯一 test）

第 1 階段在搬前移 `7f3e8e24` 與搬移後各跑一次完整 `VoiceInkTests`，兩邊都是相同 10 個唯一失敗，沒有新增。之後各階段的接受標準是「失敗清單與這份完全相同」，不是假裝 full suite 全綠，也不准把新增失敗歸給既有問題。

1. `RimeVocabularyImportServiceTests.importSelectedCandidatesOnly()`
2. `RimeVocabularyImportServiceTests.previewDoesNotMutateSwiftDataOrProtectionList()`
3. `RimeVocabularyImportServiceTests.previewSkipsDuplicateRimeCandidates()`
4. `RimeVocabularyImportServiceTests.previewSkipsUnsafeCandidates()`
5. `VoiceInkTests.backupImporterPreservesWordReplacementLearningMetadata()`
6. `VoiceInkTests.confidenceGateUsesRecentCorrectionRiskForReview()`
7. `VoiceInkTests.correctionFeedbackLearningSkipsProtectedSourceTerms()`
8. `VoiceInkTests.correctionFeedbackLearningStagesCandidateOverride()`
9. `VoiceInkTests.sessionMetricMetadataDeduplicatesReviewTriggers()`
10. `VoiceInkTests.unicodeWordBoundaryPreservesAdjacentUnicodeAndMatchesCJK()`

Python tools 測試目前為 `Ran 74 tests ... OK`，之後也要維持通過。`correctionFeedbackLearningSkipsProtectedSourceTerms` 的順序相依問題另有文件記錄，但 `-only-testing` 曾出現 `Executed 0 tests`，所以不得只憑單測命令成功就宣稱該測試通過，必須確認實際執行數。

## 7. 部署參考與目前狀態

- 第 1 階段已以 Release＋`ditto` 部署，app 30 秒後仍存活，deep verify 與 entitlements 跟備份一致，版本為 `2.0 (179)`。
- Jason 已實測新版語音輸入，TextEdit 與 Google 貼上正常，並放行 push；`3d7e23db` 已在 `origin/main`。
- 備份 `/tmp/Voco.app.bak-20261024` 保留。
- 本文件只補齊計畫，不表示第 2 階段已開始或獲准略過任何驗收。
