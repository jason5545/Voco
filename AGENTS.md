# Voco（macOS）協作指引

給所有 AI 工具（Codex、Claude Code、OpenCode）的專案層規則。`CLAUDE.md` 是本機私有筆記、不進 git；本檔才是 repo 內唯一的建置與部署規範，兩者衝突以本檔為準。

## 專案基本資料

- 產品名 **Voco**（fork 自 `Beingpax/VoiceInk`），macOS 離線語音轉文字，whisper.cpp 與 Qwen3-ASR 本機推論。
- Xcode scheme 名稱是 `VoiceInk`（歷史原因），bundle ID `com.jasonchien.Voco`。
- UI 字串一律用「Voco」；class name、UserDefaults key、URL、bundle ID、migration 服務、log 訊息維持 VoiceInk 不動。
- 記憶系統一律走 MCP memory-connect（`search_memory` / `save_memory`），不用內建記憶檔。

## 建置與測試

```bash
# 單元測試（規則助手相關 suite 範例）
xcodebuild test -scheme VoiceInk -destination 'platform=macOS' \
  -only-testing:VoiceInkTests/RuleAssistantDraftTests \
  -only-testing:VoiceInkTests/RuleAssistantIntegrationTests

# Debug build
xcodebuild -scheme VoiceInk -configuration Debug -allowProvisioningUpdates -allowProvisioningDeviceRegistration build
```

- Commit 前確認 build 通過。
- app 測試主機以系統語言（zh-Hant）執行，測試斷言不可依賴英文訊息原文。

## 部署到本機（唯一正確做法）

日常使用的是 **`/Applications/Voco.app`**，部署一律是 Release build 用 `ditto` 覆蓋它：

```bash
xcodebuild -scheme VoiceInk -configuration Release -allowProvisioningUpdates -allowProvisioningDeviceRegistration build
pkill -x Voco 2>/dev/null; sleep 1
rm -rf /Applications/Voco.app
ditto ~/Library/Developer/Xcode/DerivedData/VoiceInk-*/Build/Products/Release/Voco.app /Applications/Voco.app
open /Applications/Voco.app
```

若 build 用了自訂 `-derivedDataPath`，`ditto` 來源就換成該路徑下的 `Build/Products/Release/Voco.app`，目標不變。

- **必須用 `ditto`**：`cp -R` 會破壞 code signature，app 啟動即 SIGKILL。
- **不要放到 `~/Downloads`、`~/Applications` 或其他位置。** `Makefile` 的 `make local` 會產生 `~/Downloads/VoiceInk.app`，那是給沒有開發者憑證的外部使用者的無簽章流程，不是這台機器的部署方式。
- Release 產物的 `whisper.framework` 沿用 upstream 的 symlink 結構，`codesign --verify --deep --strict` / Gatekeeper 可能回報 symlink loop 或 rejected。這是 upstream 既有問題，不要為此 clean build、手改 bundle 或阻擋部署；驗收標準是 Release build 成功、`ditto` 安裝後 Voco 正常啟動。
- 只有改到 Swift runtime、UI、schema decoder 時才需要 rebuild 部署。新增／停用 auto-apply 規則、`activateModel`、`publishWorkerRelease` 不需要重建 app，Voco 會自動偵測 active model 檔案變更並重新載入。
- 部署目標若在文件裡找不到明確路徑，先查既有安裝位置（`/Applications`、`pgrep -fl Voco`），還不確定就停下來問，不要猜路徑。

## 規則助手（Rule Assistant）

- 設計、閘門、測試與驗收記錄在 `docs/rule-assistant-mac.md`。
- `systemPrompt` / `scanPrompt`（`VoiceInk/Services/RuleAssistant/RuleAssistantSession.swift`）與 Android Vocotype 的 `SYSTEM_PROMPT` / `SCAN_PROMPT`（`app/src/main/java/com/vocotype/ruleassistant/RuleAssistantSession.kt`）除平台字眼外必須逐字一致；改一邊就要改另一邊並各自跑測試。
- Worker（`VocoReplayLab/workers/auto-apply-sync`）改動後用 `npm test` 驗證、`npm run deploy` 部署。

## Fork 獨有功能（合併 upstream 時必須保留）

- Edit Mode：`EditModeCacheService`、`VoiceCommandService`、`enhanceForEditMode()`；`AIEnhancementService` 的 `systemMessageOverride` / `userMessageOverride` 參數是 edit mode 需要的。
- 中文後處理管線 `ChinesePostProcessingService` 與多個校正引擎。
- auto-apply runtime 內建規則 `runtime.single-prefix-restart-collapse`：Mac `VocoAutoApplyModelService.collapseSinglePrefixRestarts`、Android `VocoAutoApplyModelService.kt`、Python `tools/voco_auto_apply_control.py`、ReplayLab `voco_train_auto_apply_model.py` 四份實作必須一致。
- `ModelPrewarmService` 必須接收 `engine.serviceRegistry`（共享 registry），不能自建 `TranscriptionServiceRegistry`。
- MLX 本機推論、RNNoise 噪音抑制、SwiftyOpenCC（s2twp）。
