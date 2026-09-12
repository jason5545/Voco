# Rule Assistant on Mac Voco

Android Vocotype 的規則助手（rule assistant）與修正標記（row correction markings）移植到
Mac Voco 的落點與差異記錄。行為對齊 Android 版；本文件只記平台差異與 Mac 特有的決定。

## 落點

- `VoiceInk/Services/RuleAssistant/`
  - `RuleAssistantProtocol.swift`：常數、`RuleAssistantContext`（`init(transcription:rowPk:)`、
    `toSafeJSON`、`correctionRow`、`sourceNote`）、`RuleAssistantDraft`（parse／locateAllJSON／
    `isSafeForWrite`／`toMcpArguments`／`toPreviewArguments`）、`RuleAssistantExample`、
    `SseEventParser`、`OpenCodeDeltaParser`、`ToolCallAccumulator`、`OpenCodeMessage`、
    `nearbyRecordsToolJSON()`。
  - `RuleAssistantClients.swift`：`RuleAssistantProvider`／`RuleAssistantMCP` protocol、
    `OpenCodeGoClient`（POST SSE）、`WorkerMCPClient`（legacy SSE MCP）、
    `RuleAssistantNoRedirectDelegate`、`AsyncSerialGate`、`SseByteFeeder`。
  - `RowCorrectionMarkings.swift`：parse／label／`forRow`（fail-closed 身分比對）／fetch
    （每批 20）、`Transcription.sqliteRowPK`、`RowCorrectionMarkingRefresher`。
  - `RuleAssistantSession.swift`：`@MainActor` session 全流程（submit／auto-guess／confirm／
    resync／cancel／refreshCorrections），無 SwiftUI／SwiftData 依賴，全部可用 fake 跑。
  - `RuleAssistantSessionRegistry.swift`：同一時間只留一個 active session；panel 關閉不釋放，
    開另一筆才關閉前一個對話。
  - `RuleAssistantKeyStore.swift`：OpenCode Go key 的 Keychain 封裝
    （`KeychainService`，key `RuleAssistantOpenCodeGoKey`，`syncable: false`）。
- UI：
  - 主視窗歷史 `InlineHistoryView`：`PanelMode.ruleAssistant`、右側 panel（寬 480）、
    卡片標題列的修正標記、展開區的「Fix with AI」入口（含音檔列）、
    每次載入頁面後背景刷新修正標記。
  - 獨立歷史視窗 `TranscriptionHistoryView`：toolbar「Fix with AI」（有 rowPk 才可用）、
    第三個右側 panel（與 Info／Analysis 互斥）。
  - `TranscriptionListItem`：`TranscriptionAssistiveBadge` 的 correction-marking badge
    （review 之後、selection 之前）。
  - `TranscriptionInfoPanel`：Correction／Correction events 兩列。
  - `VoiceInk/Views/History/RuleAssistant/RuleAssistantPanelView.swift`：panel 本體。
  - `VoiceInk/Views/Settings/RuleAssistantSettingsView.swift`：設定區（Go key + Worker key 狀態），
    掛在 SettingsView 的 AutoApplyModelSettingsView 之後。

## 與 Android 的差異

- **Provenance**：`correctionSource: "voco"`、`actor: "voco-rule-assistant"`、
  `correctionRow: {platform: "voco", rowPk: <Z_PK>, recordId: <UUID>}`、`note` 尾加
  `source=voco:row:<Z_PK>`。不送 legacy 頂層 `rowPk`。
- **rowPk 取法**：SwiftData `PersistentIdentifier` 沒有公開的 `isTemporary`／
  `uriRepresentation()`，但其 JSON 編碼的 `implementation` 內含 `isTemporary`、
  `primaryKey`（`p<N>`）、`uriRepresentation`（`x-coredata://…/Transcription/p<N>`）。
  `Transcription.sqliteRowPK` 先驗 `isTemporary == false`，再從 `primaryKey`（失敗退回
  URI 末段）解析 N。`VoiceInkTests/TranscriptionRowIdentityTests` 用暫存磁碟 store +
  SQLite C API 讀 `ZTRANSCRIPTION.Z_PK`／`ZID` 對照證明。這個 namespace 與
  `tools/voco_auto_apply_control.py --row-pk`、retranscribe skill 一致。
- **鄰近紀錄**：以 `timestamp` 排序取前後各最多 5 筆（不含本筆），只送文字欄位；
  由 session 注入的 `neighborLoader` 用 ModelContext 查詢。
- **同步驗證**：寫入後呼叫 `VocoAutoApplyModelService.shared.syncFromWorker()`，
  outcome 為 installed／upToDate 且 manifest SHA 與 installed SHA 都等於發佈 SHA 才算成功；
  否則 Failed + 「Resync Mac model」。
- **SSE 讀取**：`URLSession.AsyncBytes.lines` 會丟掉空行（`\r\n` 是單一字位簇，且
  AsyncLineSequence 省略空行），直接拿來切 SSE 會讓事件永遠不派送。兩個 client 都改走
  `SseByteFeeder` 逐 byte 讀、按行餵 `SseEventParser`（保留空行、處理跨 chunk 的
  UTF-8 序列）。`SseEventParser` 本身改為 byte buffer 切行，正確處理 CRLF。
- **快取**：SSE GET 與 provider POST 都設 `cachePolicy = .reloadIgnoringLocalCacheData`；
  在 app process 內不設的話 URLSession 可能完全不吐出串流回應。
- **不跟隨 redirect**：provider 與 MCP 都掛 `RuleAssistantNoRedirectDelegate`
  （`completionHandler(nil)`），有測試覆蓋。

## 找問題與選擇題（2026-09-11）

原本的「自動猜測」只有兩種結局：有把握就出草稿，沒把握就用純文字問一句，後者得打字回答。
改成兩個通用機制，Android 同步實作、protocol 與 prompt 相同：

- **選擇題原語** `RuleAssistantQuestion`（`RuleAssistantProtocol.swift`）：模型在回覆裡附一個
  `{"question": {id, prompt, multiSelect, options[{id, label, detail?, surface?, target?}]}}`
  JSON（與草稿同一套 `locateAllJSON` 找到，一回合最多一題，多的忽略；欄位不合法就整題丟掉、
  當純文字顯示）。App 渲染成選項按鈕（多選勾、單選圓點，⌘1 到 ⌘9 切換），有 `surface`＋`target`
  的候選只顯示候選本身。按「送出選擇」把選擇組成固定格式的使用者訊息回給模型
  （`回覆問題 q1：…／選擇：[a] …／補充：無`），
  transcript 顯示的是可讀摘要（「選了：…」），wire 才是完整格式。輸入框的文字會當「補充」一起送。
  答過的題變唯讀並標「已回答」；回覆失敗或被停止時題目與勾選狀態放回去（`pendingChoiceRestore`）。
- **找問題模式** `submitScan()` 取代 `submitAutoGuess()`：固定指令要模型把可疑處列成一題多選
  question 的候選，不確定的交給使用者勾，不靠上下文硬猜；有把握的候選可同回合直接出草稿。
  開啟面板時自動跑一次（`RuleAssistantPanelView.attach` → `autoScanIfNeeded()`，每個 session
  只跑一次，`hasAutoScanned`；設定 `RuleAssistantAutoScanOnOpen` 可關，預設開）。找問題執行中輸入框
  仍可打字（`isInterruptible`：只有 scan 回合的 loadingTools／thinking 可被打斷），送出就取消掃描
  改送手動說明；停止時掃描指令不會回填到輸入框，placeholder 回合會從 transcript 移除。
- **broad 閘門留在 App 端**（`RuleAssistantSession.gateReason`）：回合分 `manual`／`scan`／
  `choice`／`rescope`。scan 回合維持自動找問題不建立 broad 或 family transaction；choice 永不放行
  broad，候選選擇只會建立 correction（有打字補充的 choice 視為 manual）。correction 草稿卡預設
  「只改這句」，使用者改成「任何語境」或「語境限定」時送出 rescope 回合；App 以
  sourcePattern／targetText 套回原句後必須精確等於修正後的內容，且語境 token 必須真的出現在原句，才放行
  replacementRule／contextLockedRule。rescope 回合不接受 replacementFamily、transaction 等其他類型；
  失敗時原 correction 草稿放回卡片。
- 模型維持 `glm-5.3-flash`：model-arena 五輪把它定位在 Opus 5 級，沒有換模型的理由；候選品質要看
  實機使用。
- 測試：`RuleAssistantQuestionParseTests`（解析、預設 id、去重、fail-closed、上限 8、與草稿互不干擾）、
  `RuleAssistantDraftTests` 覆蓋 rescope 的套回檢查、類型與 token gate；`RuleAssistantIntegrationTests`
  覆蓋 scan 出題、選擇回覆格式與閘門、rescope 成功／失敗還原、補充視為 manual、壞題目當純文字、
  題目與草稿同回合、自動掃描只跑一次、掃描中打字與取消、手動回合打字被忽略但停止會還原。

## 找問題之後的四項修正（2026-09-11，同日實機回饋）

用「複製給 reviewer」匯出的對話逐一診斷後修的，Android 同步。

- **複製給 reviewer**：草稿卡「Confirm & Publish」旁與底部按鈕列各一個按鈕，一鍵把整段對話放進
  `NSPasteboard`（Android 是 `ClipboardManager`）：客戶端標示（`voco 版本 · macOS 版本`）、紀錄各階段、
  每回合文字與題目勾選、草稿與 Worker preview／查重結果、wire history（工具呼叫、截 800 字的結果、
  thinking 截 1500 字、system prompt 只寫字數）、「UI state」段（每個選項 ☑／☐、草稿卡 scope control
  是否顯示、`canSubmitChoice`、輸入框內容、每張草稿卡確認鈕狀態）、給 Claude Code／Codex 的說明（判斷該改
  prompt、App 邏輯還是 Worker；兩端 prompt 必須一致）。`RuleAssistantSession.exportForReview(client:)`，
  純函式 `reviewExport` 可測。不含 key、音檔。
- **模型停在「請勾選：」不吐 question JSON**（glm-5.3-flash 多輪工具後常見）：`needsQuestionNudge`
  在 scan 回合沒 question 也沒草稿、或任何回合結尾像「請勾選：」「請選擇」時，自動再送一則 wire-only
  的 `questionNudgePrompt`（只輸出 question JSON、不查工具），最多一次；催促訊息不進 transcript，
  模型先前的說明文字保留在卡片上方。Prompt 明講 JSON 必須與說明同一則回答。
- **範圍改在草稿卡**：prompt 規定一個可疑處一個選項，不再把範圍拆成候選；correction 草稿卡提供
  「只改這句／語境限定／任何語境」，後兩者送出 rescope 回合。`RuleAssistantQuestion.parse` 對相同
  surface＋target 的選項只留第一個。
- **廣域規則自動反例**：literal 廣域規則是 `text.contains`＋整句 `replacingOccurrences`，沒有邊界，
  「資料架 → 資料夾」會把「資料架構」改成「資料夾構」。`RuleAssistantGuardSuggester`（`RuleAssistantProtocol.swift`）
  把來源每個 head＋tail 切法的 tail 當 prefix 查 `word_freq.tsv`（`VocoWordFrequencyLexicon.words(withPrefix:)`，
  排序陣列二分搜尋，首次建索引），組成更長的詞（資料＋架構），頻率 ≥ 100、最多 6 個，經
  `RuleAssistantSessionRegistry` 注入 session（`guardSuggester`）。replacementRule／replacementFamily
  草稿出來時自動當 `negativeExamples` 塞進草稿（總數不超過 `isSafeForWrite` 的 10），preview／查重
  都用帶反例的版本；草稿卡顯示「自動保護更長的詞」開關（`setAutoGuards`），關掉會拿掉自動反例並重跑
  Worker 檢查，模型自己寫的反例（有 context）不動。Prompt 另要求模型補語意上的反例。Worker 端對同一條規則再送 add 帶
  negativeExamples 是 metadata-only 合併，不需要 tombstone。
- **稱呼**：system prompt 與匯出說明段一律用 the user，不用人名。
- 測試：`RuleAssistantGuardSuggesterTests`（每個切點、頻率排序、上限 6、非 CJK／單字不產生、
  `withAutoGuards` 只增刪自動反例）；`RuleAssistantIntegrationTests` 加匯出內容與 UI state、催促一次
  與催促後仍無題目、手動回合不催、廣域草稿自動反例與開關重跑檢查、任何語境預覽。

## 已修好就停（2026-09-11）

- 新規則：find-issues 回合先看 `runtimeReplay`。`changed` 為 true（`fires` 不為空）就直接回
  「看起來現行規則已經修好了」，列出每個 fire 的 sourcePattern → targetText，附一題只有兩個
  way-out 選項（「好，不用改」／「還有別的錯，我來說」）的 question，然後停手：不再找其他可疑處、
  不查工具、不出草稿。`changed` 為 false 或 `runtimeReplay` 為 null 才跑原本的整筆掃描。
- 為什麼：`runtimeReplay` 剛加進來時（c48ddec7）prompt 寫的是「已被 fires 命中或 outputText
  已修好的地方不列為候選」。那等於叫模型把修好的部分排除掉、繼續在剩下的文字裡找，找不到也要
  擠出候選，結果是 overthink：一筆現行規則已經完全修好的紀錄，模型還會硬湊出不成立的可疑處。
- 為什麼一定附 question：`RuleAssistantSession.missingJSONNudge` 對 scan 回合沒 question 也沒
  草稿會再送一次 `questionNudgePrompt`。回答裡不帶 question JSON 就會多繞一輪模型呼叫，使用者也
  多等一次。所以 scanPrompt 與 system prompt 都明講要附那一題。
- 使用者要補別的需求，走輸入框或補充欄，那一則當成一般說明處理（Core behavior 的 `runtimeReplay`
  條目也補了這句：說明指到 replay 已經命中的 surface 時，直接說現行 runtime 已修好、不出草稿）。
- 兩端字串一致（`RuleAssistantSession.scanPrompt` / `SCAN_PROMPT`、`systemPrompt` / `SYSTEM_PROMPT`），
  只差平台詞。
- 測試：`RuleAssistantIntegrationTests.scanStopsWhenTheRuntimeReplayAlreadyFixedTheRecord`
  （provider 只收到一次請求、沒有草稿、pendingQuestion 兩個選項且都沒有 surface／target、
  phase 回 idle、transcript 不含 JSON，另外斷言兩份 prompt 含新句子）；Android 同名測試在
  `RuleAssistantHttpIntegrationTest`。

## tombstone 查重語意（2026-09-11，實機匯出）

- 症狀：使用者說「考迪 → 口技」其實是「口吃」，模型正確開了 tombstone＋correction 兩張草稿，
  但 tombstone 卡被 App 擋成「目前 Worker 模型已套用這條規則，不需要再寫入」，確認鈕停用，
  規則停不掉。
- 原因：Worker 的 `detect_duplicate_control_event` 對 disableRule 回的 `duplicatePolicy` 與
  `alreadyApplied` 描述的是「要停掉的那條規則」（它存在、而且來自 overlay），不是「同樣的
  tombstone 已寫過」；只有 `duplicateEvent`（shape key 含 action）才代表重複的 tombstone。
  `checkDraft` 原本對所有事件一律用 `alreadyApplied` 擋寫，對 tombstone 剛好相反。
- 修法（App 端，Worker 不動）：`RuleAssistantDuplicate.alreadyWritten(eventType:)`，tombstone
  只看 `duplicateEvents > 0`，其他事件維持 `found`。`checkDraft` 的 tombstone 分支只在有相同事件時
  擋；已停掉的規則由 preview 的 `wouldPublish=false` 擋。寫入回應遺失時的「查到就標記已消耗」
  也改用同一個判斷，避免 tombstone 在規則尚未移除時被誤判為已寫入。
- 測試：`RuleAssistantIntegrationTests` 加 `tombstoneIsNotBlockedByThePolicyItRetires`
  （alreadyApplied=true＋duplicatePolicy=1 仍可確認並呼叫 `tombstone_auto_apply_rule`）與
  `duplicateTombstoneEventStillBlocksConfirm`。Android 同步（`RuleAssistantSession.kt`、
  `RuleAssistantHttpIntegrationTest`）。

## 本機規則覆蓋（coverage）

2026-09-10 加入。Worker 收據（`RowCorrectionMarkings`）綁的是 row 身分（platform + rowPk +
recordId），從 Vocotype、claude.ai、Codex、Claude Code 修的規則永遠掛不到 Mac 的 row 上。
coverage 不看身分：對每筆歷史的原始 ASR 文字（`rawTranscript`，沒有就 `text`）用本機已安裝的
auto-apply model 跑一次 `evaluate`，直接算出「這筆現在會不會被規則改掉」。兩端各自本機重算，
model 同步後自然一致，不需要任何跨端對應。

- 落點：`VoiceInk/Services/RuleAssistant/RowCorrectionCoverage.swift`。
  `RowCorrectionCoverageEvaluator.coverage(priorHitIds:evaluation:)` 是純函式；
  `RowCorrectionCoverageStore.shared` 依 row id + 原文 + 轉錄時命中 ID 做快取，訂閱
  `VocoAutoApplyModelService.$status`，model 一換就清快取並 bump `generation`，三個歷史 UI
  都 `@ObservedObject` 它。
- 狀態：`appliedAtTranscription`（灰，會改寫，但模型輸出已等於這筆存的 `normalizedTranscript`
  或 `text`，或全部 policy 都在轉錄時的 `autoApplyPolicyHitIDs` 裡）、`fixedByCurrentRules`
  （綠，會改寫且存的文字還是錯的，即「後來補的規則現在蓋到這筆」）、`blockedByGuard`（橘，規則命中但被保護詞擋下）、
  `suggestOnly`（灰，只有 suggest policy 命中）。`runtime.currency-number-normalization`
  不算 coverage。
- 顯示：主視窗歷史卡片標題列與 `TranscriptionListItem` badge（排在 Worker 收據之前）、
  info panel「Rule coverage」「Covered policies」兩列。Worker 收據保留當 provenance。
- 限制：歷史沒有當時的 app/context hints，context 只帶原文，需要外部 context 的
  context-locked 規則不會亮。auto-apply runtime 被使用者關掉時 `evaluate` 不改文字，
  coverage 也會是空。
- 測試：`VoiceInkTests/RowCorrectionCoverageTests.swift`。

## Panel 生命週期

Session 由 `RuleAssistantSessionRegistry` 持有：關 panel、切 view 不取消進行中的請求；
再開同一筆接回原對話；開另一筆才 `closeSession()`。停止（Stop）會關 provider／MCP 連線、
還原輸入框、丟掉未完成的使用者回合，歷史保留。晚到的結果依 generation 丟棄。

## 測試

`VoiceInkTests/`：

- `RuleAssistantParserTests.swift`：SSE（UTF-8、CRLF、多行 data、尾端殘片、上限）、
  delta parser（reasoning-only、跨 chunk `<think>`、usage-only chunk、null 欄位、
  provider error 物件／字串、[DONE]）、tool call 累加器（分片、多 index、截斷不可執行、
  上限）。
- `RuleAssistantDraftTests.swift`：locateAllJSON（多物件、drafts 包裹、fenced）、
  parseAll 上限 8、isSafeForWrite 正反例、toMcpArguments／toPreviewArguments 裁切與
  provenance。
- `RowCorrectionMarkingsTests.swift`：label 各狀態、forRow 身分比對 fail-closed、
  recordId 大小寫、schema 錯誤；`sqliteRowPK` 對照 SQLite `Z_PK`。
- `RuleAssistantHarness.swift` + `RuleAssistantIntegrationTests.swift`：以 loopback
  `NWListener` 假 HTTP server 餵真實 SSE／JSON-RPC bytes，跑真的 `OpenCodeGoClient` +
  `WorkerMCPClient` + `RuleAssistantSession`：header（Bearer、UA、穩定 session id）、
  raw reasoning 與 tool_call_id 回送、純文字澄清、寫入工具被擋、preview 衝突、重複事件、
  isError 與 JSON-RPC -32602 回給模型、寫入回應遺失兩種分支、noop、publish_failed
  eventSaved、sync SHA 不一致與 resync、截斷回合回滾、部分工具呼叫整輪拒絕、
  編輯作廢草稿、多草稿逐條發佈、自動猜測拒絕 broad、nearby 夾限且不含本筆、
  取消不留 unmatched tool_calls、endpoint 異源／query／路徑拒絕、redirect 拒絕、
  close 不重開。
- `RuleAssistantLiveTests.swift`：只在環境變數 `VOCO_GO_KEY` 存在時執行
  （真 Go + 真 client，Worker 仍為假）。

注意：app 的測試主機以系統語言（zh-Hant）執行，測試斷言不可依賴英文訊息原文。

## 驗收結果（2026-09-11）

- `xcodebuild test -only-testing:VoiceInkTests/RuleAssistantIntegrationTests ... RuleAssistantGuardSuggesterTests`：
  **55 tests、3 suites，全部通過**。Release build 成功、`ditto` 部署、Voco 正常啟動。

## 驗收結果（2026-09-10）

- `xcodebuild -configuration Debug build`：成功。
- `xcodebuild test -only-testing:VoiceInkTests`：**402 tests、27 suites，全部通過**
  （`RuleAssistantLiveTests.liveProviderProducesParseableDraft` 因無 `VOCO_GO_KEY` 而
  skipped）。`docs/known-test-failures.md` 記錄的既有 order-dependent 失敗本次未復現。

## 未驗證項目

- Live 測試未跑（本機無 `VOCO_GO_KEY`）；fake 整合測試不能視為端到端真實驗證。
- 真實 Worker 的 tools/list 若缺 `get_auto_apply_row_corrections`，標記刷新會安靜略過
  （設計如此）。
