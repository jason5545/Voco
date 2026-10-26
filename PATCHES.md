# Voco fork patches

Repository: `jason5545/Voco` (`main`). Upstream: `Beingpax/VoiceInk`.

Refactor plan: [Voco/Docs/refactor-plan.md](Voco/Docs/refactor-plan.md).

## Upstream baseline

- Reviewed upstream commit: `22e5ec5479b712797c3b338fcb875961e58800b6` (`22e5ec54`).
- Merge base before stage 1: `8e31ffaacfc64d4f90805a0412e34de714d31fc9`.
- Stage 1 starts from `7f3e8e2476aaab4205cc20fbac77effa0f235fd1`.
- Pre-stage-1 trial merge: **182 unmerged paths** (`UU 82`, `AU 58`, `UD 41`, `AA 1`).
- Post-stage-1 trial merge: **121 unmerged paths** (`UU 79`, `AU 0`, `UD 41`, `AA 1`), accepted as the stage-1 gate on 2026-10-24. The reduction is 58 eliminated `AU` paths plus three eliminated `UU` paths from restoring `.gitignore`, `Makefile`, and `README.md` to upstream.

## Stage 2 record (2026-10-24)

Stage 2 is partial. It completed the low-risk D/U patches, the remaining localization-only views, and the R/E files that could move without touching the stage-3 pipeline. Core pipeline, registry, audio, paste and SwiftData schema hooks remain for stage 3.

Trial merge against `22e5ec54` after stage 2: **99 unmerged paths** (`UU 62`, `UD 37`, `AA 0`). The stage-1 baseline was 121 (`UU 79`, `UD 41`, `AA 1`). 22 paths cleared, zero new paths.

### Cleared paths

| Path | Stage-1 status | Cleared by |
|---|---|---|
| `VoiceInk/Features/Modes/Views/ModeSettingsPanelView.swift` | UU | `b3783732` localization overrides |
| `VoiceInk/Features/Modes/Views/ModeTriggerSection.swift` | UU | `b3783732` localization overrides |
| `VoiceInk/Features/Onboarding/Components/AIProviderVerificationCard.swift` | UU | `b3783732` localization overrides |
| `VoiceInk/Features/Onboarding/Views/OnboardingContextAwarenessScreen.swift` | UU | `b3783732` localization overrides |
| `VoiceInk/Infrastructure/SystemIntegration/Context/BrowserURLService.swift` | UU | `b3783732` whitespace-only diff reverted |
| `VoiceInk/Localizable.xcstrings` | AA | `b3783732`; upstream catalog restored, merged catalog lives in `Voco/Localization/` |
| `VoiceInk/Modes/ModeView.swift` | UU | `b3783732` localization overrides |
| `VoiceInk/Modes/ModeTriggerHelpers.swift` | UD | `f008c356` adopted upstream file at its new path |
| `VoiceInk/Modes/TriggerInstalledApps.swift` | UD | `f008c356` adopted upstream file at its new path |
| `VoiceInk/Services/UserDefaultsManager.swift` | UD | `f008c356` adopted upstream file at its new path (upstream has `modelUID`) |
| `VoiceInk/Features/Recording/Presentation/MiniRecorderPanel.swift` | UU | `f008c356` adopted upstream file; `RecorderScreenResolver.swift` added |
| `VoiceInk/Features/Recording/Presentation/NotchRecorderPanel.swift` | UU | `f008c356` adopted upstream file; `RecorderScreenResolver.swift` added |
| `VoiceInk/Infrastructure/Providers/Transcription/Cloud/Streaming/SonioxStreamingProvider.swift` | UU | `f008c356` adopted upstream file (`stt-rt-v5`) |
| `VoiceInk/App/AppIntents/DismissMiniRecorderIntent.swift` | UU | `b249d01a` adopted upstream file; brand via catalog |
| `VoiceInk/App/AppIntents/ToggleMiniRecorderIntent.swift` | UU | `b249d01a` adopted upstream file; brand via catalog |
| `VoiceInk/DesignSystem/Components/InfoTip.swift` | UU | `b249d01a` adopted upstream file |
| `VoiceInk/Features/Modes/Templates/StarterModeTemplate.swift` | UU | `b249d01a` adopted upstream file (`String(localized:)` already upstream) |
| `VoiceInk/Features/Onboarding/State/OnboardingPermissionModels.swift` | UU | `b249d01a` adopted upstream file |
| `VoiceInk/Infrastructure/SystemIntegration/Paste/PasteMethod.swift` | UU | `b249d01a` adopted upstream file (adds upstream `directTyping`) |
| `VoiceInk/App/MenuBar/MenuBarView.swift` | UU | `1af543b7` Voco view moved to `Voco/Sources/UI/VocoMenuBarView.swift` |
| `VoiceInk/Features/History/Export/VoiceInkCSVExportService.swift` | UU | `1af543b7` Voco service moved to `Voco/Sources/Persistence/VocoCSVExportService.swift` |
| `VoiceInk/Features/ModelLibrary/Models/LanguageDictionary.swift` | UU | `3ad2ca9c` Voco defaults moved to `Voco/Sources/ASR/VocoLanguageDefaults.swift` |

### Fork-owned code added in stage 2

- `Voco/Sources/UI/VocoMenuBarView.swift`: the fork menu bar (direct model, provider and language selection, async launch-at-login) now replaces the upstream view at the composition root in `VoiceInk/VoiceInk.swift`.
- `Voco/Sources/Persistence/VocoCSVExportService.swift`: correction-aware CSV export; the two history views call it instead of the upstream service.
- `Voco/Sources/ASR/VocoLanguageDefaults.swift`: `SelectedLanguage` key, `auto` default and `selectedLanguage(in:)` as an extension on `TranscriptionLanguageSupport`.
- `Voco/Sources/UI/QueueItemStatus+Voco.swift`: localized queue phase labels; upstream `ProcessingPhase` raw values are untouched.
- `VoiceInk/Features/Recording/Presentation/RecorderScreenResolver.swift`: upstream file added verbatim because the upstream recorder panels call it.

### Deferred D/U patches

These upstream files still carry a fork patch because the upstream replacement depends on structures that only arrive with the stage-3/4 merge.

| Path | Why it cannot be adopted yet |
|---|---|
| `VoiceInk/Services/AudioDeviceManager.swift` | upstream file needs `AudioInputChannel`, `RecordingDeviceSession`, `ClamshellStateMonitor`, `AudioInputChannelSelection`; fork keeps base plus `modelUID`/rebind |
| `VoiceInk/Transcription/Streaming/FluidAudioStreamingProvider.swift` | upstream file needs `StreamingStopDisposition`, `PCMAudioConverter`, `detectedSpeechSegments` |
| `VoiceInk/Transcription/Cloud/SonioxProvider.swift` | upstream `CloudModel` no longer takes `speed`/`accuracy`; local model still does |
| `VoiceInk/PlaybackController.swift` | upstream API requires a recording `sessionID` plumbed through `Recorder` |
| `VoiceInk/Views/Settings/AudioSetupView.swift` | local `PrioritizedDevice` carries `modelUID`; upstream version uses the new audio device types |
| `VoiceInk/Views/Recorder/MiniWindowManager.swift`, `NotchWindowManager.swift` | upstream rebuilt the panel lifecycle (`AssistantSession`, `show() -> Bool`, panel rebuilt per dictation); stage-3 recorder work |
| `VoiceInk/Views/Onboarding/OnboardingLicenseScreen.swift`, `OnboardingLicenseCards.swift`, `OnboardingExperienceModels.swift` | upstream versions need `LicenseViewModel.hasVerifiedLicense` and the new experience-card fields |
| `VoiceInk/Views/AI Models/APIKeyManagementView.swift`, `FluidAudioModelCardView.swift`, `WhisperModelCardView.swift`, `Onboarding/TranscriptionModelDownloadCard.swift` | upstream versions need `AppTheme.Action.destructiveFill/destructiveForeground`, `FluidAudioDownloadStatus.isIndeterminate`, `FluidAudioModelManager.cancelDownload`, `AIProvider.voiceInkRefine` |

### Remaining upstream patches by area

Every remaining path is a stage-3/4 hook that still needs the upstream structure from `22e5ec54`. They are counted here so a later stage can prove each one either disappears or is an intentional thin hook.

| Area | Paths | Disposition | Why it is still patched |
|---|---:|---|---|
| `VoiceInk/Features/**` | 43 | H / S / R / L | recording workflows, dictionary learning UI, model library, settings, backup; stage 3 moves the pipeline and registry hooks, stage 4 adopts the new UI shells |
| `VoiceInk/Views/**` (dashboard, history, recorder) | 15 | M / H | upstream deleted the old dashboard/history files; the replacement views are adopted in stage 4, with Voco panels and metadata hung on the new shells |
| `VoiceInk/Infrastructure/**` | 11 | H / S / U | audio, persistence models, providers, accessibility; SwiftData stored fields must stay until the schema move is proven |
| `VoiceInk/Services/**` (legacy paths) | 8 | H / I / D | identity logger patches and services whose upstream replacement is part of the new audio/persistence structure |
| `VoiceInk/Transcription/**` (legacy paths) | 5 | H / U | delivery, output filter, word replacement and provider fixes that move with the stage-3 pipeline |
| `VoiceInk/App/**` | 4 | H / R | prewarm, streaming-key migration, sidebar policy, notification actions |
| `/Xcode/config` (`pbxproj`, `Package.resolved`, `appcast.xml`) | 3 | X | resolved only when the upstream project file is merged in stage 4 |
| `VoiceInk/Models/**`, root app files, `Paste`, `DesignSystem` | 8 | H / S / I / R | legacy paths that upstream replaced; content moves in stage 3/4 |
| `VoiceInkTests/VoiceInkTests.swift` | 1 | R | 4,867-line fork test file; split into `Voco/Tests/**/*ForkTests.swift` in stage 3 |

## Stage 2 remainder and stage 3 record (2026-10-24)

Stage 2 remainder was completed by `572db7f8` (model structs/catalog, SwiftData extensions, backup DTO, settings, history metadata and fork views). Stage 3 core work is in `75309a35`, `ab3c1108`, `87349049`, `521c68da`, `30e4d748`, and `414d389b`.

At `414d389b`, Debug build exited 0. The full `VoiceInkTests` xcresult contains 508 tests: 497 passed, 10 failed, 1 skipped; the 10 unique failures exactly match the stage-6 baseline with no new failures. Python tools finished `Ran 74 tests ... OK`.

The third trial merge against `22e5ec54` is **97 unmerged paths** (`UU 60`, `UD 37`, `AA 0`), down from 99 with zero new paths. The two cleared paths are:

| Cleared path | Patch removed |
|---|---|
| `VoiceInk/Features/AudioImport/State/TranscriptionResultView.swift` | `TranscriptionTab.localizedName` moved to `Voco/Sources/UI/TranscriptionTab+Voco.swift`; enum cases/raw behavior stay unchanged |
| `VoiceInk/Features/ModelLibrary/State/TranscriptionModelManager.swift` | Voco provider download checks moved to `Voco/Sources/ASR/VocoModelDownloadStatus.swift`; the manager calls one augmentation |

### Required upstream hooks left after stage 3

These are intentional hooks, not forgotten logic. The current path is the pre-merge fork path; the upstream path is the location read at `22e5ec54` and used for the trial merge. The final column records why the hook must remain in the upstream-owned file.

| Current path / upstream path | Hook and Voco owner | Why it must remain |
|---|---|---|
| `Models/TranscriptionModel.swift` → `Features/ModelLibrary/Models/TranscriptionModel.swift` | Four provider enum cases; structs in `ASR/Models.swift` | Enum cases cannot be added in an extension; persisted raw values must stay |
| `Models/TranscriptionModelRegistry.swift` → `Features/ModelLibrary/Models/TranscriptionModelRegistry.swift` | One `VocoModelCatalog.augment` call | Upstream owns the base catalog composition |
| `Transcription/Engine/TranscriptionModelManager.swift` → `Features/ModelLibrary/State/TranscriptionModelManager.swift` | One `VocoModelDownloadStatus.isDownloaded` augmentation (conflict cleared) | Upstream owns usable-model filtering |
| `Models/Transcription.swift` → `Infrastructure/Persistence/Models/Transcription.swift` | Voco stored fields and designated init; computed/record logic in `Persistence/Transcription+Voco.swift` | SwiftData stored properties and their original names cannot move into an extension |
| `Models/SessionMetric.swift` → `Infrastructure/Persistence/Models/SessionMetric.swift` | Voco stored fields and init; calculations in `SessionMetric+Voco.swift` | Same SwiftData schema constraint |
| `Models/WordReplacement.swift` → `Infrastructure/Persistence/Models/WordReplacement.swift` | `source`, `hitCount`, `lastSeenDate` and init; learning computed state in `WordReplacement+Voco.swift` | Learning metadata must remain persisted; upstream all-enabled semantics are not adopted |
| `Services/BackupTypes.swift` → `Features/Settings/Backup/BackupTypes.swift` | Optional Voco DTO fields/CodingKeys; `WordReplacementBackup` in `Persistence/VocoBackupTypes.swift` | Backup decoding requires these optional fields in `BackupFile` |
| `Views/Settings/SettingsView.swift` → `Features/Settings/Views/SettingsView.swift` | One `VocoSettingsSections` slot | Upstream owns the Form and navigation shell |
| `EmailSupport.swift` → `Features/Settings/Support/EmailSupport.swift` | Public entry forwards to `UI/VocoSupport.swift` | Existing support entry is called by the upstream UI |
| `Views/Dictionary/VocabularyView.swift` → `Features/Dictionary/Views/VocabularyView.swift` | One `VocoRimeImportSection` slot | Rime UI is fork-owned; the base vocabulary layout is upstream-owned |
| `Views/History/TranscriptionListItem.swift` → `Features/History/Components/TranscriptionListItem.swift` | One `TranscriptionAssistiveBadgeRow` slot in `UI/VocoTranscriptionMetadataSection.swift` | Upstream owns cell selection/background |
| `Views/Common/TranscriptionInfoPanel.swift` → `Features/History/Views/TranscriptionInfoPanel.swift` | One `VocoCorrectionMetadataRows` slot | Upstream owns the generic details/AI-request panel |
| `Views/AI Models/ModelCardView.swift` → `Features/ModelLibrary/Components/ModelCardView.swift` | Default dispatch to `UI/VocoModelCardFactory.swift` | Upstream owns the common model-card dispatch |
| `Views/AI Models/ModelManagementView.swift` → `Features/ModelLibrary/Views/ModelManagementView.swift` | Local-model predicate delegates to `ASR/VocoModelCatalog.swift` | Upstream owns model-management composition |
| `Services/LastTranscriptionService.swift` → `Features/History/Workflows/LastTranscriptionService.swift` | Output selection supplied by `Persistence/VocoTranscriptionOutput.swift` | Upstream owns fetch/copy/paste/retry entries |
| `Transcription/Engine/TranscriptionServiceRegistry.swift` → `Features/Recording/Workflows/TranscriptionServiceRegistry.swift` | `VocoTranscriptionServiceRegistry` subclass overrides `service(for:)` and `cleanup()` | Upstream owns the registry protocol/composition; Voco services cannot be added without a subclass hook |
| `Transcription/Engine/VoiceInkEngine.swift` → `Features/Recording/Workflows/VoiceInkEngine.swift` | One subclass instance, `VocoEngineCoordinator`, `VocoRecordingStartFlow` | This is the composition root. Prewarm, file retranscribe and queue all use `engine.serviceRegistry`; no second registry exists |
| `Transcription/Engine/TranscriptionPipeline.swift` → `Features/Recording/Workflows/TranscriptionPipeline.swift` | Exactly four calls to `VocoTranscriptionProcessor`: `afterASR`, `postEnhancement`, `preDelivery`, `record` | Upstream retains generic transcribe/cancel/enhance/deliver/save sequencing; fork correction and Edit Mode run only in Voco |
| `Transcription/Engine/TranscriptionService.swift` → `Features/Recording/Workflows/TranscriptionService.swift` | Base context has only language/prompt; adapter option is wrapped by `ASR/VocoASRRequestContext.swift` | Qwen adapter policy must not change the upstream request struct |
| `CoreAudioRecorder.swift` → `Infrastructure/Audio/CoreAudioRecorder.swift` | `VocoAudioRenderPipeline` allocate/process/flush/reset/free calls | Only RNNoise/mono buffering is fork-specific; upstream already has ring/atomics/invalidate |
| `Paste/CursorPaster.swift` → `Infrastructure/SystemIntegration/Paste/CursorPaster.swift` | Public entries forward to `Paste/VocoPasteService.swift` | Target guard, landing retry and restore lineage are fork behavior; upstream `PasteOutcome`/directTyping migration is stage 4 |
| `Paste/ClipboardManager.swift` → `Infrastructure/SystemIntegration/Paste/ClipboardManager.swift` | Recording-context read supplied by `Paste/ClipboardCoordination.swift` | Generic set/copy/get stays upstream; Voco owns transaction/context policy |
| `Services/SelectedTextService.swift` → `Infrastructure/SystemIntegration/Accessibility/SelectedTextService.swift` | Public entries forward to `Context/VocoSelectionService.swift` | AX evidence, Electron markers and clipboard locking cannot use upstream's simplified 35-line capture |
| `Services/AIEnhancement/AIEnhancementService.swift` → `Features/Enhancement/Workflows/AIEnhancementService.swift` | `vocoContext`, system-message builder, requester and clear hooks; raw `makeRequest`/retry and override entry remain | Provider transport stays upstream; Edit Mode requires the system/user override entry and Voco owns its context/prompt policy |
| `Transcription/Engine/TranscriptionDelivery.swift` → `Features/Recording/Workflows/TranscriptionDelivery.swift` | Paste branch forwards to `Paste/VocoDelivery.swift` | Upstream owns delivery modes; Voco records paste outcome and auto-send behavior |
| `Services/AudioFileTranscriptionManager.swift` → `Features/AudioImport/Workflows/AudioFileTranscriptionManager.swift` | Uses `engine.serviceRegistry` and `VocoASRRequestContext` with adapter disabled | File import must share resident models without enabling the adapter |
| `Services/AudioFileTranscriptionService.swift` → `Features/History/Workflows/AudioFileTranscriptionService.swift` | Uses `engine.serviceRegistry` and `VocoASRRequestContext` with adapter disabled | Same shared-registry and retranscription constraints |

The other paths in the 97-path appendix are the already registered deferred D/U dependencies, stage-4 M/R shells, localization/identity patches, and Xcode/release configuration from the tables above. No unexplained path was added. Upstream `transcribeCpp`, direct typing, AutoLearn, the new Dashboard/History shells, and new audio device types were not backported during stage 3.

## Fork layout

Stage 1 is a mechanical relocation only. It does not change class names, bundle IDs, App Group, UserDefaults keys, log subsystem, appcast URL, or feature logic.

- `Voco/Sources/ASR/`: Qwen3, Whisper MLX/CoreML, shared Whisper code, ASR services.
- `Voco/Sources/Audio/`: RNNoise processor.
- `Voco/Sources/Bootstrap/`: app identifiers, migration, startup tracing.
- `Voco/Sources/Context/`: context-aware insertion and surrounding text.
- `Voco/Sources/Correction/`: Chinese post-processing, canonicalization, feedback, phonetics, diagnostics.
- `Voco/Sources/EditMode/`: edit-mode services and engine/pipeline extensions.
- `Voco/Sources/Paste/`: paste landing verification.
- `Voco/Sources/Personalization/`: personal style and Rime vocabulary import.
- `Voco/Sources/Resources/`: Chinese correction data and localized InfoPlist strings.
- `Voco/Sources/RuleAssistant/`: rule assistant engine, clients, markings, panel.
- `Voco/Sources/UI/`: model cards and settings.
- `Voco/Platforms/iOS/`: iOS app target sources, Info.plist, entitlements.
- `Voco/Platforms/Keyboard/`: keyboard extension sources, Info.plist, entitlements.
- `Voco/Packages/RNNoise/`: local RNNoise Swift package.
- `Voco/Tests/`: fork tests and harnesses, included in the existing `VoiceInkTests` target.
- `Voco/Scripts/`, `Voco/Tools/`, `Voco/Examples/`, `Voco/Docs/`: fork tooling, examples, and documentation.
- `Voco/.gitignore`: versioned ignore rules for generated files under `Voco/`. Existing local-only root paths remain in `.git/info/exclude` and are not committed.

Required exceptions left outside `Voco/`: root `AGENTS.md`, root `.gitattributes`, add/add `VoiceInk/Localizable.xcstrings` (stage 2), and `VoiceInk.xcodeproj/xcshareddata/xcschemes/VocoiOS.xcscheme` (Xcode scheme location). The root `README.md`, `.gitignore`, and `Makefile` match upstream. Voco's previous English and Traditional Chinese readmes are in `Voco/Docs/`.

## Build and test

Use a separate derived-data path for reproducible local verification.

```bash
xcodebuild -skipPackagePluginValidation -skipMacroValidation -project VoiceInk.xcodeproj -scheme VoiceInk -configuration Debug \
  -allowProvisioningUpdates -allowProvisioningDeviceRegistration \
  -derivedDataPath /tmp/voco-stage1-dd-macos build

xcodebuild -skipPackagePluginValidation -skipMacroValidation test -project VoiceInk.xcodeproj -scheme VoiceInk \
  -destination 'platform=macOS' -only-testing:VoiceInkTests \
  -derivedDataPath /tmp/voco-stage1-dd-tests
```

VocoiOS and VocoKeyboard are excluded from verification starting 2026-10-24. They may already fail to build before this relocation (`PredefinedPrompts` is absent from `HEAD`); repair them only when they are needed again. Do not alter their shared membership exceptions for this stage.

The full `VoiceInkTests` suite is required. A known order-dependent failure may be counted as pre-existing only after the same failure is reproduced from the pre-stage-1 commit in a separate worktree.

## Trial merge gate

Run against the exact upstream commit above in a temporary detached worktree, collect `git diff --name-only --diff-filter=U`, then abort and remove the worktree. Stage 1 produced **121 unmerged paths** (`UU 79`, `UD 41`, `AA 1`) after the stage-0 baseline of 182. Stage 2 produced **99** (`UU 62`, `UD 37`, `AA 0`). Stage 3 produced **97** (`UU 60`, `UD 37`, `AA 0`). Do not deploy if a stage exceeds its baseline or if a cleared path cannot be matched to a removed patch.

## Release deployment

Back up the installed app before replacement. Deploy only the Release product to `/Applications/Voco.app` with `ditto`.

```bash
xcodebuild -skipPackagePluginValidation -skipMacroValidation -project VoiceInk.xcodeproj -scheme VoiceInk -configuration Release \
  -allowProvisioningUpdates -allowProvisioningDeviceRegistration \
  -derivedDataPath /tmp/voco-stage1-dd-release build

ditto /Applications/Voco.app /tmp/Voco.app.bak-20261024
pkill -x Voco 2>/dev/null; sleep 1
rm -rf /Applications/Voco.app
ditto /tmp/voco-stage1-dd-release/Build/Products/Release/Voco.app /Applications/Voco.app
open /Applications/Voco.app
```

Verify launch, bundle ID `com.jasonchien.Voco`, unchanged version/build, `codesign --verify --deep`, and byte-identical entitlements with the backup. `codesign --verify --deep --strict` is not a deployment gate because the upstream `whisper.framework` symlink layout is already known to fail it; do not clean or hand-edit the bundle for this. The App Group directory is not a gate while the app has `com.apple.security.app-sandbox` set to false; do not create it manually. Restore `/tmp/Voco.app.bak-20261024` with `ditto` if any verification fails.

## Deployment record

On 2026-10-24 the stage-1 Release build and `ditto` deployment to `/Applications/Voco.app` succeeded. Verification followed `AGENTS.md`; `codesign --verify --deep --strict` was not used as a gate.

- Backup: `/tmp/Voco.app.bak-20261024`.
- `codesign --verify --deep` returned 0 for both backup and installed app. Entitlements were byte-identical (`diff` empty). Bundle ID and version remained `com.jasonchien.Voco`, `2.0 (179)`.
- The app stayed alive after 30 seconds with no crash. Logs included the existing `Prewarm failed: Unsupported WAV audio format` and network/window noise; `sound7.wav` had the same checksum in both bundles.
- Forced Qwen3 Edge TTS smoke produced the identical transcript before and after relocation: `世纪风电二零七二今日股价一百八十三元。` (normalized `世紀風電二零七二今日股價一百八十三元`). Runtime correction replay results were identical after excluding path fields; both returned `ready: true` and `deployReady: true`.
- `Voco/Tests/Qwen3ASRAdapterTests.swift` now resolves the repository root from its new `Voco/Tests` location so existing `LocalModels` fixtures are not skipped.

Stage 2 deployment (2026-10-24):

- App backup: `/tmp/Voco.app.bak-20261024-p2` (107 MB).
- Data backup: `/tmp/voco-data-bak-20261024-p2` (9.1 GB): `default.store`, `dictionary.store`, `stats.store` with `-wal`/`-shm`, `com.jasonchien.Voco.plist`, `Qwen3ASRContextBias/`, model directories, and `DBDeleteBackups`; `Recordings/`, `ShadowLogs/` and `AutoApplyModels/` were excluded because they are media and logs, not settings.
- Release build output: `/tmp/voco-p2-dd-release/Build/Products/Release/Voco.app`.
- Checks after deployment: 30-second liveness, `codesign --verify --deep` exit 0, entitlements byte-identical (`diff` empty), `com.jasonchien.Voco` / `2.0 (179)` unchanged, no crash report after launch. SwiftData row counts were identical before and after: transcriptions 33,095, word replacements 355, session metrics 31,521.
- Qwen3 file smoke: `Qwen3ASRAdapterTests/edgeTTSCenturyWindStockQuoteSpecialSmokeUsesAdapter` was forced with `VOCO_QWEN3_ADAPTER_EDGE_TTS_SMOKE=1` and passed (company, stock code, price and adapter assertions). Runtime replay via `voco_runtime_correction_control.py replayGate` returned `deployReady: true`, `runtimeReplayPass: true`, `rowsMatchingExpectedFinalText: 5`, `finalTextRegressionCount: 0`.

Stage 3 deployment (2026-10-24), through `c4268e5e`:

- App backup: `/tmp/Voco.app.bak-20261024-p3` (108 MB). Data backup: `/tmp/voco-data-bak-20261024-p3` (9.1 GB), with the same store/settings/model scope as p2; recordings, shadow logs and auto-apply model artifacts were excluded. Earlier backups were not modified.
- Release build exited 0; product: `/tmp/voco-p3-dd-release/Build/Products/Release/Voco.app`. Installation used `ditto` to `/Applications/Voco.app`.
- The installed app stayed alive after 30 seconds (PID 26993). `codesign --verify --deep` exited 0; entitlements were byte-identical (`/tmp/voco-p3-entitlements.diff` is empty). Bundle ID/version remained `com.jasonchien.Voco` / `2.0 (179)`. No crash report was created after deployment; logs retained the known prewarm WAV, window/network and CloudKit scheduler noise.
- SwiftData counts were identical before backup, in the backup, and after deployment: transcriptions 33,226, word replacements 355, session metrics 31,639, vocabulary 136.
- Qwen3 file smoke actually ran for 2.877 seconds and produced `世纪风电二零七二今日股价一百八十三元。` (normalized `世紀風電二零七二今日股價一百八十三元`), identical to the accepted baseline. The suite executed 21 tests with 21 passes. An earlier narrow selector returned 0 tests and was not counted; a suite run without the fixture marker also did not count as the smoke.
- Runtime replay was semantically identical before and after deployment after excluding path fields: `deployReady: true`, `runtimeReplayPass: true`, `rowsMatchingExpectedFinalText: 5`, `finalTextRegressionCount: 0`.
- Physical recording/device/paste/Edit Mode checks remain Jason's manual acceptance items. Jason accepted stage 3 on 2026-10-25 and allowed the push; `c3de55a4c15c8cf44b5a7bf0cb0e6c677990bfc0` is on `origin/main`.

Stage 4 merge and deployment (2026-10-25):

- Actual merge commit: `8692052f01b225fd626e96f48127439d35533baf`, with parents `c3de55a4` and `22e5ec54`. The real merge produced exactly 97 unmerged paths (`UU 60`, `UD 37`), matching the trial merge with no new paths. All conflicts were resolved; the final unmerged count is zero. It has not been pushed.
- Upstream's new App/Features/Infrastructure/DesignSystem shells were adopted. Voco kept its recording start flow, shared registry, four processing hooks, Edit Mode overrides, schema metadata, per-mode punctuation/lowercase settings, license policy and correction learning. Upstream AutoLearn is disabled by `VocoAutoLearnPolicy`; its defaults, configure/apply/undo paths and UI are gated.
- `CloudModelCardView` was preserved in `Voco/Sources/UI/CloudModelCardView.swift` because pinned upstream deletes the type while still calling it from `ModelCardView`. `AudioCleanupSettingsView` was preserved in `Voco/Sources/UI/Settings/AudioCleanupSettingsView.swift` for the same reason. These are not revived upstream paths.
- `Package.resolved` was regenerated from the merged dependency graph. `Localizable.xcstrings` has 2,318 keys across de/en/fr/zh-Hans/zh-Hant. `InfoPlist.xcstrings` carries Voco branding and the existing zh-Hant permission strings; legacy InfoPlist sources moved to `Voco/Config/Localization/` so they no longer form duplicate synchronized variant groups.
- Debug and Release builds exited 0. The full scheme produced 513 tests: 502 passed, 10 failed, 1 skipped. The 10 unique failures exactly match the stage-6 baseline; no new failures were accepted. The old Voco file retains all 186 `@Test` functions, upstream adds two AutoLearn tests, and the three unique upstream UI tests passed across 12 executions. An earlier run had only seven failures because three known CloudKit timing failures did not reproduce; the next complete run reproduced exactly the baseline ten. Both results are recorded rather than treating the passes as a fix.
- Python tools: `Ran 74 tests in 0.148s ... OK`. Localization generator: 7 tests OK.
- App backup: `/tmp/Voco.app.bak-20261024-p4`; data backup: `/tmp/voco-data-bak-20261024-p4`, with the same scope as p3 and no recordings or ShadowLogs. Earlier backups were not modified. Installation used the default DerivedData Release product and `ditto` to `/Applications/Voco.app`.
- The installed app stayed alive after 30 seconds (PID 12101). `codesign --verify --deep` exited 0 and entitlements were byte-identical (`/tmp/voco-p4-installed-entitlements.plist` versus backup). Bundle ID is `com.jasonchien.Voco`; upstream version advanced from `2.0 (179)` to `2.21 (221)` intentionally. No new launch crash was found; known prewarm WAV, localhost and CloudKit noise remains.
- SwiftData counts were identical before backup, in the backup, and after deployment: transcriptions 33,428, word replacements 330, session metrics 31,829, vocabulary 136.
- The forced Qwen3 adapter suite passed all 21 tests. The Edge TTS test actually ran for 2.088 seconds and produced `世纪风电二零七二今日股价一百八十三元。` (normalized `世紀風電二零七二今日股價一百八十三元`), identical to the accepted baseline. An earlier guard-skipped run was not counted as smoke.
- Runtime replay was semantically identical before and after deployment after excluding path fields: `deployReady: true`, `runtimeReplayPass: true`, `rowsMatchingExpectedFinalText: 5`, `finalTextRegressionCount: 0`.
- Jason's manual acceptance is still required for recording/device/paste/Edit Mode and the new upstream UI. Stage 4 is not pushed; stage 5 has not started.

### Ephemeral dictation hooks (2026-10-25)

The T3「無痕對話」signal and per-session privacy policy live in `Voco/Sources/Privacy/`. The following upstream files retain only the thin integration points needed by the current recorder/pipeline shells. No new local transcription, audio, evidence, learning or Worker payload is written for an ephemeral session; if the signal arrives after cloud ASR or OCR already ran, that earlier external side effect is surfaced rather than hidden. Normal sessions keep their existing records and evidence.

| Path | Hook and Voco owner | Why it must remain |
|---|---|---|
| `VoiceInk/Features/Recording/State/RecorderStateProvider.swift` | Optional `isEphemeralDictation` / `ephemeralPrivacyNotice`, backed by `VocoDictationPrivacyController` | Upstream recorder views consume this protocol; default false keeps other providers unchanged |
| `VoiceInk/Features/Recording/Presentation/RecorderUIManager.swift` | Begin/refresh privacy before Edit Mode detection and skip detection when ephemeral | Hotkey capture starts before `VoiceInkEngine.toggleRecord`; the early hook prevents selection/context capture |
| `VoiceInk/Features/Recording/Workflows/VoiceInkEngine.swift` | Composition and lifecycle hooks for `VocoDictationPrivacyController` and test boundaries | Upstream owns recording sequencing; pending-row/audio cleanup, early cloud rejection, context/Refine/AutoLearn gates and shared registry must attach here |
| `VoiceInk/Features/Recording/Workflows/TranscriptionPipeline.swift` | Privacy refresh, `VocoEvidenceScope`, enhancement/persistence gates and ephemeral output configuration | Upstream owns transcribe/enhance/deliver/save sequencing; evidence commits only after normal completion |
| `VoiceInk/Features/Recording/Context/RecordingContextSnapshot.swift` | Optional screen-text provider for the real engine capture path | Upstream owns clipboard/selection/screen capture tasks; production keeps `ScreenCaptureService`, ephemeral skips the entire capture |
| `VoiceInk/Features/Recording/Workflows/TranscriptionDelivery.swift` | Ephemeral paste-only branch and `PasteOverride` test boundary | Upstream owns follow-up/respond/custom-command delivery; ephemeral must not invoke those branches |
| `VoiceInk/Features/Recording/Components/RecorderComponents.swift` | `VocoEphemeralRecorderBadge` slot in `RecorderStatusDisplay` | Upstream owns shared mini/notch status rendering |
| `VoiceInk/Features/Recording/Views/MiniRecorderView.swift` | Pass ephemeral state to the shared badge slot | Upstream owns recorder layout |
| `VoiceInk/Features/Recording/Views/NotchRecorderView.swift` | Pass ephemeral state to the shared badge slot | Upstream owns recorder layout |
| `VoiceInk/Infrastructure/SystemIntegration/Paste/CursorPaster.swift` | Thread ephemeral flag through paste methods, suppress AutoLearn and mark clipboard writes when applicable | Upstream owns directTyping/clipboard delivery and paste outcome |
| `VoiceInk/Infrastructure/SystemIntegration/Paste/ClipboardManager.swift` | `org.nspasteboard.ConcealedType` plus Transient support supplied by the privacy session | Upstream owns generic clipboard write; Concealed is added only for actual ephemeral clipboard writes |
| `VoiceInk/Infrastructure/Providers/Transcription/Whisper/LibWhisper.swift` | Native `print_realtime` / `print_timestamps` disabled | whisper.cpp otherwise emits transcript text outside Swift `Logger` |
| `VoiceInk/Features/Recording/Processing/TranscriptionOutputFilter.swift` | Metadata-only result logging | Upstream owns the shared output filter used by recorder/import/retranscribe; raw text must not enter unified logs |
| `VoiceInk/Features/Enhancement/Workflows/AIEnhancementService.swift` | Error logging limited to provider/model/duration/error type | Provider errors can contain request or response text |

`RecordingContextSnapshot.swift` also accepts an optional screen-text provider for the real `VoiceInkEngine` capture path. Production still uses `ScreenCaptureService`; ephemeral skips capture entirely. `VocoTranscriptionServiceRegistry` remains Voco-owned and is subclassed only at test composition boundaries.

### Dictionary merge guard (2026-10-26)

Before stage 4 the launch hook called Voco's `removeExactDuplicateContent`, which only removes exact duplicates. The stage-4 resolution of `VoiceInk.swift` adopted upstream `cleanUpDictionaryContent` (upstream `d217463a`), which merges every row with the same destination and ignores `isEnabled`. On 2026-10-25 12:11 it merged disabled Edit Mode learning candidates into enabled rules (`檔案→記憶`, `出發→觸發`, `預測→預設值`, `遊的風口→Open Code`) and confirmed rules into disabled candidates (`三 11B→31B`, `COI→CLI`). The pre-merge rows are in `/tmp/voco-data-bak-20261024-p3/dictionary.store`.

`WordReplacement.destinationMergeState` (`Voco/Sources/Persistence/WordReplacement+Voco.swift`) returns nil for learning candidates and `isEnabled` otherwise. Rows only share a destination row when this value matches.

| Path | Hook | Why it must remain |
|---|---|---|
| `VoiceInk/Features/Dictionary/Workflows/DictionaryService.swift` | Cleanup keys exact duplicates and destination merges by merge state; add/edit only merge into rows with the same merge state | Upstream owns launch/import cleanup and the add/edit sheets; upstream has no learning candidates |
| `VoiceInk/Features/Dictionary/Workflows/DictionaryImportExportService.swift` | Merge-mode import only merges into enabled non-candidate rows | Same reason; imported rows are enabled |
| `VoiceInk/App/VoiceInk.swift` | One `VocoDictionaryRepairPlan.applyIfRequested` call next to the Worker sync start, skipped under tests | Repairs must run in the CloudKit-enabled app process after container creation |

Direct SQLite edits leave no persistent history, so CloudKit never uploads them and a reinstall restores the old rows. `Voco/Sources/Persistence/VocoDictionaryRepairPlan.swift` applies a JSON plan (updates, inserts with original ids and learning metadata, deletes) through SwiftData when Voco is launched with `-VocoDictionaryRepairPlanPath <file>`; the file is renamed to `*.applied`. Plans contain dictionary text and stay in `~/Library/Application Support/com.jasonchien.Voco/DictionaryRepair/`, never in git.

On 2026-10-26 the dictionary was first fixed with SQLite, reverted to the CloudKit-consistent state, then repaired through the plan: transaction 489 recorded 5 inserts, 6 updates and 1 delete (`raw→corrected`, leaked by an earlier ephemeral test). Export event 16577 succeeded; all 11 touched rows have record metadata with last exported transaction 489, the deleted row's metadata is gone, and no metadata is pending upload or cloud delete.

`VocoCanonicalizationService.wordReplacementTerms` now drops only protected aliases instead of the whole merged row, so `轉錄, 裝載, 圖案→專案` keeps `裝載` and `圖案`. Upstream AutoLearn's merge in `WordReplacementStore` is unchanged because `VocoAutoLearnPolicy` keeps AutoLearn off.

### Dashboard unit localization (2026-10-26)

Upstream `DashboardFormatters` hard-codes English suffixes, so the zh-Hant Dashboard showed `12K`, `3h 5m` and `1.2 sec`. Compact numbers now use Foundation's locale-aware `.compactName` (zh-Hant `1.2萬`, en unchanged `1.2K`); hour/minute/second strings go through `String(localized:)` with zh-Hant entries in `Voco/Localization/overrides.json`.

| Path | Hook | Why it must remain |
|---|---|---|
| `VoiceInk/Features/Dashboard/Support/DashboardFormatters.swift` | `formattedCompactNumber` uses `.compactName`; duration strings wrapped in `String(localized:)` | Upstream owns the shared Dashboard formatters and only ships English |
| `VoiceInk/Features/Dashboard/Components/ModelUsageCard.swift` | `0m` fallback reuses the localized formatter | Same reason |
| `VoiceInk/Features/Dashboard/Components/DashboardActivityCalendarCard.swift` | Best-day `%@ words` wrapped in `String(localized:)` | Same reason |

### Quick History rule assistant (2026-10-26)

Upstream Quick History (new in `22e5ec54`) had no rule assistant: the main-window entry lives on the audio player, which the quick panel hides. The quick panel now swaps its detail page for `VocoQuickHistoryRuleAssistantView` (`Voco/Sources/RuleAssistant/UI/`) and grows from 470 to 640 pt, keeping the top edge. Staying inside the floating panel keeps paste targeting the app it was opened from; the session registry keeps in-flight work when the panel closes.

| Path | Hook | Why it must remain |
|---|---|---|
| `VoiceInk/Features/History/Views/TranscriptionDetailView.swift` | Quick-panel header shows the `Fix with AI` button when `onRuleAssistant` is set | Upstream owns the shared detail view; the quick panel has no audio player to host the button |
| `VoiceInk/Features/History/Views/QuickHistoryView.swift` | Shows the rule assistant view and switches panel height | Upstream owns the quick panel layout |
| `VoiceInk/Features/History/Views/QuickHistoryViewModel.swift` | `isShowingRuleAssistant` state | Same reason |
| `VoiceInk/Features/History/Presentation/QuickHistoryController.swift` | Resizes the panel, passes Return through to the composer, Escape returns to details; hosting `sizingOptions = []` | Upstream intercepts Return for paste and Escape for back/dismiss |
| `VoiceInk/App/Windows/PersistentQuickPanel.swift` | Keys pass to the text view while an IME composition is open; resize stays on screen | Without it Return/Escape during 注音 composition paste or close the panel |

### Modifier-only toggle starts on press (2026-10-26)

Upstream `ee6cf208` (in `22e5ec54`) waits for the release of a modifier-only shortcut in toggle mode before it starts recording. Jason's primary shortcut is Fn alone in toggle mode and he speaks right after pressing it, so capture began one key-hold later. Across 1,480 recordings from 10-23 to the merge deployment (10-25 12:29) and 954 after it, recordings whose speech is already loud within the first 200 ms rose from 15.7% to 39.2%, and the median speech onset moved from 390 ms to 250 ms into the file. The jump starts in the 10-25 12:00 hour; stage 2/3 deployments on 10-24 did not move it. The first syllable was lost in those recordings, e.g. 「不用管」→「用管」.

`Voco/Sources/Shortcuts/VocoShortcutActivationPolicy.swift` keeps the release-only behavior for double tap and starts toggle, hybrid and push-to-talk on press, as before the merge. A chord such as Fn+arrow still goes through the existing interruption path, which cancels an accidental start.

| Path | Hook | Why it must remain |
|---|---|---|
| `VoiceInk/Features/Shortcuts/Coordination/RecordingShortcutManager.swift` | `standaloneModifierActions` asks `VocoShortcutActivationPolicy` | Upstream owns the recording shortcut monitor setup |
| `VoiceInk/Features/Shortcuts/Coordination/ModeShortcutManager.swift` | Same policy for mode shortcuts | Upstream owns the mode shortcut monitor setup |

### Appendix: remaining patched upstream paths (97)

Generated from the final stage-3 trial merge at `414d389b`. Every path below is either an intentional hook registered above or an already classified stage-4/config patch; status is the merge state against `22e5ec54`.

- `UU` `VoiceInk.xcodeproj/project.pbxproj`
- `UU` `VoiceInk.xcodeproj/project.xcworkspace/xcshareddata/swiftpm/Package.resolved`
- `UU` `VoiceInk/App/Lifecycle/ModelPrewarmService.swift`
- `UU` `VoiceInk/App/Migrations/StreamingKeysMigration.swift`
- `UU` `VoiceInk/App/Navigation/AppSidebar.swift`
- `UU` `VoiceInk/App/Notifications/NotificationManager.swift`
- `UD` `VoiceInk/AppDefaults.swift`
- `UU` `VoiceInk/DesignSystem/Components/AppControls.swift`
- `UU` `VoiceInk/Features/AudioImport/Views/AudioFileRow.swift`
- `UU` `VoiceInk/Features/AudioImport/Workflows/AudioFileTranscriptionManager.swift`
- `UU` `VoiceInk/Features/Dictionary/Views/DictionarySettingsView.swift`
- `UU` `VoiceInk/Features/Dictionary/Views/VocabularyView.swift`
- `UU` `VoiceInk/Features/Dictionary/Views/WordReplacementView.swift`
- `UU` `VoiceInk/Features/Enhancement/Workflows/AIEnhancementService.swift`
- `UU` `VoiceInk/Features/History/Workflows/AudioFileTranscriptionService.swift`
- `UU` `VoiceInk/Features/History/Workflows/LastTranscriptionService.swift`
- `UU` `VoiceInk/Features/Licensing/Views/LicenseManagementView.swift`
- `UU` `VoiceInk/Features/ModelLibrary/Components/LanguageSelectionView.swift`
- `UU` `VoiceInk/Features/ModelLibrary/Components/ModelCardView.swift`
- `UU` `VoiceInk/Features/ModelLibrary/Models/TranscriptionModel.swift`
- `UU` `VoiceInk/Features/ModelLibrary/Models/TranscriptionModelRegistry.swift`
- `UU` `VoiceInk/Features/ModelLibrary/State/CustomCloudModelManager.swift`
- `UU` `VoiceInk/Features/ModelLibrary/Views/APIKeyManagementView.swift`
- `UU` `VoiceInk/Features/ModelLibrary/Views/CustomProviderManagementView.swift`
- `UU` `VoiceInk/Features/ModelLibrary/Views/ModelManagementView.swift`
- `UU` `VoiceInk/Features/ModelLibrary/Views/ProviderCloudManagementView.swift`
- `UU` `VoiceInk/Features/ModelLibrary/Views/ProviderDetailPanel.swift`
- `UU` `VoiceInk/Features/ModelLibrary/Views/ProviderLocalManagementView.swift`
- `UU` `VoiceInk/Features/Modes/State/ModeConfig.swift`
- `UU` `VoiceInk/Features/Onboarding/Components/OnboardingLicenseCards.swift`
- `UU` `VoiceInk/Features/Onboarding/Components/TranscriptionModelDownloadCard.swift`
- `UU` `VoiceInk/Features/Onboarding/Models/OnboardingExperienceModels.swift`
- `UU` `VoiceInk/Features/Recording/Components/AudioVisualizerView.swift`
- `UU` `VoiceInk/Features/Recording/Components/RecorderComponents.swift`
- `UU` `VoiceInk/Features/Recording/Presentation/RecorderUIManager.swift`
- `UU` `VoiceInk/Features/Recording/Streaming/StreamingTranscriptionService.swift`
- `UU` `VoiceInk/Features/Recording/Views/MiniRecorderView.swift`
- `UU` `VoiceInk/Features/Recording/Views/NotchRecorderView.swift`
- `UU` `VoiceInk/Features/Recording/Workflows/TranscriptionPipeline.swift`
- `UU` `VoiceInk/Features/Recording/Workflows/TranscriptionService.swift`
- `UU` `VoiceInk/Features/Recording/Workflows/TranscriptionServiceRegistry.swift`
- `UU` `VoiceInk/Features/Recording/Workflows/VoiceInkEngine.swift`
- `UU` `VoiceInk/Features/Settings/Backup/BackupImporter.swift`
- `UU` `VoiceInk/Features/Settings/Backup/BackupTypes.swift`
- `UU` `VoiceInk/Features/Settings/Backup/ImportExportService.swift`
- `UU` `VoiceInk/Features/Settings/Support/EmailSupport.swift`
- `UU` `VoiceInk/Features/Settings/Views/SettingsView.swift`
- `UU` `VoiceInk/Features/Shortcuts/Coordination/RecordingShortcutManager.swift`
- `UU` `VoiceInk/Features/Shortcuts/State/ShortcutValidator.swift`
- `UU` `VoiceInk/Infrastructure/Audio/AudioFileProcessor.swift`
- `UU` `VoiceInk/Infrastructure/Audio/CoreAudioRecorder.swift`
- `UU` `VoiceInk/Infrastructure/Persistence/Models/SessionMetric.swift`
- `UU` `VoiceInk/Infrastructure/Persistence/Models/Transcription.swift`
- `UU` `VoiceInk/Infrastructure/Persistence/Models/WordReplacement.swift`
- `UU` `VoiceInk/Infrastructure/Persistence/Statistics/SessionMetricRecorder.swift`
- `UU` `VoiceInk/Infrastructure/Providers/Transcription/Cloud/SonioxProvider.swift`
- `UU` `VoiceInk/Infrastructure/Providers/Transcription/FluidAudio/Streaming/FluidAudioStreamingProvider.swift`
- `UU` `VoiceInk/Infrastructure/Providers/Transcription/Whisper/LibWhisper.swift`
- `UU` `VoiceInk/Infrastructure/Providers/Transcription/Whisper/WhisperPrompt.swift`
- `UU` `VoiceInk/Infrastructure/SystemIntegration/Accessibility/SelectedTextService.swift`
- `UD` `VoiceInk/Models/AIPrompts.swift`
- `UD` `VoiceInk/Models/LicenseViewModel.swift`
- `UD` `VoiceInk/Paste/CursorPaster.swift`
- `UD` `VoiceInk/PlaybackController.swift`
- `UD` `VoiceInk/Recorder.swift`
- `UD` `VoiceInk/Services/AudioDeviceConfiguration.swift`
- `UD` `VoiceInk/Services/AudioDeviceManager.swift`
- `UD` `VoiceInk/Services/DictionaryService.swift`
- `UD` `VoiceInk/Services/KeychainService.swift`
- `UD` `VoiceInk/Services/LogExporter.swift`
- `UD` `VoiceInk/Services/SessionMetricMigrationService.swift`
- `UD` `VoiceInk/Services/SystemInfoService.swift`
- `UD` `VoiceInk/Services/TranscriptionAutoCleanupService.swift`
- `UD` `VoiceInk/Transcription/Engine/TranscriptionDelivery.swift`
- `UD` `VoiceInk/Transcription/FluidAudio/FluidAudioTranscriptionService.swift`
- `UD` `VoiceInk/Transcription/Native/NativeAppleTranscriptionService.swift`
- `UD` `VoiceInk/Transcription/Processing/TranscriptionOutputFilter.swift`
- `UD` `VoiceInk/Transcription/Processing/WordReplacementService.swift`
- `UD` `VoiceInk/Views/AI Models/CloudModelCardView.swift`
- `UD` `VoiceInk/Views/AI Models/ModelSettingsPanel.swift`
- `UD` `VoiceInk/Views/Common/TranscriptionInfoPanel.swift`
- `UD` `VoiceInk/Views/Dashboard/DashboardContent.swift`
- `UD` `VoiceInk/Views/Dashboard/DashboardPromotionsSection.swift`
- `UD` `VoiceInk/Views/Dashboard/DashboardStatCard.swift`
- `UD` `VoiceInk/Views/Dashboard/ModelPerformancePanel.swift`
- `UD` `VoiceInk/Views/Dashboard/PerformanceAnalysisPanelView.swift`
- `UD` `VoiceInk/Views/Dashboard/PerformanceAnalysisView.swift`
- `UD` `VoiceInk/Views/DashboardView.swift`
- `UD` `VoiceInk/Views/History/InlineHistoryView.swift`
- `UD` `VoiceInk/Views/History/TranscriptionDetailView.swift`
- `UD` `VoiceInk/Views/History/TranscriptionHistoryView.swift`
- `UD` `VoiceInk/Views/Recorder/MiniWindowManager.swift`
- `UD` `VoiceInk/Views/Recorder/NotchWindowManager.swift`
- `UD` `VoiceInk/VoiceInk.swift`
- `UD` `VoiceInk/WindowManager.swift`
- `UD` `VoiceInkTests/VoiceInkTests.swift`
- `UU` `appcast.xml`
