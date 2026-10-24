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
xcodebuild -project VoiceInk.xcodeproj -scheme VoiceInk -configuration Debug \
  -allowProvisioningUpdates -allowProvisioningDeviceRegistration \
  -derivedDataPath /tmp/voco-stage1-dd-macos build

xcodebuild test -project VoiceInk.xcodeproj -scheme VoiceInk \
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
xcodebuild -project VoiceInk.xcodeproj -scheme VoiceInk -configuration Release \
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
