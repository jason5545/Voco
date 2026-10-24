# Voco fork patches

Repository: `jason5545/Voco` (`main`). Upstream: `Beingpax/VoiceInk`.

Refactor plan: [Voco/Docs/refactor-plan.md](Voco/Docs/refactor-plan.md).

## Upstream baseline

- Reviewed upstream commit: `22e5ec5479b712797c3b338fcb875961e58800b6` (`22e5ec54`).
- Merge base before stage 1: `8e31ffaacfc64d4f90805a0412e34de714d31fc9`.
- Stage 1 starts from `7f3e8e2476aaab4205cc20fbac77effa0f235fd1`.
- Pre-stage-1 trial merge: **182 unmerged paths** (`UU 82`, `AU 58`, `UD 41`, `AA 1`).
- Post-stage-1 trial merge: **121 unmerged paths** (`UU 79`, `AU 0`, `UD 41`, `AA 1`), accepted as the stage-1 gate on 2026-10-24. The reduction is 58 eliminated `AU` paths plus three eliminated `UU` paths from restoring `.gitignore`, `Makefile`, and `README.md` to upstream.

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

Run against the exact upstream commit above in a temporary detached worktree, collect `git diff --name-only --diff-filter=U`, then abort and remove the worktree. Stage 1 must produce exactly **124 unmerged paths**, with all 58 stage-0 `AU` paths gone. Do not deploy if this gate fails.

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

On 2026-10-24 the Release build and `ditto` deployment to `/Applications/Voco.app` succeeded. Verification followed `AGENTS.md`; `codesign --verify --deep --strict` was not used as a gate.

- Backup: `/tmp/Voco.app.bak-20261024`.
- `codesign --verify --deep` returned 0 for both backup and installed app. Entitlements were byte-identical (`diff` empty). Bundle ID and version remained `com.jasonchien.Voco`, `2.0 (179)`.
- The app stayed alive after 30 seconds with no crash. Logs included the existing `Prewarm failed: Unsupported WAV audio format` and network/window noise; `sound7.wav` had the same checksum in both bundles.
- Forced Qwen3 Edge TTS smoke produced the identical transcript before and after relocation: `世纪风电二零七二今日股价一百八十三元。` (normalized `世紀風電二零七二今日股價一百八十三元`). Runtime correction replay results were identical after excluding path fields; both returned `ready: true` and `deployReady: true`.
- `Voco/Tests/Qwen3ASRAdapterTests.swift` now resolves the repository root from its new `Voco/Tests` location so existing `LocalModels` fixtures are not skipped.
