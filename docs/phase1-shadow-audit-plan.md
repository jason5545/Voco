# Phase 1 Shadow Evidence Audit Plan

## Scope Guard

Phase 1 is analysis-only. It may log shadow evidence, classify evidence purity, extract phonetic features, audit copied local databases, and replay JSONL metrics. It must not change final inserted text, auto-apply candidates, redesign Rescue Mode, upgrade ASR models, or write learned replacements into production data.

## Files And Entry Points Found

- App entry point: `VoiceInk/VoiceInk.swift`
- Main transcription pipeline: `VoiceInk/Transcription/Engine/TranscriptionPipeline.swift`
- Fork-specific edit-mode and context-aware insertion hooks: `VoiceInk/Transcription/Engine/TranscriptionPipeline+ForkFeatures.swift`
- Final paste/delivery path: `VoiceInk/Transcription/Engine/TranscriptionDelivery.swift`
- Engine orchestration: `VoiceInk/Transcription/Engine/VoiceInkEngine.swift`
- Transcription service routing: `VoiceInk/Transcription/Engine/TranscriptionServiceRegistry.swift`
- Transcription protocol: `VoiceInk/Transcription/Engine/TranscriptionService.swift`
- Chinese post-processing pipeline: `VoiceInk/Services/ChinesePostProcessing/ChinesePostProcessingService.swift`
- Existing correction engines:
  - `VoiceInk/Services/ChinesePostProcessing/PinyinCorrector.swift`
  - `VoiceInk/Services/ChinesePostProcessing/HomophoneCorrectionEngine.swift`
  - `VoiceInk/Services/ChinesePostProcessing/NasalCorrectionEngine.swift`
  - `VoiceInk/Services/ChinesePostProcessing/PersonalCorrectionEngine.swift`
- Confidence routing: `VoiceInk/Services/VocoConfidenceGateService.swift`
- Candidate review feedback: `VoiceInk/Services/VocoCandidateReviewService.swift`
- Correction feedback staging: `VoiceInk/Services/CorrectionFeedbackLearningService.swift`
- Canonicalization metadata recording: `VoiceInk/Services/VocoCanonicalizationPipeline.swift`
- SwiftData models:
  - `VoiceInk/Models/Transcription.swift`
  - `VoiceInk/Models/WordReplacement.swift`
  - `VoiceInk/Models/CorrectionFeedback.swift`
  - `VoiceInk/Models/VocoCanonicalization.swift`
  - `VoiceInk/Models/SessionMetric.swift`
  - `VoiceInk/Models/VocabularyWord.swift`
- Local-only audit context found: `phonetic-confusion-audit.md`
- Existing tests:
  - `VoiceInkTests/VoiceInkTests.swift`
  - `VoiceInkTests/ModelPrewarmServiceTests.swift`
  - `VoiceInkUITests/VoiceInkUITests.swift`
  - `VoiceInkUITests/VoiceInkUITestsLaunchTests.swift`
- Existing scripts with direct SQLite reads:
  - `scripts/mine_personal_corrections.py`
  - `scripts/analyze_comma_cleanup.py`
  - `scripts/validate_suspicious_words.py`
  - `scripts/asr_eval/prepare_test_set.py`

## Actual Pipeline Hook Points

- Raw ASR output: immediately after `TranscriptionServiceRegistry.transcribe(...)` or `TranscriptionSession.transcribe(...)` in `TranscriptionPipeline.run()`.
- Filtered output: after `TranscriptionOutputFilter.filter(...)`.
- Formatting output: after `ParagraphFormatter.format(...)`, if enabled.
- Chinese post-processing output: `ChinesePostProcessingService.process(_:)`; currently returns only final `processedText`, so Phase 1 needs a trace object that records intermediate snapshots without changing output.
- OpenCC/Pinyin/Homophone/Nasal/Personal correction snapshots: inside `ChinesePostProcessingService.process(_:)`, around existing steps and `dataDrivenEngines`.
- Canonicalization/confidence route: `VocoCanonicalizationPipeline.normalizeWithAssessment(...)` and returned `VocoConfidenceAssessment`.
- Review selection: `VocoCandidateReviewService.acceptCandidate(...)` call inside `TranscriptionPipeline.run()`.
- LLM enhancement output: after `enhancementService.enhance(...)` and `validateEnhancedText(...)` in `TranscriptionPipeline.run()`.
- Edit Mode correction feedback: `TranscriptionPipeline+ForkFeatures.handleEditMode(...)`, where `recordCorrectionFeedback(...)` is already called.
- Final insertion text: just before `TranscriptionDelivery.deliver(...)`, and paste metadata in `TranscriptionDelivery.paste(...)`.
- Rollback hook: no explicit rollback path found in the main transcription pipeline during Step 0; Phase 1 should log `null`/`unknown` rather than create behavior.

## Data Model And SQLite Store Names

SwiftData stores are created in `VoiceInk/VoiceInk.swift`:

- `default.store`
  - SwiftData entity: `Transcription`
  - SQLite table observed in existing code: `ZTRANSCRIPTION`
  - Relevant columns implied by model properties include `ZTEXT`, `ZENHANCEDTEXT`, `ZRAWTRANSCRIPT`, `ZNORMALIZEDTRANSCRIPT`, `ZFINALPASTEDTEXT`, `ZCONFIDENCEROUTE`, `ZCONFIDENCESCORE`, `ZCORRECTIONFEEDBACKJSON`.
- `dictionary.store`
  - SwiftData entities: `VocabularyWord`, `WordReplacement`
  - SQLite table used by existing code: `ZWORDREPLACEMENT`
  - Relevant columns used by existing code: `ZORIGINALTEXT`, `ZREPLACEMENTTEXT`, `ZSOURCE`, `ZHITCOUNT`, `ZLASTSEENDATE`.
- `stats.store`
  - SwiftData entity: `SessionMetric`

Important current behavior:

- `PersonalCorrectionEngine` reads trusted replacements from `ZWORDREPLACEMENT` where `ZSOURCE IN ('editMode', 'correctionFeedback')`.
- `ZTEXT -> ZENHANCEDTEXT` differences are not trusted user confirmation.
- `phonetic-confusion-audit.md` marks `69 輪` as correct content and not a correction target.

## Build And Test Commands

- Project discovery: `xcodebuild -list`
- Unit test command: `xcodebuild test -project VoiceInk.xcodeproj -scheme VoiceInk -destination 'platform=macOS'`
- Release build command: `xcodebuild -project VoiceInk.xcodeproj -scheme VoiceInk -configuration Release -destination 'platform=macOS' build`
- Local build helper: `make local`
- There is no root `Package.swift`; the root app is Xcode-project based. A local package exists at `Packages/RNNoise/Package.swift`.

## Files Planned For Phase 1 Changes

- `docs/phase1-shadow-audit-plan.md`
- `scripts/voco_db_audit.py`
- `scripts/voco_shadow_replay.py`
- `VoiceInk/Phonetics/PhoneticFeatureExtractor.swift`
- `VoiceInk/Personalization/CorrectionEvidenceClassifier.swift`
- `VoiceInk/Diagnostics/PhoneticShadowLogger.swift`
- `VoiceInk/Services/ChinesePostProcessing/ChinesePostProcessingService.swift`
- `VoiceInk/Transcription/Engine/TranscriptionPipeline.swift`
- `VoiceInk/Transcription/Engine/TranscriptionPipeline+ForkFeatures.swift`
- `VoiceInk/Transcription/Engine/TranscriptionDelivery.swift`
- `VoiceInkTests/PhoneticFeatureExtractorTests.swift`
- `VoiceInkTests/CorrectionEvidenceClassifierTests.swift`
- `VoiceInkTests/PhoneticShadowLoggerTests.swift`
- `VoiceInkTests/Phase1NoOutputChangeTests.swift`

The Xcode project uses `PBXFileSystemSynchronizedRootGroup` for `VoiceInk` and `VoiceInkTests`, so new files under those roots should be picked up without manually adding file references.
