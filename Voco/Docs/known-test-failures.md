# Known Test Failures

## VoiceInkTests.correctionFeedbackLearningSkipsProtectedSourceTerms

- Status: open
- Observed: `xcodebuild test -project VoiceInk.xcodeproj -scheme VoiceInk -destination 'platform=macOS' -only-testing:VoiceInkTests`
- Failure: `VoiceInkTests.correctionFeedbackLearningSkipsProtectedSourceTerms()`
- Retest: the same test passes when run alone with `-only-testing:VoiceInkTests/correctionFeedbackLearningSkipsProtectedSourceTerms`
- Current read: likely order-dependent shared state pollution in the broader `VoiceInkTests` suite.
- Phase 1 impact: not treated as a phonetic shadow regression. Phase 1 focused tests pass, and candidate application remains disabled.
- TODO: isolate the shared mutable defaults/model state that leaks into the protected-source-terms test, then make the full suite order-independent.
