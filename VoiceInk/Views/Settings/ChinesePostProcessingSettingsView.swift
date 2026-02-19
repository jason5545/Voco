import SwiftUI

struct ChinesePostProcessingSettingsView: View {
    @ObservedObject private var service = ChinesePostProcessingService.shared

    // Expansion states
    @State private var isMainExpanded = false
    @State private var isAdvancedExpanded = false

    var body: some View {
        Section {
            // Main toggle
            ExpandableSettingsRow(
                isExpanded: $isMainExpanded,
                isEnabled: $service.isEnabled,
                label: "Chinese Post-Processing",
                infoMessage: "Process transcription through OpenCC conversion, pinyin correction, spoken punctuation conversion, and repetition detection for Taiwanese Mandarin."
            ) {
                // Sub-feature toggles
                Toggle("OpenCC Simplified → Traditional", isOn: $service.isOpenCCEnabled)

                Toggle("Pinyin Correction", isOn: $service.isPinyinCorrectionEnabled)

                Toggle("Spoken Punctuation Conversion", isOn: $service.isSpokenPunctuationEnabled)

                Toggle("Half-Width → Full-Width Punctuation", isOn: $service.isHalfWidthConversionEnabled)

                Toggle("Repetition Detection", isOn: $service.isRepetitionDetectionEnabled)

                Divider()
                    .padding(.vertical, 4)

                // Advanced section
                DisclosureGroup("Advanced", isExpanded: $isAdvancedExpanded) {
                    Toggle(isOn: $service.isConfidenceRoutingEnabled) {
                        HStack(spacing: 4) {
                            Text("Confidence Routing")
                            InfoTip("Skip AI Enhancement for high-confidence transcriptions. Uses log-prob for both Whisper and Qwen3, with text heuristic fallback for Qwen3.")
                        }
                    }

                    if service.isConfidenceRoutingEnabled {
                        LabeledContent("Log-Prob Threshold") {
                            HStack {
                                Slider(value: $service.logProbThreshold, in: -1.0...0.0, step: 0.05)
                                    .frame(width: 120)
                                Text(String(format: "%.2f", service.logProbThreshold))
                                    .foregroundColor(.secondary)
                                    .frame(width: 40)
                            }
                        }

                        LabeledContent("Qwen3 Skip Threshold") {
                            HStack {
                                Slider(value: Binding(
                                    get: { Double(service.qwen3SkipThreshold) },
                                    set: { service.qwen3SkipThreshold = Int($0) }
                                ), in: 10...80, step: 5)
                                    .frame(width: 120)
                                Text("\(service.qwen3SkipThreshold)")
                                    .foregroundColor(.secondary)
                                    .frame(width: 40)
                            }
                        }

                    }

                    Toggle("Context Memory", isOn: $service.isContextMemoryEnabled)

                    Toggle(isOn: $service.isLLMValidationEnabled) {
                        HStack(spacing: 4) {
                            Text("LLM Response Validation")
                            InfoTip("Reject invalid LLM responses (blacklisted phrases, excessive length) and fall back to pre-LLM text.")
                        }
                    }
                }
            }
        } header: {
            Text("Chinese Post-Processing")
        } footer: {
            Text("Optimized for Taiwanese Mandarin speech recognition with XVoice-derived corrections.")
        }
    }
}
