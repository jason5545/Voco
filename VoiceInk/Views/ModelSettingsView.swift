import SwiftUI

struct ModelSettingsView: View {
    @ObservedObject var whisperPrompt: WhisperPrompt
    @AppStorage(TranscriptionLanguageSupport.selectedLanguageKey) private var selectedLanguage: String = TranscriptionLanguageSupport.defaultLanguageCode
    @AppStorage("IsTextFormattingEnabled") private var isTextFormattingEnabled = true
    @AppStorage(PunctuationCleanupMode.userDefaultsKey) private var punctuationCleanupModeRaw = PunctuationCleanupMode.current().rawValue
    @AppStorage("LowercaseTranscription") private var lowercaseTranscription = false
    @AppStorage("IsVADEnabled") private var isVADEnabled = true
    @AppStorage("AppendTrailingSpace") private var appendTrailingSpace = true
    @AppStorage("ContextAwareInsertionEnabled") private var contextAwareInsertionEnabled = false
    @AppStorage("ContextAwareLLMMergeEnabled") private var contextAwareLLMMergeEnabled = false
    @AppStorage("PrewarmModelOnWake") private var prewarmModelOnWake = true
    @AppStorage("KeepModelAlive") private var keepModelAlive = false
    @AppStorage("KeepModelAliveOnBattery") private var keepModelAliveOnBattery = false
    @AppStorage("showLiveTextPreview") private var showLiveTextPreview = true
    @State private var customPrompt: String = ""
    @State private var isEditing: Bool = false

    private var punctuationCleanupMode: Binding<PunctuationCleanupMode> {
        Binding(
            get: {
                PunctuationCleanupMode(rawValue: punctuationCleanupModeRaw) ?? PunctuationCleanupMode.current()
            },
            set: { newMode in
                punctuationCleanupModeRaw = newMode.rawValue
                PunctuationCleanupMode.setCurrent(newMode)
            }
        )
    }

    var body: some View {
        Form {
            Section {
                VStack(alignment: .leading, spacing: 8) {
                    if isEditing {
                        TextEditor(text: $customPrompt)
                            .font(.system(size: 12))
                            .frame(minHeight: 40, maxHeight: 80)
                            .fixedSize(horizontal: false, vertical: true)
                            .scrollContentBackground(.hidden)

                        Button("Save") {
                            whisperPrompt.setCustomPrompt(customPrompt, for: selectedLanguage)
                            isEditing = false
                        }
                    } else {
                        Text(whisperPrompt.getLanguagePrompt(for: selectedLanguage))
                            .font(.system(size: 12))
                            .foregroundColor(.secondary)
                            .frame(maxWidth: .infinity, alignment: .leading)

                        Button("Edit") {
                            customPrompt = whisperPrompt.getLanguagePrompt(for: selectedLanguage)
                            isEditing = true
                        }
                    }
                }
            } header: {
                HStack(spacing: 4) {
                    Text("Output Format")
                    InfoTip(
                        "Only supported for local Whisper models. Unlike GPT, Voice Models(whisper) follows the style of your prompt rather than instructions. Use examples of your desired output format instead of commands.",
                        learnMoreURL: "https://cookbook.openai.com/examples/whisper_prompting_guide#comparison-with-gpt-prompting"
                    )
                }
            }

            Section {
                Toggle(isOn: $isTextFormattingEnabled) {
                    HStack(spacing: 4) {
                        Text("Paragraph breaks")
                        InfoTip("Apply intelligent text formatting to break large block of text into paragraphs.")
                    }
                }
                .toggleStyle(.switch)

                Toggle(isOn: $contextAwareInsertionEnabled) {
                    HStack(spacing: 4) {
                        Text("Context-Aware Insertion")
                        InfoTip("Read surrounding text at the cursor position and automatically adjust spacing, capitalization, and punctuation for natural insertion.")
                    }
                }
                .toggleStyle(.switch)

                if contextAwareInsertionEnabled {
                    Toggle(isOn: $contextAwareLLMMergeEnabled) {
                        HStack(spacing: 4) {
                            Text("LLM Merge")
                            InfoTip("When inserting in the middle of existing text, use AI to merge the inserted text seamlessly with surrounding content. Requires AI enhancement to be configured.")
                        }
                    }
                    .toggleStyle(.switch)
                    .padding(.leading, 20)
                }

                Picker(selection: punctuationCleanupMode) {
                    ForEach(PunctuationCleanupMode.allCases) { mode in
                        Text(mode.displayName).tag(mode)
                    }
                } label: {
                    HStack(spacing: 4) {
                        Text("Punctuation")
                        InfoTip("Keep preserves punctuation as transcribed. Remove all strips punctuation marks from the transcribed text. Remove trailing period only removes a final period from the transcribed text.")
                    }
                }
                .pickerStyle(.menu)

                Toggle(isOn: $lowercaseTranscription) {
                    HStack(spacing: 4) {
                        Text("Lowercase output")
                        InfoTip("Convert transcription output to lowercase.")
                    }
                }
                .toggleStyle(.switch)

                FillerWordsSettingsView()
            } header: {
                Text("Transcript Formatting")
            }

            Section {
                Toggle(isOn: $appendTrailingSpace) {
                    HStack(spacing: 4) {
                        Text("Add Space After Paste")
                        InfoTip("Add a trailing space after pasted transcription output.")
                    }
                }
                .toggleStyle(.switch)

                Toggle(isOn: $isVADEnabled) {
                    HStack(spacing: 4) {
                        Text("Voice Activity Detection (VAD)")
                        InfoTip("Detect speech segments and filter out silence to improve accuracy of local models.")
                    }
                }
                .toggleStyle(.switch)

                Toggle(isOn: $prewarmModelOnWake) {
                    HStack(spacing: 4) {
                        Text("Prewarm model (Experimental)")
                        InfoTip("Turn this on if transcriptions with local models (including Whisper MLX) are taking longer than expected. Runs silent background transcription on app launch and wake to trigger optimization.")
                    }
                }
                .toggleStyle(.switch)

                if prewarmModelOnWake {
                    Toggle(isOn: $keepModelAlive) {
                        HStack(spacing: 4) {
                            Text("Keep model in memory")
                            InfoTip("Periodically touch the model's memory pages (every 5 minutes) to prevent macOS from swapping them to disk. Keeps the first transcription after idle as fast as subsequent ones.")
                        }
                    }
                    .toggleStyle(.switch)
                    .padding(.leading, 20)

                    if keepModelAlive {
                        Toggle(isOn: $keepModelAliveOnBattery) {
                            HStack(spacing: 4) {
                                Text("Keep alive on battery")
                                InfoTip("By default, keep-alive pauses when running on battery to save power. Enable this to keep the model resident even on battery.")
                            }
                        }
                        .toggleStyle(.switch)
                        .padding(.leading, 40)
                    }
                }

                Toggle(isOn: $showLiveTextPreview) {
                    HStack(spacing: 4) {
                        Text("Show Live Text Preview")
                        InfoTip("Displays the live transcript preview in the recorder while speaking. Only applies when using real-time streaming models.")
                    }
                }
                .toggleStyle(.switch)
            } header: {
                Text("Advanced")
            }
        }
        .formStyle(.grouped)
        .scrollContentBackground(.hidden)
        .onChange(of: selectedLanguage) { oldValue, newValue in
            if isEditing {
                customPrompt = whisperPrompt.getLanguagePrompt(for: selectedLanguage)
            }
        }
    }
}
