// LLMSettingsView.swift
// LLM enhancement settings for VocoKeyboard, stored in App Group UserDefaults
// [AI-Claude: 2026-03-02]

import SwiftUI

struct LLMSettingsView: View {
    private let defaults = UserDefaults(suiteName: AppIdentifiers.appGroupID)

    @State private var isEnabled: Bool = false
    @State private var selectedProvider: String = "Groq"
    @State private var apiKey: String = ""
    @State private var modelName: String = ""
    @State private var selectedPromptId: String = PredefinedPrompts.taiwaneseChinesePromptId.uuidString

    private let providers = ["Cerebras", "Groq", "Gemini", "Anthropic", "OpenAI", "OpenRouter"]

    var body: some View {
        NavigationStack {
            Form {
                Section {
                    Toggle("AI Enhancement", isOn: $isEnabled)
                        .onChange(of: isEnabled) { _, newValue in
                            defaults?.set(newValue, forKey: "KeyboardLLMEnabled")
                        }
                } footer: {
                    Text("Enhance transcriptions with cloud AI (requires network)")
                }

                Section("Provider") {
                    Picker("Provider", selection: $selectedProvider) {
                        ForEach(providers, id: \.self) { provider in
                            Text(provider).tag(provider)
                        }
                    }
                    .onChange(of: selectedProvider) { _, newValue in
                        defaults?.set(newValue, forKey: "KeyboardLLMProvider")
                        // Reset model to provider default
                        modelName = ""
                        defaults?.set("", forKey: "KeyboardLLMModel")
                    }
                }

                Section("API Key") {
                    SecureField("Enter API Key", text: $apiKey)
                        .textContentType(.password)
                        .autocorrectionDisabled()
                        .textInputAutocapitalization(.never)
                        .onChange(of: apiKey) { _, newValue in
                            defaults?.set(newValue, forKey: "KeyboardLLMAPIKey")
                        }
                }

                Section {
                    TextField("Model name (leave empty for default)", text: $modelName)
                        .autocorrectionDisabled()
                        .textInputAutocapitalization(.never)
                        .onChange(of: modelName) { _, newValue in
                            defaults?.set(newValue, forKey: "KeyboardLLMModel")
                        }
                } header: {
                    Text("Model")
                } footer: {
                    Text("Default: \(defaultModelForProvider(selectedProvider))")
                }

                Section("Prompt") {
                    Picker("Enhancement Prompt", selection: $selectedPromptId) {
                        ForEach(PredefinedPrompts.all) { prompt in
                            Label(prompt.title, systemImage: prompt.icon)
                                .tag(prompt.id.uuidString)
                        }
                    }
                    .onChange(of: selectedPromptId) { _, newValue in
                        defaults?.set(newValue, forKey: "KeyboardLLMPromptId")
                    }
                }

                Section {
                    statusRow
                }
            }
            .navigationTitle("AI Enhancement")
            .onAppear(perform: loadSettings)
        }
    }

    @ViewBuilder
    private var statusRow: some View {
        let configured = isEnabled && !apiKey.isEmpty
        HStack {
            Image(systemName: configured ? "checkmark.circle.fill" : "exclamationmark.triangle.fill")
                .foregroundStyle(configured ? .green : .orange)
            Text(configured ? "Ready" : "Not configured — enter API key to enable")
                .font(.footnote)
                .foregroundStyle(.secondary)
        }
    }

    private func loadSettings() {
        isEnabled = defaults?.bool(forKey: "KeyboardLLMEnabled") ?? false
        selectedProvider = defaults?.string(forKey: "KeyboardLLMProvider") ?? "Groq"
        apiKey = defaults?.string(forKey: "KeyboardLLMAPIKey") ?? ""
        modelName = defaults?.string(forKey: "KeyboardLLMModel") ?? ""
        selectedPromptId = defaults?.string(forKey: "KeyboardLLMPromptId") ?? PredefinedPrompts.taiwaneseChinesePromptId.uuidString
    }

    private func defaultModelForProvider(_ provider: String) -> String {
        switch provider {
        case "Cerebras":   return "gpt-oss-120b"
        case "Groq":       return "openai/gpt-oss-120b"
        case "Gemini":     return "gemini-2.5-flash-lite"
        case "Anthropic":  return "claude-sonnet-4-6"
        case "OpenAI":     return "gpt-5.2"
        case "OpenRouter":  return "openai/gpt-oss-120b"
        default:           return ""
        }
    }
}
