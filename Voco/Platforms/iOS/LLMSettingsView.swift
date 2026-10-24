// LLMSettingsView.swift
// LLM enhancement settings for VocoKeyboard, stored in App Group UserDefaults
// [AI-Claude: 2026-03-02]

import SwiftUI

struct LLMSettingsView: View {
    private let defaults = UserDefaults(suiteName: AppIdentifiers.appGroupID)

    @State private var isEnabled: Bool = false
    @State private var selectedProvider: String = "Groq"
    @State private var apiKey: String = ""
    @State private var apiKeyValidationMessage: String?
    @State private var isValidatingKey: Bool = false
    @State private var modelName: String = ""
    @State private var selectedPromptId: String = PredefinedPrompts.taiwaneseChinesePromptId.uuidString
    
    @State private var openRouterModels: [OpenRouterModel] = []
    @State private var isLoadingModels: Bool = false
    
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
                        
                        // Fetch OpenRouter models when switching to OpenRouter
                        if newValue == "OpenRouter" && !apiKey.isEmpty && apiKey.count >= 20 {
                            Task {
                                await fetchOpenRouterModels()
                            }
                        }
                    }
                }

                Section("API Key") {
                    SecureField("Enter API Key", text: $apiKey)
                        .textContentType(.password)
                        .autocorrectionDisabled()
                        .textInputAutocapitalization(.never)
                        .onChange(of: apiKey) { _, newValue in
                            defaults?.set(newValue, forKey: "KeyboardLLMAPIKey")
                            validateAPIKey(newValue)
                            
                            // Fetch OpenRouter models when key is entered for OpenRouter
                            if selectedProvider == "OpenRouter" && newValue.count >= 20 {
                                Task {
                                    await fetchOpenRouterModels()
                                }
                            }
                        }
                    if let message = apiKeyValidationMessage {
                        Text(message)
                            .font(.caption)
                            .foregroundStyle(message.contains("Valid") ? .green : .orange)
                    }
                }

                Section {
                    if selectedProvider == "OpenRouter" && !openRouterModels.isEmpty {
                        Picker("Model", selection: $modelName) {
                            Text("Default (recommended)").tag("")
                            ForEach(openRouterModels, id: \.id) { model in
                                Text(model.displayName).tag(model.id)
                            }
                        }
                        .onChange(of: modelName) { _, newValue in
                            defaults?.set(newValue, forKey: "KeyboardLLMModel")
                        }
                        if isLoadingModels {
                            HStack {
                                ProgressView()
                                    .scaleEffect(0.8)
                                Text("Loading models...")
                                    .font(.caption)
                                    .foregroundStyle(.secondary)
                            }
                        }
                    } else {
                        TextField("Model name (leave empty for default)", text: $modelName)
                            .autocorrectionDisabled()
                            .textInputAutocapitalization(.never)
                            .onChange(of: modelName) { _, newValue in
                                defaults?.set(newValue, forKey: "KeyboardLLMModel")
                            }
                    }
                } header: {
                    Text("Model")
                } footer: {
                    Text(selectedProvider == "OpenRouter" && !openRouterModels.isEmpty 
                         ? "Select a model or leave empty for recommended"
                         : "Default: \(defaultModelForProvider(selectedProvider))")
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
        
        if !apiKey.isEmpty {
            validateAPIKey(apiKey)
        }
        
        // Auto-fetch OpenRouter models on appear if OpenRouter is selected
        if selectedProvider == "OpenRouter" && !apiKey.isEmpty && apiKey.count >= 20 {
            Task {
                await fetchOpenRouterModels()
            }
        }
    }

    private func validateAPIKey(_ key: String) {
        let trimmed = key.trimmingCharacters(in: .whitespacesAndNewlines)
        
        guard !trimmed.isEmpty else {
            apiKeyValidationMessage = nil
            return
        }
        
        switch selectedProvider {
        case "Anthropic":
            if trimmed.hasPrefix("sk-ant-") && trimmed.count >= 20 {
                apiKeyValidationMessage = "Valid Anthropic key format"
            } else {
                apiKeyValidationMessage = "Key should start with 'sk-ant-'"
            }
        case "OpenAI":
            if trimmed.hasPrefix("sk-") && trimmed.count >= 30 {
                apiKeyValidationMessage = "Valid OpenAI key format"
            } else {
                apiKeyValidationMessage = "Key should start with 'sk-'"
            }
        case "OpenRouter":
            if trimmed.count >= 20 {
                apiKeyValidationMessage = "Validating..."
                Task {
                    await validateOpenRouterKey(trimmed)
                }
            } else {
                apiKeyValidationMessage = "Key seems too short"
            }
        case "Groq":
            if trimmed.count >= 20 {
                apiKeyValidationMessage = "Validating..."
                Task {
                    await validateGroqKey(trimmed)
                }
            } else {
                apiKeyValidationMessage = "Key seems too short"
            }
        case "Gemini":
            if trimmed.count >= 20 {
                apiKeyValidationMessage = "Key length OK"
            } else {
                apiKeyValidationMessage = "Key seems too short"
            }
        default:
            apiKeyValidationMessage = nil
        }
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
    
    private func fetchOpenRouterModels() async {
        isLoadingModels = true
        
        do {
            let url = URL(string: "https://openrouter.ai/api/v1/models")!
            var request = URLRequest(url: url)
            request.setValue("Bearer \(apiKey)", forHTTPHeaderField: "Authorization")
            request.setValue("application/json", forHTTPHeaderField: "Content-Type")
            request.timeoutInterval = 15
            
            let (data, response) = try await URLSession.shared.data(for: request)
            
            guard let httpResponse = response as? HTTPURLResponse, httpResponse.statusCode == 200 else {
                isLoadingModels = false
                return
            }
            
            let result = try JSONDecoder().decode(OpenRouterModelsResponse.self, from: data)
            openRouterModels = result.data.sorted { $0.id < $1.id }
        } catch {
            openRouterModels = []
        }
        
        isLoadingModels = false
    }
    
    private func validateOpenRouterKey(_ key: String) async {
        isValidatingKey = true
        
        do {
            let url = URL(string: "https://openrouter.ai/api/v1/auth/key")!
            var request = URLRequest(url: url)
            request.setValue("Bearer \(key)", forHTTPHeaderField: "Authorization")
            request.timeoutInterval = 10
            
            let (_, response) = try await URLSession.shared.data(for: request)
            
            guard let httpResponse = response as? HTTPURLResponse else {
                apiKeyValidationMessage = "Validation failed"
                isValidatingKey = false
                return
            }
            
            if httpResponse.statusCode == 200 {
                apiKeyValidationMessage = "✓ Valid key"
                // Also fetch models on successful validation
                await fetchOpenRouterModels()
            } else if httpResponse.statusCode == 401 {
                apiKeyValidationMessage = "✗ Invalid key"
            } else {
                apiKeyValidationMessage = "Validation error (\(httpResponse.statusCode))"
            }
        } catch {
            apiKeyValidationMessage = "Validation failed: \(error.localizedDescription)"
        }
        
        isValidatingKey = false
    }
    
    private func validateGroqKey(_ key: String) async {
        isValidatingKey = true
        
        do {
            let url = URL(string: "https://api.groq.com/openai/v1/models")!
            var request = URLRequest(url: url)
            request.setValue("Bearer \(key)", forHTTPHeaderField: "Authorization")
            request.timeoutInterval = 10
            
            let (_, response) = try await URLSession.shared.data(for: request)
            
            guard let httpResponse = response as? HTTPURLResponse else {
                apiKeyValidationMessage = "Validation failed"
                isValidatingKey = false
                return
            }
            
            if httpResponse.statusCode == 200 {
                apiKeyValidationMessage = "✓ Valid key"
            } else if httpResponse.statusCode == 401 {
                apiKeyValidationMessage = "✗ Invalid key"
            } else {
                apiKeyValidationMessage = "Validation error (\(httpResponse.statusCode))"
            }
        } catch {
            apiKeyValidationMessage = "Validation failed: \(error.localizedDescription)"
        }
        
        isValidatingKey = false
    }
}

struct OpenRouterModel: Identifiable, Codable {
    let id: String
    let name: String?
    
    var displayName: String {
        if let name = name, !name.isEmpty {
            return name
        }
        return id.replacingOccurrences(of: "/", with: " • ")
    }
}

struct OpenRouterModelsResponse: Codable {
    let data: [OpenRouterModel]
}
