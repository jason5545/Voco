// KeyboardEnhancementService.swift
// Lightweight LLM enhancement service for VocoKeyboard
// Shares Chinese post-processing engines + prompts + validator with macOS app,
// but uses App Group UserDefaults for config (no SwiftData/AppKit dependency).
// [AI-Claude: 2026-03-02]

import Foundation
import LLMkit
import os

/// LLM provider options for keyboard enhancement (subset of macOS AIProvider)
enum KeyboardLLMProvider: String, CaseIterable {
    case cerebras = "Cerebras"
    case groq = "Groq"
    case gemini = "Gemini"
    case anthropic = "Anthropic"
    case openAI = "OpenAI"
    case openRouter = "OpenRouter"

    var baseURL: String {
        switch self {
        case .cerebras:  return "https://api.cerebras.ai/v1/chat/completions"
        case .groq:      return "https://api.groq.com/openai/v1/chat/completions"
        case .gemini:    return "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"
        case .anthropic: return "https://api.anthropic.com/v1/messages"
        case .openAI:    return "https://api.openai.com/v1/chat/completions"
        case .openRouter: return "https://openrouter.ai/api/v1/chat/completions"
        }
    }

    var defaultModel: String {
        switch self {
        case .cerebras:   return "gpt-oss-120b"
        case .groq:       return "openai/gpt-oss-120b"
        case .gemini:     return "gemini-2.5-flash-lite"
        case .anthropic:  return "claude-sonnet-4-6"
        case .openAI:     return "gpt-5.2"
        case .openRouter: return "openai/gpt-oss-120b"
        }
    }
}

@MainActor
class KeyboardEnhancementService {
    static let shared = KeyboardEnhancementService()

    private let logger = Logger(subsystem: AppIdentifiers.subsystem, category: "KeyboardEnhancement")
    private let defaults = UserDefaults(suiteName: AppIdentifiers.appGroupID)
    private let baseTimeout: TimeInterval = 15

    // MARK: - App Group UserDefaults Keys

    private enum Keys {
        static let provider = "KeyboardLLMProvider"
        static let apiKey = "KeyboardLLMAPIKey"
        static let model = "KeyboardLLMModel"
        static let promptId = "KeyboardLLMPromptId"
        static let enabled = "KeyboardLLMEnabled"
    }

    // MARK: - Config Accessors

    var isEnabled: Bool {
        defaults?.bool(forKey: Keys.enabled) ?? false
    }

    var provider: KeyboardLLMProvider {
        guard let raw = defaults?.string(forKey: Keys.provider),
              let p = KeyboardLLMProvider(rawValue: raw) else { return .groq }
        return p
    }

    var apiKey: String {
        defaults?.string(forKey: Keys.apiKey) ?? ""
    }

    var model: String {
        let m = defaults?.string(forKey: Keys.model) ?? ""
        return m.isEmpty ? provider.defaultModel : m
    }

    var isConfigured: Bool {
        isEnabled && !apiKey.isEmpty
    }

    private var activePrompt: CustomPrompt? {
        guard let idStr = defaults?.string(forKey: Keys.promptId),
              let id = UUID(uuidString: idStr) else {
            return PredefinedPrompts.all.first { $0.id == PredefinedPrompts.taiwaneseChinesePromptId }
        }
        return PredefinedPrompts.all.first { $0.id == id }
    }

    // MARK: - Main Enhancement Pipeline

    /// Process text through Chinese post-processing + optional LLM enhancement
    func enhance(_ text: String, language: String?) async -> String {
        guard !text.isEmpty else { return text }

        // Step 1: Chinese post-processing (always runs for zh)
        let isChinese = language == "zh" || containsCJK(text)
        let postProcessing = ChinesePostProcessingService.shared
        var result = text

        if isChinese && postProcessing.isEnabled {
            let postResult = postProcessing.process(text)
            result = postResult.processedText

            // Step 2: LLM enhancement (if needed)
            if postResult.needsLLMCorrection && isConfigured {
                if let enhanced = await callLLMWithValidation(original: text, processed: result) {
                    return enhanced
                }
            }
        } else if isConfigured {
            // Non-Chinese: still try LLM if configured
            if let enhanced = await callLLMWithValidation(original: text, processed: result) {
                return enhanced
            }
        }

        return result
    }

    // MARK: - LLM Call

    private func callLLMWithValidation(original: String, processed: String) async -> String? {
        do {
            let enhanced = try await callLLM(text: processed)
            let validation = LLMResponseValidator.shared.validate(response: enhanced, original: processed)

            if validation.isValid {
                return AIEnhancementOutputFilter.filter(enhanced)
            }

            // Conservative retry if retryable
            if validation.isRetryable {
                logger.info("LLM validation failed (retryable): \(validation.reasons.joined(separator: ", "))")
                if let retried = try? await callLLMConservative(text: processed) {
                    let retryValidation = LLMResponseValidator.shared.validate(response: retried, original: processed)
                    if retryValidation.isValid {
                        return AIEnhancementOutputFilter.filter(retried)
                    }
                }
            } else {
                logger.warning("LLM validation failed: \(validation.reasons.joined(separator: ", "))")
            }
        } catch {
            logger.error("LLM call failed: \(error)")
        }
        return nil
    }

    private func callLLM(text: String) async throws -> String {
        guard let prompt = activePrompt else {
            throw KeyboardEnhancementError.noPrompt
        }

        let systemMessage = prompt.finalPromptText
        let userMessage = "\n<TRANSCRIPT>\n\(text)\n</TRANSCRIPT>"

        let result: String
        switch provider {
        case .anthropic:
            result = try await AnthropicLLMClient.chatCompletion(
                apiKey: apiKey,
                model: model,
                messages: [.user(userMessage)],
                systemPrompt: systemMessage,
                timeout: baseTimeout
            )
        default:
            guard let baseURL = URL(string: provider.baseURL) else {
                throw KeyboardEnhancementError.invalidURL
            }
            let temperature = model.lowercased().hasPrefix("gpt-5") ? 1.0 : 0.3
            let reasoningEffort = ReasoningConfig.getReasoningParameter(for: model)
            result = try await OpenAILLMClient.chatCompletion(
                baseURL: baseURL,
                apiKey: apiKey,
                model: model,
                messages: [.user(userMessage)],
                systemPrompt: systemMessage,
                temperature: temperature,
                reasoningEffort: reasoningEffort,
                timeout: baseTimeout
            )
        }
        return AIEnhancementOutputFilter.filter(result.trimmingCharacters(in: .whitespacesAndNewlines))
    }

    private func callLLMConservative(text: String) async throws -> String {
        let systemMessage = AIPrompts.conservativeRetryPrompt(uncertainWords: [])
        let userMessage = "\n<TRANSCRIPT>\n\(text)\n</TRANSCRIPT>"

        let result: String
        switch provider {
        case .anthropic:
            result = try await AnthropicLLMClient.chatCompletion(
                apiKey: apiKey,
                model: model,
                messages: [.user(userMessage)],
                systemPrompt: systemMessage,
                timeout: baseTimeout
            )
        default:
            guard let baseURL = URL(string: provider.baseURL) else {
                throw KeyboardEnhancementError.invalidURL
            }
            result = try await OpenAILLMClient.chatCompletion(
                baseURL: baseURL,
                apiKey: apiKey,
                model: model,
                messages: [.user(userMessage)],
                systemPrompt: systemMessage,
                temperature: 0.3,
                timeout: baseTimeout
            )
        }
        return result.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    // MARK: - Helpers

    private func containsCJK(_ text: String) -> Bool {
        text.unicodeScalars.contains {
            (0x4E00...0x9FFF).contains($0.value) || (0x3400...0x4DBF).contains($0.value)
        }
    }
}

enum KeyboardEnhancementError: LocalizedError {
    case noPrompt
    case invalidURL

    var errorDescription: String? {
        switch self {
        case .noPrompt: return "No enhancement prompt configured"
        case .invalidURL: return "Invalid LLM API endpoint URL"
        }
    }
}
