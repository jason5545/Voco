// WhisperTokens.swift
// Shared Whisper special token IDs and language constants
// Used by both WhisperMLX (macOS) and WhisperCoreML (iOS) engines
// [AI-Claude: 2026-03-02]

import Foundation

/// Special token IDs for Whisper
struct WhisperTokens {
    // Base tokens
    static let eotTokenId = 50257         // <|endoftext|> (also used as EOS/PAD/BOS)
    static let sotTokenId = 50258         // <|startoftranscript|>
    static let translateTokenId = 50358   // <|translate|>
    static let transcribeTokenId = 50359  // <|transcribe|>
    static let noSpeechTokenId = 50362    // <|nospeech|>
    static let noTimestampsTokenId = 50363 // <|notimestamps|>

    // Language token range: 50259..50357 (99 languages for v1/v2), 50259..50358 (100 for v3)
    static let firstLanguageTokenId = 50259

    /// Number of language tokens for the model version.
    /// v1/v2 (nMels=80): 99 languages (token 50358 is <|translate|>, not a language)
    /// v3 (nMels=128): 100 languages (includes "yue")
    static func languageCount(nMels: Int) -> Int {
        return nMels >= 128 ? 100 : 99
    }

    /// Map language code to token ID
    static func languageTokenId(for code: String) -> Int? {
        guard let index = languageOrder.firstIndex(of: code) else { return nil }
        return firstLanguageTokenId + index
    }

    /// Map token ID to language code
    static func languageCode(for tokenId: Int) -> String? {
        let index = tokenId - firstLanguageTokenId
        guard index >= 0, index < languageOrder.count else { return nil }
        return languageOrder[index]
    }

    /// Whisper language order (same as tiktoken/openai-whisper)
    static let languageOrder: [String] = [
        "en", "zh", "de", "es", "ru", "ko", "fr", "ja", "pt", "tr",
        "pl", "ca", "nl", "ar", "sv", "it", "id", "hi", "fi", "vi",
        "he", "uk", "el", "ms", "cs", "ro", "da", "hu", "ta", "no",
        "th", "ur", "hr", "bg", "lt", "la", "mi", "ml", "cy", "sk",
        "te", "fa", "lv", "bn", "sr", "az", "sl", "kn", "et", "mk",
        "br", "eu", "is", "hy", "ne", "mn", "bs", "kk", "sq", "sw",
        "gl", "mr", "pa", "si", "km", "sn", "yo", "so", "af", "oc",
        "ka", "be", "tg", "sd", "gu", "am", "yi", "lo", "uz", "fo",
        "ht", "ps", "tk", "nn", "mt", "sa", "lb", "my", "bo", "tl",
        "mg", "as", "tt", "haw", "ln", "ha", "ba", "jw", "su", "yue",
    ]
}
