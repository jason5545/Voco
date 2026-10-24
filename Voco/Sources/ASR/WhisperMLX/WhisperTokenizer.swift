// WhisperTokenizer.swift
// Tokenizer for Whisper: supports both tiktoken (.tiktoken) and BPE (vocab.json + merges.txt)
// [AI-Claude: 2026-02-27]

import Foundation
import os

enum WhisperTokenizerError: Error, LocalizedError {
    case noTokenizerFound(URL)
    case invalidFormat(String)

    var errorDescription: String? {
        switch self {
        case .noTokenizerFound(let dir):
            return "No tokenizer files found in: \(dir.path)"
        case .invalidFormat(let reason):
            return "Invalid tokenizer format: \(reason)"
        }
    }
}

/// Whisper tokenizer supporting both tiktoken and BPE formats
class WhisperTokenizer {
    private static let logger = Logger(subsystem: AppIdentifiers.subsystem, category: "WhisperTokenizer")

    private var idToToken: [Int: String] = [:]
    private var tokenToId: [String: Int] = [:]
    private var bpeMerges: [(String, String)] = []
    private var bpeMergeRanks: [String: Int] = [:]

    init() {}

    /// Load tokenizer from model directory. Tries tiktoken first, then BPE.
    func load(from directory: URL) throws {
        // Try tiktoken format first
        let tiktokenFile = directory.appendingPathComponent("multilingual.tiktoken")
        if FileManager.default.fileExists(atPath: tiktokenFile.path) {
            try loadTiktoken(from: tiktokenFile)
            return
        }

        // Try BPE format (vocab.json + merges.txt)
        let vocabFile = directory.appendingPathComponent("vocab.json")
        if FileManager.default.fileExists(atPath: vocabFile.path) {
            try loadBPE(from: vocabFile)
            return
        }

        throw WhisperTokenizerError.noTokenizerFound(directory)
    }

    // MARK: - Tiktoken Loading

    private func loadTiktoken(from url: URL) throws {
        let content = try String(contentsOf: url, encoding: .utf8)
        let lines = content.components(separatedBy: .newlines)

        for line in lines {
            let trimmed = line.trimmingCharacters(in: .whitespaces)
            if trimmed.isEmpty { continue }

            let parts = trimmed.components(separatedBy: " ")
            guard parts.count == 2,
                  let tokenData = Data(base64Encoded: parts[0]),
                  let rank = Int(parts[1]) else { continue }

            let token = String(data: tokenData, encoding: .utf8) ?? ""
            idToToken[rank] = token
            tokenToId[token] = rank
        }

        // Add special tokens that aren't in the tiktoken file
        let specialTokens: [(String, Int)] = [
            ("<|endoftext|>", 50257),
            ("<|startoftranscript|>", 50258),
            ("<|translate|>", 50358),
            ("<|transcribe|>", 50359),
            ("<|startoflm|>", 50360),
            ("<|startofprev|>", 50361),
            ("<|nospeech|>", 50362),
            ("<|notimestamps|>", 50363),
        ]

        // Add language tokens
        for (i, lang) in WhisperTokens.languageOrder.enumerated() {
            let id = WhisperTokens.firstLanguageTokenId + i
            let token = "<|\(lang)|>"
            idToToken[id] = token
            tokenToId[token] = id
        }

        for (token, id) in specialTokens {
            idToToken[id] = token
            tokenToId[token] = id
        }

        Self.logger.info("Loaded tiktoken tokenizer with \(self.idToToken.count) tokens")
    }

    // MARK: - BPE Loading

    private func loadBPE(from vocabURL: URL) throws {
        let data = try Data(contentsOf: vocabURL)
        guard let vocab = try JSONSerialization.jsonObject(with: data) as? [String: Int] else {
            throw WhisperTokenizerError.invalidFormat("Expected {token: id} dictionary")
        }

        for (token, id) in vocab {
            idToToken[id] = token
            tokenToId[token] = id
        }

        // Load added tokens from tokenizer_config.json
        let configURL = vocabURL.deletingLastPathComponent().appendingPathComponent("tokenizer_config.json")
        if FileManager.default.fileExists(atPath: configURL.path) {
            try loadAddedTokens(from: configURL)
        }

        // Load merges
        let mergesURL = vocabURL.deletingLastPathComponent().appendingPathComponent("merges.txt")
        if FileManager.default.fileExists(atPath: mergesURL.path) {
            try loadMerges(from: mergesURL)
        }

        Self.logger.info("Loaded BPE tokenizer with \(self.idToToken.count) tokens, \(self.bpeMerges.count) merges")
    }

    private func loadAddedTokens(from url: URL) throws {
        let data = try Data(contentsOf: url)
        guard let config = try JSONSerialization.jsonObject(with: data) as? [String: Any] else { return }

        if let addedTokens = config["added_tokens_decoder"] as? [String: [String: Any]] {
            for (idString, tokenInfo) in addedTokens {
                guard let id = Int(idString),
                      let content = tokenInfo["content"] as? String else { continue }
                idToToken[id] = content
                tokenToId[content] = id
            }
        }
    }

    private func loadMerges(from url: URL) throws {
        let content = try String(contentsOf: url, encoding: .utf8)
        let lines = content.components(separatedBy: .newlines)

        for (index, line) in lines.enumerated() {
            if line.hasPrefix("#") || line.isEmpty { continue }
            let parts = line.components(separatedBy: " ")
            guard parts.count == 2 else { continue }
            bpeMerges.append((parts[0], parts[1]))
            bpeMergeRanks["\(parts[0]) \(parts[1])"] = index
        }
    }

    // MARK: - Decode

    func decode(tokens: [Int]) -> String {
        // For tiktoken: tokens map directly to UTF-8 text
        if bpeMerges.isEmpty {
            return decodeTiktoken(tokens: tokens)
        }
        // For BPE: byte-level decoding
        return decodeBPE(tokens: tokens)
    }

    private func decodeTiktoken(tokens: [Int]) -> String {
        var result = ""
        for tokenId in tokens {
            guard let token = idToToken[tokenId] else { continue }
            // Skip special tokens
            if token.hasPrefix("<|") && token.hasSuffix("|>") { continue }
            result += token
        }
        return result.trimmingCharacters(in: .whitespaces)
    }

    private func decodeBPE(tokens: [Int]) -> String {
        var allBytes: [UInt8] = []
        var result = ""

        for tokenId in tokens {
            guard let token = idToToken[tokenId] else { continue }
            // Skip special tokens
            if token.hasPrefix("<|") && token.hasSuffix("|>") { continue }

            for char in token {
                if let byte = Self.unicodeToByte[char] {
                    allBytes.append(byte)
                } else {
                    allBytes.append(contentsOf: String(char).utf8)
                }
            }
        }

        if !allBytes.isEmpty {
            if let decoded = String(bytes: allBytes, encoding: .utf8) {
                result = decoded
            }
        }

        return result.trimmingCharacters(in: .whitespaces)
    }

    // MARK: - Encode (for prompts)

    func encode(_ text: String) -> [Int] {
        guard !bpeMerges.isEmpty else {
            // Tiktoken: simple character-level fallback (prompts are typically short)
            return characterEncode(text)
        }
        // BPE encoding
        let words = preTokenize(text)
        var tokens: [Int] = []
        for word in words {
            let bpeTokens = bpe(word)
            for bpeToken in bpeTokens {
                if let id = tokenToId[bpeToken] {
                    tokens.append(id)
                }
            }
        }
        return tokens
    }

    private func characterEncode(_ text: String) -> [Int] {
        var tokens: [Int] = []
        for char in text {
            if let id = tokenToId[String(char)] {
                tokens.append(id)
            }
        }
        return tokens
    }

    private func preTokenize(_ text: String) -> [String] {
        var words: [String] = []
        var current = ""
        for char in text {
            if char == " " || char == "\n" || char == "\t" {
                if !current.isEmpty {
                    words.append(encodeByteLevelToken(current))
                    current = ""
                }
                current = String(char)
            } else {
                current.append(char)
            }
        }
        if !current.isEmpty {
            words.append(encodeByteLevelToken(current))
        }
        return words
    }

    private func encodeByteLevelToken(_ text: String) -> String {
        var result = ""
        for byte in text.utf8 {
            if let char = Self.byteToUnicode[byte] {
                result.append(char)
            }
        }
        return result
    }

    private func bpe(_ word: String) -> [String] {
        var pieces = word.map { String($0) }
        while pieces.count > 1 {
            var bestPair: (String, String)?
            var bestRank = Int.max
            for i in 0..<(pieces.count - 1) {
                let pair = "\(pieces[i]) \(pieces[i + 1])"
                if let rank = bpeMergeRanks[pair], rank < bestRank {
                    bestRank = rank
                    bestPair = (pieces[i], pieces[i + 1])
                }
            }
            guard let (first, second) = bestPair else { break }
            var newPieces: [String] = []
            var i = 0
            while i < pieces.count {
                if i < pieces.count - 1 && pieces[i] == first && pieces[i + 1] == second {
                    newPieces.append(first + second)
                    i += 2
                } else {
                    newPieces.append(pieces[i])
                    i += 1
                }
            }
            pieces = newPieces
        }
        return pieces
    }

    // MARK: - Byte-level BPE maps

    private static var byteToUnicode: [UInt8: Character] = {
        var mapping: [UInt8: Character] = [:]
        var n = 0
        let ranges: [(ClosedRange<UInt8>)] = [
            (UInt8(ascii: "!")...UInt8(ascii: "~")),
            (0xA1...0xAC),
            (0xAE...0xFF),
        ]
        for range in ranges {
            for b in range { mapping[b] = Character(UnicodeScalar(b)) }
        }
        for b: UInt8 in 0...255 {
            if mapping[b] == nil {
                mapping[b] = Character(UnicodeScalar(0x100 + n)!)
                n += 1
            }
        }
        return mapping
    }()

    private static var unicodeToByte: [Character: UInt8] = {
        var reverse: [Character: UInt8] = [:]
        for (byte, char) in byteToUnicode { reverse[char] = byte }
        return reverse
    }()
}
