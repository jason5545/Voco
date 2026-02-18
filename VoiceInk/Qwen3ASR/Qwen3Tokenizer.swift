// Qwen3Tokenizer.swift
// Adapted from qwen3-asr-swift Tokenizer.swift
// Fixed: print() → os.Logger
// [AI-Claude: 2025-02-18]

import Foundation
import os

enum Qwen3TokenizerError: Error, LocalizedError {
    case invalidFormat(String)

    var errorDescription: String? {
        switch self {
        case .invalidFormat(let reason):
            return "Invalid tokenizer format: \(reason)"
        }
    }
}

/// BPE tokenizer for Qwen3, loads from vocab.json
class Qwen3Tokenizer {
    private static let logger = Logger(subsystem: "com.jasonchien.voco", category: "Qwen3Tokenizer")

    private var idToToken: [Int: String] = [:]
    private var tokenToId: [String: Int] = [:]
    private var bpeMerges: [(String, String)] = []
    private var bpeMergeRanks: [String: Int] = [:]

    var eosTokenId: Int = 151643
    var padTokenId: Int = 151643
    var bosTokenId: Int = 151644

    init() {}

    func load(from url: URL) throws {
        let data = try Data(contentsOf: url)

        guard let vocab = try JSONSerialization.jsonObject(with: data) as? [String: Int] else {
            throw Qwen3TokenizerError.invalidFormat("Expected {token: id} dictionary")
        }

        for (token, id) in vocab {
            idToToken[id] = token
            tokenToId[token] = id
        }

        let configUrl = url.deletingLastPathComponent().appendingPathComponent("tokenizer_config.json")
        if FileManager.default.fileExists(atPath: configUrl.path) {
            try loadAddedTokens(from: configUrl)
        }

        let mergesUrl = url.deletingLastPathComponent().appendingPathComponent("merges.txt")
        if FileManager.default.fileExists(atPath: mergesUrl.path) {
            try loadMerges(from: mergesUrl)
        }

        Self.logger.info("Loaded tokenizer with \(self.idToToken.count) tokens, \(self.bpeMerges.count) merges")
    }

    private func loadAddedTokens(from url: URL) throws {
        let data = try Data(contentsOf: url)

        guard let config = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            return
        }

        if let addedTokens = config["added_tokens_decoder"] as? [String: [String: Any]] {
            var addedCount = 0
            for (idString, tokenInfo) in addedTokens {
                guard let id = Int(idString),
                      let content = tokenInfo["content"] as? String else {
                    continue
                }
                idToToken[id] = content
                tokenToId[content] = id
                addedCount += 1
            }
            Self.logger.debug("Loaded \(addedCount) added tokens from tokenizer_config.json")
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

    func decode(tokens: [Int]) -> String {
        var allBytes: [UInt8] = []
        var result = ""

        for tokenId in tokens {
            if let token = idToToken[tokenId] {
                // Skip special tokens like <|im_start|>, <|im_end|>
                if token.hasPrefix("<|") && token.hasSuffix("|>") {
                    continue
                }

                // Keep markers like <asr_text> — flush accumulated bytes first
                if token.hasPrefix("<") && token.hasSuffix(">") && !token.contains("|") {
                    if !allBytes.isEmpty {
                        if let decoded = String(bytes: allBytes, encoding: .utf8) {
                            result += decoded
                        }
                        allBytes.removeAll()
                    }
                    result += token
                    continue
                }

                // Accumulate bytes via byte-level BPE reverse mapping
                for char in token {
                    if let byte = Self.unicodeToByte[char] {
                        allBytes.append(byte)
                    } else {
                        allBytes.append(contentsOf: String(char).utf8)
                    }
                }
            }
        }

        // Flush remaining bytes
        if !allBytes.isEmpty {
            if let decoded = String(bytes: allBytes, encoding: .utf8) {
                result += decoded
            }
        }

        return result.trimmingCharacters(in: .whitespaces)
    }

    // MARK: - Byte-level BPE

    private static var byteToUnicode: [UInt8: Character] = {
        var mapping: [UInt8: Character] = [:]
        var n = 0

        let ranges: [(ClosedRange<UInt8>)] = [
            (UInt8(ascii: "!")...UInt8(ascii: "~")),
            (0xA1...0xAC),
            (0xAE...0xFF),
        ]

        for range in ranges {
            for b in range {
                mapping[b] = Character(UnicodeScalar(b))
            }
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
        for (byte, char) in byteToUnicode {
            reverse[char] = byte
        }
        return reverse
    }()

    private func decodeByteLevelToken(_ token: String) -> String {
        var bytes: [UInt8] = []

        for char in token {
            if let byte = Self.unicodeToByte[char] {
                bytes.append(byte)
            } else {
                bytes.append(contentsOf: String(char).utf8)
            }
        }

        if let decoded = String(bytes: bytes, encoding: .utf8) {
            return decoded
        } else {
            return token
        }
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

    func encode(_ text: String) -> [Int] {
        guard !bpeMerges.isEmpty else {
            return characterEncode(text)
        }

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

    private func characterEncode(_ text: String) -> [Int] {
        var tokens: [Int] = []
        for char in text {
            if let id = tokenToId[String(char)] {
                tokens.append(id)
            }
        }
        return tokens
    }
}
