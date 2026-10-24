import Foundation
import Testing
@testable import Voco

struct Qwen3ASRContextBiasStoreTests {
    @Test func decodeProfileAcceptsCurrentSchemaAndDeduplicatesTerms() throws {
        let profile = try Qwen3ASRContextBiasStore.decodeProfile(
            from: fixtureData(terms: [" repo ", "MCP", "repo", "", "JSONL"], boost: 4),
            sourceKind: .downloaded,
            sha256: "fixture-sha",
            fileURL: URL(fileURLWithPath: "/tmp/context-hotword-bias.json")
        )

        #expect(profile.sourceKind == .downloaded)
        #expect(profile.artifactId == "context-hotword-bias-20260705")
        #expect(profile.terms == ["repo", "MCP", "JSONL"])
        #expect(profile.boost == 4)
        #expect(profile.maxTermsPerDecode == 8)
        #expect(profile.repeatNgramSize == 4)
        #expect(profile.repeatNgramMaxCount == 2)
        #expect(profile.sha256 == "fixture-sha")
    }

    @Test func decodeProfileRejectsUnsupportedSchema() throws {
        assertDecodeError(
            fixtureData(schema: "bad.schema", terms: ["repo"], boost: 4),
            equals: .unsupportedSchema("bad.schema")
        )
    }

    @Test func decodeProfileRejectsEmptyTerms() throws {
        assertDecodeError(
            fixtureData(terms: [" ", ""], boost: 4),
            equals: .emptyTerms
        )
    }

    @Test func decodeProfileRejectsUnsafeBoost() throws {
        assertDecodeError(
            fixtureData(terms: ["repo"], boost: 24),
            equals: .invalidBoost(24)
        )
    }

    @Test @MainActor func storeFallsBackToBuiltinProfileWithoutDownloadedOverride() throws {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("Qwen3ASRContextBiasStoreTests-\(UUID().uuidString)", isDirectory: true)
        let defaults = try temporaryDefaults()
        let store = Qwen3ASRContextBiasStore(
            fileURL: directory.appendingPathComponent("context-hotword-bias.json"),
            defaults: defaults
        )

        #expect(store.activeProfile().sourceKind == .builtin)
        #expect(store.status.sourceKind == .builtin)
        #expect(store.isEnabled == true)
    }
}

private func assertDecodeError(
    _ data: Data,
    equals expected: Qwen3ASRContextBiasStoreError
) {
    do {
        _ = try Qwen3ASRContextBiasStore.decodeProfile(
            from: data,
            sourceKind: .downloaded,
            sha256: nil,
            fileURL: nil
        )
        Issue.record("Expected decode error \(expected)")
    } catch let error as Qwen3ASRContextBiasStoreError {
        #expect(error == expected)
    } catch {
        Issue.record("Unexpected error \(error)")
    }
}

private func fixtureData(
    schema: String = Qwen3ASRContextBiasStore.supportedSchema,
    terms: [String],
    boost: Double
) -> Data {
    let json = """
    {
      "schema": "\(schema)",
      "artifactId": "context-hotword-bias-20260705",
      "createdAt": "2026-07-05T14:30:00+08:00",
      "decodeBias": {
        "terms": \(termsJSON(terms)),
        "boost": \(boost),
        "maxTermsPerDecode": 8,
        "repetitionGuard": {
          "repeatNgramSize": 4,
          "repeatNgramMaxCount": 2
        }
      }
    }
    """
    return Data(json.utf8)
}

private func termsJSON(_ terms: [String]) -> String {
    let data = try! JSONSerialization.data(withJSONObject: terms)
    return String(data: data, encoding: .utf8)!
}

private func temporaryDefaults() throws -> UserDefaults {
    let suiteName = "Qwen3ASRContextBiasStoreTests-\(UUID().uuidString)"
    return try #require(UserDefaults(suiteName: suiteName))
}
