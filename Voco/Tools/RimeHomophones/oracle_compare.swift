import Foundation
import Darwin

// These two shims let swiftc compile the production service unchanged outside the app target.
enum AppIdentifiers {
    static var appSupportDirectory: URL {
        FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("com.jasonchien.Voco", isDirectory: true)
    }
}

struct VocoAutoApplyPolicyFire {
    init(
        policyId: String,
        policyType: String,
        autoApplyMode: String,
        sourcePattern: String,
        targetText: String,
        sourceSlices: [String]
    ) {}
}

private struct Change: Hashable, Comparable, CustomStringConvertible {
    let pk: Int
    let offset: Int
    let source: String
    let target: String

    static func < (lhs: Self, rhs: Self) -> Bool {
        (lhs.pk, lhs.offset, lhs.source, lhs.target) < (rhs.pk, rhs.offset, rhs.source, rhs.target)
    }

    var description: String { "(\(pk), \(offset), \(source), \(target))" }
}

@main
private enum RimeOracleCompare {
    static func main() throws {
        let environment = ProcessInfo.processInfo.environment
        let dataURL = try requiredURL("VOCO_RIME_GENERATED_DATA", environment: environment)
        let service = VocoRimeHomophoneService(dataURL: dataURL)
        if environment["VOCO_RIME_SMOKE"] == "1" {
            let status = service.dataStatus()
            let input = environment["VOCO_RIME_SMOKE_INPUT"] ?? "粉絲專業"
            let expectedOutput = environment["VOCO_RIME_SMOKE_EXPECTED"] ?? "粉絲專頁"
            let result = service.evaluate(input)
            let modificationDate = status.modificationDate?.description ?? "missing"
            let reason = result.applied.isEmpty ? "none" : VocoRimeHomophoneService.reason
            print(
                "rime_data path=\(status.path) loaded=\(status.isLoaded) " +
                    "learned_words=\(status.learnedWordCount) characters=\(status.characterReadingCount) " +
                    "essay_words=\(status.essayEntryCount) mtime=\(modificationDate)"
            )
            print(
                "smoke input=\(input) output=\(result.outputText) " +
                    "reason=\(reason)"
            )
            fflush(stdout)
            guard status.isLoaded, result.outputText == expectedOutput else { throw CompareError.mismatch }
            return
        }

        let rowsURL = try requiredURL("VOCO_RIME_ROWS_S2TWP_JSON", environment: environment)
        let oracleURL = try requiredURL("VOCO_RIME_ORACLE_DUMP", environment: environment)
        let rowsRoot = try JSONSerialization.jsonObject(with: Data(contentsOf: rowsURL))
        let rows = (rowsRoot as? [String: Any])?["rows"] as? [[Any]] ?? rowsRoot as? [[Any]] ?? []
        let oracleRoot = try JSONSerialization.jsonObject(with: Data(contentsOf: oracleURL))
        let oracleChanges = (oracleRoot as? [String: Any])?["changes"] as? [[Any]] ?? []

        var expected = oracleChanges.compactMap { values -> Change? in
            guard values.count >= 4,
                  let pk = (values[0] as? NSNumber)?.intValue,
                  let offset = (values[1] as? NSNumber)?.intValue,
                  let source = values[2] as? String,
                  let target = values[3] as? String
            else { return nil }
            return Change(pk: pk, offset: offset, source: source, target: target)
        }

        var actual: [Change] = []
        var changedRows = Set<Int>()
        for values in rows {
            guard values.count >= 2,
                  let pk = (values[0] as? NSNumber)?.intValue,
                  let text = values[1] as? String
            else { continue }
            let result = service.evaluate(text)
            guard result.changed else { continue }
            changedRows.insert(pk)
            for fire in result.applied {
                let prefix = (text as NSString).substring(to: fire.range.location)
                actual.append(
                    Change(
                        pk: pk,
                        offset: prefix.unicodeScalars.count,
                        source: fire.sourceText,
                        target: fire.targetText
                    )
                )
            }
        }

        expected.sort()
        actual.sort()
        let missing = Set(expected).subtracting(actual).sorted()
        let extra = Set(actual).subtracting(expected).sorted()
        print("rows=\(rows.count) oracle_spans=\(expected.count) swift_spans=\(actual.count) swift_rows=\(changedRows.count) missing=\(missing.count) extra=\(extra.count)")
        for change in missing { print("MISSING \(change)") }
        for change in extra { print("EXTRA \(change)") }
        fflush(stdout)
        guard rows.count == 34_190,
              expected.count == 62,
              Set(expected.map(\.pk)).count == 59,
              actual.count == 62,
              changedRows.count == 59,
              missing.isEmpty,
              extra.isEmpty
        else { throw CompareError.mismatch }
        print("diff=empty")
    }

    private static func requiredURL(_ name: String, environment: [String: String]) throws -> URL {
        guard let value = environment[name]?.trimmingCharacters(in: .whitespacesAndNewlines),
              !value.isEmpty
        else { throw CompareError.missingEnvironment(name) }
        return URL(fileURLWithPath: value)
    }
}

private enum CompareError: Error, CustomStringConvertible {
    case missingEnvironment(String)
    case mismatch

    var description: String {
        switch self {
        case .missingEnvironment(let name): return "missing environment variable \(name)"
        case .mismatch: return "Swift output does not match the oracle"
        }
    }
}
