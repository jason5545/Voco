import Foundation
import Testing
@testable import Voco

struct VocoRimeHomophoneServiceTests {
    @Test func repairsSyntheticHomophoneButLeavesProfessionalAlone() throws {
        let service = VocoRimeHomophoneService(dataURL: try writeData(fixtureData()))

        #expect(service.evaluate("我喜歡粉絲專業。\n").outputText == "我喜歡粉絲專頁。\n")
        #expect(service.evaluate("這很專業。").outputText == "這很專業。")
    }

    @Test func missingDataFileLeavesTextUnchanged() throws {
        let directory = temporaryDirectory()
        let service = VocoRimeHomophoneService(dataURL: directory.appendingPathComponent("missing.json"))

        let result = service.evaluate("粉絲專業")
        #expect(result.outputText == "粉絲專業")
        #expect(result.applied.isEmpty)
    }

    @Test func reloadsGeneratedDataWhenModificationDateChanges() async throws {
        let url = try writeData(fixtureData())
        let service = VocoRimeHomophoneService(dataURL: url)
        #expect(service.evaluate("粉絲專業").outputText == "粉絲專頁")

        try await Task.sleep(nanoseconds: 50_000_000)
        try JSONEncoder().encode(fixtureData(includeLearnedWords: false)).write(to: url, options: .atomic)

        #expect(service.evaluate("粉絲專業").outputText == "粉絲專業")
    }

    @Test func defersToContextLockedAutoApplySource() throws {
        let service = VocoRimeHomophoneService(dataURL: try writeData(fixtureData()))

        let result = service.evaluate("粉絲專業", deferredSourcePatterns: ["粉絲專業"])
        #expect(result.outputText == "粉絲專業")
        #expect(result.applied.isEmpty)
    }

    @Test func excludesPronounAndTaiwanVariantDifferences() throws {
        let service = VocoRimeHomophoneService(dataURL: try writeData(fixtureData()))

        #expect(service.evaluate("你愛你").outputText == "你愛你")
        #expect(service.evaluate("台灣高速公路").outputText == "台灣高速公路")
    }

    @Test func recordsRimeReasonAndSkipPolicyDisablesTheLayer() throws {
        let dataURL = try writeData(fixtureData())
        let canonicalization = VocoCanonicalizationService(
            contextPacks: [],
            autoApplyModelService: disabledAutoApplyModelService(),
            runtimeCorrectionModelService: disabledRuntimeCorrectionModelService(),
            phoneticCorrectionService: VocoPhoneticCorrectionService(
                rulesURL: temporaryDirectory().appendingPathComponent("missing-phonetic-rules.json"),
                seedRules: []
            ),
            rimeHomophoneService: VocoRimeHomophoneService(dataURL: dataURL)
        )

        let full = canonicalization.normalize("粉絲專業", activeContextIDs: [])
        #expect(full.normalizedText == "粉絲專頁")
        #expect(full.replacements.map(\.reason) == [VocoRimeHomophoneService.reason])

        let skipped = canonicalization.normalize(
            "粉絲專業",
            activeContextIDs: [],
            correctionPolicy: .skipPostASRCorrectionModels
        )
        #expect(skipped.normalizedText == "粉絲專業")
        #expect(skipped.replacements.isEmpty)
    }

    private func fixtureData(includeLearnedWords: Bool = true) -> VocoRimeHomophoneData {
        let syllables: [String: String] = [
            "粉": "fen3", "絲": "si1", "專": "zhuan1", "業": "ye4", "頁": "ye4",
            "你": "ni3", "妳": "ni3", "愛": "ai4", "台": "tai2", "臺": "tai2",
            "灣": "wan1", "高": "gao1", "速": "su4", "公": "gong1", "路": "lu4"
        ]
        let readings = Dictionary(uniqueKeysWithValues: syllables.map { entry in
            (entry.key, [VocoRimeHomophoneCharacterReading(syllable: entry.value, weightPercent: 100)])
        })
        return VocoRimeHomophoneData(
            format: VocoRimeHomophoneService.expectedFormat,
            schemaVersion: 1,
            generatedAt: "synthetic-fixture",
            readingWeightMinimumPercent: 5,
            learnedWords: includeLearnedWords ? [
                VocoRimeHomophoneLearnedWord(word: "粉絲專頁", syllables: ["fen3", "si1", "zhuan1", "ye4"], commits: 3),
                VocoRimeHomophoneLearnedWord(word: "妳愛你", syllables: ["ni3", "ai4", "ni3"], commits: 3),
                VocoRimeHomophoneLearnedWord(word: "臺灣高速公路", syllables: ["tai2", "wan1", "gao1", "su4", "gong1", "lu4"], commits: 3)
            ] : [],
            characterReadings: readings,
            essayFrequencies: [:],
            taiwanToOpenCC: ["臺": "台"],
            openCCToTaiwan: ["台": "臺"],
            deferredSourcePatterns: []
        )
    }

    private func writeData(_ value: VocoRimeHomophoneData) throws -> URL {
        let url = temporaryDirectory().appendingPathComponent("rime-homophones-\(UUID().uuidString).json")
        try JSONEncoder().encode(value).write(to: url, options: .atomic)
        return url
    }

    private func temporaryDirectory() -> URL {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("VocoRimeHomophoneTests-\(UUID().uuidString)", isDirectory: true)
        try? FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        return url
    }

    private func disabledAutoApplyModelService() -> VocoAutoApplyModelService {
        VocoAutoApplyModelService(
            modelURL: temporaryDirectory().appendingPathComponent("missing-auto-apply-model.json"),
            defaults: UserDefaults(suiteName: "VocoRimeHomophoneTests-\(UUID().uuidString)") ?? .standard
        )
    }

    private func disabledRuntimeCorrectionModelService() -> VocoRuntimeCorrectionModelService {
        let root = temporaryDirectory()
        return VocoRuntimeCorrectionModelService(
            artifactURL: root.appendingPathComponent("missing-runtime-model.json"),
            eventLogURL: nil,
            defaults: UserDefaults(suiteName: "VocoRimeHomophoneRuntimeTests-\(UUID().uuidString)") ?? .standard
        )
    }
}
