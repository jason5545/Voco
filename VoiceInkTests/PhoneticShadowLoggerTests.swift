import Foundation
import Testing
@testable import Voco

struct PhoneticShadowLoggerTests {
    @Test func eventJSONContainsRequiredSchemaKeysAndNulls() async throws {
        let event = sampleEvent()
        let object = event.jsonObject()

        for key in [
            "schemaVersion", "eventId", "createdAt", "appVersion", "buildGitSha",
            "eventType", "utteranceId", "transcriptionDbId", "audio", "pipeline",
            "userAction", "uiContext", "classification", "phonetics",
            "shadowCandidates", "safety", "featureFlags",
        ] {
            #expect(object.keys.contains(key))
        }

        let pipeline = try #require(object["pipeline"] as? [String: Any])
        for key in [
            "asrEngine", "rawASR", "afterOpenCC", "afterPinyinCorrector",
            "afterHomophoneCorrection", "afterNasalCorrection", "afterPersonalCorrection",
            "llmEnhanced", "finalInserted", "route", "confidenceScore", "avgLogprob",
            "noSpeechProb", "compressionRatio", "posteriorGap", "latencyMs",
        ] {
            #expect(pipeline.keys.contains(key))
        }
        #expect(pipeline["llmEnhanced"] is NSNull)

        let safety = try #require(object["safety"] as? [String: Any])
        #expect(safety["autoApplied"] as? Bool == false)
        #expect(safety["wouldHaveChangedFinalOutput"] as? Bool == false)
    }

    @Test func loggerWritesOneJSONLEvent() async throws {
        let directory = try temporaryDirectory()
        let logger = PhoneticShadowLogger(logDirectory: directory)
        let event = sampleEvent()

        logger.log(event, force: true)
        logger.flush()

        let logFile = directory.appendingPathComponent("phonetic-shadow-2026-06-10.jsonl")
        let content = try String(contentsOf: logFile, encoding: .utf8)
        let lines = content.split(separator: "\n")
        #expect(lines.count == 1)

        let data = try #require(lines.first?.data(using: .utf8))
        let object = try #require(JSONSerialization.jsonObject(with: data) as? [String: Any])
        let safety = try #require(object["safety"] as? [String: Any])
        #expect(object["eventType"] as? String == "pipelineSnapshot")
        #expect(safety["autoApplied"] as? Bool == false)
    }

    @Test func loggerFailureDoesNotThrowIntoCaller() async throws {
        let parent = try temporaryDirectory()
        let fileURL = parent.appendingPathComponent("not-a-directory")
        try Data("x".utf8).write(to: fileURL)
        let logger = PhoneticShadowLogger(logDirectory: fileURL)

        logger.log(sampleEvent(), force: true)
        logger.flush()

        let data = try Data(contentsOf: fileURL)
        #expect(String(data: data, encoding: .utf8) == "x")
    }

    @Test func loggerRotatesWhenFileExceedsLimit() async throws {
        let directory = try temporaryDirectory()
        let logger = PhoneticShadowLogger(logDirectory: directory, maxFileSizeBytes: 1)

        logger.log(sampleEvent(eventId: "first"), force: true)
        logger.log(sampleEvent(eventId: "second"), force: true)
        logger.flush()

        let current = directory.appendingPathComponent("phonetic-shadow-2026-06-10.jsonl")
        let rotated = directory.appendingPathComponent("phonetic-shadow-2026-06-10.1.jsonl")
        #expect(FileManager.default.fileExists(atPath: current.path))
        #expect(FileManager.default.fileExists(atPath: rotated.path))
    }

    @Test func defaultCandidateApplicationFlagIsFalse() async throws {
        #expect(AppDefaults.defaultValues[PhoneticShadowLogger.candidateApplicationEnabledKey] as? Bool == false)
        let object = sampleEvent().jsonObject()
        let safety = try #require(object["safety"] as? [String: Any])
        #expect(safety["autoApplied"] as? Bool == false)
        #expect(safety["wouldHaveChangedFinalOutput"] as? Bool == false)
    }

    private func sampleEvent(eventId: String = "event-1") -> PhoneticShadowEvent {
        PhoneticShadowEvent.pipelineSnapshot(
            utteranceId: "utt-1",
            transcriptionDbId: "db-1",
            pipeline: PhoneticShadowPipeline(
                asrEngine: "Qwen3-ASR",
                rawASR: "修正",
                afterOpenCC: "修正",
                afterPinyinCorrector: "修正",
                afterHomophoneCorrection: "修正",
                afterNasalCorrection: "修正",
                afterPersonalCorrection: "修正",
                finalInserted: "修正",
                route: "directInsertion",
                confidenceScore: 0.99,
                latencyMs: 120
            )
        ).with(eventId: eventId, createdAt: fixedDate())
    }

    private func fixedDate() -> Date {
        Date(timeIntervalSince1970: 1_781_020_800)
    }

    private func temporaryDirectory() throws -> URL {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("PhoneticShadowLoggerTests-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        return url
    }
}

private extension PhoneticShadowEvent {
    func with(eventId: String, createdAt: Date) -> PhoneticShadowEvent {
        var copy = self
        copy.eventId = eventId
        copy.createdAt = createdAt
        return copy
    }
}
