import Foundation
import Testing
@testable import Voco

@MainActor
struct Phase1NoOutputChangeTests {
    @Test func shadowLoggingFlagDoesNotChangePostProcessingOutput() async throws {
        let service = ChinesePostProcessingService.shared
        let oldShadowFlag = UserDefaults.standard.object(forKey: PhoneticShadowLogger.shadowLoggingEnabledKey)
        defer {
            if let oldShadowFlag {
                UserDefaults.standard.set(oldShadowFlag, forKey: PhoneticShadowLogger.shadowLoggingEnabledKey)
            } else {
                UserDefaults.standard.removeObject(forKey: PhoneticShadowLogger.shadowLoggingEnabledKey)
            }
        }

        for sample in ["修正", "69 輪", "Load Fail"] {
            UserDefaults.standard.set(false, forKey: PhoneticShadowLogger.shadowLoggingEnabledKey)
            let outputWithShadowOff = service.process(sample).processedText

            UserDefaults.standard.set(true, forKey: PhoneticShadowLogger.shadowLoggingEnabledKey)
            let outputWithShadowOn = service.process(sample).processedText

            #expect(outputWithShadowOn == outputWithShadowOff)
        }
    }

    @Test func shadowSnapshotAndLoggerDoNotMutateFinalInsertedText() async throws {
        let directory = try temporaryDirectory()
        let logger = PhoneticShadowLogger(logDirectory: directory)

        for finalText in ["修正", "69 輪", "Load Fail"] {
            let pipeline = PhoneticShadowPipeline(
                rawASR: finalText,
                finalInserted: finalText,
                route: "directInsertion"
            )
            let event = PhoneticShadowEvent.pipelineSnapshot(pipeline: pipeline)
            let before = finalText

            logger.log(event, force: true)
            logger.flush()

            #expect(before == finalText)
            #expect(event.pipeline.finalInserted == finalText)
            let safety = try #require(event.jsonObject()["safety"] as? [String: Any])
            #expect(safety["autoApplied"] as? Bool == false)
            #expect(safety["wouldHaveChangedFinalOutput"] as? Bool == false)
        }
    }

    private func temporaryDirectory() throws -> URL {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("Phase1NoOutputChangeTests-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        return url
    }
}
