import Foundation
import Testing
@testable import Voco

struct Qwen3ASRAdapterTests {
    @Test func specialistRouterTriggersFirmwareWithUnicode() {
        let trigger = Qwen3ASRSpecialistRouter.triggerDecision(
            baselineTranscript: "軟體定位在 Unicode。",
            recentTranscriptions: []
        )
        #expect(trigger.triggered)
        #expect(trigger.requiredSurfaces == ["韌體"])
    }

    @Test func specialistRouterUsesRecentSupportContext() {
        let trigger = Qwen3ASRSpecialistRouter.triggerDecision(
            baselineTranscript: "這個軟體資源很差。",
            recentTranscriptions: ["也就是應該要變成軟體 support 的那個東西。"]
        )
        #expect(trigger.requiredSurfaces == ["支援"])
    }

    @Test func specialistRouterPreservesLegitimateResourceManagement() {
        let trigger = Qwen3ASRSpecialistRouter.triggerDecision(
            baselineTranscript: "但是資源管理要做好。",
            recentTranscriptions: []
        )
        #expect(!trigger.triggered)
    }

    @Test func specialistRouterRejectsNonTargetDrift() {
        let trigger = Qwen3ASRSpecialistTriggerDecision(
            triggered: true,
            reasons: ["supportAmbiguity:資源"],
            requiredSurfaces: ["支援"]
        )
        let selection = Qwen3ASRSpecialistRouter.selectionDecision(
            baselineTranscript: "但是 Slack 資源 Cloud 開頭。",
            specialistTranscript: "但是 slug 支援 Cloud 開頭。",
            trigger: trigger
        )
        #expect(!selection.selectSpecialist)
    }

    @Test func specialistDiscoveryUsesSeparateDirectory() throws {
        let modelDirectory = try temporaryDirectory()
        let directory = Qwen3ASRAudioAdapterLoader.specialistAdaptersDirectory(in: modelDirectory)
            .appendingPathComponent(Qwen3ASRSpecialistRouter.specialistID, isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        try validConfigData().write(to: directory.appendingPathComponent("adapter_config.json"))
        try Data("fixture-safetensors".utf8).write(to: directory.appendingPathComponent("adapters.safetensors"))

        let discovery = Qwen3ASRAudioAdapterLoader.discoverSpecialist(
            in: modelDirectory,
            specialistID: Qwen3ASRSpecialistRouter.specialistID
        )
        #expect(discovery.descriptor?.directory == directory)
        #expect(Qwen3ASRAudioAdapterLoader.discover(in: modelDirectory).descriptor == nil)
    }
    @Test func discoveryReturnsUnavailableWhenModelOrAdaptersDirectoryIsMissing() throws {
        let modelDirectory = try temporaryDirectory()

        let missingAdapters = Qwen3ASRAudioAdapterLoader.discover(in: modelDirectory)
        #expect(missingAdapters.adapterDetected == false)
        #expect(missingAdapters.adapterPath == nil)

        let missingModelDirectory = modelDirectory.appendingPathComponent("missing-model", isDirectory: true)
        let missingModel = Qwen3ASRAudioAdapterLoader.discover(in: missingModelDirectory)
        #expect(missingModel.adapterDetected == false)
        #expect(missingModel.error == nil)
    }

    @Test func discoveryDetectsValidAdapterOnlyWhenBothRequiredFilesExist() throws {
        let adapterDirectory = try makeAdapterDirectory(named: "current-promoted-adapter")
        try validConfigData().write(to: adapterDirectory.appendingPathComponent("adapter_config.json"))
        try Data("fixture-safetensors".utf8).write(to: adapterDirectory.appendingPathComponent("adapters.safetensors"))

        let discovery = Qwen3ASRAudioAdapterLoader.discover(in: adapterDirectory.deletingLastPathComponent().deletingLastPathComponent())

        #expect(discovery.adapterDetected == true)
        #expect(discovery.descriptor?.directory.resolvingSymlinksInPath() == adapterDirectory.resolvingSymlinksInPath())
        #expect(discovery.error == nil)
    }

    @Test func discoveryRejectsMultipleValidAdaptersInsteadOfChoosingArbitrarily() throws {
        let first = try makeAdapterDirectory(named: "adapter-a")
        let modelDirectory = first.deletingLastPathComponent().deletingLastPathComponent()
        let second = modelDirectory
            .appendingPathComponent("adapters", isDirectory: true)
            .appendingPathComponent("adapter-b", isDirectory: true)
        try FileManager.default.createDirectory(at: second, withIntermediateDirectories: true)

        for directory in [first, second] {
            try validConfigData().write(to: directory.appendingPathComponent("adapter_config.json"))
            try Data("fixture-safetensors".utf8).write(to: directory.appendingPathComponent("adapters.safetensors"))
        }

        let discovery = Qwen3ASRAudioAdapterLoader.discover(in: modelDirectory)
        #expect(discovery.adapterDetected == false)
        #expect(discovery.error?.contains("multiple valid") == true)
        #expect(discovery.error?.contains("adapter-a") == true)
        #expect(discovery.error?.contains("adapter-b") == true)
    }

    @Test func discoveryRejectsMissingConfigOrWeights() throws {
        let missingConfigDirectory = try makeAdapterDirectory()
        try Data("fixture-safetensors".utf8).write(to: missingConfigDirectory.appendingPathComponent("adapters.safetensors"))
        let missingConfig = Qwen3ASRAudioAdapterLoader.discover(
            in: missingConfigDirectory.deletingLastPathComponent().deletingLastPathComponent()
        )
        #expect(missingConfig.adapterDetected == false)
        #expect(missingConfig.error?.contains("adapter_config.json") == true)

        let missingWeightsDirectory = try makeAdapterDirectory()
        try validConfigData().write(to: missingWeightsDirectory.appendingPathComponent("adapter_config.json"))
        let missingWeights = Qwen3ASRAudioAdapterLoader.discover(
            in: missingWeightsDirectory.deletingLastPathComponent().deletingLastPathComponent()
        )
        #expect(missingWeights.adapterDetected == false)
        #expect(missingWeights.error?.contains("adapters.safetensors") == true)
    }

    @Test func fingerprintTracksAdapterFileChanges() throws {
        let adapterDirectory = try makeAdapterDirectory()
        let modelDirectory = adapterDirectory.deletingLastPathComponent().deletingLastPathComponent()
        let configURL = adapterDirectory.appendingPathComponent("adapter_config.json")
        let weightsURL = adapterDirectory.appendingPathComponent("adapters.safetensors")

        try validConfigData().write(to: configURL)
        try Data("fixture-safetensors".utf8).write(to: weightsURL)

        let initial = try #require(Qwen3ASRAudioAdapterLoader.fingerprint(in: modelDirectory))
        #expect(
            URL(fileURLWithPath: initial.directoryPath).resolvingSymlinksInPath()
                == adapterDirectory.resolvingSymlinksInPath()
        )
        #expect(initial.config.exists == true)
        #expect(initial.weights.exists == true)

        sleep(1)
        try Data("updated-fixture-safetensors".utf8).write(to: weightsURL)

        let updated = try #require(Qwen3ASRAudioAdapterLoader.fingerprint(in: modelDirectory))
        #expect(updated != initial)
        #expect(updated.weights.size != initial.weights.size)
    }

    @Test func fingerprintIsUnavailableWhenAdapterDirectoryIsMissing() throws {
        let modelDirectory = try temporaryDirectory()
        #expect(Qwen3ASRAudioAdapterLoader.fingerprint(in: modelDirectory) == nil)
    }

    @Test func coordinatorLoadsAndAppliesValidAdapter() throws {
        let modelDirectory = try temporaryDirectory()
        let descriptor = try makeDescriptor(in: modelDirectory)
        var appliedDescriptor: Qwen3ASRAdapterDescriptor?

        let metadata = Qwen3ASRAdapterCoordinator.loadIfAvailable(
            modelDirectory: modelDirectory,
            discover: { _ in .available(descriptor) },
            apply: { descriptor in
                appliedDescriptor = descriptor
                return descriptor.config.loraParameters.keys.count
            }
        )

        #expect(appliedDescriptor == descriptor)
        #expect(metadata.adapterDetected == true)
        #expect(metadata.adapterLoaded == true)
        #expect(metadata.adapterApplied == true)
        #expect(metadata.adapterPath == descriptor.directory.path)
        #expect(metadata.adapterLoadError == nil)
    }

    @Test func coordinatorTreatsAdapterLoadFailureAsNonFatalFallback() throws {
        let modelDirectory = try temporaryDirectory()
        let descriptor = try makeDescriptor(in: modelDirectory)

        let metadata = Qwen3ASRAdapterCoordinator.loadIfAvailable(
            modelDirectory: modelDirectory,
            discover: { _ in .available(descriptor) },
            apply: { _ in throw FixtureError.loadFailed }
        )

        #expect(metadata.adapterDetected == true)
        #expect(metadata.adapterLoaded == false)
        #expect(metadata.adapterApplied == false)
        #expect(metadata.adapterPath == descriptor.directory.path)
        #expect(metadata.adapterLoadError?.contains("fixture load failed") == true)
    }

    @Test func runtimeGuardOnlyProbesLongAdapterActionCommandOutputs() throws {
        let adapterMetadata = Qwen3ASRAdapterMetadata(
            adapterDetected: true,
            adapterLoaded: true,
            adapterApplied: true,
            adapterPath: "/tmp/adapter",
            adapterLoadError: nil
        )

        #expect(Qwen3ASRAdapterRuntimeGuard.shouldProbeBaseFallback(
            adapterTranscript: "全部刪除。",
            adapterMetadata: adapterMetadata,
            audioDurationSeconds: 9.664
        ))
        #expect(!Qwen3ASRAdapterRuntimeGuard.shouldProbeBaseFallback(
            adapterTranscript: "全部刪除。",
            adapterMetadata: adapterMetadata,
            audioDurationSeconds: 2.464
        ))
        #expect(!Qwen3ASRAdapterRuntimeGuard.shouldProbeBaseFallback(
            adapterTranscript: "Repo 內的 Markdown。",
            adapterMetadata: adapterMetadata,
            audioDurationSeconds: 9.664
        ))
    }

    @Test func runtimeGuardUsesBaseFallbackOnlyWhenBaseRestoresSurroundingActionCommandContext() throws {
        #expect(Qwen3ASRAdapterRuntimeGuard.shouldUseBaseFallback(
            adapterTranscript: "全部刪除。",
            baseTranscript: "好了，Happy Graduation！全部删除。"
        ))
        #expect(!Qwen3ASRAdapterRuntimeGuard.shouldUseBaseFallback(
            adapterTranscript: "全部刪除。",
            baseTranscript: "全部删除。"
        ))
        #expect(!Qwen3ASRAdapterRuntimeGuard.shouldUseBaseFallback(
            adapterTranscript: "Repo 內的 Markdown。",
            baseTranscript: "Repo 內的 Markdown。"
        ))
    }

    @Test func requestContextDefaultsToAdapterButAudioFileCallersCanDisableIt() throws {
        let defaultContext = TranscriptionRequestContext(language: "Chinese", prompt: nil)
        #expect(defaultContext.usesQwen3AudioAdapter)

        let fileContext = defaultContext.withQwen3AudioAdapter(false)
        #expect(fileContext.language == "Chinese")
        #expect(fileContext.prompt == nil)
        #expect(!fileContext.usesQwen3AudioAdapter)
    }

    @Test func actualWAVSmokeConfirmsAdapterMetadataAndNoRegression() async throws {
        guard ProcessInfo.processInfo.environment["VOCO_QWEN3_ADAPTER_SMOKE"] == "1"
                || FileManager.default.fileExists(
                    atPath: projectRootURL()
                        .appendingPathComponent("LocalModels/.run-qwen3-adapter-smoke")
                        .path
                ) else {
            return
        }

        let fixtureRows = [12687, 12133]
        let fixtures = try ReplayLabAdapterFixtures.load(rows: fixtureRows)
        let model = try #require(
            TranscriptionModelRegistry.models.compactMap { $0 as? Qwen3Model }
                .first(where: { $0.name == "qwen3-asr-1.7b-8bit" })
        )

        let service = Qwen3TranscriptionService()
        defer {
            Task {
                await service.cleanup()
            }
        }

        for fixture in fixtures {
            let transcript = try await service.transcribe(
                audioURL: fixture.audioURL,
                model: model,
                context: TranscriptionRequestContext(language: "Chinese", prompt: nil)
            )
            let metadata = service.lastAdapterMetadata
            #expect(metadata.adapterDetected == true)
            #expect(metadata.adapterLoaded == true)
            #expect(metadata.adapterApplied == true)
            #expect(metadata.adapterPath != nil)
            #expect(metadata.adapterSHA256?.count == 64)
            #expect(metadata.adapterLoadError == nil)

            #expect(
                transcript == fixture.adapterPrediction,
                "row \(fixture.rowPk) drifted from ReplayLab adapter prediction: expected=\(fixture.adapterPrediction), actual=\(transcript)"
            )

            if fixture.rowPk == 12133 {
                #expect(normalizedActionCommand(transcript) == "全部刪除")
                continue
            }

            let actualCER = cer(transcript, fixture.targetTranscript)
            let baseCER = cer(fixture.basePrediction, fixture.targetTranscript)
            print("Qwen3 adapter smoke row \(fixture.rowPk): actualCER=\(actualCER), baseCER=\(baseCER), transcript=\(transcript)")
            #expect(
                actualCER <= baseCER,
                "row \(fixture.rowPk) regressed: actualCER=\(actualCER), baseCER=\(baseCER), transcript=\(transcript)"
            )
        }
    }

    @Test func actualWAVSmokeSelectsSupportFirmwareSpecialist() async throws {
        guard let audioPath = ProcessInfo.processInfo.environment["VOCO_QWEN3_SPECIALIST_SMOKE_AUDIO"],
              !audioPath.isEmpty else {
            return
        }
        let audioURL = URL(fileURLWithPath: audioPath)
        #expect(FileManager.default.fileExists(atPath: audioURL.path))
        let model = try #require(
            TranscriptionModelRegistry.models.compactMap { $0 as? Qwen3Model }
                .first(where: { $0.name == "qwen3-asr-1.7b-8bit" })
        )
        let service = Qwen3TranscriptionService()
        defer { Task { await service.cleanup() } }

        let transcript = try await service.transcribe(
            audioURL: audioURL,
            model: model,
            context: TranscriptionRequestContext(language: "Chinese", prompt: nil)
        )
        let routing = try #require(service.lastSpecialistRoutingMetadata)
        #expect(routing.trigger.triggered)
        #expect(routing.selection.selectSpecialist)
        #expect(routing.specialistAdapter?.adapterApplied == true)
        #expect(routing.specialistAdapter?.adapterSHA256?.count == 64)
        #expect(transcript.contains("韌體"))
        #expect(transcript.contains("Unicode"))
    }

    @Test func edgeTTSCenturyWindStockQuoteSpecialSmokeUsesAdapter() async throws {
        guard EdgeTTSCenturyWindFixture.isEnabled else {
            return
        }

        let audioURL = EdgeTTSCenturyWindFixture.audioURL
        #expect(
            FileManager.default.fileExists(atPath: audioURL.path),
            "Missing Edge TTS fixture WAV at \(audioURL.path). Run scripts/generate_qwen3_adapter_edge_tts_fixture.sh first."
        )

        let model = try #require(
            TranscriptionModelRegistry.models.compactMap { $0 as? Qwen3Model }
                .first(where: { $0.name == "qwen3-asr-1.7b-8bit" })
        )

        let service = Qwen3TranscriptionService()
        defer {
            Task {
                await service.cleanup()
            }
        }

        let transcript = try await service.transcribe(
            audioURL: audioURL,
            model: model,
            context: TranscriptionRequestContext(language: "Chinese", prompt: nil)
        )
        let metadata = service.lastAdapterMetadata
        #expect(metadata.adapterDetected == true)
        #expect(metadata.adapterLoaded == true)
        #expect(metadata.adapterApplied == true)
        #expect(metadata.adapterPath != nil)
        #expect(metadata.adapterSHA256?.count == 64)
        #expect(metadata.adapterLoadError == nil)

        let normalized = normalizeStockQuoteTranscript(transcript)
        print("Qwen3 adapter Edge TTS special smoke: transcript=\(transcript), normalized=\(normalized)")

        #expect(normalized.contains("世紀風電"), "Expected company name in transcript: \(transcript)")
        #expect(
            normalized.contains("2072") || normalized.contains("二零七二") || normalized.contains("兩千零七十二"),
            "Expected stock code 2072 in transcript: \(transcript)"
        )
        #expect(
            normalized.contains("今日股價") || (normalized.contains("今日") && normalized.contains("股價")),
            "Expected stock-price wording in transcript: \(transcript)"
        )
        #expect(
            normalized.contains("183") || normalized.contains("一百八十三"),
            "Expected price 183 in transcript: \(transcript)"
        )
        #expect(normalized.contains("元"), "Expected currency unit in transcript: \(transcript)")
    }
}

private enum FixtureError: LocalizedError {
    case loadFailed

    var errorDescription: String? {
        "fixture load failed"
    }
}

private struct ReplayLabAdapterFixture {
    let rowPk: Int
    let audioURL: URL
    let targetTranscript: String
    let basePrediction: String
    let adapterPrediction: String
}

private enum ReplayLabAdapterFixtures {
    static var artifactDirectory: URL {
        if let override = ProcessInfo.processInfo.environment["VOCO_REPLAYLAB_ADAPTER_ARTIFACT_DIR"],
           !override.isEmpty {
            return URL(fileURLWithPath: override, isDirectory: true)
        }

        return projectRootURL()
            .appendingPathComponent("LocalModels/ReplayLab/audio-adapter-training-data", isDirectory: true)
    }

    static func load(rows: [Int]) throws -> [ReplayLabAdapterFixture] {
        let evalRows = try keyedRows(from: artifactDirectory.appendingPathComponent("asr-adapter-eval.jsonl"))
        let baseRows = try keyedRows(from: artifactDirectory.appendingPathComponent("adapter-predictions.base-local.eval231.jsonl"))
        let adapterRows = try keyedRows(from: artifactDirectory.appendingPathComponent("adapter-predictions.audio-lora-balanced64.eval231.jsonl"))

        return try rows.map { rowPk in
            let eval = try #require(evalRows[rowPk], "Missing eval fixture row \(rowPk)")
            let base = try #require(baseRows[rowPk], "Missing base prediction row \(rowPk)")
            let adapter = try #require(adapterRows[rowPk], "Missing adapter prediction row \(rowPk)")
            let audioPath = try #require(eval["audioFilePath"] as? String)
            let target = try #require(eval["targetTranscript"] as? String ?? eval["baseRawASR"] as? String)
            let basePrediction = try #require(base["adapterTranscript"] as? String)
            let adapterPrediction = try #require(adapter["adapterTranscript"] as? String)
            return ReplayLabAdapterFixture(
                rowPk: rowPk,
                audioURL: URL(fileURLWithPath: audioPath),
                targetTranscript: target,
                basePrediction: basePrediction,
                adapterPrediction: adapterPrediction
            )
        }
    }

    private static func keyedRows(from url: URL) throws -> [Int: [String: Any]] {
        let data = try String(contentsOf: url, encoding: .utf8)
        var rows: [Int: [String: Any]] = [:]
        for line in data.split(separator: "\n") {
            let object = try JSONSerialization.jsonObject(with: Data(line.utf8)) as? [String: Any]
            if let object, let rowPk = object["rowPk"] as? Int {
                rows[rowPk] = object
            }
        }
        return rows
    }
}

private enum EdgeTTSCenturyWindFixture {
    static var isEnabled: Bool {
        ProcessInfo.processInfo.environment["VOCO_QWEN3_ADAPTER_EDGE_TTS_SMOKE"] == "1"
            || FileManager.default.fileExists(
                atPath: projectRootURL()
                    .appendingPathComponent("LocalModels/.run-qwen3-adapter-edge-tts-smoke")
                    .path
            )
    }

    static var audioURL: URL {
        if let override = ProcessInfo.processInfo.environment["VOCO_QWEN3_ADAPTER_EDGE_TTS_AUDIO"], !override.isEmpty {
            return URL(fileURLWithPath: override)
        }

        return projectRootURL()
            .appendingPathComponent("LocalModels/EdgeTTSSmoke/century-wind-2072-stock-183.wav")
    }
}

private func makeAdapterDirectory(named name: String = "test-audio-adapter") throws -> URL {
    let modelDirectory = try temporaryDirectory()
    let adapterDirectory = modelDirectory
        .appendingPathComponent("adapters", isDirectory: true)
        .appendingPathComponent(name, isDirectory: true)
    try FileManager.default.createDirectory(at: adapterDirectory, withIntermediateDirectories: true)
    return adapterDirectory
}

private func makeDescriptor(in modelDirectory: URL) throws -> Qwen3ASRAdapterDescriptor {
    let adapterDirectory = modelDirectory
        .appendingPathComponent("adapters", isDirectory: true)
        .appendingPathComponent("test-audio-adapter", isDirectory: true)
    try FileManager.default.createDirectory(at: adapterDirectory, withIntermediateDirectories: true)
    let configURL = adapterDirectory.appendingPathComponent("adapter_config.json")
    let weightsURL = adapterDirectory.appendingPathComponent("adapters.safetensors")
    try validConfigData().write(to: configURL)
    try Data("fixture-safetensors".utf8).write(to: weightsURL)
    let config = try JSONDecoder().decode(Qwen3ASRAdapterConfig.self, from: validConfigData())
    return Qwen3ASRAdapterDescriptor(
        directory: adapterDirectory,
        configURL: configURL,
        weightsURL: weightsURL,
        config: config
    )
}

private func validConfigData() -> Data {
    Data(
        """
        {
          "schema": "voco.qwen3-asr-audio-lora-adapter-config.v1",
          "base_model": "local-models/qwen3-asr-1.7b-8bit",
          "boundary": "audio-side Qwen3-ASR LoRA; not text cleanup LoRA",
          "fine_tune_type": "lora",
          "lora_parameters": {
            "rank": 4,
            "scale": 8,
            "dropout": 0,
            "keys": [
              "audio_tower.proj1"
            ]
          }
        }
        """.utf8
    )
}

private func temporaryDirectory() throws -> URL {
    let directory = FileManager.default.temporaryDirectory
        .appendingPathComponent("Qwen3ASRAdapterTests-\(UUID().uuidString)", isDirectory: true)
    try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    return directory
}

private func projectRootURL() -> URL {
    URL(fileURLWithPath: #filePath)
        .deletingLastPathComponent()
        .deletingLastPathComponent()
}

private func cer(_ hypothesis: String, _ reference: String) -> Double {
    let h = Array(hypothesis)
    let r = Array(reference)
    guard !r.isEmpty else {
        return h.isEmpty ? 0 : 1
    }

    var previous = Array(0...r.count)
    for (i, hChar) in h.enumerated() {
        var current = [i + 1] + Array(repeating: 0, count: r.count)
        for (j, rChar) in r.enumerated() {
            if hChar == rChar {
                current[j + 1] = previous[j]
            } else {
                current[j + 1] = min(previous[j], previous[j + 1], current[j]) + 1
            }
        }
        previous = current
    }
    return Double(previous[r.count]) / Double(r.count)
}

private func normalizedActionCommand(_ text: String) -> String {
    text
        .replacingOccurrences(of: "删除", with: "刪除")
        .filter { !$0.isWhitespace && !"。.!！?？".contains($0) }
}

private func normalizeStockQuoteTranscript(_ text: String) -> String {
    let digitMap: [Character: Character] = [
        "０": "0", "１": "1", "２": "2", "３": "3", "４": "4",
        "５": "5", "６": "6", "７": "7", "８": "8", "９": "9"
    ]
    let mapped = String(text.map { digitMap[$0] ?? $0 })
    return mapped
        .replacingOccurrences(of: "世纪", with: "世紀")
        .replacingOccurrences(of: "风电", with: "風電")
        .replacingOccurrences(of: "股价", with: "股價")
        .replacingOccurrences(of: "價格", with: "價")
        .filter { !$0.isWhitespace && !"，,。.!！?？：:；;、".contains($0) }
}
