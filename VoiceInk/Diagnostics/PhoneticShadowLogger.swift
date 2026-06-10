import Foundation
import OSLog

enum PhoneticShadowEventType: String, Codable, Equatable {
    case pipelineSnapshot
    case userCorrection
    case reviewSelection
    case rollback
    case candidateShadow
}

struct PhoneticShadowAudio: Equatable {
    var audioAssetId: String?
    var durationMs: Double?
    var sampleRate: Double?
    var audioHashPrefix: String?

    static let empty = PhoneticShadowAudio()
}

struct PhoneticShadowPipeline: Equatable {
    var asrEngine: String?
    var rawASR: String?
    var afterOpenCC: String?
    var afterPinyinCorrector: String?
    var afterHomophoneCorrection: String?
    var afterNasalCorrection: String?
    var afterPersonalCorrection: String?
    var llmEnhanced: String?
    var finalInserted: String?
    var route: String?
    var confidenceScore: Double?
    var avgLogprob: Double?
    var noSpeechProb: Double?
    var compressionRatio: Double?
    var posteriorGap: Double?
    var latencyMs: Double?

    static let empty = PhoneticShadowPipeline()
}

struct PhoneticShadowUserAction: Equatable {
    var source: String?
    var targetText: String?
    var selectedCandidateText: String?
    var rejectedCandidateText: String?
    var selectedRangeLength: Int?
    var timeSinceUtteranceMs: Int?
    var estimatedClickCount: Int?
    var repeatRequested: Bool?

    static let none = PhoneticShadowUserAction(source: "none")
}

struct PhoneticShadowUIContext: Equatable {
    var activeAppBundleId: String?
    var windowTitleHash: String?
    var focusedElementRole: String?
    var selectionTextBefore: String?
    var selectionTextAfter: String?
    var anchorBeforeHash: String?
    var anchorAfterHash: String?

    static let empty = PhoneticShadowUIContext()
}

struct PhoneticShadowClassification: Equatable {
    var lengthBucket: PhoneticLengthBucket
    var scriptMode: PhoneticScriptMode
    var languageMode: PhoneticLanguageMode
    var isCommandLike: Bool
    var isTechnicalTermCandidate: Bool
    var evidenceTier: CorrectionEvidenceTier
    var noiseFlags: [CorrectionEvidenceNoiseFlag]
    var isPurePhoneticCandidate: Bool

    static let empty = PhoneticShadowClassification(
        lengthBucket: .unknown,
        scriptMode: .unknown,
        languageMode: .unknown,
        isCommandLike: false,
        isTechnicalTermCandidate: false,
        evidenceTier: .none,
        noiseFlags: [],
        isPurePhoneticCandidate: false
    )
}

struct PhoneticShadowPhonetics: Equatable {
    var rawNormalized: String?
    var targetNormalized: String?
    var rawPhones: [String]
    var targetPhones: [String]
    var weightedPhoneEditDistance: Double?
    var pinyinSimilarity: Double?
    var confusionPairs: [PhoneticConfusionPair]

    static let empty = PhoneticShadowPhonetics(
        rawNormalized: nil,
        targetNormalized: nil,
        rawPhones: [],
        targetPhones: [],
        weightedPhoneEditDistance: nil,
        pinyinSimilarity: nil,
        confusionPairs: []
    )
}

struct PhoneticShadowCandidate: Equatable {
    var text: String
    var source: String
    var rank: Int
    var score: Double?
    var wouldChangeOutput: Bool
    var requiresReview: Bool
    var reason: String?
}

struct PhoneticShadowSafety: Equatable {
    var blockedBecauseLlmOnly: Bool
    var blockedBecauseShortPhraseRisk: Bool
    var blockedBecauseNoiseSuspected: Bool
    var blockedBecauseNegativeEvidence: Bool

    static let safe = PhoneticShadowSafety(
        blockedBecauseLlmOnly: false,
        blockedBecauseShortPhraseRisk: false,
        blockedBecauseNoiseSuspected: false,
        blockedBecauseNegativeEvidence: false
    )
}

struct PhoneticShadowFeatureFlags: Equatable {
    var shadowLoggingEnabled: Bool
    var candidateApplicationEnabled: Bool
}

struct PhoneticShadowEvent: Equatable {
    var schemaVersion = 1
    var eventId: String
    var createdAt: Date
    var appVersion: String?
    var buildGitSha: String?
    var eventType: PhoneticShadowEventType
    var utteranceId: String?
    var transcriptionDbId: String?
    var audio: PhoneticShadowAudio
    var pipeline: PhoneticShadowPipeline
    var userAction: PhoneticShadowUserAction
    var uiContext: PhoneticShadowUIContext
    var classification: PhoneticShadowClassification
    var phonetics: PhoneticShadowPhonetics
    var shadowCandidates: [PhoneticShadowCandidate]
    var safety: PhoneticShadowSafety
    var featureFlags: PhoneticShadowFeatureFlags

    init(
        eventId: String = UUID().uuidString,
        createdAt: Date = Date(),
        appVersion: String? = Bundle.main.infoDictionary?["CFBundleShortVersionString"] as? String,
        buildGitSha: String? = Bundle.main.infoDictionary?["VocoBuildGitSHA"] as? String,
        eventType: PhoneticShadowEventType,
        utteranceId: String? = nil,
        transcriptionDbId: String? = nil,
        audio: PhoneticShadowAudio = .empty,
        pipeline: PhoneticShadowPipeline = .empty,
        userAction: PhoneticShadowUserAction = .none,
        uiContext: PhoneticShadowUIContext = .empty,
        classification: PhoneticShadowClassification = .empty,
        phonetics: PhoneticShadowPhonetics = .empty,
        shadowCandidates: [PhoneticShadowCandidate] = [],
        safety: PhoneticShadowSafety = .safe,
        featureFlags: PhoneticShadowFeatureFlags = PhoneticShadowFeatureFlags.current()
    ) {
        self.eventId = eventId
        self.createdAt = createdAt
        self.appVersion = appVersion
        self.buildGitSha = buildGitSha
        self.eventType = eventType
        self.utteranceId = utteranceId
        self.transcriptionDbId = transcriptionDbId
        self.audio = audio
        self.pipeline = pipeline
        self.userAction = userAction
        self.uiContext = uiContext
        self.classification = classification
        self.phonetics = phonetics
        self.shadowCandidates = shadowCandidates
        self.safety = safety
        self.featureFlags = featureFlags
    }

    static func pipelineSnapshot(
        utteranceId: String? = nil,
        transcriptionDbId: String? = nil,
        pipeline: PhoneticShadowPipeline,
        audio: PhoneticShadowAudio = .empty
    ) -> PhoneticShadowEvent {
        let rawText = pipeline.rawASR ?? pipeline.finalInserted ?? ""
        let targetText = pipeline.finalInserted ?? pipeline.llmEnhanced ?? pipeline.afterPersonalCorrection ?? rawText
        let comparison = PhoneticFeatureExtractor.compare(raw: rawText, target: targetText)
        let evidence = CorrectionEvidenceClassifier.classify(
            CorrectionEvidenceInput(
                source: .none,
                rawText: rawText,
                targetText: targetText
            )
        )

        return PhoneticShadowEvent(
            eventType: .pipelineSnapshot,
            utteranceId: utteranceId,
            transcriptionDbId: transcriptionDbId,
            audio: audio,
            pipeline: pipeline,
            classification: PhoneticShadowClassification(
                lengthBucket: comparison.raw.lengthBucket,
                scriptMode: comparison.raw.scriptMode,
                languageMode: comparison.languageMode,
                isCommandLike: comparison.raw.isCommandLike,
                isTechnicalTermCandidate: comparison.raw.isTechnicalTermCandidate || comparison.target.isTechnicalTermCandidate,
                evidenceTier: evidence.evidenceTier,
                noiseFlags: evidence.noiseFlags,
                isPurePhoneticCandidate: evidence.isPurePhoneticCandidate
            ),
            phonetics: PhoneticShadowPhonetics(
                rawNormalized: comparison.raw.normalized,
                targetNormalized: comparison.target.normalized,
                rawPhones: comparison.raw.phones,
                targetPhones: comparison.target.phones,
                weightedPhoneEditDistance: comparison.weightedPhoneticEditDistance,
                pinyinSimilarity: comparison.pinyinSimilarity,
                confusionPairs: comparison.confusionPairs
            )
        )
    }

    static func userCorrection(
        signal: CorrectionFeedbackSignal,
        eventType: PhoneticShadowEventType = .userCorrection,
        source: CorrectionEvidenceSource,
        utteranceId: String? = nil,
        transcriptionDbId: String? = nil,
        selectedRangeLength: Int? = nil,
        timeSinceUtteranceMs: Int? = nil
    ) -> PhoneticShadowEvent {
        let classification = CorrectionEvidenceClassifier.classify(
            CorrectionEvidenceInput(
                source: source,
                rawText: signal.sourceText,
                targetText: signal.acceptedText,
                proposedText: signal.proposedText,
                selectedRangeLength: selectedRangeLength,
                timeSinceUtteranceMs: timeSinceUtteranceMs
            )
        )
        let comparison = classification.phoneticComparison
        let featureSource = comparison?.raw ?? PhoneticFeatureExtractor.extract(signal.sourceText)
        let featureTarget = comparison?.target ?? PhoneticFeatureExtractor.extract(signal.acceptedText)
        let flags = classification.noiseFlags

        return PhoneticShadowEvent(
            eventType: eventType,
            utteranceId: utteranceId,
            transcriptionDbId: transcriptionDbId,
            pipeline: PhoneticShadowPipeline(
                rawASR: signal.sourceText,
                llmEnhanced: source == .llmEnhancement || source == .ztextEnhancedDifference ? signal.acceptedText : nil,
                finalInserted: signal.acceptedText,
                confidenceScore: signal.confidenceScore
            ),
            userAction: PhoneticShadowUserAction(
                source: source.rawValue,
                targetText: signal.acceptedText,
                selectedCandidateText: signal.kind == .candidateSelection ? signal.acceptedText : nil,
                selectedRangeLength: selectedRangeLength,
                timeSinceUtteranceMs: timeSinceUtteranceMs
            ),
            classification: PhoneticShadowClassification(
                lengthBucket: featureSource.lengthBucket,
                scriptMode: featureSource.scriptMode,
                languageMode: comparison?.languageMode ?? featureSource.languageMode,
                isCommandLike: featureSource.isCommandLike,
                isTechnicalTermCandidate: featureSource.isTechnicalTermCandidate || featureTarget.isTechnicalTermCandidate,
                evidenceTier: classification.evidenceTier,
                noiseFlags: flags,
                isPurePhoneticCandidate: classification.isPurePhoneticCandidate
            ),
            phonetics: PhoneticShadowPhonetics(
                rawNormalized: featureSource.normalized,
                targetNormalized: featureTarget.normalized,
                rawPhones: featureSource.phones,
                targetPhones: featureTarget.phones,
                weightedPhoneEditDistance: comparison?.weightedPhoneticEditDistance,
                pinyinSimilarity: comparison?.pinyinSimilarity,
                confusionPairs: comparison?.confusionPairs ?? []
            ),
            safety: PhoneticShadowSafety(
                blockedBecauseLlmOnly: flags.contains(.llmOnly),
                blockedBecauseShortPhraseRisk: featureSource.lengthBucket == .oneToFour,
                blockedBecauseNoiseSuspected: !flags.isEmpty,
                blockedBecauseNegativeEvidence: classification.evidenceTier == .negativeEvidence
            )
        )
    }

    func jsonObject() -> [String: Any] {
        [
            "schemaVersion": schemaVersion,
            "eventId": eventId,
            "createdAt": ISO8601DateFormatter.phoneticShadow.string(from: createdAt),
            "appVersion": jsonValue(appVersion),
            "buildGitSha": jsonValue(buildGitSha),
            "eventType": eventType.rawValue,
            "utteranceId": jsonValue(utteranceId),
            "transcriptionDbId": jsonValue(transcriptionDbId),
            "featureFlags": featureFlags.jsonObject(),
            "audio": audio.jsonObject(),
            "pipeline": pipeline.jsonObject(),
            "userAction": userAction.jsonObject(),
            "uiContext": uiContext.jsonObject(),
            "classification": classification.jsonObject(),
            "phonetics": phonetics.jsonObject(),
            "shadowCandidates": shadowCandidates.map { $0.jsonObject() },
            "safety": safety.jsonObject(),
        ]
    }
}

final class PhoneticShadowLogger {
    static let shared = PhoneticShadowLogger()

    static let shadowLoggingEnabledKey = "VocoPhoneticShadowLoggingEnabled"
    static let candidateApplicationEnabledKey = "VocoPhoneticCandidateApplicationEnabled"

    private let logger = Logger(subsystem: AppIdentifiers.subsystem, category: "PhoneticShadowLogger")
    private let queue = DispatchQueue(label: "com.jasonchien.Voco.phoneticShadowLogger", qos: .utility)
    private let fileManager: FileManager
    private let logDirectory: URL
    private let maxFileSizeBytes: UInt64

    init(
        logDirectory: URL = AppIdentifiers.appSupportDirectory.appendingPathComponent("ShadowLogs", isDirectory: true),
        maxFileSizeBytes: UInt64 = 10 * 1024 * 1024,
        fileManager: FileManager = .default
    ) {
        self.logDirectory = logDirectory
        self.maxFileSizeBytes = maxFileSizeBytes
        self.fileManager = fileManager
    }

    func log(_ event: PhoneticShadowEvent, force: Bool = false) {
        guard force || Self.isShadowLoggingEnabled else { return }
        queue.async { [self] in
            do {
                try append(event)
            } catch {
                logger.error("Failed to write phonetic shadow event: \(error.localizedDescription, privacy: .public)")
            }
        }
    }

    func flush() {
        queue.sync {}
    }

    static var isShadowLoggingEnabled: Bool {
        UserDefaults.standard.bool(forKey: shadowLoggingEnabledKey)
    }

    static var isCandidateApplicationEnabled: Bool {
        UserDefaults.standard.bool(forKey: candidateApplicationEnabledKey)
    }

    static func logDirectory() -> URL {
        AppIdentifiers.appSupportDirectory.appendingPathComponent("ShadowLogs", isDirectory: true)
    }

    private func append(_ event: PhoneticShadowEvent) throws {
        try fileManager.createDirectory(at: logDirectory, withIntermediateDirectories: true)
        let url = activeLogURL(for: event.createdAt)
        try rotateIfNeeded(url)

        let object = event.jsonObject()
        let data = try JSONSerialization.data(withJSONObject: object, options: [.sortedKeys])
        var line = data
        line.append(0x0A)

        if fileManager.fileExists(atPath: url.path) {
            let handle = try FileHandle(forWritingTo: url)
            defer { try? handle.close() }
            try handle.seekToEnd()
            try handle.write(contentsOf: line)
            try handle.synchronize()
        } else {
            try line.write(to: url, options: .atomic)
        }
    }

    private func activeLogURL(for date: Date) -> URL {
        let day = Self.dayFormatter.string(from: date)
        return logDirectory.appendingPathComponent("phonetic-shadow-\(day).jsonl")
    }

    private func rotatedLogURL(for baseURL: URL, index: Int) -> URL {
        let name = baseURL.deletingPathExtension().lastPathComponent
        return baseURL.deletingLastPathComponent().appendingPathComponent("\(name).\(index).jsonl")
    }

    private func rotateIfNeeded(_ url: URL) throws {
        guard let size = try? fileSize(url), size >= maxFileSizeBytes else { return }

        for index in stride(from: 4, through: 1, by: -1) {
            let source = rotatedLogURL(for: url, index: index)
            let destination = rotatedLogURL(for: url, index: index + 1)
            if fileManager.fileExists(atPath: destination.path) {
                try? fileManager.removeItem(at: destination)
            }
            if fileManager.fileExists(atPath: source.path) {
                try fileManager.moveItem(at: source, to: destination)
            }
        }

        let firstRotated = rotatedLogURL(for: url, index: 1)
        if fileManager.fileExists(atPath: firstRotated.path) {
            try? fileManager.removeItem(at: firstRotated)
        }
        try fileManager.moveItem(at: url, to: firstRotated)
    }

    private func fileSize(_ url: URL) throws -> UInt64 {
        let values = try url.resourceValues(forKeys: [.fileSizeKey])
        return UInt64(values.fileSize ?? 0)
    }

    private static let dayFormatter: DateFormatter = {
        let formatter = DateFormatter()
        formatter.locale = Locale(identifier: "en_US_POSIX")
        formatter.timeZone = TimeZone.current
        formatter.dateFormat = "yyyy-MM-dd"
        return formatter
    }()
}

private extension PhoneticShadowFeatureFlags {
    static func current() -> PhoneticShadowFeatureFlags {
        PhoneticShadowFeatureFlags(
            shadowLoggingEnabled: PhoneticShadowLogger.isShadowLoggingEnabled,
            candidateApplicationEnabled: PhoneticShadowLogger.isCandidateApplicationEnabled
        )
    }

    func jsonObject() -> [String: Any] {
        [
            PhoneticShadowLogger.shadowLoggingEnabledKey: shadowLoggingEnabled,
            PhoneticShadowLogger.candidateApplicationEnabledKey: candidateApplicationEnabled,
        ]
    }
}

private extension PhoneticShadowAudio {
    func jsonObject() -> [String: Any] {
        [
            "audioAssetId": jsonValue(audioAssetId),
            "durationMs": jsonValue(durationMs),
            "sampleRate": jsonValue(sampleRate),
            "audioHashPrefix": jsonValue(audioHashPrefix),
        ]
    }
}

private extension PhoneticShadowPipeline {
    func jsonObject() -> [String: Any] {
        [
            "asrEngine": jsonValue(asrEngine ?? "unknown"),
            "rawASR": jsonValue(rawASR),
            "afterOpenCC": jsonValue(afterOpenCC),
            "afterPinyinCorrector": jsonValue(afterPinyinCorrector),
            "afterHomophoneCorrection": jsonValue(afterHomophoneCorrection),
            "afterNasalCorrection": jsonValue(afterNasalCorrection),
            "afterPersonalCorrection": jsonValue(afterPersonalCorrection),
            "llmEnhanced": jsonValue(llmEnhanced),
            "finalInserted": jsonValue(finalInserted),
            "route": jsonValue(route ?? "unknown"),
            "confidenceScore": jsonValue(confidenceScore),
            "avgLogprob": jsonValue(avgLogprob),
            "noSpeechProb": jsonValue(noSpeechProb),
            "compressionRatio": jsonValue(compressionRatio),
            "posteriorGap": jsonValue(posteriorGap),
            "latencyMs": jsonValue(latencyMs),
        ]
    }
}

private extension PhoneticShadowUserAction {
    func jsonObject() -> [String: Any] {
        [
            "source": jsonValue(source ?? "none"),
            "targetText": jsonValue(targetText),
            "selectedCandidateText": jsonValue(selectedCandidateText),
            "rejectedCandidateText": jsonValue(rejectedCandidateText),
            "selectedRangeLength": jsonValue(selectedRangeLength),
            "timeSinceUtteranceMs": jsonValue(timeSinceUtteranceMs),
            "estimatedClickCount": jsonValue(estimatedClickCount),
            "repeatRequested": jsonValue(repeatRequested),
        ]
    }
}

private extension PhoneticShadowUIContext {
    func jsonObject() -> [String: Any] {
        [
            "activeAppBundleId": jsonValue(activeAppBundleId),
            "windowTitleHash": jsonValue(windowTitleHash),
            "focusedElementRole": jsonValue(focusedElementRole),
            "selectionTextBefore": jsonValue(selectionTextBefore),
            "selectionTextAfter": jsonValue(selectionTextAfter),
            "anchorBeforeHash": jsonValue(anchorBeforeHash),
            "anchorAfterHash": jsonValue(anchorAfterHash),
        ]
    }
}

private extension PhoneticShadowClassification {
    func jsonObject() -> [String: Any] {
        [
            "lengthBucket": lengthBucket.rawValue,
            "scriptMode": scriptMode.rawValue,
            "languageMode": languageMode.rawValue,
            "isCommandLike": isCommandLike,
            "isTechnicalTermCandidate": isTechnicalTermCandidate,
            "evidenceTier": evidenceTier.rawValue,
            "noiseFlags": noiseFlags.map(\.rawValue),
            "isPurePhoneticCandidate": isPurePhoneticCandidate,
        ]
    }
}

private extension PhoneticShadowPhonetics {
    func jsonObject() -> [String: Any] {
        [
            "rawNormalized": jsonValue(rawNormalized),
            "targetNormalized": jsonValue(targetNormalized),
            "rawPhones": rawPhones,
            "targetPhones": targetPhones,
            "weightedPhoneEditDistance": jsonValue(weightedPhoneEditDistance),
            "pinyinSimilarity": jsonValue(pinyinSimilarity),
            "confusionPairs": confusionPairs.map { $0.jsonObject() },
        ]
    }
}

private extension PhoneticConfusionPair {
    func jsonObject() -> [String: Any] {
        [
            "raw": raw,
            "target": target,
            "operation": operation.rawValue,
            "position": jsonValue(position),
        ]
    }
}

private extension PhoneticShadowCandidate {
    func jsonObject() -> [String: Any] {
        [
            "text": text,
            "source": source,
            "rank": rank,
            "score": jsonValue(score),
            "wouldChangeOutput": wouldChangeOutput,
            "requiresReview": requiresReview,
            "reason": jsonValue(reason),
        ]
    }
}

private extension PhoneticShadowSafety {
    func jsonObject() -> [String: Any] {
        [
            "wouldHaveChangedFinalOutput": false,
            "autoApplied": false,
            "blockedBecauseLlmOnly": blockedBecauseLlmOnly,
            "blockedBecauseShortPhraseRisk": blockedBecauseShortPhraseRisk,
            "blockedBecauseNoiseSuspected": blockedBecauseNoiseSuspected,
            "blockedBecauseNegativeEvidence": blockedBecauseNegativeEvidence,
        ]
    }
}

private func jsonValue(_ value: String?) -> Any {
    value ?? NSNull()
}

private func jsonValue(_ value: Double?) -> Any {
    value ?? NSNull()
}

private func jsonValue(_ value: Int?) -> Any {
    value ?? NSNull()
}

private func jsonValue(_ value: Bool?) -> Any {
    value ?? NSNull()
}

private extension ISO8601DateFormatter {
    static let phoneticShadow: ISO8601DateFormatter = {
        let formatter = ISO8601DateFormatter()
        formatter.formatOptions = [.withInternetDateTime, .withFractionalSeconds]
        return formatter
    }()
}
