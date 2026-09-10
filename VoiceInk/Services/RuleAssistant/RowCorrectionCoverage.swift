import Combine
import Foundation

/// Local rule coverage for a history row: whether the auto-apply model installed on this Mac
/// would rewrite the row's original ASR text today.
///
/// This is the cross-client "is it fixed" signal. Worker receipts (`RowCorrectionMarkings`) are
/// keyed by row identity, so a fix made from Vocotype, claude.ai, Codex, or Claude Code can never
/// attach to a Mac row. Coverage sidesteps identity entirely: every client syncs the same model,
/// so every client can recompute this locally and agree. No network is involved.
enum RowCorrectionCoverageState: Equatable {
    /// The current model rewrites the source text with at least one policy that had not fired
    /// when the row was transcribed. This is the "a later fix now covers this row" case.
    case fixedByCurrentRules
    /// The model rewrites the source text and every firing policy already fired at transcription time.
    case appliedAtTranscription
    /// A rule matched but a protected-term guard blocked the rewrite.
    case blockedByGuard
    /// Only suggest-mode policies match; nothing is rewritten.
    case suggestOnly
}

struct RowCorrectionCoverage: Equatable {
    let state: RowCorrectionCoverageState
    /// Policy IDs behind the state (guard IDs for `.blockedByGuard`).
    let policyIds: [String]
    let modelVersion: String?

    var label: (text: String, tone: RowCorrectionMarkingTone) {
        switch state {
        case .fixedByCurrentRules:
            return (String(localized: "Fixed by current rules"), .green)
        case .appliedAtTranscription:
            return (String(localized: "Rule applied at transcription"), .secondary)
        case .blockedByGuard:
            return (String(localized: "Rule blocked by guard"), .orange)
        case .suggestOnly:
            return (String(localized: "Suggest-only rule matches"), .secondary)
        }
    }

    var badgeIcon: String {
        switch state {
        case .fixedByCurrentRules: return "checkmark.shield.fill"
        case .appliedAtTranscription: return "checkmark.shield"
        case .blockedByGuard: return "exclamationmark.shield"
        case .suggestOnly: return "lightbulb"
        }
    }
}

enum RowCorrectionCoverageEvaluator {
    /// Runtime-only normalizations that are not user rules and must not count as coverage.
    static let ignoredPolicyIds: Set<String> = [
        VocoAutoApplyModelService.currencyNumberNormalizationPolicyId
    ]

    /// The text a rule would have to fix: the pre-normalization ASR output when the row kept it,
    /// otherwise the stored text. Empty and canceled rows have no coverage.
    static func sourceText(for transcription: Transcription) -> String? {
        let candidates = [transcription.rawTranscript, transcription.text]
        for candidate in candidates {
            guard let candidate else { continue }
            let trimmed = candidate.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !trimmed.isEmpty, trimmed != Transcription.canceledTranscriptionText else { continue }
            return candidate
        }
        return nil
    }

    /// Pure classification of one evaluation against what the row already recorded.
    ///
    /// A rewrite counts as `appliedAtTranscription` when the row's stored text already equals the
    /// model output (the row ended up where the rules take it), or when every firing policy is in
    /// the hit IDs recorded at transcription time. Anything else is a later fix now covering the row.
    static func coverage(
        priorHitIds: [String],
        storedTexts: [String?] = [],
        evaluation: VocoAutoApplyEvaluation
    ) -> RowCorrectionCoverage? {
        let applied = evaluation.applied.filter { !ignoredPolicyIds.contains($0.policyId) }
        if evaluation.changed, !applied.isEmpty {
            let ids = uniqueInOrder(applied.map(\.policyId))
            let prior = Set(priorHitIds)
            let output = normalized(evaluation.outputText)
            let alreadyStored = storedTexts.contains { text in
                guard let text else { return false }
                let candidate = normalized(text)
                return !candidate.isEmpty && candidate == output
            }
            let state: RowCorrectionCoverageState = alreadyStored || ids.allSatisfy { prior.contains($0) }
                ? .appliedAtTranscription
                : .fixedByCurrentRules
            return RowCorrectionCoverage(state: state, policyIds: ids, modelVersion: evaluation.modelVersion)
        }
        if !evaluation.guardBlocks.isEmpty {
            return RowCorrectionCoverage(
                state: .blockedByGuard,
                policyIds: uniqueInOrder(evaluation.guardBlocks.map(\.guardId)),
                modelVersion: evaluation.modelVersion
            )
        }
        let suggestions = evaluation.suggestions.filter { !ignoredPolicyIds.contains($0.policyId) }
        if !suggestions.isEmpty {
            return RowCorrectionCoverage(
                state: .suggestOnly,
                policyIds: uniqueInOrder(suggestions.map(\.policyId)),
                modelVersion: evaluation.modelVersion
            )
        }
        return nil
    }

    static func coverage(
        for transcription: Transcription,
        service: VocoAutoApplyModelService
    ) -> RowCorrectionCoverage? {
        guard let source = sourceText(for: transcription) else { return nil }
        // History has no app/context hints, so context-locked rules only see the utterance itself.
        let evaluation = service.evaluate(source, context: source)
        return coverage(
            priorHitIds: transcription.autoApplyPolicyHitIDs,
            storedTexts: [transcription.normalizedTranscript, transcription.text],
            evaluation: evaluation
        )
    }

    private static func normalized(_ text: String) -> String {
        text.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    private static func uniqueInOrder(_ ids: [String]) -> [String] {
        var seen: Set<String> = []
        return ids.filter { seen.insert($0).inserted }
    }
}

/// Memoizes coverage per row and invalidates whenever the local model changes, so history views
/// re-render after a Worker sync without any per-row network work.
final class RowCorrectionCoverageStore: ObservableObject {
    static let shared = RowCorrectionCoverageStore(service: .shared)

    /// Bumps on every model status change; views observe it to recompute badges.
    @Published private(set) var generation: Int = 0

    private let service: VocoAutoApplyModelService
    private let lock = NSLock()
    private var cache: [String: RowCorrectionCoverage?] = [:]
    private var cancellable: AnyCancellable?
    private static let cacheLimit = 4000

    init(service: VocoAutoApplyModelService) {
        self.service = service
        cancellable = service.$status
            .removeDuplicates()
            .dropFirst()
            .receive(on: DispatchQueue.main)
            .sink { [weak self] _ in self?.invalidate() }
    }

    func invalidate() {
        lock.lock()
        cache.removeAll()
        lock.unlock()
        if Thread.isMainThread {
            generation &+= 1
        } else {
            DispatchQueue.main.async { [weak self] in self?.generation &+= 1 }
        }
    }

    func coverage(for transcription: Transcription) -> RowCorrectionCoverage? {
        guard let source = RowCorrectionCoverageEvaluator.sourceText(for: transcription) else { return nil }
        let key = "\(transcription.id.uuidString)|\(source.hashValue)|\(transcription.text.hashValue)|\(transcription.normalizedTranscript?.hashValue ?? 0)|\(transcription.autoApplyPolicyHitIDsJSON ?? "")"
        lock.lock()
        if let cached = cache[key] {
            lock.unlock()
            return cached
        }
        lock.unlock()

        let value = RowCorrectionCoverageEvaluator.coverage(for: transcription, service: service)

        lock.lock()
        if cache.count >= Self.cacheLimit { cache.removeAll() }
        cache[key] = value
        lock.unlock()
        return value
    }
}

extension Transcription {
    /// Local rule coverage from the installed auto-apply model (nil when no rule touches this row).
    var correctionCoverage: RowCorrectionCoverage? {
        RowCorrectionCoverageStore.shared.coverage(for: self)
    }
}
