import Foundation

// MARK: - Phase & state

enum RuleAssistantPhase: Equatable {
    case idle
    case loadingTools
    case thinking
    /// Deterministic preview + duplicate check of the exact draft the user will confirm.
    case checking
    case draftReady
    case confirming
    case syncing
    case published
    case failed(String)

    var isBusy: Bool {
        switch self {
        case .loadingTools, .thinking, .checking, .confirming, .syncing:
            return true
        case .idle, .draftReady, .published, .failed:
            return false
        }
    }
}

/// Who authored the user side of a turn. The App-side gate distinguishes typed instructions,
/// automatic scans, candidate choices, and explicit scope changes on a correction draft.
enum RuleAssistantTurnKind: Equatable {
    case manual
    case scan
    /// `candidateChosen` is true when at least one ticked option was a correction candidate: the
    /// reply then owes the user a draft (or another question), never plain text.
    case choice(candidateChosen: Bool = false, candidates: [RuleAssistantChoiceCandidate] = [])
    case rescope(scope: RuleAssistantScope, sourceText: String, targetText: String)

    var isManual: Bool { self == .manual }
}

/// Why a JSON-less answer gets one follow-up request before it is shown as a dead end.
enum RuleAssistantNudge: Equatable {
    /// The turn had to end in a question (scan) or its last line promised options (「請勾選：」).
    case question
    /// The answer listed plan lines ([exact] / [context-locked] ...) but no draft JSON.
    case draft
    /// The user ticked a candidate and the reply carries neither a draft nor a question.
    case choice

    var prompt: String {
        switch self {
        case .question: return RuleAssistantSession.questionNudgePrompt
        case .draft: return RuleAssistantSession.draftNudgePrompt
        case .choice: return RuleAssistantSession.choiceNudgePrompt
        }
    }

    var status: String {
        switch self {
        case .question: return String(localized: "The AI announced options without the question JSON; asking it to add them.")
        case .draft: return String(localized: "The AI listed a rule plan without the draft JSON; asking it to add them.")
        case .choice: return String(localized: "The AI answered the choice without a draft or question JSON; asking it to add one.")
        }
    }
}

struct RuleAssistantTurn: Equatable {
    var role: String
    var text: String
    /// Structured question carried by an assistant turn (rendered as option buttons).
    var question: RuleAssistantQuestion? = nil
    /// Filled in once the question was answered; the card then renders read-only.
    var answeredOptionIds: [String] = []
    var answeredOptionScopes: [String: RuleAssistantScope] = [:]
}

struct RuleAssistantPreview: Equatable {
    var wouldPublish: Bool
    var realtimeSupported: Bool
    var conflicts: Int
    var skipped: Int
    var unsupported: Int
    var reason: String?
    var baseModelSha256: String?
    /// "policies-changed" / "metadata-only" / "none": what the Worker says this event does to the runtime.
    var runtimeEffect: String?
    var policiesAdded: Int

    var ok: Bool { wouldPublish && realtimeSupported && conflicts == 0 && unsupported == 0 }

    /// The preview proved a new runtime policy appears. A duplicate hit on the same pattern (an existing
    /// or already retired rule) then describes a different policy, so it must not gate the card away.
    var addsRuntimePolicy: Bool { runtimeEffect == "policies-changed" || policiesAdded > 0 }

    init(json: [String: Any]) {
        wouldPublish = json.raBool("wouldPublish")
        realtimeSupported = json.raBool("realtimeSupported")
        conflicts = json.raArray("conflicts")?.count ?? 0
        skipped = json.raArray("skipped")?.count ?? 0
        unsupported = json.raArray("unsupported")?.count ?? 0
        reason = json.raNonBlankString("reason")
        baseModelSha256 = json.raNonBlankString("baseModelSha256")
        runtimeEffect = json.raNonBlankString("runtimeEffect")
        policiesAdded = Int(json.raInt64("policiesAdded") ?? 0)
    }
}

struct RuleAssistantDuplicate: Equatable {
    var alreadyApplied: Bool
    var duplicatePolicies: Int
    /// Policies with the same source/target that a tombstone already retired; they no longer apply at runtime.
    var retiredPolicies: Int
    var duplicateEvents: Int
    var suggestedNextAction: String?

    var found: Bool { alreadyApplied || duplicatePolicies > 0 || duplicateEvents > 0 }

    /// Only a retired rule matches: the same source/target exists in the model but a tombstone took it out,
    /// so re-adding it really does create a runtime policy and the old saved event is not a duplicate of it.
    var onlyRetiredPolicyMatched: Bool { !alreadyApplied && duplicatePolicies == 0 && retiredPolicies > 0 }

    /// Whether the Worker already holds this exact event. For add events any match counts. For a
    /// tombstone the Worker's duplicatePolicy / alreadyApplied describe the rule being retired (it has
    /// to exist for the tombstone to do anything), so only an identical tombstone event counts.
    /// A match that only describes a retired rule never proves the lost write landed.
    func alreadyWritten(eventType: String) -> Bool {
        if onlyRetiredPolicyMatched { return false }
        return eventType == "tombstone" ? duplicateEvents > 0 : found
    }

    init(json: [String: Any]) {
        alreadyApplied = json.raBool("alreadyApplied")
        if let policy = json.raDict("duplicatePolicy") {
            duplicatePolicies = policy.raBool("found") ? max(1, Int(policy.raInt64("count") ?? 0)) : 0
        } else {
            duplicatePolicies = 0
        }
        if let retired = json.raDict("retiredPolicy") {
            retiredPolicies = retired.raBool("found") ? max(1, Int(retired.raInt64("count") ?? 0)) : 0
        } else {
            retiredPolicies = 0
        }
        if let event = json.raDict("duplicateEvent") {
            duplicateEvents = event.raBool("found") ? max(1, Int(event.raInt64("count") ?? 0)) : 0
        } else {
            duplicateEvents = 0
        }
        suggestedNextAction = json.raNonBlankString("suggestedNextAction")
    }
}

/// Result of checking one exact draft (identified by nonce) against the Worker before the user confirms.
struct RuleAssistantDraftCheck: Equatable {
    var draftNonce: String
    var preview: RuleAssistantPreview?
    var duplicate: RuleAssistantDuplicate?
    var blockedReason: String?
    /// True when the model could plausibly fix this block itself (conflict, duplicate, unsupported type), so
    /// the App sends the reason back instead of leaving Jason to copy it into the note field. A transport
    /// failure, an already-applied rule, or a no-op preview is never retried.
    var retryable = false

    var ok: Bool { blockedReason == nil }
}

/// One proposed rule with its own Worker check and write outcome; an answer may carry several.
struct RuleAssistantDraftEntry: Equatable {
    var draft: RuleAssistantDraft
    var check: RuleAssistantDraftCheck?
    /// True once a write for this draft reached the Worker (or its outcome shows it did).
    var consumed = false
    var publishedSha256: String?
    /// Short per-draft result shown on its card (published / why it stopped).
    var outcome: String?
    /// Longer lexicon words containing a broad rule's source, added as negative examples by the App.
    var autoGuards: [String] = []
    var autoGuardsEnabled = true
}

struct RuleAssistantUIState: Equatable {
    var context: RuleAssistantContext
    var userText = ""
    var transcript: [RuleAssistantTurn] = []
    var reasoning = ""
    var answer = ""
    var toolStatus: [String] = []
    /// Every draft from the latest answer, in plan order; each is confirmed and published on its own.
    var drafts: [RuleAssistantDraftEntry] = []
    var phase: RuleAssistantPhase = .idle
    var goKeyConfigured = false
    var syncConfigured = false
    /// Number of neighbouring records already shared with the model in this conversation (0 = only the selected one).
    var neighborsShared = 0
    var publishedSha256: String?
    var publishMessage: String?
    /// Question from the latest answer that still waits for the user's choice.
    var pendingQuestion: RuleAssistantQuestion?
    var selectedOptionIds: [String] = []
    var selectedOptionScopes: [String: RuleAssistantScope] = [:]
    var optionScopeHints: [String: String] = [:]
    /// Set once the panel triggered the automatic find-issues turn, so reopening never rescans.
    var hasAutoScanned = false
    /// Kind of the turn currently in flight (nil when idle).
    var busyTurnKind: RuleAssistantTurnKind?

    var canSubmitChoice: Bool {
        pendingQuestion != nil && !selectedOptionIds.isEmpty && !phase.isBusy
    }

    /// A running find-issues turn may be interrupted by typing: it carries nothing of Jason's.
    /// Draft checks, publishing and syncing are never interruptible.
    var isInterruptible: Bool {
        busyTurnKind == .scan && (phase == .loadingTools || phase == .thinking)
    }

    /// The draft a bare confirm acts on: the first one still waiting, else the last one (for its outcome).
    var activeDraft: RuleAssistantDraftEntry? {
        drafts.first(where: { !$0.consumed }) ?? drafts.last
    }

    var draft: RuleAssistantDraft? { activeDraft?.draft }
    var draftCheck: RuleAssistantDraftCheck? { activeDraft?.check }
    var draftConsumed: Bool { activeDraft?.consumed == true }

    /// Confirm stays available after a recoverable failure (and after a sibling draft was published);
    /// every confirm re-runs the Worker gates first.
    var canConfirm: Bool {
        guard let activeDraft else { return false }
        return canConfirm(activeDraft)
    }

    func canConfirm(_ entry: RuleAssistantDraftEntry) -> Bool {
        !phase.isBusy && !entry.consumed && entry.check?.draftNonce == entry.draft.nonce && entry.check?.ok == true
    }
}

/// Minimal sync verdict the session needs; adapted from VocoAutoApplyWorkerSyncOutcome by the caller.
struct RuleAssistantSyncResult: Equatable {
    enum Outcome: Equatable {
        case installed
        case upToDate
        case failed
    }

    var outcome: Outcome
    var message: String
    var remoteSha256: String?
    var installedSha256: String?

    init(outcome: Outcome, message: String, remoteSha256: String?, installedSha256: String?) {
        self.outcome = outcome
        self.message = message
        self.remoteSha256 = remoteSha256
        self.installedSha256 = installedSha256
    }

    init(workerSync outcome: VocoAutoApplyWorkerSyncOutcome) {
        switch outcome.state {
        case .installed:
            self.outcome = .installed
        case .upToDate:
            self.outcome = .upToDate
        case .keptLocal:
            self.outcome = .failed
        }
        message = outcome.message
        remoteSha256 = outcome.manifest?.modelSha256
        installedSha256 = outcome.installedModelSha256
    }
}

// MARK: - Session

/// Orchestrates one rule-assistant conversation for one transcription record. No SwiftUI/SwiftData
/// dependency, so the whole flow (provider stream, tool rounds, draft gates, write, sync check)
/// runs against fake clients in tests. Callers run submit/confirm/resync inside a cancellable
/// Task and call cancel() to abort them.
@MainActor
final class RuleAssistantSession: ObservableObject {
    @Published private(set) var state: RuleAssistantUIState

    private let providerFactory: () -> RuleAssistantProvider?
    private let mcpFactory: () -> RuleAssistantMCP?
    private let syncNow: () async -> RuleAssistantSyncResult
    /// Loads records around the selected row (before, after), text fields only.
    private let neighborLoader: (_ before: Int, _ after: Int) -> [RuleAssistantContext]
    private let onCorrections: (RuleAssistantContext, String) -> Void
    /// Longer lexicon words containing a broad source (see RuleAssistantGuardSuggester); empty in tests without a lexicon.
    private let guardSuggester: (String) -> [String]
    /// Returns true when a candidate surface is absent from the lexicon and has no known longer word.
    private let lexiconProbe: (String) -> Bool
    /// Text re-run through the runtime installed right now; injectable so tests need no model on disk.
    private let runtimeReplayer: (String?) -> RuleAssistantRuntimeReplay?
    /// Re-reads the installed model before a replay. The Mac runtime only picks a new file up through its
    /// debounced file watcher, so a replay taken right after a publish would still describe the old model.
    private let runtimeReloader: () -> Void

    private var provider: RuleAssistantProvider?
    private var mcp: RuleAssistantMCP?
    private var toolJSON: [[String: Any]]?
    /// Committed wire history; a failed or cancelled round never leaves dangling tool calls here.
    private var history: [OpenCodeMessage] = []
    private var generation: Int64 = 0
    /// The user turn in flight: wire text sent to the model, display text shown in the transcript, kind.
    private var pendingTurn: (wire: String, display: String, kind: RuleAssistantTurnKind, restore: String?)?
    /// Question state to put back when a choice reply fails or is cancelled.
    private var pendingChoiceRestore: (question: RuleAssistantQuestion, selected: [String], scopes: [String: RuleAssistantScope])?
    /// Correction draft to put back when a scope-change reply fails or is cancelled.
    private var pendingRescopeRestore: RuleAssistantDraftEntry?
    private var turnKind: RuleAssistantTurnKind = .manual
    /// A publish changed the installed model and the recomputed replay has not reached the model yet.
    private var runtimeReplayIsFresh = false
    /// Automatic follow-up rounds spent on the current user action (reset by every submit).
    private var autoRetries = 0
    /// Gate keys of drafts already refused in this user action; the same draft again stops the retry loop.
    private var blockedDraftKeys: Set<String> = []

    init(
        context: RuleAssistantContext,
        providerFactory: @escaping () -> RuleAssistantProvider?,
        mcpFactory: @escaping () -> RuleAssistantMCP?,
        syncNow: @escaping () async -> RuleAssistantSyncResult,
        neighborLoader: @escaping (_ before: Int, _ after: Int) -> [RuleAssistantContext] = { _, _ in [] },
        onCorrections: @escaping (RuleAssistantContext, String) -> Void = { _, _ in },
        guardSuggester: @escaping (String) -> [String] = { _ in [] },
        lexiconProbe: @escaping (String) -> Bool = { _ in false },
        runtimeReplayer: @escaping (String?) -> RuleAssistantRuntimeReplay? = { RuleAssistantRuntimeReplay.current(inputText: $0) },
        runtimeReloader: @escaping () -> Void = { VocoAutoApplyModelService.shared.reload() }
    ) {
        state = RuleAssistantUIState(context: context)
        self.providerFactory = providerFactory
        self.mcpFactory = mcpFactory
        self.syncNow = syncNow
        self.neighborLoader = neighborLoader
        self.onCorrections = onCorrections
        self.guardSuggester = guardSuggester
        self.lexiconProbe = lexiconProbe
        self.runtimeReplayer = runtimeReplayer
        self.runtimeReloader = runtimeReloader
    }

    func setConfig(goKeyConfigured: Bool, syncConfigured: Bool) {
        state.goKeyConfigured = goKeyConfigured
        state.syncConfigured = syncConfigured
    }

    func setUserText(_ value: String) {
        let current = state
        if value == current.userText { return }
        // Typing while a find-issues turn runs is allowed (the caller cancels it on send);
        // typing during any other busy phase is ignored.
        if current.phase.isBusy {
            if current.isInterruptible { state.userText = value }
            return
        }
        // Editing after a draft was proposed invalidates that draft; the next submit starts a new turn.
        if current.drafts.contains(where: { !$0.consumed }) {
            state.userText = value
            state.drafts = []
            state.phase = .idle
        } else {
            state.userText = value
        }
    }

    /// The Go key changed: drop the provider so the next request authenticates with the new key.
    func onProviderKeyChanged() {
        provider?.close()
        provider = nil
    }

    /// Abort any in-flight work. The caller must also cancel the Task running the flow; this closes
    /// the network clients so blocked reads return, bumps the generation so a late completion cannot
    /// publish, and restores the input so the user can resend. Conversation history is kept.
    func cancel() {
        generation += 1
        closeClients()
        let current = state
        let aborted = pendingTurn
        if current.phase.isBusy {
            state.phase = .idle
            state.busyTurnKind = nil
            // Only text Jason typed goes back into the composer; a scan or choice reply has nothing to restore.
            if let restore = aborted?.restore { state.userText = restore }
            state.reasoning = ""
            state.answer = ""
            state.toolStatus.append(String(localized: "Stopped."))
            if let aborted {
                dropTrailingUserTurn(aborted.display)
            }
            restorePendingChoice()
            restorePendingRescope()
        }
        pendingTurn = nil
        pendingChoiceRestore = nil
        pendingRescopeRestore = nil
    }

    func closeClients() {
        provider?.close()
        provider = nil
        mcp?.close()
        mcp = nil
        toolJSON = nil
    }

    // MARK: Submit

    /// Text Jason typed himself.
    func submit(_ prompt: String) async {
        let clean = prompt.trimmingCharacters(in: .whitespacesAndNewlines)
        await submitTurn(wire: clean, display: clean, kind: .manual, restore: clean)
    }

    /// `restore` is the text put back into the composer when the turn fails or is stopped:
    /// what Jason typed (never a scan prompt or a choice reply's wire text).
    private func submitTurn(wire: String, display: String, kind: RuleAssistantTurnKind, restore: String?) async {
        let clean = wire
        if clean.isEmpty || state.phase.isBusy { return }
        let gen = generation
        pendingTurn = (wire: clean, display: display, kind: kind, restore: restore)
        turnKind = kind
        state.userText = ""
        state.phase = .loadingTools
        state.busyTurnKind = kind
        state.reasoning = ""
        state.answer = ""
        state.toolStatus = []
        state.drafts = []
        state.publishedSha256 = nil
        state.publishMessage = nil
        state.pendingQuestion = nil
        state.selectedOptionIds = []
        state.selectedOptionScopes = [:]
        state.optionScopeHints = [:]
        autoRetries = 0
        blockedDraftKeys = []
        state.transcript.append(RuleAssistantTurn(role: "user", text: display))
        var transaction = history
        do {
            if provider == nil { provider = providerFactory() }
            guard let provider else {
                throw RuleAssistantFailure(String(localized: "Save the OpenCode Go API key first."))
            }
            if mcp == nil { mcp = mcpFactory() }
            guard let worker = mcp else {
                throw RuleAssistantFailure(String(localized: "The Worker sync key is not configured on this Mac."))
            }
            let tools: [[String: Any]]
            if let cached = toolJSON {
                tools = cached
            } else {
                let remote = try await worker.tools()
                if remote.isEmpty {
                    throw RuleAssistantFailure(String(localized: "The Worker did not offer any rule tools."))
                }
                tools = remote.map { $0.asOpenAITool() } + [nearbyRecordsToolJSON()]
                toolJSON = tools
            }
            await refreshCorrections(worker: worker, gen: gen)
            try guardGeneration(gen)
            if transaction.isEmpty {
                transaction.append(OpenCodeMessage(role: "system", content: Self.systemPrompt))
            }
            transaction.append(OpenCodeMessage(
                role: "user",
                content: {
                    if case .rescope = kind { return clean }
                    return buildUserPrompt(context: state.context, instruction: clean, first: history.isEmpty)
                }()
            ))
            var nudged = false
            var promisedAnswer = ""
            for _ in 0..<Self.maxRounds {
                state.phase = .thinking
                let outcome = try await streamRound(provider: provider, messages: transaction, tools: tools, gen: gen)
                try guardGeneration(gen)
                transaction.append(OpenCodeMessage(
                    role: "assistant",
                    content: outcome.rawContent,
                    reasoningContent: outcome.rawReasoning.isEmpty ? nil : outcome.rawReasoning,
                    toolCalls: outcome.calls
                ))
                if outcome.calls.isEmpty {
                    // The model sometimes announces a question (「請勾選：」) or lists plan lines and stops
                    // without the JSON, especially after tool rounds. Ask once for the JSON alone instead of
                    // showing a dead end.
                    if !nudged, let nudge = Self.missingJSONNudge(answer: outcome.visibleAnswer, kind: kind) {
                        nudged = true
                        promisedAnswer = outcome.visibleAnswer.trimmingCharacters(in: .whitespacesAndNewlines)
                        status(nudge.status)
                        transaction.append(OpenCodeMessage(role: "user", content: nudge.prompt))
                        continue
                    }
                    let retry = try await finishTurn(answer: outcome.visibleAnswer, promisedAnswer: promisedAnswer, transaction: transaction, worker: worker, gen: gen)
                    try guardGeneration(gen)
                    guard let retry else { return }
                    // The App speaks for itself here: shown as an App turn, printed as [app] in the export,
                    // and sent with the same turn kind so the gate keeps applying.
                    state.transcript.append(RuleAssistantTurn(role: "app", text: retry))
                    // The refused cards stay visible with their reason until the next answer replaces them.
                    transaction = history
                    transaction.append(OpenCodeMessage(role: "user", content: retry))
                    promisedAnswer = ""
                    continue
                }
                for call in outcome.calls {
                    guard let function = call.raDict("function"),
                          let name = function.raString("name"),
                          let callId = call.raString("id")
                    else { continue }
                    let args = RAJSON.parseObject(function.raString("arguments") ?? "{}") ?? [:]
                    let result: String
                    if name == RuleAssistantConstants.nearbyTool {
                        result = RAJSON.serialize(loadNearby(args))
                    } else if RuleAssistantConstants.isMCPWriteTool(name) {
                        status(String(localized: "Paused write tool \(name): user confirmation required"))
                        result = RAJSON.serialize([
                            "blocked": true,
                            "reason": "Write tools can only run inside the App after the user confirms. Please output a JSON draft for the user to confirm instead.",
                        ])
                    } else {
                        status(String(localized: "Querying \(name)"))
                        do {
                            let body = try await worker.call(name, args: enrichedModelToolArguments(name: name, args: args))
                            result = RAJSON.serialize(body)
                        } catch let toolError as McpToolError {
                            status(String(localized: "\(name) reported an error: \(toolError.message)"))
                            result = RAJSON.serialize(["isError": true, "reason": toolError.message])
                        } catch is RuleAssistantClientError {
                            status(String(localized: "Refused a tool that is not allowlisted: \(name)"))
                            result = RAJSON.serialize(["isError": true, "reason": "tool is not allowlisted"])
                        }
                    }
                    try guardGeneration(gen)
                    transaction.append(OpenCodeMessage(
                        role: "tool",
                        content: String(result.prefix(Self.maxToolResultChars)),
                        toolCallId: callId
                    ))
                }
            }
            throw RuleAssistantFailure(String(localized: "The AI used more than \(Self.maxRounds) tool rounds; please rephrase and try again."))
        } catch is CancellationError {
            // cancel() already reset the visible state.
        } catch {
            if gen != generation { return }
            let message = Self.message(of: error)
            state.phase = .failed(message)
            state.busyTurnKind = nil
            if let restore { state.userText = restore }
            dropTrailingUserTurn(display)
            restorePendingChoice()
            restorePendingRescope()
        }
        if gen == generation {
            pendingTurn = nil
            pendingChoiceRestore = nil
            pendingRescopeRestore = nil
            state.busyTurnKind = nil
        }
    }

    private func enrichedModelToolArguments(name: String, args: [String: Any]) -> [String: Any] {
        guard name == "preview_auto_apply_control_event" || name == "detect_duplicate_control_event" else { return args }
        var enriched = args
        if enriched.raNonBlankString("actor") == nil { enriched["actor"] = RuleAssistantConstants.mcpClientName }
        if enriched.raNonBlankString("correctionSource") == nil { enriched["correctionSource"] = "voco" }
        if enriched["correctionRow"] == nil || enriched["correctionRow"] is NSNull { enriched["correctionRow"] = state.context.correctionRow() }
        return enriched
    }

    /// Find-issues mode: no explanation from the user. The model lists the places it suspects as
    /// options on a question card; the user picks instead of typing.
    func submitScan() async {
        state.hasAutoScanned = true
        await submitTurn(
            wire: Self.scanPrompt,
            display: String(localized: "(Looking for issues automatically)"),
            kind: .scan,
            restore: nil
        )
    }

    /// Run the automatic find-issues turn once per session, only on a fresh, configured session.
    func autoScanIfNeeded() async {
        let current = state
        guard !current.hasAutoScanned, current.transcript.isEmpty, current.phase == .idle,
              current.goKeyConfigured, current.syncConfigured
        else { return }
        await submitScan()
    }

    // MARK: Questions

    func toggleOption(_ optionId: String) {
        let current = state
        guard let question = current.pendingQuestion,
              question.options.contains(where: { $0.id == optionId }),
              !current.phase.isBusy
        else { return }
        if current.selectedOptionIds.contains(optionId) {
            state.selectedOptionIds.removeAll { $0 == optionId }
        } else if question.multiSelect {
            state.selectedOptionIds.append(optionId)
        } else {
            state.selectedOptionIds = [optionId]
        }
        if let option = question.options.first(where: { $0.id == optionId }), option.isCandidate,
           state.selectedOptionScopes[optionId] == nil {
            state.selectedOptionScopes[optionId] = defaultScope(for: option)
        }
    }

    private func defaultScope(for option: RuleAssistantQuestionOption) -> RuleAssistantScope {
        guard let surface = option.surface?.trimmingCharacters(in: .whitespacesAndNewlines),
              surface.count >= 2,
              surface.allSatisfy(Self.isCJKCharacter),
              lexiconProbe(surface)
        else { return .sentence }
        return .broad
    }

    private static func isCJKCharacter(_ character: Character) -> Bool {
        guard let scalar = character.unicodeScalars.first else { return false }
        switch scalar.value {
        case 0x3400...0x4DBF, 0x4E00...0x9FFF, 0xF900...0xFAFF, 0x20000...0x2FA1F:
            return true
        default:
            return false
        }
    }

    func setOptionScope(_ optionId: String, scope: RuleAssistantScope) {
        guard let question = state.pendingQuestion,
              question.options.contains(where: { $0.id == optionId && $0.isCandidate }),
              state.selectedOptionIds.contains(optionId), !state.phase.isBusy
        else { return }
        state.selectedOptionScopes[optionId] = scope
    }

    /// Send the selected options (plus any note in the composer) as the reply to the pending question.
    func submitChoice() async {
        let current = state
        guard let question = current.pendingQuestion, !current.selectedOptionIds.isEmpty, !current.phase.isBusy else { return }
        let chosen = question.options.filter { current.selectedOptionIds.contains($0.id) }
        guard !chosen.isEmpty else { return }
        let note = current.userText.trimmingCharacters(in: .whitespacesAndNewlines)
        var wireLines = ["回覆問題 \(question.id)：\(question.prompt)"]
        var displayParts: [String] = []
        var candidates: [RuleAssistantChoiceCandidate] = []
        for option in chosen {
            let scope = current.selectedOptionScopes[option.id] ?? defaultScope(for: option)
            let scopeText = option.isCandidate ? "（範圍：\(scope.wireLabel)）" : ""
            let line = "[\(option.id)] \(option.label)\(scopeText)"
            wireLines.append("選擇：\(line)")
            displayParts.append(option.isCandidate ? "\(option.label)（\(scope.wireLabel)）" : option.label)
            if let surface = option.surface, let target = option.target {
                candidates.append(RuleAssistantChoiceCandidate(surface: surface, target: target, scope: scope))
            }
        }
        wireLines.append("補充：\(note.isEmpty ? "無" : note)")
        var display = String(localized: "Chose: \(displayParts.joined(separator: "、"))")
        if !note.isEmpty { display += "\n\(note)" }
        // A typed note is Jason's own statement: the turn counts as manual and the broad gate is off.
        let kind: RuleAssistantTurnKind = note.isEmpty ? .choice(candidateChosen: !candidates.isEmpty, candidates: candidates) : .manual
        pendingChoiceRestore = (question: question, selected: current.selectedOptionIds, scopes: current.selectedOptionScopes)
        markQuestionAnswered(question.id, selected: current.selectedOptionIds, scopes: current.selectedOptionScopes)
        await submitTurn(wire: wireLines.joined(separator: "\n"), display: display, kind: kind, restore: note.isEmpty ? nil : note)
    }

    /// Widen a correction draft's scope. The draft is restored when the provider fails or the turn is stopped.
    func rescope(nonce: String, scope: RuleAssistantScope) async {
        guard scope != .sentence, !state.phase.isBusy,
              let entry = state.drafts.first(where: { $0.draft.nonce == nonce && !$0.consumed && $0.draft.eventType == "correction" }),
              let sourceText = entry.draft.sourceText, !sourceText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty,
              let targetText = entry.draft.targetText, !targetText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
        else { return }
        let wire: String
        let display: String
        switch scope {
        case .broad:
            wire = "範圍變更：任何語境\n原句：\(sourceText)\n修正後：\(targetText)\n請只針對錯誤片段出一條 replacementRule：sourcePattern 是原句裡最短的錯誤片段，targetText 是對應的正確片段，把 sourcePattern → targetText 套回原句必須正好得到修正後的句子。附 positiveExamples（原句 → 修正後）與 negativeExamples。不要問範圍，不要出其他類型的草稿。"
            display = String(localized: "Change scope to Any context")
        case .context:
            wire = "範圍變更：語境限定\n原句：\(sourceText)\n修正後：\(targetText)\n請只針對錯誤片段出一條 contextLockedRule：sourcePattern 是原句裡最短的錯誤片段，targetText 是對應的正確片段，把 sourcePattern → targetText 套回原句必須正好得到修正後的句子。contextTokensAny 從原句挑；不確定就只出一題 question，選項是 token 候選。不要問範圍。"
            display = String(localized: "Change scope to Context-locked")
        case .sentence:
            return
        }
        pendingRescopeRestore = entry
        state.drafts.removeAll { $0.draft.nonce == nonce }
        await submitTurn(wire: wire, display: display, kind: .rescope(scope: scope, sourceText: sourceText, targetText: targetText), restore: nil)
    }

    private func markQuestionAnswered(_ questionId: String, selected: [String], scopes: [String: RuleAssistantScope]) {
        guard let index = state.transcript.lastIndex(where: { $0.role == "assistant" && $0.question?.id == questionId }) else { return }
        state.transcript[index].answeredOptionIds = selected
        state.transcript[index].answeredOptionScopes = scopes
    }

    /// A failed or cancelled choice reply puts the question back so the user can answer again.
    private func restorePendingChoice() {
        guard let restore = pendingChoiceRestore else { return }
        pendingChoiceRestore = nil
        state.pendingQuestion = restore.question
        state.selectedOptionIds = restore.selected
        state.selectedOptionScopes = restore.scopes
        if let index = state.transcript.lastIndex(where: { $0.role == "assistant" && $0.question?.id == restore.question.id }) {
            state.transcript[index].answeredOptionIds = []
        }
    }

    private func restorePendingRescope() {
        guard let restore = pendingRescopeRestore else { return }
        pendingRescopeRestore = nil
        guard !state.drafts.contains(where: { $0.draft.nonce == restore.draft.nonce }) else { return }
        state.drafts = [restore] + state.drafts
        if state.phase == .idle { state.phase = .draftReady }
    }

    // MARK: Confirm / write / sync

    /// Confirm one draft (by nonce), or the first draft still waiting when nonce is nil.
    func confirm(nonce: String? = nil) async {
        let current = state
        let entry: RuleAssistantDraftEntry?
        if let nonce {
            entry = current.drafts.first(where: { $0.draft.nonce == nonce })
        } else {
            entry = current.activeDraft
        }
        guard let entry, current.canConfirm(entry) else { return }
        let draft = entry.draft
        let gen = generation
        if mcp == nil { mcp = mcpFactory() }
        guard let worker = mcp else {
            state.phase = .failed(String(localized: "The Worker sync key is not configured on this Mac."))
            return
        }
        state.phase = .confirming
        state.publishMessage = nil
        do {
            if !draft.isSafeForWrite() {
                throw RuleAssistantFailure(String(localized: "This draft does not satisfy the safety field limits; please send again."))
            }
            if draft.eventType == "tombstone" &&
                (draft.sourcePattern?.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty != false ||
                 draft.targetText?.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty != false) {
                throw RuleAssistantFailure(String(localized: "The tombstone has no sourcePattern/targetText; not writing it."))
            }
            // Re-check the same draft at confirm time; the model's earlier answers are not trusted here.
            let recheck = try await checkDraft(worker: worker, draft: draft)
            try guardGeneration(gen)
            updateEntry(nonce: draft.nonce) { $0.check = recheck }
            if let blocked = recheck.blockedReason {
                throw RuleAssistantFailure(blocked)
            }
            guard let toolName = draft.writeToolName() else {
                throw RuleAssistantFailure(String(localized: "Unknown rule type: \(draft.eventType)"))
            }
            let previewArgs = draft.toPreviewArguments(context: current.context)
            let writeArgs = draft.toMcpArguments(context: current.context)
            status(String(localized: "Writing \(toolName)"))
            try Task.checkCancellation()
            try guardGeneration(gen)
            let result: [String: Any]
            do {
                result = try await worker.call(toolName, args: writeArgs)
            } catch let toolError as McpToolError {
                if toolError.body.raBool("eventSaved") {
                    updateEntry(nonce: draft.nonce) { $0.consumed = true }
                    await refreshCorrections(worker: worker, gen: gen)
                    let eventId = String((toolError.body.raString("eventId") ?? "").prefix(24))
                    throw RuleAssistantFailure(String(localized: "The Worker recorded the event (\(eventId)) but did not publish: \(toolError.message). The event stays pending on the Worker and will not be resent; resolve the cause first (for example, retire the conflicting rule)."))
                }
                throw RuleAssistantFailure(String(localized: "The Worker rejected the write: \(toolError.message)"))
            } catch is CancellationError {
                throw CancellationError()
            } catch {
                // The POST may have reached the Worker even though its response was lost. Resolve with the
                // same draft's duplicate check; never blindly resend.
                await refreshCorrections(worker: worker, gen: gen)
                if draft.isTransaction {
                    updateEntry(nonce: draft.nonce) { $0.consumed = true }
                    throw RuleAssistantFailure(String(localized: "The Worker write outcome is unknown (\(Self.message(of: error))). Moves and merges cannot be duplicate-checked, so the transaction was not resent; confirm with lookup_auto_apply_policy or the Worker reconcile state before asking again."))
                }
                let after = try? await worker.call("detect_duplicate_control_event", args: previewArgs)
                let duplicate = after.map { RuleAssistantDuplicate(json: $0) }
                if duplicate?.alreadyWritten(eventType: draft.eventType) == true {
                    updateEntry(nonce: draft.nonce) { $0.consumed = true }
                    throw RuleAssistantFailure(String(localized: "The Worker write response was lost, but the duplicate check shows the rule already exists; it was not resent. Use Resync Mac model shortly to confirm the Mac model."))
                }
                throw RuleAssistantFailure(String(localized: "The Worker write outcome is unknown (\(Self.message(of: error))); the duplicate check did not find this rule, so nothing was resent. Press Confirm & Publish again — the full check runs before writing."))
            }
            updateEntry(nonce: draft.nonce) { $0.consumed = true }
            await refreshCorrections(worker: worker, gen: gen)
            // Worker publishes are cumulative: realtimeOverlay.published can be true because other pending
            // events went out, while this event itself was a no-op or skipped. Judge this rule by its own
            // effect first.
            let ownEffect = result.raString("runtimeEffect")
            if result.raBool("noop") || ownEffect == "none" {
                let suggestion = result.raNonBlankString("suggestedNextAction") ?? String(localized: "it may already exist or not match")
                throw RuleAssistantFailure(String(localized: "The Worker recorded the event, but this rule did not change the model (\(String(suggestion.prefix(200)))). It will not be resent."))
            }
            let overlay = result.raDict("realtimeOverlay")
            if overlay?.raBool("published") != true {
                let suggestion = result.raNonBlankString("suggestedNextAction") ?? String(localized: "check again shortly")
                throw RuleAssistantFailure(String(localized: "The Worker recorded the event (pending) but did not publish it in realtime: \(String(suggestion.prefix(200))). It will not be resent."))
            }
            guard let publishedSha = overlay?.raString("modelSha256"), Self.isSha256(publishedSha) else {
                throw RuleAssistantFailure(String(localized: "The Worker realtime overlay did not carry a valid published SHA."))
            }
            let warnings = result.raArray("warnings")?.count ?? 0
            if warnings > 0 {
                let first = (result.raArray("warnings")?.first as? String) ?? ""
                status(String(localized: "Worker warnings: \(warnings) — \(String(first.prefix(160)))"))
            }
            updateEntry(nonce: draft.nonce) {
                $0.publishedSha256 = publishedSha
                $0.outcome = String(localized: "Published (SHA \(publishedSha.prefix(12)))")
            }
            state.publishedSha256 = publishedSha
            state.publishMessage = String(localized: "The Worker published \(publishedSha.prefix(12)); syncing the Mac model…")
            try await syncAndVerify(publishedSha: publishedSha, gen: gen)
            // This publish moved the Worker head: a sibling card whose check ran against the old head (for
            // example a rule that was blocked by the rule this one just retired) is now stale. Re-check every
            // draft still waiting, so its button reflects the model the next confirm would actually write to.
            await recheckWaitingDrafts(worker: worker, gen: gen, excluding: draft.nonce)
        } catch is CancellationError {
            return
        } catch {
            if gen != generation { return }
            let message = Self.message(of: error)
            updateEntry(nonce: draft.nonce) { $0.outcome = message }
            state.phase = .failed(message)
        }
    }

    /// Re-run the Worker check for every draft that is still waiting, after the head moved.
    private func recheckWaitingDrafts(worker: RuleAssistantMCP, gen: Int64, excluding nonce: String) async {
        let waiting = state.drafts.filter { !$0.consumed && $0.draft.nonce != nonce }
        guard !waiting.isEmpty else { return }
        for entry in waiting {
            if gen != generation { return }
            do {
                let check = try await checkDraft(worker: worker, draft: entry.draft)
                try guardGeneration(gen)
                updateEntry(nonce: entry.draft.nonce) { $0.check = check }
            } catch is CancellationError {
                return
            } catch {
                status(String(localized: "Could not re-check the remaining draft (\(String(Self.message(of: error).prefix(100))))"))
            }
        }
    }

    // MARK: Corrections refresh

    func refreshCorrections() async {
        if mcp == nil { mcp = mcpFactory() }
        guard let worker = mcp else { return }
        await refreshCorrections(worker: worker, gen: generation)
    }

    private func refreshCorrections(worker: RuleAssistantMCP, gen: Int64) async {
        do {
            let tools = try await worker.tools()
            guard tools.contains(where: { $0.name == RowCorrectionMarkings.tool }) else { return }
            let context = state.context
            guard let markings = try await RowCorrectionMarkings.fetch(worker: worker, contexts: [context])[context.rowPk] else {
                throw RuleAssistantTransportError("Worker returned no correction row for this record")
            }
            try guardGeneration(gen)
            onCorrections(context, markings)
            try guardGeneration(gen)
            state.context.correctionsJSON = markings
        } catch is CancellationError {
            return
        } catch {
            status(String(localized: "Correction markings not refreshed; showing last result (\(String(Self.message(of: error).prefix(100))))"))
        }
    }

    // MARK: Resync

    /// Re-run the Mac sync against a SHA the Worker already published (after a sync failure).
    func resync() async {
        guard let sha = state.publishedSha256, !state.phase.isBusy else { return }
        let gen = generation
        do {
            await refreshCorrections()
            try await syncAndVerify(publishedSha: sha, gen: gen)
        } catch is CancellationError {
            return
        } catch {
            if gen != generation { return }
            state.phase = .failed(Self.message(of: error))
        }
    }

    private func syncAndVerify(publishedSha: String, gen: Int64) async throws {
        state.phase = .syncing
        let sync = await syncNow()
        try Task.checkCancellation()
        try guardGeneration(gen)
        guard sync.outcome != .failed else {
            throw RuleAssistantFailure(String(localized: "The Worker published \(publishedSha.prefix(12)), but the Mac sync did not complete: \(sync.message)"))
        }
        guard sync.remoteSha256 == publishedSha, sync.installedSha256 == publishedSha else {
            let remote = sync.remoteSha256.map { String($0.prefix(12)) } ?? "-"
            let installed = sync.installedSha256.map { String($0.prefix(12)) } ?? "-"
            throw RuleAssistantFailure(String(localized: "The Worker published \(publishedSha.prefix(12)), but the Mac sync SHA does not match (remote=\(remote), installed=\(installed))."))
        }
        state.phase = .published
        state.publishMessage = String(localized: "The Worker published and the Mac model is in sync (SHA \(publishedSha.prefix(12))). The canonical ReplayLab merge is handled separately.")
        refreshRuntimeReplay()
    }

    /// The installed model changed, so the record's runtimeReplay is stale. Recompute it and mark it so the
    /// next turn tells the model what the current rules already fix (otherwise it keeps drafting for a span a
    /// card published one minute ago already handles).
    private func refreshRuntimeReplay() {
        runtimeReloader()
        guard let replay = runtimeReplayer(state.context.normalizedTranscript ?? state.context.text) else { return }
        guard replay != state.context.runtimeReplay else { return }
        state.context.runtimeReplay = replay
        runtimeReplayIsFresh = true
        status(String(localized: "Re-ran this record through the updated runtime: \(replay.fires.count) rule(s) fire now."))
    }

    // MARK: Internals

    private func loadNearby(_ args: [String: Any]) -> [String: Any] {
        let before = min(max(Int(args.raInt64("before") ?? Int64(RuleAssistantConstants.nearbyMax)), 0), RuleAssistantConstants.nearbyMax)
        let after = min(max(Int(args.raInt64("after") ?? Int64(RuleAssistantConstants.nearbyMax)), 0), RuleAssistantConstants.nearbyMax)
        let context = state.context
        let records = neighborLoader(before, after)
            .filter { $0.rowPk != context.rowPk }
            .sorted { $0.timestampMs < $1.timestampMs }
        let beforeCount = records.filter { $0.timestampMs < context.timestampMs }.count
        status(String(localized: "Loaded nearby records: \(beforeCount) before, \(records.count - beforeCount) after"))
        state.neighborsShared = max(state.neighborsShared, records.count)
        return [
            "selectedRowPk": context.rowPk,
            "records": records.map { $0.toSafeJSON() },
            "note": records.isEmpty
                ? "no nearby records available"
                : "rows are in chronological order; audio is never included",
        ]
    }

    private struct RoundOutcome {
        var rawContent: String
        var rawReasoning: String
        var visibleAnswer: String
        var calls: [[String: Any]]
    }

    private func streamRound(
        provider: RuleAssistantProvider,
        messages: [OpenCodeMessage],
        tools: [[String: Any]],
        gen: Int64
    ) async throws -> RoundOutcome {
        var reasoning = ""
        var answer = ""
        var wireReasoning = ""
        var wireContent = ""
        let accumulator = ToolCallAccumulator()
        var finishReason: String?
        try await provider.stream(messages: messages, tools: tools) { delta in
            try self.guardGeneration(gen)
            if let reason = delta.finishReason { finishReason = reason }
            if reasoning.count + delta.reasoning.count > Self.maxTextChars
                || answer.count + delta.content.count > Self.maxTextChars
                || wireReasoning.count + delta.rawReasoningContent.count > Self.maxTextChars
                || wireContent.count + delta.rawContent.count > Self.maxTextChars {
                throw RuleAssistantTransportError(String(localized: "The AI response exceeded the safety length limit."))
            }
            reasoning.append(delta.reasoning)
            answer.append(delta.content)
            wireReasoning.append(delta.rawReasoningContent)
            wireContent.append(delta.rawContent)
            delta.toolCalls.forEach(accumulator.add)
            state.reasoning = reasoning
            state.answer = answer
        }
        try Task.checkCancellation()
        if !accumulator.isEmpty() {
            if finishReason != "tool_calls" && finishReason != "stop" {
                let actual = finishReason ?? String(localized: "unknown")
                throw RuleAssistantFailure(String(localized: "The provider finished with [\(actual)]; the tool calls may be incomplete, so this round was not executed."))
            }
            let problems = accumulator.problems()
            if !problems.isEmpty {
                throw RuleAssistantFailure(String(localized: "The tool calls were incomplete; none of this round was executed: \(problems.joined(separator: "; "))"))
            }
        } else if finishReason == "length" {
            throw RuleAssistantFailure(String(localized: "The AI reply was cut off by the length limit; please send again."))
        }
        return RoundOutcome(
            rawContent: wireContent,
            rawReasoning: wireReasoning,
            visibleAnswer: answer,
            calls: accumulator.isEmpty() ? [] : (try? accumulator.complete()) ?? []
        )
    }

    /// An answer without any draft or question JSON is a dead end in three shapes, each nudged once:
    /// plan lines without their drafts, a scan (or a 「請勾選：」 tail) without its question, and a
    /// candidate/scope-change turn answered in plain text. Returns nil when the answer already carries JSON
    /// or nothing was promised.
    static func missingJSONNudge(answer: String, kind: RuleAssistantTurnKind) -> RuleAssistantNudge? {
        let located = RuleAssistantDraft.locateAllJSON(in: answer)
        if RuleAssistantQuestion.parseFirst(located) != nil || !RuleAssistantDraft.parseAll(located).isEmpty { return nil }
        if case .rescope = kind { return .choice }
        if answer.range(of: planLinePattern, options: .regularExpression) != nil { return .draft }
        if case .scan = kind { return .question }
        let tail = String(answer.trimmingCharacters(in: .whitespacesAndNewlines).suffix(40))
        if tail.hasSuffix("：") || tail.hasSuffix(":") { return .question }
        if tail.range(of: "(請|请)(勾選|勾选|選擇|选择|選|选)", options: .regularExpression) != nil { return .question }
        if case .choice(let candidateChosen, _) = kind, candidateChosen { return .choice }
        return nil
    }

    /// A plan line as the system prompt defines it: 「[context-locked] <source> → <target>」 at the start of a line.
    static let planLinePattern = "(?m)^[ \\t]*\\[(exact|broad|context-locked|family|tombstone|move|merge|replace)\\]"

    /// Commit one completed answer. Returns the wire text of an automatic follow-up turn when the App-side
    /// gate or the Worker check refused every draft and the model can still fix it; nil when the turn is done.
    private func finishTurn(
        answer: String,
        promisedAnswer: String = "",
        transaction: [OpenCodeMessage],
        worker: RuleAssistantMCP,
        gen: Int64
    ) async throws -> String? {
        let located = RuleAssistantDraft.locateAllJSON(in: answer)
        let proposed = RuleAssistantDraft.parseAll(located)
        let question = RuleAssistantQuestion.parseFirst(located)
        // Cut every JSON object out of the visible reply (only when at least one parsed as a draft or question).
        var display: String
        if !proposed.isEmpty || question != nil {
            var trimmed = answer
            for item in located.sorted(by: { $0.range.lowerBound > $1.range.lowerBound }) {
                trimmed.removeSubrange(item.range)
            }
            display = trimmed
                .replacingOccurrences(of: "```[A-Za-z]*\\s*```", with: "", options: .regularExpression)
                .trimmingCharacters(in: .whitespacesAndNewlines)
        } else {
            display = answer.trimmingCharacters(in: .whitespacesAndNewlines)
        }
        // After a nudge the model usually returns the JSON alone; keep its earlier explanation visible.
        if !promisedAnswer.isEmpty {
            display = display.isEmpty ? promisedAnswer : promisedAnswer + "\n" + display
        }
        // Commit the wire history only for a normally completed turn.
        history = transaction
        let fallback: String
        if question != nil {
            fallback = String(localized: "(Question for you)")
        } else if proposed.isEmpty {
            fallback = String(localized: "(No text reply)")
        } else {
            fallback = String(localized: "(Rule draft proposed)")
        }
        state.transcript.append(RuleAssistantTurn(role: "assistant", text: display.isEmpty ? fallback : display, question: question))
        // The live-answer bubble in the panel is for streaming only; once the turn is
        // committed to the transcript, keeping state.answer would render the reply twice.
        state.answer = ""
        // The choice/rescope reply that led here is committed; a later failure must not resurrect it.
        pendingChoiceRestore = nil
        pendingRescopeRestore = nil
        if let question {
            state.pendingQuestion = question
            state.selectedOptionIds = []
            var scopes: [String: RuleAssistantScope] = [:]
            var hints: [String: String] = [:]
            for option in question.options where option.isCandidate {
                let scope = defaultScope(for: option)
                scopes[option.id] = scope
                if scope == .broad { hints[option.id] = "詞庫查無，建議任何語境" }
            }
            state.selectedOptionScopes = scopes
            state.optionScopeHints = hints
        }
        if proposed.isEmpty {
            state.phase = .idle
            return nil
        }
        var rejected: [String] = []
        let kind = turnKind
        // A scope change is judged against 原句 and against 原句 after the rules that are already live.
        var runtimeBaseline: String?
        if case .rescope(_, let rescopeSource, _) = kind {
            runtimeBaseline = runtimeReplayer(rescopeSource)?.outputText
        }
        let accepted = proposed.filter { draft in
            if !draft.isSafeForWrite() {
                rejected.append(String(localized: "The AI proposed a draft with incomplete or unsafe fields (\(draft.eventType)); it was not listed for confirmation. Add more detail and send again."))
                return false
            }
            if let reason = Self.gateReason(for: draft, kind: kind, runtimeBaseline: runtimeBaseline, recordText: state.context.combinedText) {
                rejected.append(reason)
                return false
            }
            return true
        }
        for reason in rejected {
            status(String(localized: "Skipped a draft: \(String(reason.prefix(120)))"))
        }
        guard !accepted.isEmpty else {
            // A question alongside only-rejected drafts still deserves an answer; keep the turn usable.
            if question != nil {
                state.phase = .idle
                return nil
            }
            if let retry = autoRetryPrompt(reasons: rejected, blocked: proposed) { return retry }
            return nil
        }
        let entries = accepted.map { draft -> RuleAssistantDraftEntry in
            var entry = RuleAssistantDraftEntry(draft: draft)
            let guards = Self.autoGuards(for: draft, suggester: guardSuggester)
            if !guards.isEmpty {
                entry.autoGuards = guards
                entry.draft = Self.withAutoGuards(draft, guards: guards, enabled: true)
                status(String(localized: "Added negative examples for longer words: \(guards.joined(separator: "\u{3001}"))"))
            }
            return entry
        }
        state.drafts = entries
        state.phase = .checking
        for entry in entries {
            var draft = entry.draft
            do {
                draft = try await resolveTombstoneIdentity(worker: worker, draft: draft)
                if draft != entry.draft {
                    updateEntry(nonce: entry.draft.nonce) { $0.draft = draft }
                }
            } catch is CancellationError {
                throw CancellationError()
            } catch {
                let reason = Self.message(of: error)
                updateEntry(nonce: entry.draft.nonce) {
                    $0.check = RuleAssistantDraftCheck(draftNonce: entry.draft.nonce, preview: nil, duplicate: nil, blockedReason: reason)
                }
                continue
            }
            let check = try await checkDraft(worker: worker, draft: draft)
            try guardGeneration(gen)
            updateEntry(nonce: entry.draft.nonce) { $0.check = check }
        }
        // Every card the Worker refused for a reason the model can act on goes back to the model once,
        // so Jason does not have to copy the block message into the note field.
        let checkedDrafts = state.drafts
        if question == nil, !checkedDrafts.isEmpty,
           checkedDrafts.allSatisfy({ $0.check?.retryable == true }),
           let retry = autoRetryPrompt(
               reasons: checkedDrafts.compactMap { $0.check?.blockedReason },
               blocked: checkedDrafts.map { $0.draft }
           ) {
            return retry
        }
        if case .failed = state.phase {} else { state.phase = .draftReady }
        return nil
    }

    /// The gate or the Worker refused every draft: send the reasons back to the model as an App turn instead
    /// of leaving Jason to retype them. Returns nil (and sets the failed phase) when the retry budget is
    /// spent or the model just re-sent a draft that was already refused in this user action.
    private func autoRetryPrompt(reasons: [String], blocked: [RuleAssistantDraft]) -> String? {
        let first = reasons.first ?? String(localized: "No confirmable draft.")
        let keys = Set(blocked.map(Self.draftGateKey))
        if !keys.isEmpty, keys.isSubset(of: blockedDraftKeys) {
            state.phase = .failed(String(localized: "\(first)（已自動重試 \(autoRetries) 次，模型重出相同草稿）"))
            return nil
        }
        blockedDraftKeys.formUnion(keys)
        guard autoRetries < Self.maxAutoRetries else {
            state.phase = .failed(String(localized: "\(first)（已自動重試 \(autoRetries) 次仍未通過檢查）"))
            return nil
        }
        autoRetries += 1
        status(String(localized: "Sent the check result back to the AI (attempt \(autoRetries)/\(Self.maxAutoRetries))"))
        let lines = reasons.map { "- \($0)" }.joined(separator: "\n")
        return "\(Self.gateFeedbackPrefix)：\n\(lines)\n請依這些理由重新出草稿，不要問範圍、不要重複同一份草稿。"
    }

    /// Identity of a draft for the retry loop: the same rule proposed again, whatever its wording.
    static func draftGateKey(_ draft: RuleAssistantDraft) -> String {
        [
            draft.eventType,
            draft.sourcePattern ?? draft.sourceText ?? "",
            draft.targetText ?? "",
            draft.contextTokensAny.sorted().joined(separator: "\u{1}"),
            draft.aliases.sorted().joined(separator: "\u{1}"),
            draft.policyId ?? ""
        ].joined(separator: "|")
    }

    /// Tombstones that carry a policyId must also carry the canonical text keys before they can be written.
    /// The resolved keys remain on the draft for confirm's final safety guard and re-check.
    private func resolveTombstoneIdentity(worker: RuleAssistantMCP, draft: RuleAssistantDraft) async throws -> RuleAssistantDraft {
        guard draft.eventType == "tombstone",
              let policyId = draft.policyId?.trimmingCharacters(in: .whitespacesAndNewlines), !policyId.isEmpty
        else { return draft }

        let lookup: [String: Any]
        do {
            lookup = try await worker.call("lookup_auto_apply_policy", args: ["policyId": policyId, "limit": 1])
        } catch is CancellationError {
            throw CancellationError()
        } catch let error as McpToolError {
            throw RuleAssistantFailure(String(localized: "Worker policy lookup failed: \(error.message)"))
        } catch {
            throw RuleAssistantFailure(String(localized: "Worker policy lookup failed: \(Self.message(of: error))"))
        }
        let policies = lookup.raDictArray("policies")
        guard lookup.raInt64("matchedPoliciesCount") != 0,
              let policy = policies.first,
              let source = policy.raNonBlankString("sourcePattern")?.trimmingCharacters(in: .whitespacesAndNewlines),
              let target = policy.raNonBlankString("targetText")?.trimmingCharacters(in: .whitespacesAndNewlines),
              !source.isEmpty,
              !target.isEmpty
        else {
            throw RuleAssistantFailure(String(localized: "The Worker has no policy with policyId \(policyId), so sourcePattern/targetText cannot be filled in; propose it again with sourcePattern + targetText."))
        }

        let normalizedSource = source.trimmingCharacters(in: .whitespacesAndNewlines)
        let normalizedTarget = target.trimmingCharacters(in: .whitespacesAndNewlines)
        let draftSource = draft.sourcePattern?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
        let draftTarget = draft.targetText?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
        if (!draftSource.isEmpty && draftSource != normalizedSource) || (!draftTarget.isEmpty && draftTarget != normalizedTarget) {
            throw RuleAssistantFailure(String(localized: "The policyId and sourcePattern/targetText point to different policies; check them and propose again."))
        }
        var resolved = draft
        let wasMissing = draftSource.isEmpty || draftTarget.isEmpty
        if draftSource.isEmpty { resolved.sourcePattern = normalizedSource }
        if draftTarget.isEmpty { resolved.targetText = normalizedTarget }
        if wasMissing {
            status(String(localized: "Filled the policy text from the Worker: \(normalizedSource) → \(normalizedTarget)"))
        }
        return resolved
    }

    // MARK: Automatic negative guards

    /// Longer lexicon words that contain a broad draft's source(s), deduplicated, at most 8.
    static func autoGuards(for draft: RuleAssistantDraft, suggester: (String) -> [String]) -> [String] {
        guard draft.isBroad else { return [] }
        let sources: [String]
        if draft.eventType == "replacementFamily" {
            sources = draft.aliases
        } else {
            sources = [draft.sourcePattern ?? ""]
        }
        var seen = Set<String>()
        var guards: [String] = []
        for source in sources where !source.isEmpty {
            for word in suggester(source) where !seen.contains(word) {
                seen.insert(word)
                guards.append(word)
            }
        }
        return Array(guards.prefix(8))
    }

    /// The draft with the automatic guards present (enabled) or removed (disabled); model-authored examples stay.
    static func withAutoGuards(_ draft: RuleAssistantDraft, guards: [String], enabled: Bool) -> RuleAssistantDraft {
        var updated = draft
        let guardSet = Set(guards)
        updated.negativeExamples.removeAll { guardSet.contains($0.text) && $0.context.isEmpty }
        if enabled {
            let existing = Set(updated.negativeExamples.map(\.text))
            // isSafeForWrite allows at most 10 negative examples; never push the draft past it.
            for word in guards where !existing.contains(word) && updated.negativeExamples.count < 10 {
                updated.negativeExamples.append(RuleAssistantExample(text: word, context: "", expectedText: word))
            }
        }
        return updated
    }

    /// Turn the automatic guards of one draft on or off; the Worker check re-runs on the edited draft.
    func setAutoGuards(nonce: String, enabled: Bool) async {
        let current = state
        guard !current.phase.isBusy,
              let entry = current.drafts.first(where: { $0.draft.nonce == nonce }),
              !entry.consumed, !entry.autoGuards.isEmpty, entry.autoGuardsEnabled != enabled
        else { return }
        let gen = generation
        let updated = Self.withAutoGuards(entry.draft, guards: entry.autoGuards, enabled: enabled)
        updateEntry(nonce: nonce) {
            $0.draft = updated
            $0.autoGuardsEnabled = enabled
            $0.check = nil
        }
        if mcp == nil { mcp = mcpFactory() }
        guard let worker = mcp else {
            updateEntry(nonce: nonce) { $0.check = RuleAssistantDraftCheck(draftNonce: nonce, preview: nil, duplicate: nil, blockedReason: String(localized: "The Worker sync key is not configured on this Mac.")) }
            return
        }
        state.phase = .checking
        do {
            let check = try await checkDraft(worker: worker, draft: updated)
            try guardGeneration(gen)
            updateEntry(nonce: nonce) { $0.check = check }
            state.phase = .draftReady
        } catch is CancellationError {
        } catch {
            if gen != generation { return }
            updateEntry(nonce: nonce) { $0.check = RuleAssistantDraftCheck(draftNonce: nonce, preview: nil, duplicate: nil, blockedReason: Self.message(of: error)) }
            state.phase = .draftReady
        }
    }

    /// Preview and duplicate-check the exact draft; Worker tool failures are reported as a blocked
    /// reason, never thrown. Transport failures still propagate (the turn itself is broken).
    private func checkDraft(worker: RuleAssistantMCP, draft: RuleAssistantDraft) async throws -> RuleAssistantDraftCheck {
        let previewArgs = draft.toPreviewArguments(context: state.context)
        status(String(localized: "Worker preview (\(draft.eventType))"))
        let preview: RuleAssistantPreview
        do {
            preview = RuleAssistantPreview(json: try await worker.call("preview_auto_apply_control_event", args: previewArgs))
        } catch let error as McpToolError {
            return RuleAssistantDraftCheck(
                draftNonce: draft.nonce,
                preview: nil,
                duplicate: nil,
                blockedReason: String(localized: "Worker preview failed: \(error.message)")
            )
        }
        try Task.checkCancellation()
        // detect_duplicate_control_event only knows the five single-event types; for transactions the
        // Worker's preview already returns noop/wouldPublish=false when there is nothing to do.
        var duplicate: RuleAssistantDuplicate?
        if !draft.isTransaction {
            status(String(localized: "Worker duplicate check"))
            do {
                duplicate = RuleAssistantDuplicate(json: try await worker.call("detect_duplicate_control_event", args: previewArgs))
            } catch let error as McpToolError {
                return RuleAssistantDraftCheck(
                    draftNonce: draft.nonce,
                    preview: preview,
                    duplicate: nil,
                    blockedReason: String(localized: "Worker duplicate check failed: \(error.message)")
                )
            }
        }
        let blocked: String?
        // Blocks the model can act on itself (pick another event type, tombstone first, narrow the scope).
        var retryable = false
        if !preview.realtimeSupported {
            retryable = true
            blocked = String(localized: "The Worker cannot publish this event type in realtime (unsupported=\(preview.unsupported), skipped=\(preview.skipped)): \(preview.reason ?? "")")
        } else if preview.conflicts > 0 {
            retryable = true
            blocked = String(localized: "Conflicts with existing rules (\(preview.conflicts)): \(preview.reason ?? String(localized: "Retire the conflicting rule or narrow the scope first"))")
        } else if !preview.wouldPublish {
            blocked = String(localized: "The Worker preview says this would not change the model: \(preview.reason ?? "")")
        } else if let duplicate, draft.eventType == "tombstone" {
            // The policy match is the rule this tombstone retires, not a duplicate; a second identical
            // tombstone event is the only thing to refuse (an already retired rule fails preview above).
            if duplicate.duplicateEvents > 0 {
                retryable = true
                blocked = String(localized: "The Worker already has an identical event (\(duplicate.duplicateEvents)); not writing again.")
            } else {
                blocked = nil
            }
        } else if let duplicate, preview.addsRuntimePolicy {
            // The Worker's duplicate check matches by source pattern, so an existing or retired rule for the
            // same pattern shows up even when this draft adds a genuinely new runtime policy (a wider context
            // lock, for example). The preview is the deterministic answer; only an identical saved event stops it.
            //
            // Except when the only policy match is a retired one: the saved event the Worker points at is the
            // very add that tombstone took out (record voco:row:24701), so blocking on it makes the rule
            // impossible to re-send. Newer Workers already drop those events from duplicateEvent; this keeps
            // an older Worker from wedging the card too.
            if duplicate.duplicateEvents > 0 && !duplicate.onlyRetiredPolicyMatched {
                retryable = true
                blocked = String(localized: "The Worker already has an identical event (\(duplicate.duplicateEvents)); not writing again.")
            } else {
                blocked = nil
                if duplicate.duplicateEvents > 0 && duplicate.onlyRetiredPolicyMatched {
                    status(String(localized: "The Worker's matching event belongs to a rule a tombstone already retired, and the preview confirms this adds a runtime policy; letting it through."))
                } else if duplicate.alreadyApplied || duplicate.duplicatePolicies > 0 {
                    status(String(localized: "The Worker duplicate check matched an existing or retired rule for the same pattern, but the preview confirms this adds a runtime policy; letting it through."))
                }
            }
        } else if let duplicate {
            if duplicate.alreadyApplied {
                blocked = String(localized: "This rule is already applied in the current Worker model; no need to write it again.")
            } else if duplicate.duplicateEvents > 0 {
                retryable = true
                blocked = String(localized: "The Worker already has an identical event (\(duplicate.duplicateEvents)); not writing again.")
            } else if duplicate.duplicatePolicies > 0 {
                retryable = true
                blocked = String(localized: "An existing policy already covers this (\(duplicate.duplicatePolicies)): \(duplicate.suggestedNextAction ?? String(localized: "Check whether the old rule should be retired first"))")
            } else {
                blocked = nil
            }
        } else {
            blocked = nil
        }
        return RuleAssistantDraftCheck(
            draftNonce: draft.nonce,
            preview: preview,
            duplicate: duplicate,
            blockedReason: blocked,
            retryable: blocked != nil && retryable
        )
    }

    // MARK: Review export

    /// Status line after the export was put on the clipboard.
    func noteCopiedForReview() {
        status(String(localized: "Copied the conversation for review."))
    }

    /// The whole conversation as text for Claude Code / Codex to review: record, every turn with
    /// questions and choices, drafts with Worker checks, the wire history (tool calls and truncated
    /// results), and a note telling the reviewer what to look at. Never includes keys or audio.
    func exportForReview(client: String, date: Date = Date()) -> String {
        Self.reviewExport(state: state, history: history, client: client, date: date)
    }

    static func reviewExport(state: RuleAssistantUIState, history: [OpenCodeMessage], client: String, date: Date) -> String {
        var out: [String] = []
        let stamp = ISO8601DateFormatter().string(from: date)
        out.append("# Voco rule assistant · turn export for review")
        out.append("client: \(client)")
        out.append("model: \(RuleAssistantConstants.model)")
        out.append("record: \(state.context.sourceNote())" + (state.context.recordId.map { " · recordId \($0)" } ?? ""))
        out.append("exported: \(stamp)")
        out.append("phase: \(state.phase)")
        if let version = state.context.autoApplyModelVersion { out.append("auto-apply model: \(version)") }
        out.append("")
        out.append("## Record (what the model saw)")
        let record: [(String, String?)] = [
            ("rawTranscript", state.context.rawTranscript),
            ("text", state.context.text),
            ("normalizedTranscript", state.context.normalizedTranscript),
            ("enhancedText", state.context.enhancedText),
            ("selectedCandidate", state.context.selectedCandidate),
            ("finalPastedText", state.context.finalPastedText),
            ("transcriptionModel", state.context.transcriptionModelName),
        ]
        for (label, value) in record {
            if let value, !value.isEmpty { out.append("- \(label): \(value)") }
        }
        let markings = RowCorrectionMarkings.parse(state.context.correctionsJSON)
        if !markings.isEmpty {
            out.append("- corrections: \(RAJSON.serializeArray(markings.map { $0.toJSONObject() }))")
        }
        if state.neighborsShared > 0 { out.append("- nearby records shared with the model: \(state.neighborsShared)") }
        if let replay = state.context.runtimeReplay {
            out.append("")
            out.append("## Runtime replay (current runtime)")
            out.append("- outputText: \(replay.outputText)")
            out.append("- changed: \(replay.changed)")
            if let version = replay.modelVersion { out.append("- model version: \(version)") }
            if replay.fires.isEmpty {
                out.append("- no policy fired")
            } else {
                for fire in replay.fires {
                    out.append("- \(fire.policyId): \(fire.sourcePattern) → \(fire.targetText)")
                }
            }
        } else {
            out.append("- runtime replay: unavailable")
        }
        out.append("")
        out.append("## Conversation (as shown in the App)")
        for turn in state.transcript {
            out.append("[\(turn.role)] \(turn.text)")
            if let question = turn.question {
                out.append("  question \(question.id) (\(question.multiSelect ? "multi" : "single")): \(question.prompt)")
                for option in question.options {
                    var line = "    - [\(option.id)] \(option.label)"
                    if let surface = option.surface, let target = option.target { line += " {\(surface) → \(target)}" }
                    if let detail = option.detail { line += " · \(detail)" }
                    if turn.answeredOptionIds.contains(option.id) {
                        if let scope = turn.answeredOptionScopes[option.id], option.isCandidate { line += " ✓ chosen scope=\(scope.wireLabel)" }
                        else { line += " ✓ chosen" }
                    }
                    out.append(line)
                }
                if turn.answeredOptionIds.isEmpty, state.pendingQuestion?.id == question.id {
                    out.append("    (still unanswered; currently ticked: \(state.selectedOptionIds.joined(separator: ", ")))")
                }
            }
        }
        if !state.answer.isEmpty { out.append("[assistant · streaming] \(state.answer)") }
        out.append("")
        out.append("## UI state (what the card actually shows)")
        out.append("- phase=\(state.phase) busyTurnKind=\(state.busyTurnKind.map { "\($0)" } ?? "nil") interruptible=\(state.isInterruptible) hasAutoScanned=\(state.hasAutoScanned) goKey=\(state.goKeyConfigured) syncKey=\(state.syncConfigured)")
        out.append("- composer text: \(state.userText.isEmpty ? "(empty)" : state.userText)")
        if let question = state.pendingQuestion {
            out.append("- pending question \(question.id): interactive card, canSubmitChoice=\(state.canSubmitChoice), ticked=[\(state.selectedOptionIds.joined(separator: ", "))]")
            for option in question.options {
                let ticked = state.selectedOptionIds.contains(option.id)
                var line = "  - [\(option.id)] \(ticked ? "☑" : "☐") \"\(option.label)\""
                line += option.isCandidate ? " (candidate)" : " (plain option)"
                if let scope = state.selectedOptionScopes[option.id], option.isCandidate { line += " scope=\(scope.wireLabel)" }
                if let hint = state.optionScopeHints[option.id] { line += " · \(hint)" }
                out.append(line)
            }
        } else {
            out.append("- no pending question (no interactive card)")
        }
        for (index, entry) in state.drafts.enumerated() {
            let scopeControl = entry.draft.eventType == "correction" && !entry.consumed && !state.phase.isBusy ? "shown" : "hidden"
            out.append("- draft card \(index + 1): confirm button \(state.canConfirm(entry) ? "enabled" : "disabled"), consumed=\(entry.consumed), check=\(entry.check == nil ? "pending" : (entry.check?.ok == true ? "ok" : "blocked")), scope control=\(scopeControl)")
        }
        out.append("")
        if !state.drafts.isEmpty {
            out.append("## Drafts")
            for (index, entry) in state.drafts.enumerated() {
                let draft = entry.draft
                var head = "\(index + 1). \(draft.eventType)"
                if let source = draft.sourcePattern ?? draft.sourceText { head += ": \(source)" }
                if !draft.aliases.isEmpty { head += " [\(draft.aliases.joined(separator: "、"))]" }
                if let target = draft.targetText { head += " → \(target)" }
                out.append(head)
                if let check = entry.check {
                    var line = "   check:"
                    if let preview = check.preview {
                        line += " preview wouldPublish=\(preview.wouldPublish) conflicts=\(preview.conflicts) skipped=\(preview.skipped) unsupported=\(preview.unsupported)"
                        if let reason = preview.reason { line += " reason=\(reason)" }
                    }
                    if let duplicate = check.duplicate {
                        line += " · duplicate found=\(duplicate.found) applied=\(duplicate.alreadyApplied) policies=\(duplicate.duplicatePolicies) retired=\(duplicate.retiredPolicies) events=\(duplicate.duplicateEvents)"
                    }
                    if let blocked = check.blockedReason { line += " · BLOCKED: \(blocked)" }
                    out.append(line)
                } else {
                    out.append("   check: not run")
                }
                if !entry.autoGuards.isEmpty {
                    out.append("   auto negative guards (\(entry.autoGuardsEnabled ? "on" : "off")): \(entry.autoGuards.joined(separator: "\u{3001}"))")
                }
                out.append("   consumed=\(entry.consumed)" + (entry.publishedSha256.map { " publishedSha=\($0.prefix(12))" } ?? "") + (entry.outcome.map { " outcome=\($0)" } ?? ""))
                out.append("   worker args: \(RAJSON.serialize(draft.toPreviewArguments(context: state.context)))")
            }
            out.append("")
        }
        if let message = state.publishMessage { out.append("publish: \(message)"); out.append("") }
        if !state.toolStatus.isEmpty {
            out.append("## App status lines")
            for line in state.toolStatus { out.append("- \(line)") }
            out.append("")
        }
        out.append("## Wire history (what actually went to the model; system prompt omitted, long fields truncated)")
        for message in history {
            switch message.role {
            case "system":
                out.append("[system] (system prompt, \(message.content?.count ?? 0) chars; see RuleAssistantSession systemPrompt)")
            case "tool":
                out.append("[tool \(message.toolCallId ?? "?")] \(Self.truncate(message.content ?? "", 800))")
            case "assistant":
                if let reasoning = message.reasoningContent, !reasoning.isEmpty {
                    out.append("[assistant · reasoning] \(Self.truncate(reasoning, 1500))")
                }
                if let content = message.content, !content.isEmpty { out.append("[assistant] \(content)") }
                for call in message.toolCalls {
                    let function = call.raDict("function")
                    out.append("[assistant → tool] \(function?.raString("name") ?? "?")(\(Self.truncate(function?.raString("arguments") ?? "", 300))) id=\(call.raString("id") ?? "?")")
                }
            default:
                out.append("[\(message.role)] \(Self.truncate(message.content ?? "", 1500))")
            }
        }
        out.append("")
        out.append("## For the reviewer (Claude Code / Codex)")
        out.append("""
            The user exported this rule-assistant conversation from the App because something in it looked wrong or worth improving. Judge the model's behaviour against the App's rules: did it find the right candidates, ask with a proper question JSON instead of free text, draft correction for a choice, honor a rescope turn only when its replacement reproduces the supplied sentence, avoid inventing targets, and respect the correction markings and the App-side gate? Then decide where the fix belongs:
            - Prompt: `systemPrompt` / `scanPrompt` in VoiceInk/Services/RuleAssistant/RuleAssistantSession.swift (Mac) and `SYSTEM_PROMPT` / `SCAN_PROMPT` in app/src/main/java/com/vocotype/ruleassistant/RuleAssistantSession.kt (Android). The two must stay identical apart from platform words.
            - App logic: question/draft parsing in RuleAssistantProtocol.swift / .kt, the turn-kind gate `gateReason`, the choice reply format in `submitChoice`, or the panel/screen UI. A `[app]` turn is the App's own automatic follow-up carrying the gate / Worker refusal reasons, not something Jason typed.
            - Worker: preview / duplicate / write results above come from the Worker MCP; a wrong check result is a Worker issue, not a prompt issue.
            Reply with concrete suggestions, or make the change directly on both platforms and run the RuleAssistant test suites.
            """)
        return out.joined(separator: "\n")
    }

    private static func truncate(_ text: String, _ limit: Int) -> String {
        text.count <= limit ? text : String(text.prefix(limit)) + "… (\(text.count) chars)"
    }

    /// Apply the App-side authorization for each kind of user turn. `runtimeBaseline` is 原句 re-run through
    /// the runtime installed now; a scope change may legitimately only cover what the live rules leave behind.
    static func gateReason(
        for draft: RuleAssistantDraft,
        kind: RuleAssistantTurnKind,
        runtimeBaseline: String? = nil,
        recordText: String? = nil
    ) -> String? {
        switch kind {
        case .manual:
            return nil
        case .scan:
            guard draft.isBroad || draft.isTransaction else { return nil }
            let source = draft.sourcePattern ?? draft.aliases.joined(separator: "\u{3001}")
            return String(localized: "Find-issues never creates broad rules or moves/merges families (\(draft.eventType): \(source) → \(draft.targetText ?? "")). Tick the candidate first and change the correction draft's scope on its card, or state yourself that the source is never valid and send again.")
        case .choice(let candidateChosen, let candidates):
            if draft.isFamilyTransaction {
                return String(localized: "Moving or merging families needs your own words (\(draft.eventType)); it is not created from a choice.")
            }
            // Replacing a context lock is already a context-scoped transaction; it retires the old lock and
            // re-adds it with more tokens, so it needs no broad authorisation.
            if draft.eventType == "replaceContextLockedRule" { return nil }
            if draft.eventType == "contextLockedRule" {
                if candidates.isEmpty { return nil }
                let source = draft.sourcePattern?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
                let target = draft.targetText?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
                guard let candidate = candidates.first(where: { $0.scope == .context && $0.surface.trimmingCharacters(in: .whitespacesAndNewlines) == source && $0.target.trimmingCharacters(in: .whitespacesAndNewlines) == target }) else {
                    return String(localized: "語境限定規則只會為勾選時已選「語境限定」的候選建立（sourcePattern=\(source)，targetText=\(target)）。")
                }
                let text = recordText ?? ""
                let missing = draft.contextTokensAny.map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }.filter { !$0.isEmpty && !text.contains($0) }
                if !missing.isEmpty {
                    return String(localized: "context token 不在紀錄文字裡：\(missing.joined(separator: "、"))。")
                }
                _ = candidate
                return nil
            }
            guard draft.isBroad else { return nil }
            guard candidateChosen else { return String(localized: "廣泛規則不會從勾選建立；請先在候選旁選擇範圍。") }
            if draft.eventType == "replacementRule" {
                let source = draft.sourcePattern?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
                let target = draft.targetText?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
                guard let candidate = candidates.first(where: { $0.scope == .broad && $0.surface.trimmingCharacters(in: .whitespacesAndNewlines) == source && $0.target.trimmingCharacters(in: .whitespacesAndNewlines) == target }) else {
                    return String(localized: "廣泛規則只會為勾選時已選「任何語境」的候選建立（sourcePattern=\(source)，targetText=\(target)）。")
                }
                _ = candidate
                return nil
            }
            return String(localized: "廣泛規則不會從勾選建立；replacementFamily／family transaction 仍需要你親自說明。")
        case .rescope(let scope, let sourceText, let targetText):
            if draft.eventType == "correction" { return nil }
            let expectedType = scope == .broad ? "replacementRule" : "contextLockedRule"
            guard draft.eventType == expectedType else {
                return String(localized: "範圍變更回合只接受 \(expectedType)，收到 \(draft.eventType)。")
            }
            // Two bases count as 原句: the sentence as recognised, and the same sentence after the rules that
            // are already live. Part of 修正後 may already be produced by a rule published earlier in this
            // conversation, and then only the remaining wrong span belongs in this draft.
            var bases = [sourceText]
            if let runtimeBaseline = runtimeBaseline?.trimmingCharacters(in: .whitespacesAndNewlines),
               !runtimeBaseline.isEmpty, runtimeBaseline != sourceText {
                bases.append(runtimeBaseline)
            }
            guard let rawPattern = draft.sourcePattern?.trimmingCharacters(in: .whitespacesAndNewlines), !rawPattern.isEmpty,
                  let rawTarget = draft.targetText?.trimmingCharacters(in: .whitespacesAndNewlines), !rawTarget.isEmpty,
                  bases.contains(where: { $0.contains(rawPattern) })
            else {
                let patternDescription = draft.sourcePattern ?? ""
                let targetDescription = draft.targetText ?? ""
                return String(localized: "範圍變更回合的 sourcePattern 必須是原句中的錯誤片段（sourcePattern=\(patternDescription)，targetText=\(targetDescription)）。")
            }
            let expected = targetText.trimmingCharacters(in: .whitespacesAndNewlines)
            let results = bases.map { base in
                VocoAutoApplyModelService.applyLiteralReplacement(
                    rawPattern,
                    with: rawTarget,
                    in: base
                ).trimmingCharacters(in: .whitespacesAndNewlines)
            }
            guard results.contains(expected) else {
                if results.contains(where: { Self.onlyCJKLatinBoundarySpacingDifference($0, expected) }) {
                    return String(localized: "套回原句後只差中英之間的空格：targetText 只放正確的英文片段本身，sourcePattern 只放對應的錯誤片段，不要把周圍的中文一起放進來。")
                }
                let applied = results.first ?? ""
                return String(localized: "套回原句不等於修正後：sourcePattern=\(rawPattern)，targetText=\(rawTarget)，套用結果=\(applied)，修正後=\(expected)。")
            }
            if scope == .context {
                let missing = draft.contextTokensAny.map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }.filter { !$0.isEmpty && !sourceText.contains($0) }
                if !missing.isEmpty {
                    return String(localized: "context token 不在原句裡：\(missing.joined(separator: "、"))。")
                }
            }
            return nil
        }
    }

    private static func onlyCJKLatinBoundarySpacingDifference(_ lhs: String, _ rhs: String) -> Bool {
        guard lhs.replacingOccurrences(of: " ", with: "") == rhs.replacingOccurrences(of: " ", with: "") else { return false }
        return [lhs, rhs].allSatisfy { value in
            let characters = Array(value)
            for index in characters.indices where characters[index] == " " {
                guard index > characters.startIndex,
                      index < characters.index(before: characters.endIndex)
                else { return false }
                let left = characters[characters.index(before: index)]
                let right = characters[characters.index(after: index)]
                let leftHan = Self.isHanCharacterForSpacing(left)
                let rightHan = Self.isHanCharacterForSpacing(right)
                let leftASCII = Self.isASCIIAlphaNumericForSpacing(left)
                let rightASCII = Self.isASCIIAlphaNumericForSpacing(right)
                guard (leftHan && rightASCII) || (leftASCII && rightHan) else { return false }
            }
            return true
        }
    }

    private static func isHanCharacterForSpacing(_ character: Character) -> Bool {
        guard character.unicodeScalars.count == 1,
              let value = character.unicodeScalars.first?.value
        else { return false }
        return (0x3400...0x4DBF).contains(value)
            || (0x4E00...0x9FFF).contains(value)
            || (0x20000...0x3134F).contains(value)
    }

    private static func isASCIIAlphaNumericForSpacing(_ character: Character) -> Bool {
        guard character.unicodeScalars.count == 1,
              let value = character.unicodeScalars.first?.value
        else { return false }
        return (48...57).contains(value) || (65...90).contains(value) || (97...122).contains(value)
    }

    private func updateEntry(nonce: String, transform: (inout RuleAssistantDraftEntry) -> Void) {
        state.drafts = state.drafts.map { entry in
            var entry = entry
            if entry.draft.nonce == nonce { transform(&entry) }
            return entry
        }
    }

    private func guardGeneration(_ gen: Int64) throws {
        if gen != generation { throw CancellationError() }
    }

    private func status(_ line: String) {
        state.toolStatus = Array((state.toolStatus + [line]).suffix(Self.maxStatusLines))
    }

    private func dropTrailingUserTurn(_ text: String) {
        while let last = state.transcript.last, last.role == "user", last.text == text {
            state.transcript.removeLast()
        }
    }

    private func buildUserPrompt(context: RuleAssistantContext, instruction: String, first: Bool) -> String {
        if first {
            return "選取紀錄（只限此筆）：\(RAJSON.serialize(context.toSafeJSON()))\n使用者說明：\(instruction)"
        }
        let markings = RAJSON.serializeArray(RowCorrectionMarkings.parse(state.context.correctionsJSON).map { $0.toJSONObject() })
        var replayLine = ""
        if runtimeReplayIsFresh, let replay = state.context.runtimeReplay {
            runtimeReplayIsFresh = false
            replayLine = "此筆現行 runtime 重放（剛發佈的規則已納入）：\(RAJSON.serialize(replay.toJSONObject()))\n"
        }
        return "\(replayLine)此筆修正標記（Worker 狀態）：\(markings)\n使用者補充：\(instruction)"
    }

    static func message(of error: Error) -> String {
        if let failure = error as? RuleAssistantFailure { return failure.message }
        if let transport = error as? RuleAssistantTransportError { return transport.message }
        if let toolError = error as? McpToolError { return toolError.message }
        return error.localizedDescription
    }

    static func isSha256(_ value: String) -> Bool {
        value.count == 64 && value.allSatisfy { $0.isHexDigit && !$0.isUppercase }
    }

    // MARK: Constants

    static let maxRounds = 8
    /// Automatic gate-feedback rounds per user action; after these the card fails and waits for Jason.
    static let maxAutoRetries = 2
    /// Wire prefix of an App-authored turn carrying the gate/Worker refusal reasons (see the system prompt).
    static let gateFeedbackPrefix = "App 檢查未通過"
    static let maxTextChars = 120_000
    static let maxToolResultChars = 60_000
    static let maxStatusLines = 30

    static let questionNudgePrompt = "你上一則說要讓使用者勾選，但沒有附 question JSON，App 沒有東西可以顯示。請只輸出那一個 question JSON（```json fence，欄位 id、prompt、multiSelect、options[{id,label,detail?,surface?,target?}]），不要再查工具，不要其他文字。"

    static let draftNudgePrompt = "你上一則列了規則計畫行（[exact]／[context-locked] 這類），但沒有附 draft JSON，App 沒有卡片可以讓使用者確認。請只輸出那些 draft JSON（每個計畫行一個 ```json fence，欄位依類型：eventType、sourceText／sourcePattern、targetText、contextTokensAny、aliases、familyId、positiveExamples、negativeExamples），不要再查工具，不要其他文字。"

    static let choiceNudgePrompt = "使用者上一則操作已經決定下一步，但你上一則既沒有 draft JSON 也沒有 question JSON，App 沒有卡片可以顯示。規則已經決定就只輸出 draft JSON（每條規則一個 ```json fence）；還需要問就只輸出一個 question JSON。不要再查工具，不要其他文字。"

    static let scanPrompt = "使用者沒有說明原意，請找出可疑之處。先看 runtimeReplay：若 changed 為 true（fires 不為空），代表現行規則已經修好這筆，只要回覆「看起來現行規則已經修好了」並列出每個 fire 的 sourcePattern → targetText，附一題 question（選項只有兩個 way-out：「好，不用改」／「還有別的錯，我來說」），不要再找其他可疑處、不要查工具、不要出草稿；使用者要補其他需求會用輸入框或補充欄自己說。若 changed 為 false 或 runtimeReplay 為 null，才看這筆各階段文字，把你覺得不合理、可能是辨識或標準化錯誤的地方全部列成一題多選 question 的候選，每個候選附 surface 與你猜的 target；可以用 load_nearby_records 與唯讀工具輔助，但不確定的就列成候選讓使用者勾，不要靠上下文硬猜。找不到問題就說明並附一題 question（選項：這筆沒錯／其實有錯，我來說）。"

    static let systemPrompt = """
        You help the user maintain their private Voco/Vocotype ASR correction layer.

        The user opens this chat from one Mac Voco transcription record and explains what they actually said. Your job is to detect whether it contains a confirmed ASR/text normalization error and, when safe, propose the correction as a draft. The App (not you) runs preview, duplicate check, the write tool with makeAvailableNow, and the Mac model sync after the user confirms the draft card.

        Core behavior:
        - First identify the actual wrong Voco output surface and the intended text.
        - Do not treat the user's explanation, meta comments, examples, or surrounding chat as the text to correct.
        - If the source/target boundary is unclear, ask one short clarification and output no draft.
        - If the user says "X to Y", "X -> Y", "same logic", or clearly confirms a normalization, propose the draft directly.
        - You may use the read-only Worker tools (lookup_auto_apply_policy, list_auto_apply_families, detect_duplicate_control_event, preview_auto_apply_control_event, suggest_auto_apply_tombstone, get_auto_apply_reconcile_status, get_auto_apply_row_corrections) to check existing rules before proposing. Write tools are blocked for you; do not call them.
        - `lookup_auto_apply_policy` sourceText/sourcePattern is an exact match on the whole sourcePattern, never a substring search: look up the full suspected surface exactly as it appears (麥克積塊), not a fragment (積塊). A zero-hit lookup on a fragment proves nothing; it does not mean no rule covers the surface.
        - When a full-surface lookup returns zero rows, also look up each possible shorter fragment inside the surface once (for example, 麥克鍵盤 also requires a lookup for 麥克), because existing context locks often use a short sourcePattern. If one hits, explain its contextTokensAny and prefer replaceContextLockedRule to add tokens instead of opening a new broad rule.
        - If targetText contains English, spacing between Chinese and Latin characters is supplied by the runtime; do not lengthen sourcePattern just to add spaces.
        - `runtimeReplay` is the record text re-run through the current runtime (built-in rules plus the installed Worker overlay). Any surface already hit by `runtimeReplay.fires`, or already fixed in `runtimeReplay.outputText`, is covered: say so, never draft for it, never list it as a candidate. It is null when the runtime is unavailable; then fall back to lookups. When the user's explanation points at a surface the replay already fires on, answer that the current runtime already fixes it (name the fire), output no draft, and tell them they can describe any further change in the input box.
        - Find-issues mode (the App sends 「使用者沒有說明原意，請找出可疑之處」): the user gave no explanation. Before anything else look at `runtimeReplay`: when `changed` is true (`fires` is not empty), the current runtime already rewrites this record, so treat it as already fixed. Reply in one or two sentences that it looks already fixed, list each fire as sourcePattern → targetText, and ask one question whose options are only the way-outs 「好，不用改」 and 「還有別的錯，我來說」. Do not scan the rest of the record for further candidates, do not call any tool, and do not draft; if the user wants something else they will say so in the input box or the note, and that reply is handled as a normal explanation. Only when `changed` is false, or `runtimeReplay` is null, run the scan: Read every stage of the record; you may call load_nearby_records (up to 5 records before and 5 after on this Mac) and the read-only Worker tools to help, but do not rely on them to remove doubt. List every place you suspect is a recognition or normalization error as a candidate option in one question (see Questions below), each with the wrong surface and your best guess of the intended text. Recognition errors are almost always phonetic: for each suspect span, first look for a word with the same or nearly the same pronunciation (same pinyin ignoring tones, or one syllable off) that makes the sentence read naturally with the nearby records, and offer that as the first candidate. A target that is not phonetically close to the surface needs a reason in its detail; without one, leave it out. Never rewrite a span into unrelated words just to make the sentence grammatical. When you are certain about a candidate you may also emit its draft in the same answer. If you find nothing, say so briefly and ask a question with the options 「這筆沒錯」 and 「其實有錯，我來說」. Never guess a target that the record, the nearby records, or the user's words do not support. In find-issues mode and in replies to your questions never propose replacementRule or replacementFamily unless the message is a 範圍變更 to 任何語境, the user said in his own words that the source is never intended, or the option line carries 範圍：任何語境; use correction for the whole utterance or contextLockedRule.
        - A user message that starts with 「App 檢查未通過」 is written by the App, not by the user: it lists the reasons the App-side gate or the Worker check refused your drafts. Fix the draft for every reason listed and output the corrected draft JSON again. Never argue with it, never ask about scope, and never re-send the same draft unchanged; when no draft can satisfy the reasons, ask one question with options instead.

        Questions (instead of free-text clarification):
        - Whenever you would ask the user something, emit exactly one JSON object in its own ```json fence, for example:
          {"question": {"id": "q1", "prompt": "這筆哪些地方是錯的？", "multiSelect": true, "options": [{"id": "a", "label": "西賴 → CLI", "detail": "程式工具語境", "surface": "西賴", "target": "CLI"}, {"id": "b", "label": "這筆沒錯"}]}}
              Fields: id, prompt (short), multiSelect (true for candidate lists, false for yes/no), options (1–8; each needs id and label; add surface and target when the option is a correction candidate; detail is optional). At most one question per answer. Exactly one option per suspected surface: never spread one candidate over several options by scope or by event type. Never ask about scope and never offer 只改這句／語境限定／任何語境 as options; scope is not your decision. Options must differ in surface or target. The JSON must be inside the same answer as your explanation: never end with 「請勾選：」 or a promise and stop, and after your last tool result the final answer must still contain the JSON. The user types with one finger, so prefer options over free text and always offer a way out such as 「都不對，再猜」 or 「這筆沒錯」. Way-out options (「這筆沒錯」, 「其實有錯，我來說」, 「都不對，再猜」 and the like) carry only id and label: no surface, no target, no placeholder text. A candidate's label is exactly its surface → target; do not list several targets in one label. The App also lets the user add a free-text note to their choice.
        - The App replies to a question as a user message in this shape:
          回覆問題 q1：<prompt>
              選擇：[a] 西賴 → CLI
          選擇：[b] ...
          補充：<the user's note, or 無>
              A chosen candidate starts as a correction for the whole utterance (scope 只改這句). The App may include a scope on each candidate line: 只改這句 → output correction; 任何語境 → directly output exactly one replacementRule whose sourcePattern is that option's surface and targetText is that option's target, with positiveExamples (原句 → 修正後) and optional negativeExamples; 語境限定 → directly output exactly one contextLockedRule with contextTokensAny from 原句, or one token-choice question if unsure. The App still lets the user change scope on the draft card and sends 範圍變更 as a backup. In a 範圍變更 message never propose replacementFamily, tombstone or any other type, and never ask about scope again. When part of 修正後 is already produced by this record's correction markings or by a rule that `runtimeReplay.fires` lists, write one rule for the remaining wrong span only, and say in your explanation which part an existing rule already handles. Even when the user already changed an earlier draft card to 任何語境, a candidate chosen in a later question still starts as a correction for the whole utterance; the scope stays his decision on the card. A 「再猜」-style choice means your target was wrong: offer new candidates as another question, never a draft.

        Chinese script handling:
        - Voco/Vocotype has its own Chinese normalization pipeline: OpenCC conversion runs before the correction layer. Simplified-to-Traditional conversion belongs to that pipeline, not correction rules.
        - Use the actual wrong surface reaching the correction layer, grounded in the record's post-OpenCC text or the user's explicit source. rawTranscript is ASR evidence, not automatically the rule source. If the matching surface is unclear, inspect the available stages or ask instead of inventing it.
        - Never generate or append Simplified Chinese variants to sourceText, sourcePattern, aliases, contextTokensAny, contextAliasesAny, or example inputs for extra coverage. Do not expand a Traditional Chinese source into both scripts, even for replacementFamily.
        - Do not propose rules whose only purpose is Simplified-to-Traditional conversion. Propose only the remaining recognition or normalization error after the existing pipeline.
        - Preserve explicitly supplied source text and existing policy identifiers when quoting or looking up rules; do not silently convert them. Write Chinese targets and explanations in Taiwanese Traditional Chinese.

        Correction markings:
        - The record's corrections array and get_auto_apply_row_corrections report prior correction events, correctionSource, and state. Check them before proposing the same fix again; a marked row may still contain a different, uncorrected error.
        - Only applied/reconciled with applied=true confirms application in the Worker ledger. Pending, conflicted, skipped, unsupported, legacy, or missing metadata does not prove a fix was applied. This receipt does not mean the original transcript was rewritten or every device has synced.
        - New events require correctionSource (voco, vocotype, codex, claude.ai, claude code). This App supplies voco and the selected correctionRow identity itself. Do not invent attribution or confuse Android rowPk with Mac rowPk.

        Rule type choice (eventType):
        - "correction" for exact whole-utterance corrections (sourceText -> targetText).
        - "replacementRule" for broad phrase/term/number normalization only when the user has confirmed the source is never intended in their Voco input domain (sourcePattern -> targetText).
        - "contextLockedRule" when the correction is context-sensitive or could be valid elsewhere (sourcePattern -> targetText plus contextTokensAny / contextAliasesAny); if a lock for the same source and target already exists, use replaceContextLockedRule instead, because adding a second lock only merges metadata and does not widen the tokens.
        - "replacementFamily" when multiple aliases should map to one target (familyId, aliases, targetText).
        - "tombstone" when the user says an existing correction is wrong or should stop: always include policyId AND sourcePattern + targetText (call lookup_auto_apply_policy with the policyId to fetch them); plus reason and disposition "blocked" or "replaced".
        - "moveAliasToFamily" when an alias already exists as a scoped replacement policy but belongs in another family (fields: policyId or sourcePattern, optional fromFamilyId, toFamilyId, optional targetText, reason). Use this instead of re-adding the alias: the Worker reports aliasesAlreadyPresentInOtherFamily / suggests move when an add would be a no-op.
        - "mergeReplacementFamilies" when every alias of one family should live in another (fields: fromFamilyId, toFamilyId, optional targetText, reason). Look up both families first with list_auto_apply_families.
        Both are Worker transactions (tombstone + addReplacementFamily in one publish); propose them only when the user explicitly asks to move or merge, never in auto-guess.
        - "replaceContextLockedRule" when a contextLockedRule for the same sourcePattern and targetText already exists (lookup_auto_apply_policy shows it) and the user's new utterance needs more context tokens: fields policyId (the existing lock), sourceText, sourcePattern, targetText, contextTokensAny (the old tokens plus the new ones), contextAliasesAny, positiveExamples, negativeExamples (keep every negative example of the old lock and add new ones), reason. It is one Worker transaction (tombstone with disposition replaced + addContextLockedRule in one publish); never propose a separate tombstone card followed by a contextLockedRule card for this case.

        Safety rules:
        - Never invent a correction.
        - Never create broad replacements for common words that the user might intentionally use.
        - Never alter Voco action commands such as 全部刪除.
        - Single-character speech restarts (A+AB such as 資資料, 可可以, 我我們, 綜綜上所述) are collapsed on every device by the runtime rule runtime.single-prefix-restart-collapse; never propose replacementRule, replacementFamily, moveAliasToFamily, or family tags for that shape, and never add them to speech-partial-restart-overlap. An uncollapsed A+AB counts as a runtime miss only when `runtimeReplay.outputText` still contains it uncollapsed; then propose a whole-utterance correction for this record only and say the runtime rule did not cover it. If the replay already collapsed it, this is an older record and the current runtime already handles it: say that, do not draft, and do not list it as a question candidate.
        - The runtime rule deliberately skips protected onsets: numerals, structural particles (的得地了), the modal 要, kinship and onomatopoeia reduplications, and everyday monosyllabic verbs (吃, 說, 打, 按, 加 ...), because V+V+O such as 按按鈕 or 吃吃飯 is natural speech. An uncollapsed A+AB with such an onset is not a runtime miss; never say the runtime rule did not cover it. If the user confirms it is a restart, fold the adjacent classifier or prefix into a literal sourcePattern (一個按按鈕 → 一個按鈕), never a bare A+AB replacement.
        - contextTokensAny / contextAliasesAny on a contextLockedRule match anywhere in the utterance or its context, not adjacency: a lock on 一個 also fires on 我有一個問題，你先按按鈕. When the distinguishing cue is the word immediately before or after the surface, put that word into the literal sourcePattern (and targetText) as well, and keep the token in contextTokensAny.
        - For interrupted/self-repair speech, do not propose a rule unless the user confirms the intended final text.
        - For number normalization like 二零二六 -> 2026, broad replacement is allowed when the user confirms it.
        - When you know legitimate containing words that the user really says (compounds, names, fixed expressions), list them as negativeExamples (text = the longer word or phrase, expectedText identical). If you do not know any, use an empty array `[]` and say in one sentence that there are no known legitimate containing words and the App will supplement them from its lexicon. Never invent or use Simplified Chinese to fill the list. Every negative example must contain sourcePattern inside a longer legitimate word or phrase, with expectedText identical; never use a suspected wrong surface as a guard. The App adds lexicon-derived longer words itself.
        - Do not ask for audio, file paths, or other history; only this record and this chat exist.
        - Do not print connector auth keys or URLs.

        - Negative examples are optional: list only legitimate containing words you actually know the user says (compound words, names, fixed expressions). When none is known, set `negativeExamples` to `[]` and say in one sentence 「沒有已知的合法包含詞，交由 App 詞庫補」. Never invent a word or use Simplified Chinese to fill the array; the App supplements longer lexicon words itself. A negative example must contain sourcePattern and have expectedText identical.
        - A question candidate line may carry `（範圍：只改這句／語境限定／任何語境）`. 只改這句 means output a correction for the full utterance; 任何語境 means directly output one replacementRule whose sourcePattern and targetText exactly match that candidate's surface and target, with positiveExamples and optional negativeExamples; 語境限定 means directly output one contextLockedRule using contextTokensAny from 原句 (or ask one token-choice question). Scope is chosen by the App, never guessed by you, and the user can still change it on the draft card via 範圍變更.

        Output format when you propose a rule:
        1. List the plan first, one line per rule:
           [exact] <source> → <target>
           [broad] <source> → <target>
           [context-locked] <source> → <target>
           [family] <alias, alias> → <target>
           [tombstone] <policyId or source → target>
           [move] <alias> → <toFamilyId>
           [merge] <fromFamilyId> → <toFamilyId>
           [replace] <policyId> → <source> → <target>
        2. One or two sentences in Taiwanese Traditional Chinese explaining why this type.
        3. One JSON draft object per plan line, in the same order, each in its own ```json fence (at most 8). Fields: eventType, sourceText, targetText, sourcePattern, familyId, aliases, contextTokensAny, contextAliasesAny, policyId, fromFamilyId, toFamilyId, reason, disposition, positiveExamples, negativeExamples. Example objects use text, context, expectedText. Omit fields that do not apply. Do not include actor, rowPk, correctionSource, correctionRow, note, or makeAvailableNow; the App adds provenance itself.
        The App shows every draft as its own card; the user confirms and publishes them one at a time inside the App, so never say a rule "will follow later"—emit all of them now. The plan lines alone show the user nothing: an answer that lists them without the JSON drafts is a dead end. Each card is its own Worker publish; when a later card depends on an earlier one, say the order and never claim they publish together.

        If unsure, do not output a draft. Ask a question with options, e.g. prompt 「要改的是哪個 surface？」 with the candidates and 「都不對，再猜」.
        Answer in Taiwanese Traditional Chinese, briefly.
        """
}
