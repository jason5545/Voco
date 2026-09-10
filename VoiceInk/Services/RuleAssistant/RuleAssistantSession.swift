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

struct RuleAssistantTurn: Equatable {
    var role: String
    var text: String
}

struct RuleAssistantPreview: Equatable {
    var wouldPublish: Bool
    var realtimeSupported: Bool
    var conflicts: Int
    var skipped: Int
    var unsupported: Int
    var reason: String?
    var baseModelSha256: String?

    var ok: Bool { wouldPublish && realtimeSupported && conflicts == 0 && unsupported == 0 }

    init(json: [String: Any]) {
        wouldPublish = json.raBool("wouldPublish")
        realtimeSupported = json.raBool("realtimeSupported")
        conflicts = json.raArray("conflicts")?.count ?? 0
        skipped = json.raArray("skipped")?.count ?? 0
        unsupported = json.raArray("unsupported")?.count ?? 0
        reason = json.raNonBlankString("reason")
        baseModelSha256 = json.raNonBlankString("baseModelSha256")
    }
}

struct RuleAssistantDuplicate: Equatable {
    var alreadyApplied: Bool
    var duplicatePolicies: Int
    var duplicateEvents: Int
    var suggestedNextAction: String?

    var found: Bool { alreadyApplied || duplicatePolicies > 0 || duplicateEvents > 0 }

    init(json: [String: Any]) {
        alreadyApplied = json.raBool("alreadyApplied")
        if let policy = json.raDict("duplicatePolicy") {
            duplicatePolicies = policy.raBool("found") ? max(1, Int(policy.raInt64("count") ?? 0)) : 0
        } else {
            duplicatePolicies = 0
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

    private var provider: RuleAssistantProvider?
    private var mcp: RuleAssistantMCP?
    private var toolJSON: [[String: Any]]?
    /// Committed wire history; a failed or cancelled round never leaves dangling tool calls here.
    private var history: [OpenCodeMessage] = []
    private var generation: Int64 = 0
    private var pendingPrompt: String?
    private var autoGuessTurn = false

    init(
        context: RuleAssistantContext,
        providerFactory: @escaping () -> RuleAssistantProvider?,
        mcpFactory: @escaping () -> RuleAssistantMCP?,
        syncNow: @escaping () async -> RuleAssistantSyncResult,
        neighborLoader: @escaping (_ before: Int, _ after: Int) -> [RuleAssistantContext] = { _, _ in [] },
        onCorrections: @escaping (RuleAssistantContext, String) -> Void = { _, _ in }
    ) {
        state = RuleAssistantUIState(context: context)
        self.providerFactory = providerFactory
        self.mcpFactory = mcpFactory
        self.syncNow = syncNow
        self.neighborLoader = neighborLoader
        self.onCorrections = onCorrections
    }

    func setConfig(goKeyConfigured: Bool, syncConfigured: Bool) {
        state.goKeyConfigured = goKeyConfigured
        state.syncConfigured = syncConfigured
    }

    func setUserText(_ value: String) {
        let current = state
        if current.phase.isBusy { return }
        if value == current.userText { return }
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
        let aborted = pendingPrompt
        if current.phase.isBusy {
            state.phase = .idle
            if let aborted { state.userText = aborted }
            state.reasoning = ""
            state.answer = ""
            state.toolStatus.append(String(localized: "Stopped."))
            if let aborted {
                dropTrailingUserTurn(aborted)
            }
        }
        pendingPrompt = nil
    }

    func closeClients() {
        provider?.close()
        provider = nil
        mcp?.close()
        mcp = nil
        toolJSON = nil
    }

    // MARK: Submit

    func submit(_ prompt: String) async {
        let clean = prompt.trimmingCharacters(in: .whitespacesAndNewlines)
        if clean.isEmpty || state.phase.isBusy { return }
        let gen = generation
        pendingPrompt = clean
        state.userText = ""
        state.phase = .loadingTools
        state.reasoning = ""
        state.answer = ""
        state.toolStatus = []
        state.drafts = []
        state.publishedSha256 = nil
        state.publishMessage = nil
        state.transcript.append(RuleAssistantTurn(role: "user", text: clean))
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
                content: buildUserPrompt(context: state.context, instruction: clean, first: history.isEmpty)
            ))
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
                    try await finishTurn(answer: outcome.visibleAnswer, transaction: transaction, worker: worker, gen: gen)
                    return
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
                            let body = try await worker.call(name, args: args)
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
            state.userText = clean
            dropTrailingUserTurn(clean)
        }
        pendingPrompt = nil
    }

    /// Auto-guess mode: no explanation from the user; the model works from the record, nearby
    /// records, and must ask when unsure.
    func submitAutoGuess() async {
        autoGuessTurn = true
        defer { autoGuessTurn = false }
        await submit(Self.autoGuessPrompt)
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
                if duplicate?.found == true {
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
        } catch is CancellationError {
            return
        } catch {
            if gen != generation { return }
            let message = Self.message(of: error)
            updateEntry(nonce: draft.nonce) { $0.outcome = message }
            state.phase = .failed(message)
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

    private func finishTurn(
        answer: String,
        transaction: [OpenCodeMessage],
        worker: RuleAssistantMCP,
        gen: Int64
    ) async throws {
        let located = RuleAssistantDraft.locateAllJSON(in: answer)
        let proposed = RuleAssistantDraft.parseAll(located)
        // Cut every draft object out of the visible reply (only when at least one parsed as a draft).
        var display: String
        if !proposed.isEmpty {
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
        // Commit the wire history only for a normally completed turn.
        history = transaction
        let fallback = proposed.isEmpty
            ? String(localized: "(No text reply)")
            : String(localized: "(Rule draft proposed)")
        state.transcript.append(RuleAssistantTurn(role: "assistant", text: display.isEmpty ? fallback : display))
        // The live-answer bubble in the panel is for streaming only; once the turn is
        // committed to the transcript, keeping state.answer would render the reply twice.
        state.answer = ""
        if proposed.isEmpty {
            state.phase = .idle
            return
        }
        var rejected: [String] = []
        let accepted = proposed.filter { draft in
            if !draft.isSafeForWrite() {
                rejected.append(String(localized: "The AI proposed a draft with incomplete or unsafe fields (\(draft.eventType)); it was not listed for confirmation. Add more detail and send again."))
                return false
            }
            // Broad rules need Jason's explicit statement that the source is never intended; a guess is not that.
            if autoGuessTurn && (draft.isBroad || draft.isTransaction) {
                let source = draft.sourcePattern ?? draft.aliases.joined(separator: "\u{3001}")
                rejected.append(String(localized: "Auto-guess never creates broad rules or moves/merges families (\(draft.eventType): \(source) → \(draft.targetText ?? "")). For a batch replacement, state yourself that the source is never valid in any context and send again, or use a whole-utterance correction or a context-locked rule."))
                return false
            }
            return true
        }
        for reason in rejected {
            status(String(localized: "Skipped a draft: \(String(reason.prefix(120)))"))
        }
        guard !accepted.isEmpty else {
            state.phase = .failed(rejected.first ?? String(localized: "No confirmable draft."))
            return
        }
        state.drafts = accepted.map { RuleAssistantDraftEntry(draft: $0) }
        state.phase = .checking
        for draft in accepted {
            let check = try await checkDraft(worker: worker, draft: draft)
            try guardGeneration(gen)
            updateEntry(nonce: draft.nonce) { $0.check = check }
        }
        state.phase = .draftReady
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
        if !preview.realtimeSupported {
            blocked = String(localized: "The Worker cannot publish this event type in realtime (unsupported=\(preview.unsupported), skipped=\(preview.skipped)): \(preview.reason ?? "")")
        } else if preview.conflicts > 0 {
            blocked = String(localized: "Conflicts with existing rules (\(preview.conflicts)): \(preview.reason ?? String(localized: "Retire the conflicting rule or narrow the scope first"))")
        } else if !preview.wouldPublish {
            blocked = String(localized: "The Worker preview says this would not change the model: \(preview.reason ?? "")")
        } else if let duplicate {
            if duplicate.alreadyApplied {
                blocked = String(localized: "This rule is already applied in the current Worker model; no need to write it again.")
            } else if duplicate.duplicateEvents > 0 {
                blocked = String(localized: "The Worker already has an identical event (\(duplicate.duplicateEvents)); not writing again.")
            } else if duplicate.duplicatePolicies > 0 {
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
            blockedReason: blocked
        )
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
        return "此筆修正標記（Worker 狀態）：\(markings)\n使用者補充：\(instruction)"
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

    static let maxRounds = 6
    static let maxTextChars = 120_000
    static let maxToolResultChars = 60_000
    static let maxStatusLines = 30

    static let autoGuessPrompt = "使用者沒有說明原意，請自動判斷。先看這筆各階段文字；單筆看不出來就用 load_nearby_records 讀前後紀錄；仍不確定就直接問一句，不要輸出草稿。"

    static let systemPrompt = """
        You help Jason maintain his private Voco/Vocotype ASR correction layer.

        Jason opens this chat from one Mac Voco transcription record and explains what he actually said. Your job is to detect whether it contains a confirmed ASR/text normalization error and, when safe, propose the correction as a draft. The App (not you) runs preview, duplicate check, the write tool with makeAvailableNow, and the Mac model sync after Jason confirms the draft card.

        Core behavior:
        - First identify the actual wrong Voco output surface and the intended text.
        - Do not treat Jason's explanation, meta comments, examples, or surrounding chat as the text to correct.
        - If the source/target boundary is unclear, ask one short clarification and output no draft.
        - If Jason says "X to Y", "X -> Y", "same logic", or clearly confirms a normalization, propose the draft directly.
        - You may use the read-only Worker tools (lookup_auto_apply_policy, list_auto_apply_families, detect_duplicate_control_event, preview_auto_apply_control_event, suggest_auto_apply_tombstone, get_auto_apply_reconcile_status, get_auto_apply_row_corrections) to check existing rules before proposing. Write tools are blocked for you; do not call them.
        - Auto-guess: when Jason gives no explanation (or the single record is not enough to tell the wrong surface from the intended text), call load_nearby_records to read up to 5 records before and 5 after on this Mac and use them as context (the same term often appears correctly nearby). If it is still unclear after that, ask one short question and output no draft. Never guess a target that the record, the nearby records, or Jason's words do not support. In auto-guess (no explanation from Jason) never propose replacementRule or replacementFamily: use correction for the whole utterance, or contextLockedRule when the term recurs; you may mention that a broad rule is possible if Jason confirms it explicitly.

        Chinese script handling:
        - Voco/Vocotype has its own Chinese normalization pipeline: OpenCC conversion runs before the correction layer. Simplified-to-Traditional conversion belongs to that pipeline, not correction rules.
        - Use the actual wrong surface reaching the correction layer, grounded in the record's post-OpenCC text or Jason's explicit source. rawTranscript is ASR evidence, not automatically the rule source. If the matching surface is unclear, inspect the available stages or ask instead of inventing it.
        - Never generate or append Simplified Chinese variants to sourceText, sourcePattern, aliases, contextTokensAny, contextAliasesAny, or example inputs for extra coverage. Do not expand a Traditional Chinese source into both scripts, even for replacementFamily.
        - Do not propose rules whose only purpose is Simplified-to-Traditional conversion. Propose only the remaining recognition or normalization error after the existing pipeline.
        - Preserve explicitly supplied source text and existing policy identifiers when quoting or looking up rules; do not silently convert them. Write Chinese targets and explanations in Taiwanese Traditional Chinese.

        Correction markings:
        - The record's corrections array and get_auto_apply_row_corrections report prior correction events, correctionSource, and state. Check them before proposing the same fix again; a marked row may still contain a different, uncorrected error.
        - Only applied/reconciled with applied=true confirms application in the Worker ledger. Pending, conflicted, skipped, unsupported, legacy, or missing metadata does not prove a fix was applied. This receipt does not mean the original transcript was rewritten or every device has synced.
        - New events require correctionSource (voco, vocotype, codex, claude.ai, claude code). This App supplies voco and the selected correctionRow identity itself. Do not invent attribution or confuse Android rowPk with Mac rowPk.

        Rule type choice (eventType):
        - "correction" for exact whole-utterance corrections (sourceText -> targetText).
        - "replacementRule" for broad phrase/term/number normalization only when Jason has confirmed the source is never intended in his Voco input domain (sourcePattern -> targetText).
        - "contextLockedRule" when the correction is context-sensitive or could be valid elsewhere (sourcePattern -> targetText plus contextTokensAny / contextAliasesAny).
        - "replacementFamily" when multiple aliases should map to one target (familyId, aliases, targetText).
        - "tombstone" when Jason says an existing correction is wrong or should stop (policyId, or sourcePattern + targetText; plus reason and disposition "blocked" or "replaced").
        - "moveAliasToFamily" when an alias already exists as a scoped replacement policy but belongs in another family (fields: policyId or sourcePattern, optional fromFamilyId, toFamilyId, optional targetText, reason). Use this instead of re-adding the alias: the Worker reports aliasesAlreadyPresentInOtherFamily / suggests move when an add would be a no-op.
        - "mergeReplacementFamilies" when every alias of one family should live in another (fields: fromFamilyId, toFamilyId, optional targetText, reason). Look up both families first with list_auto_apply_families.
        Both are Worker transactions (tombstone + addReplacementFamily in one publish); propose them only when Jason explicitly asks to move or merge, never in auto-guess.

        Safety rules:
        - Never invent a correction.
        - Never create broad replacements for common words that Jason might intentionally use.
        - Never alter Voco action commands such as 全部刪除.
        - Single-character speech restarts (A+AB such as 資資料, 可可以, 我我們, 綜綜上所述) are collapsed on every device by the runtime rule runtime.single-prefix-restart-collapse; never propose replacementRule, replacementFamily, moveAliasToFamily, or family tags for that shape, and never add them to speech-partial-restart-overlap. If the runtime rule missed one, propose a whole-utterance correction for this record only and say the runtime rule did not cover it.
        - For interrupted/self-repair speech, do not propose a rule unless Jason confirms the intended final text.
        - For number normalization like 二零二六 -> 2026, broad replacement is allowed when Jason confirms it.
        - Do not ask for audio, file paths, or other history; only this record and this chat exist.
        - Do not print connector auth keys or URLs.

        Output format when you propose a rule:
        1. List the plan first, one line per rule:
           [exact] <source> → <target>
           [broad] <source> → <target>
           [context-locked] <source> → <target>
           [family] <alias, alias> → <target>
           [tombstone] <policyId or source → target>
           [move] <alias> → <toFamilyId>
           [merge] <fromFamilyId> → <toFamilyId>
        2. One or two sentences in Taiwanese Traditional Chinese explaining why this type.
        3. One JSON draft object per plan line, in the same order, each in its own ```json fence (at most 8). Fields: eventType, sourceText, targetText, sourcePattern, familyId, aliases, contextTokensAny, contextAliasesAny, policyId, fromFamilyId, toFamilyId, reason, disposition, positiveExamples, negativeExamples. Example objects use text, context, expectedText. Omit fields that do not apply. Do not include actor, rowPk, correctionSource, correctionRow, note, or makeAvailableNow; the App adds provenance itself.
        The App shows every draft as its own card; Jason confirms and publishes them one at a time inside the App, so never say a rule "will follow later"—emit all of them now.

        If unsure, do not output a draft. Ask: 「我不確定要改的是哪個 surface：A -> B 對嗎？」
        Answer in Taiwanese Traditional Chinese, briefly.
        """
}
