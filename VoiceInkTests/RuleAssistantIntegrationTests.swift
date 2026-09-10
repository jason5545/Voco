import Foundation
import Testing
@testable import Voco

/// Runs the real OpenCodeGoClient + WorkerMCPClient + RuleAssistantSession against in-memory
/// fake servers (URLProtocol), matching the Android harness tests.
@MainActor
@Suite(.serialized)
struct RuleAssistantIntegrationTests {
    private let server = FakeMCPServer.shared
    private let sha = String(repeating: "a", count: 64)

    // MARK: Helpers

    private func correctionDraft(_ source: String, _ target: String) -> String {
        RuleAssistantTestJSON.string([
            "eventType": "correction",
            "sourceText": source,
            "targetText": target,
            "reason": "測試",
        ])
    }

    private func draftAnswer(_ drafts: [String]) -> String {
        "我建議：\n" + drafts.joined(separator: "\n")
    }

    private func providerMessages(_ index: Int) -> [[String: Any]] {
        let recorded = FakeGoProvider.recorded
        guard recorded.indices.contains(index) else { return [] }
        return recorded[index].body["messages"] as? [[String: Any]] ?? []
    }

    private func waitForRequests(_ count: Int, timeout: TimeInterval = 10) async -> Bool {
        let deadline = Date().addingTimeInterval(timeout)
        while Date() < deadline {
            if FakeGoProvider.recorded.count >= count { return true }
            try? await Task.sleep(for: .milliseconds(20))
        }
        return FakeGoProvider.recorded.count >= count
    }

    // MARK: Full flow

    @Test func fullFlowPublishesAndSyncs() async throws {
        server.reset()
        server.toolHandler = { name, args in
            if name == "add_auto_apply_correction" {
                return .result(FakeMCPServer.writePublished(sha: String(repeating: "a", count: 64)))
            }
            return FakeMCPServer.defaultToolHandler(name, args)
        }
        FakeGoProvider.reset(scripts: [
            .stream([
                FakeGoProvider.chunk(reasoning: "先想一下"),
                FakeGoProvider.chunk(content: draftAnswer([correctionDraft("小振", "小鎮")])),
                FakeGoProvider.chunk(finish: "stop"),
            ]),
        ])
        var writtenCorrections: [String] = []
        let session = makeRuleAssistantSession(server: server) { _, json in
            writtenCorrections.append(json)
        }
        await session.submit("把小振改成小鎮")
        guard case .draftReady = session.state.phase else {
            Issue.record("expected draftReady, got \(session.state.phase)")
            return
        }
        #expect(session.state.drafts.count == 1)
        #expect(session.state.drafts[0].check?.ok == true)
        #expect(session.state.reasoning.contains("先想一下"))
        // Provider request headers and body shape.
        let request = try #require(FakeGoProvider.recorded.first)
        #expect(request.authorization == "Bearer test-go-key")
        #expect(request.userAgent?.hasPrefix("voco-rule-assistant/") == true)
        #expect(request.sessionId == "fixed-session-id")
        #expect(request.body["model"] as? String == "glm-5.3-flash")
        #expect(request.body["stream"] as? Bool == true)
        #expect(request.body["tool_choice"] as? String == "auto")
        let tools = request.body["tools"] as? [[String: Any]] ?? []
        #expect(tools.contains { ($0["function"] as? [String: Any])?["name"] as? String == "load_nearby_records" })
        // Corrections refresh happened before the round trip.
        #expect(server.toolCalls.contains { $0.name == "get_auto_apply_row_corrections" })
        #expect(writtenCorrections.isEmpty == false)
        await session.confirm()
        guard case .published = session.state.phase else {
            Issue.record("expected published, got \(session.state.phase)")
            return
        }
        #expect(session.state.publishedSha256 == sha)
        #expect(session.state.drafts[0].consumed)
        #expect(session.state.drafts[0].publishedSha256 == sha)
        let write = server.toolCalls.first { $0.name == "add_auto_apply_correction" }
        let args = try #require(write?.args)
        #expect(args["correctionSource"] as? String == "voco")
        #expect(args["actor"] as? String == "voco-rule-assistant")
        #expect(args["makeAvailableNow"] as? Bool == true)
        #expect((args["note"] as? String)?.contains("source=voco:row:42") == true)
        #expect(args["rowPk"] == nil)
        let row = args["correctionRow"] as? [String: Any]
        #expect(row?["platform"] as? String == "voco")
        #expect((row?["rowPk"] as? Int64) == 42)
    }

    // MARK: Plain clarification

    @Test func plainTextAnswerNeedsNoConfirmation() async {
        server.reset()
        FakeGoProvider.reset(scripts: [
            .stream([
                FakeGoProvider.chunk(content: "你說的「小振」是人名嗎？"),
                FakeGoProvider.chunk(finish: "stop"),
            ]),
        ])
        let session = makeRuleAssistantSession(server: server)
        await session.submit("把小振改成小鎮")
        #expect(session.state.phase == .idle)
        #expect(session.state.drafts.isEmpty)
        #expect(session.state.transcript.last?.text.contains("小振") == true)
    }

    // MARK: Write tools are never executed for the model

    @Test func modelWriteToolCallIsBlocked() async {
        server.reset()
        FakeGoProvider.reset(scripts: [
            .stream([
                FakeGoProvider.chunk(toolCalls: [
                    FakeGoProvider.toolCallFragment(
                        index: 0,
                        id: "call_write",
                        name: "add_auto_apply_correction",
                        arguments: "{\"sourceText\":\"小振\",\"targetText\":\"小鎮\"}"
                    ),
                ]),
                FakeGoProvider.chunk(finish: "tool_calls"),
            ]),
            .stream([
                FakeGoProvider.chunk(content: draftAnswer([correctionDraft("小振", "小鎮")])),
                FakeGoProvider.chunk(finish: "stop"),
            ]),
        ])
        let session = makeRuleAssistantSession(server: server)
        await session.submit("把小振改成小鎮")
        // The Worker never saw the write tool.
        #expect(!server.toolCalls.contains { $0.name == "add_auto_apply_correction" })
        // The model was told the call is blocked.
        let messages = providerMessages(1)
        let toolMessage = messages.first { $0["role"] as? String == "tool" }
        let content = toolMessage?["content"] as? String ?? ""
        #expect(content.contains("blocked"))
        #expect(session.state.phase == .draftReady)
    }

    // MARK: Wire history keeps raw reasoning and tool_call_id

    @Test func readToolRoundTripKeepsRawWire() async {
        server.reset()
        FakeGoProvider.reset(scripts: [
            .stream([
                FakeGoProvider.chunk(reasoning: "原始思考"),
                FakeGoProvider.chunk(toolCalls: [
                    FakeGoProvider.toolCallFragment(index: 0, id: "call_1", name: "lookup_auto_apply_policy", arguments: "{\"sourceText\":\"小振\"}"),
                ]),
                FakeGoProvider.chunk(finish: "tool_calls"),
            ]),
            .stream([
                FakeGoProvider.chunk(content: "查到了"),
                FakeGoProvider.chunk(finish: "stop"),
            ]),
        ])
        let session = makeRuleAssistantSession(server: server)
        await session.submit("查一下小振")
        #expect(session.state.phase == .idle)
        let messages = providerMessages(1)
        let assistant = messages.first { $0["role"] as? String == "assistant" && $0["tool_calls"] != nil }
        #expect(assistant?["reasoning_content"] as? String == "原始思考")
        let calls = assistant?["tool_calls"] as? [[String: Any]]
        #expect(calls?.first?["id"] as? String == "call_1")
        let tool = messages.first { $0["role"] as? String == "tool" }
        #expect(tool?["tool_call_id"] as? String == "call_1")
        #expect(server.toolCalls.contains { $0.name == "lookup_auto_apply_policy" })
    }

    // MARK: Draft gates

    @Test func previewConflictBlocksConfirm() async {
        server.reset()
        server.toolHandler = { name, args in
            if name == "preview_auto_apply_control_event" {
                var preview = FakeMCPServer.previewOK
                preview["conflicts"] = [["policyId": "old1"]]
                preview["reason"] = "與現有規則衝突"
                return .result(preview)
            }
            return FakeMCPServer.defaultToolHandler(name, args)
        }
        FakeGoProvider.reset(scripts: [
            .stream([
                FakeGoProvider.chunk(content: draftAnswer([correctionDraft("小振", "小鎮")])),
                FakeGoProvider.chunk(finish: "stop"),
            ]),
        ])
        let session = makeRuleAssistantSession(server: server)
        await session.submit("把小振改成小鎮")
        guard case .draftReady = session.state.phase else {
            Issue.record("expected draftReady, got \(session.state.phase)")
            return
        }
        let entry = session.state.drafts[0]
        #expect(entry.check?.blockedReason != nil)
        #expect(!session.state.canConfirm(entry))
        await session.confirm()
        // Confirm is gated: nothing was written, phase unchanged.
        #expect(!server.toolCalls.contains { $0.name == "add_auto_apply_correction" })
        #expect(session.state.phase == .draftReady)
    }

    @Test func duplicateBlocksConfirm() async {
        server.reset()
        server.toolHandler = { name, args in
            if name == "detect_duplicate_control_event" {
                var duplicate = FakeMCPServer.duplicateNone
                duplicate["alreadyApplied"] = true
                return .result(duplicate)
            }
            return FakeMCPServer.defaultToolHandler(name, args)
        }
        FakeGoProvider.reset(scripts: [
            .stream([
                FakeGoProvider.chunk(content: draftAnswer([correctionDraft("小振", "小鎮")])),
                FakeGoProvider.chunk(finish: "stop"),
            ]),
        ])
        let session = makeRuleAssistantSession(server: server)
        await session.submit("把小振改成小鎮")
        #expect(session.state.drafts[0].check?.blockedReason != nil)
        #expect(!session.state.canConfirm)
    }

    // MARK: Worker errors go back to the model

    @Test func toolIsErrorGoesBackToModel() async {
        server.reset()
        server.toolHandler = { name, args in
            if name == "lookup_auto_apply_policy" {
                return .isError(["reason": "查詢壞掉"])
            }
            return FakeMCPServer.defaultToolHandler(name, args)
        }
        FakeGoProvider.reset(scripts: [
            .stream([
                FakeGoProvider.chunk(toolCalls: [
                    FakeGoProvider.toolCallFragment(index: 0, id: "c1", name: "lookup_auto_apply_policy", arguments: "{}"),
                ]),
                FakeGoProvider.chunk(finish: "tool_calls"),
            ]),
            .stream([
                FakeGoProvider.chunk(content: "Worker 說查詢壞掉，換個方式"),
                FakeGoProvider.chunk(finish: "stop"),
            ]),
        ])
        let session = makeRuleAssistantSession(server: server)
        await session.submit("查一下小振")
        #expect(session.state.phase == .idle)
        let tool = providerMessages(1).first { $0["role"] as? String == "tool" }
        #expect((tool?["content"] as? String)?.contains("查詢壞掉") == true)
    }

    @Test func rpcErrorIsConvertedToToolError() async {
        server.reset()
        server.toolHandler = { name, args in
            if name == "lookup_auto_apply_policy" {
                return .rpcError(-32602, "invalid params")
            }
            return FakeMCPServer.defaultToolHandler(name, args)
        }
        FakeGoProvider.reset(scripts: [
            .stream([
                FakeGoProvider.chunk(toolCalls: [
                    FakeGoProvider.toolCallFragment(index: 0, id: "c1", name: "lookup_auto_apply_policy", arguments: "{}"),
                ]),
                FakeGoProvider.chunk(finish: "tool_calls"),
            ]),
            .stream([
                FakeGoProvider.chunk(content: "了解，參數錯誤"),
                FakeGoProvider.chunk(finish: "stop"),
            ]),
        ])
        let session = makeRuleAssistantSession(server: server)
        await session.submit("查一下小振")
        #expect(session.state.phase == .idle)
        let tool = providerMessages(1).first { $0["role"] as? String == "tool" }
        #expect((tool?["content"] as? String)?.contains("-32602") == true)
    }

    // MARK: Lost write responses

    @Test func lostWriteWithDuplicateFoundIsConsumedNotResent() async {
        server.reset()
        server.toolHandler = { name, args in
            switch name {
            case "add_auto_apply_correction":
                return .httpError(500)
            case "detect_duplicate_control_event":
                // The duplicate only exists once the write actually reached the Worker.
                let attempted = FakeMCPServer.shared.toolCalls.contains { $0.name == "add_auto_apply_correction" }
                var duplicate = FakeMCPServer.duplicateNone
                duplicate["duplicateEvent"] = ["found": attempted, "count": attempted ? 1 : 0]
                return .result(duplicate)
            default:
                return FakeMCPServer.defaultToolHandler(name, args)
            }
        }
        FakeGoProvider.reset(scripts: [
            .stream([
                FakeGoProvider.chunk(content: draftAnswer([correctionDraft("小振", "小鎮")])),
                FakeGoProvider.chunk(finish: "stop"),
            ]),
        ])
        let session = makeRuleAssistantSession(server: server)
        await session.submit("把小振改成小鎮")
        await session.confirm()
        guard case .failed(let message) = session.state.phase else {
            Issue.record("expected failed, got \(session.state.phase)")
            return
        }
        #expect(!message.isEmpty)
        #expect(session.state.drafts[0].consumed)
        // Never resent: exactly one write attempt.
        #expect(server.toolCalls.filter { $0.name == "add_auto_apply_correction" }.count == 1)
    }

    @Test func lostWriteWithoutDuplicateIsNotConsumed() async {
        server.reset()
        server.toolHandler = { name, args in
            if name == "add_auto_apply_correction" {
                return .httpError(500)
            }
            return FakeMCPServer.defaultToolHandler(name, args)
        }
        FakeGoProvider.reset(scripts: [
            .stream([
                FakeGoProvider.chunk(content: draftAnswer([correctionDraft("小振", "小鎮")])),
                FakeGoProvider.chunk(finish: "stop"),
            ]),
        ])
        let session = makeRuleAssistantSession(server: server)
        await session.submit("把小振改成小鎮")
        await session.confirm()
        guard case .failed(let message) = session.state.phase else {
            Issue.record("expected failed, got \(session.state.phase)")
            return
        }
        #expect(message.contains("MCP HTTP 500"))
        #expect(!session.state.drafts[0].consumed)
        // The draft stays confirmable (the full check re-runs before any retry).
        #expect(session.state.canConfirm)
    }

    // MARK: Noop and publish-failed outcomes

    @Test func noopWriteIsConsumedWithoutSync() async {
        server.reset()
        server.toolHandler = { name, args in
            if name == "add_auto_apply_correction" {
                return .result(["noop": true, "suggestedNextAction": "規則已存在"])
            }
            return FakeMCPServer.defaultToolHandler(name, args)
        }
        FakeGoProvider.reset(scripts: [
            .stream([
                FakeGoProvider.chunk(content: draftAnswer([correctionDraft("小振", "小鎮")])),
                FakeGoProvider.chunk(finish: "stop"),
            ]),
        ])
        let session = makeRuleAssistantSession(server: server)
        await session.submit("把小振改成小鎮")
        await session.confirm()
        guard case .failed(let message) = session.state.phase else {
            Issue.record("expected failed, got \(session.state.phase)")
            return
        }
        #expect(message.contains("規則已存在"))
        #expect(session.state.drafts[0].consumed)
        #expect(session.state.publishedSha256 == nil)
    }

    @Test func publishFailedEventSavedIsConsumedNotResent() async {
        server.reset()
        server.toolHandler = { name, args in
            if name == "add_auto_apply_correction" {
                return .isError([
                    "publishStatus": "publish_failed",
                    "eventSaved": true,
                    "eventId": "evt123",
                    "reason": "no publish channel",
                ])
            }
            return FakeMCPServer.defaultToolHandler(name, args)
        }
        FakeGoProvider.reset(scripts: [
            .stream([
                FakeGoProvider.chunk(content: draftAnswer([correctionDraft("小振", "小鎮")])),
                FakeGoProvider.chunk(finish: "stop"),
            ]),
        ])
        let session = makeRuleAssistantSession(server: server)
        await session.submit("把小振改成小鎮")
        await session.confirm()
        guard case .failed(let message) = session.state.phase else {
            Issue.record("expected failed, got \(session.state.phase)")
            return
        }
        #expect(message.contains("evt123"))
        #expect(session.state.drafts[0].consumed)
        #expect(server.toolCalls.filter { $0.name == "add_auto_apply_correction" }.count == 1)
    }

    // MARK: Sync verification and resync

    @Test func syncShaMismatchFailsThenResyncSucceeds() async {
        server.reset()
        server.toolHandler = { name, args in
            if name == "add_auto_apply_correction" {
                return .result(FakeMCPServer.writePublished(sha: String(repeating: "a", count: 64)))
            }
            return FakeMCPServer.defaultToolHandler(name, args)
        }
        FakeGoProvider.reset(scripts: [
            .stream([
                FakeGoProvider.chunk(content: draftAnswer([correctionDraft("小振", "小鎮")])),
                FakeGoProvider.chunk(finish: "stop"),
            ]),
        ])
        var syncResult = RuleAssistantSyncResult(
            outcome: .installed,
            message: "安裝了別的",
            remoteSha256: String(repeating: "a", count: 64),
            installedSha256: String(repeating: "c", count: 64)
        )
        let session = RuleAssistantSession(
            context: RuleAssistantContext(rowPk: 42, timestampMs: 1_700_000_000_000, text: "我們去小振家"),
            providerFactory: { FakeGoProvider.makeClient() },
            mcpFactory: { server.makeClient() },
            syncNow: { syncResult }
        )
        await session.submit("把小振改成小鎮")
        await session.confirm()
        guard case .failed(let message) = session.state.phase else {
            Issue.record("expected failed, got \(session.state.phase)")
            return
        }
        #expect(message.contains("cccccccccccc"))
        #expect(session.state.publishedSha256 == sha)
        syncResult = RuleAssistantSyncResult(
            outcome: .upToDate,
            message: "同步完成",
            remoteSha256: sha,
            installedSha256: sha
        )
        await session.resync()
        #expect(session.state.phase == .published)
    }

    // MARK: Broken rounds never execute

    @Test func truncatedToolRoundIsNotExecuted() async {
        server.reset()
        FakeGoProvider.reset(scripts: [
            .stream([
                FakeGoProvider.chunk(toolCalls: [
                    FakeGoProvider.toolCallFragment(index: 0, id: "c1", name: "lookup_auto_apply_policy", arguments: "{}"),
                ]),
                FakeGoProvider.chunk(finish: "length"),
            ]),
        ])
        let session = makeRuleAssistantSession(server: server)
        await session.submit("查一下小振")
        guard case .failed(let message) = session.state.phase else {
            Issue.record("expected failed, got \(session.state.phase)")
            return
        }
        #expect(message.contains("length"))
        #expect(!server.toolCalls.contains { $0.name == "lookup_auto_apply_policy" })
        // The failed turn is rolled back: the user text is restored and no dangling turn stays.
        #expect(session.state.userText == "查一下小振")
        #expect(session.state.transcript.isEmpty)
    }

    @Test func partialToolCallsRefuseTheWholeRound() async {
        server.reset()
        FakeGoProvider.reset(scripts: [
            .stream([
                FakeGoProvider.chunk(toolCalls: [
                    FakeGoProvider.toolCallFragment(index: 0, id: "c1", name: "lookup_auto_apply_policy", arguments: "{}"),
                    FakeGoProvider.toolCallFragment(index: 1, name: "lookup_auto_apply_policy", arguments: "{}"),
                ]),
                FakeGoProvider.chunk(finish: "tool_calls"),
            ]),
        ])
        let session = makeRuleAssistantSession(server: server)
        await session.submit("查一下小振")
        guard case .failed(let message) = session.state.phase else {
            Issue.record("expected failed, got \(session.state.phase)")
            return
        }
        #expect(message.contains("incomplete"))
        #expect(!server.toolCalls.contains { $0.name == "lookup_auto_apply_policy" })
    }

    // MARK: Draft lifecycle

    @Test func editingInputInvalidatesPendingDrafts() async {
        server.reset()
        FakeGoProvider.reset(scripts: [
            .stream([
                FakeGoProvider.chunk(content: draftAnswer([correctionDraft("小振", "小鎮")])),
                FakeGoProvider.chunk(finish: "stop"),
            ]),
        ])
        let session = makeRuleAssistantSession(server: server)
        await session.submit("把小振改成小鎮")
        #expect(session.state.phase == .draftReady)
        session.setUserText("改成別的")
        #expect(session.state.drafts.isEmpty)
        #expect(session.state.phase == .idle)
    }

    @Test func multipleDraftsPublishOneByOne() async {
        server.reset()
        var writes = 0
        server.toolHandler = { name, args in
            if name == "add_auto_apply_correction" {
                writes += 1
                return .result(FakeMCPServer.writePublished(sha: String(repeating: "a", count: 64)))
            }
            return FakeMCPServer.defaultToolHandler(name, args)
        }
        FakeGoProvider.reset(scripts: [
            .stream([
                FakeGoProvider.chunk(content: draftAnswer([
                    correctionDraft("小振", "小鎮"),
                    correctionDraft("失重", "釋出"),
                ])),
                FakeGoProvider.chunk(finish: "stop"),
            ]),
        ])
        let session = makeRuleAssistantSession(server: server)
        await session.submit("兩個都要改")
        guard case .draftReady = session.state.phase else {
            Issue.record("expected draftReady, got \(session.state.phase)")
            return
        }
        #expect(session.state.drafts.count == 2)
        #expect(session.state.drafts.allSatisfy { $0.check?.ok == true })
        await session.confirm()
        #expect(writes == 1)
        #expect(session.state.drafts[0].consumed)
        #expect(!session.state.drafts[1].consumed)
        await session.confirm()
        #expect(writes == 2)
        #expect(session.state.drafts[1].consumed)
        #expect(session.state.phase == .published)
    }

    @Test func autoGuessRejectsBroadDrafts() async {
        server.reset()
        let broad = RuleAssistantTestJSON.string([
            "eventType": "replacementRule",
            "sourcePattern": "小振",
            "targetText": "小鎮",
        ])
        FakeGoProvider.reset(scripts: [
            .stream([
                FakeGoProvider.chunk(content: draftAnswer([broad])),
                FakeGoProvider.chunk(finish: "stop"),
            ]),
        ])
        let session = makeRuleAssistantSession(server: server)
        await session.submitScan()
        guard case .failed = session.state.phase else {
            Issue.record("expected failed, got \(session.state.phase)")
            return
        }
        #expect(session.state.drafts.isEmpty)
        #expect(session.state.hasAutoScanned)
        // The scan prompt never lands in the composer; the transcript shows the placeholder, not the wire text.
        #expect(session.state.userText.isEmpty)
        #expect(session.state.transcript.first?.text.contains("請找出可疑之處") == false)
    }

    // MARK: Questions and choices

    private func questionJSON(id: String = "q1", multiSelect: Bool = true, options: [[String: Any]]) -> String {
        RuleAssistantTestJSON.string([
            "question": [
                "id": id,
                "prompt": "這筆哪些地方是錯的？",
                "multiSelect": multiSelect,
                "options": options,
            ] as [String: Any],
        ])
    }

    private var candidateOptions: [[String: Any]] {
        [
            ["id": "a", "label": "小振 → 小鎮", "detail": "地名", "surface": "小振", "target": "小鎮"],
            ["id": "b", "label": "去 → 趣", "surface": "去", "target": "趣"],
            ["id": "c", "label": "這筆沒錯"],
        ]
    }

    private func lastUserMessage(_ index: Int) -> String {
        providerMessages(index).last { $0["role"] as? String == "user" }?["content"] as? String ?? ""
    }

    @Test func scanQuestionIsParsedAndWaitsForChoice() async {
        server.reset()
        FakeGoProvider.reset(scripts: [
            .stream([
                FakeGoProvider.chunk(content: "我找到兩個可疑處。\n```json\n" + questionJSON(options: candidateOptions) + "\n```"),
                FakeGoProvider.chunk(finish: "stop"),
            ]),
        ])
        let session = makeRuleAssistantSession(server: server)
        await session.submitScan()
        #expect(session.state.phase == .idle)
        #expect(session.state.drafts.isEmpty)
        let question = session.state.pendingQuestion
        #expect(question?.id == "q1")
        #expect(question?.multiSelect == true)
        #expect(question?.options.map(\.id) == ["a", "b", "c"])
        #expect(question?.options[0].isCandidate == true)
        #expect(question?.options[2].isCandidate == false)
        // The JSON is cut out of the visible reply; the card hangs off the assistant turn.
        let assistant = session.state.transcript.last
        #expect(assistant?.role == "assistant")
        #expect(assistant?.text == "我找到兩個可疑處。")
        #expect(assistant?.question?.id == "q1")
        #expect(session.state.transcript.first?.text.contains("自動") == true || session.state.transcript.first?.text.contains("automatically") == true)
        #expect(!server.toolCalls.contains { $0.name == "preview_auto_apply_control_event" })
        // Only the scan prompt went to the model.
        #expect(lastUserMessage(0).contains("請找出可疑之處"))
    }

    @Test func choiceReplyCarriesSelectionsAndBlocksUnscopedBroadDraft() async {
        server.reset()
        let broad = RuleAssistantTestJSON.string([
            "eventType": "replacementRule",
            "sourcePattern": "小振",
            "targetText": "小鎮",
        ])
        FakeGoProvider.reset(scripts: [
            .stream([
                FakeGoProvider.chunk(content: questionJSON(options: candidateOptions)),
                FakeGoProvider.chunk(finish: "stop"),
            ]),
            .stream([
                FakeGoProvider.chunk(content: draftAnswer([correctionDraft("我們去小振家", "我們去小鎮家"), broad])),
                FakeGoProvider.chunk(finish: "stop"),
            ]),
        ])
        let session = makeRuleAssistantSession(server: server)
        await session.submitScan()
        #expect(session.state.pendingQuestion != nil)
        // Single-select semantics are per question; this one is multi-select.
        session.toggleOption("a")
        session.toggleOption("c")
        session.toggleOption("c")
        session.toggleOption("b")
        #expect(session.state.selectedOptionIds == ["a", "b"])
        session.setScope(.context, for: "b")
        await session.submitChoice()
        guard case .draftReady = session.state.phase else {
            Issue.record("expected draftReady, got \(session.state.phase)")
            return
        }
        // The exact correction is offered; the broad one was not scoped as any-context.
        #expect(session.state.drafts.count == 1)
        #expect(session.state.drafts[0].draft.eventType == "correction")
        #expect(session.state.toolStatus.contains { $0.contains("Skipped") || $0.contains("略過") })
        // Wire reply shape.
        let reply = lastUserMessage(1)
        #expect(reply.contains("回覆問題 q1：這筆哪些地方是錯的？"))
        #expect(reply.contains("選擇：[a] 小振 → 小鎮（範圍：只改這句）"))
        #expect(reply.contains("選擇：[b] 去 → 趣（範圍：語境限定）"))
        #expect(reply.contains("補充：無"))
        #expect(!reply.contains("[c]"))
        // Question is consumed and marked answered on its turn.
        #expect(session.state.pendingQuestion == nil)
        let questionTurn = session.state.transcript.first { $0.question != nil }
        #expect(questionTurn?.answeredOptionIds == ["a", "b"])
        #expect(questionTurn?.answeredScopes["b"] == .context)
        // The user turn shows a readable summary, not the wire text.
        let userTurn = session.state.transcript.last { $0.role == "user" }
        #expect(userTurn?.text.contains("回覆問題") == false)
        #expect(userTurn?.text.contains("小振 → 小鎮") == true)
    }

    @Test func choiceScopedAnyContextAllowsMatchingBroadDraftOnly() async {
        server.reset()
        let broadA = RuleAssistantTestJSON.string(["eventType": "replacementRule", "sourcePattern": "小振", "targetText": "小鎮"])
        let broadB = RuleAssistantTestJSON.string(["eventType": "replacementRule", "sourcePattern": "去", "targetText": "趣"])
        FakeGoProvider.reset(scripts: [
            .stream([FakeGoProvider.chunk(content: questionJSON(options: candidateOptions)), FakeGoProvider.chunk(finish: "stop")]),
            .stream([FakeGoProvider.chunk(content: draftAnswer([broadA, broadB])), FakeGoProvider.chunk(finish: "stop")]),
        ])
        let session = makeRuleAssistantSession(server: server)
        await session.submitScan()
        session.toggleOption("a")
        session.toggleOption("b")
        session.setScope(.broad, for: "a")
        await session.submitChoice()
        guard case .draftReady = session.state.phase else {
            Issue.record("expected draftReady, got \(session.state.phase)")
            return
        }
        #expect(session.state.drafts.map { $0.draft.sourcePattern } == ["小振"])
        #expect(lastUserMessage(1).contains("[a] 小振 → 小鎮（範圍：任何語境）"))
    }

    @Test func choiceWithTypedNoteCountsAsJasonsOwnWords() async {
        server.reset()
        let broad = RuleAssistantTestJSON.string(["eventType": "replacementRule", "sourcePattern": "小振", "targetText": "小鎮"])
        FakeGoProvider.reset(scripts: [
            .stream([FakeGoProvider.chunk(content: questionJSON(multiSelect: false, options: candidateOptions)), FakeGoProvider.chunk(finish: "stop")]),
            .stream([FakeGoProvider.chunk(content: draftAnswer([broad])), FakeGoProvider.chunk(finish: "stop")]),
        ])
        let session = makeRuleAssistantSession(server: server)
        await session.submitScan()
        session.toggleOption("b")
        session.toggleOption("a")
        // Single-select: the second pick replaces the first.
        #expect(session.state.selectedOptionIds == ["a"])
        session.setUserText("小振在我這裡永遠是小鎮")
        await session.submitChoice()
        guard case .draftReady = session.state.phase else {
            Issue.record("expected draftReady, got \(session.state.phase)")
            return
        }
        #expect(session.state.drafts.first?.draft.eventType == "replacementRule")
        #expect(lastUserMessage(1).contains("補充：小振在我這裡永遠是小鎮"))
        #expect(session.state.userText.isEmpty)
    }

    @Test func failedChoiceReplyRestoresTheQuestion() async {
        server.reset()
        FakeGoProvider.reset(scripts: [
            .stream([FakeGoProvider.chunk(content: questionJSON(options: candidateOptions)), FakeGoProvider.chunk(finish: "stop")]),
            .httpError(500),
        ])
        let session = makeRuleAssistantSession(server: server)
        await session.submitScan()
        session.toggleOption("a")
        session.setScope(.context, for: "a")
        await session.submitChoice()
        guard case .failed = session.state.phase else {
            Issue.record("expected failed, got \(session.state.phase)")
            return
        }
        #expect(session.state.pendingQuestion?.id == "q1")
        #expect(session.state.selectedOptionIds == ["a"])
        #expect(session.state.optionScopes["a"] == .context)
        #expect(session.state.transcript.first { $0.question != nil }?.answeredOptionIds.isEmpty == true)
        #expect(session.state.transcript.last?.role == "assistant")
        // The wire reply never lands in the composer; only a typed note would.
        #expect(session.state.userText.isEmpty)
    }

    @Test func failedChoiceReplyWithNoteRestoresOnlyTheNote() async {
        server.reset()
        FakeGoProvider.reset(scripts: [
            .stream([FakeGoProvider.chunk(content: questionJSON(options: candidateOptions)), FakeGoProvider.chunk(finish: "stop")]),
            .httpError(500),
        ])
        let session = makeRuleAssistantSession(server: server)
        await session.submitScan()
        session.toggleOption("a")
        session.setUserText("其實是小鎮")
        await session.submitChoice()
        guard case .failed = session.state.phase else {
            Issue.record("expected failed, got \(session.state.phase)")
            return
        }
        #expect(session.state.userText == "其實是小鎮")
        #expect(session.state.pendingQuestion?.id == "q1")
        #expect(session.state.selectedOptionIds == ["a"])
    }

    @Test func malformedQuestionIsShownAsPlainText() async {
        server.reset()
        let bad = RuleAssistantTestJSON.string(["question": ["prompt": "沒有選項", "options": [] as [Any]] as [String: Any]])
        FakeGoProvider.reset(scripts: [
            .stream([FakeGoProvider.chunk(content: "看不出來。" + bad), FakeGoProvider.chunk(finish: "stop")]),
        ])
        let session = makeRuleAssistantSession(server: server)
        await session.submit("看看")
        #expect(session.state.phase == .idle)
        #expect(session.state.pendingQuestion == nil)
        #expect(session.state.transcript.last?.text.contains("看不出來") == true)
    }

    @Test func questionAndDraftInOneAnswerBothSurvive() async {
        server.reset()
        FakeGoProvider.reset(scripts: [
            .stream([
                FakeGoProvider.chunk(content: draftAnswer([correctionDraft("我們去小振家", "我們去小鎮家")]) + "\n" + questionJSON(options: [["id": "x", "label": "去 → 趣", "surface": "去", "target": "趣"], ["label": "沒錯"]])),
                FakeGoProvider.chunk(finish: "stop"),
            ]),
        ])
        let session = makeRuleAssistantSession(server: server)
        await session.submitScan()
        guard case .draftReady = session.state.phase else {
            Issue.record("expected draftReady, got \(session.state.phase)")
            return
        }
        #expect(session.state.drafts.count == 1)
        #expect(session.state.pendingQuestion?.options.map(\.id) == ["x", "b"])
        #expect(session.state.canSubmitChoice == false)
        session.toggleOption("x")
        #expect(session.state.canSubmitChoice)
    }

    @Test func autoScanRunsOnceAndOnlyOnFreshConfiguredSession() async {
        server.reset()
        FakeGoProvider.reset(scripts: [
            .stream([FakeGoProvider.chunk(content: "沒有問題。"), FakeGoProvider.chunk(finish: "stop")]),
        ])
        let session = makeRuleAssistantSession(server: server)
        // Not configured: nothing happens.
        await session.autoScanIfNeeded()
        #expect(FakeGoProvider.recorded.isEmpty)
        #expect(!session.state.hasAutoScanned)
        session.setConfig(goKeyConfigured: true, syncConfigured: true)
        await session.autoScanIfNeeded()
        #expect(FakeGoProvider.recorded.count == 1)
        #expect(session.state.hasAutoScanned)
        await session.autoScanIfNeeded()
        #expect(FakeGoProvider.recorded.count == 1)
    }

    @Test func typingDuringScanIsKeptAndScanCancelRestoresNothing() async {
        server.reset()
        // The provider never answers, so the scan stays in flight until cancelled.
        FakeGoProvider.reset(scripts: [.hang])
        let session = makeRuleAssistantSession(server: server)
        let task = Task { @MainActor in await session.submitScan() }
        _ = await waitForRequests(1)
        #expect(session.state.phase.isBusy)
        #expect(session.state.isInterruptible)
        #expect(session.state.busyTurnKind == .scan)
        session.setUserText("其實是小鎮")
        #expect(session.state.userText == "其實是小鎮")
        session.cancel()
        task.cancel()
        _ = await task.value
        // Typed text survives; the scan placeholder turn is dropped and its prompt never lands in the composer.
        #expect(session.state.userText == "其實是小鎮")
        #expect(session.state.phase == .idle)
        #expect(session.state.busyTurnKind == nil)
        #expect(session.state.transcript.isEmpty)
    }

    @Test func typingDuringManualTurnIsIgnoredButCancelRestoresIt() async {
        server.reset()
        FakeGoProvider.reset(scripts: [.hang])
        let session = makeRuleAssistantSession(server: server)
        let task = Task { @MainActor in await session.submit("小振是小鎮") }
        _ = await waitForRequests(1)
        #expect(!session.state.isInterruptible)
        session.setUserText("不該進去")
        #expect(session.state.userText.isEmpty)
        session.cancel()
        task.cancel()
        _ = await task.value
        #expect(session.state.userText == "小振是小鎮")
    }

    // MARK: Nearby records

    @Test func nearbyRecordsClampAndExcludeSelf() async {
        server.reset()
        var loaderBefore = -1
        var loaderAfter = -1
        let neighbors = [40, 41, 42, 43, 44, 45].map {
            RuleAssistantContext(rowPk: Int64($0), timestampMs: Int64($0) * 1000, text: "第 \($0) 筆")
        }
        FakeGoProvider.reset(scripts: [
            .stream([
                FakeGoProvider.chunk(toolCalls: [
                    FakeGoProvider.toolCallFragment(
                        index: 0,
                        id: "c1",
                        name: "load_nearby_records",
                        arguments: "{\"before\":99,\"after\":99}"
                    ),
                ]),
                FakeGoProvider.chunk(finish: "tool_calls"),
            ]),
            .stream([
                FakeGoProvider.chunk(content: "看完了"),
                FakeGoProvider.chunk(finish: "stop"),
            ]),
        ])
        let session = RuleAssistantSession(
            context: RuleAssistantContext(rowPk: 42, timestampMs: 42_000, text: "本筆"),
            providerFactory: { FakeGoProvider.makeClient() },
            mcpFactory: { server.makeClient() },
            syncNow: { RuleAssistantSyncResult(outcome: .upToDate, message: "ok", remoteSha256: nil, installedSha256: nil) },
            neighborLoader: { before, after in
                loaderBefore = before
                loaderAfter = after
                return neighbors
            }
        )
        await session.submit("看前後")
        // Counts are clamped to 5 before the loader runs.
        #expect(loaderBefore == 5)
        #expect(loaderAfter == 5)
        let tool = providerMessages(1).first { $0["role"] as? String == "tool" }
        let content = tool?["content"] as? String ?? ""
        let result = RuleAssistantTestJSON.object(content)
        let records = result["records"] as? [[String: Any]] ?? []
        #expect(records.count == 5)
        #expect(!records.contains { ($0["rowPk"] as? Int) == 42 })
        #expect(session.state.neighborsShared == 5)
    }

    // MARK: Cancel

    @Test func cancelLeavesNoUnmatchedToolCalls() async {
        server.reset()
        FakeGoProvider.reset(scripts: [
            .stream([
                FakeGoProvider.chunk(toolCalls: [
                    FakeGoProvider.toolCallFragment(index: 0, id: "c1", name: "lookup_auto_apply_policy", arguments: "{}"),
                ]),
                FakeGoProvider.chunk(finish: "tool_calls"),
            ]),
            .hang,
        ])
        let session = makeRuleAssistantSession(server: server)
        let task = Task { @MainActor in
            await session.submit("查一下小振")
        }
        // Wait for round 2 to start (the hanging request), then stop.
        let started = await waitForRequests(2)
        #expect(started)
        session.cancel()
        await task.value
        #expect(session.state.phase == .idle)
        #expect(session.state.userText == "查一下小振")
        #expect(session.state.transcript.isEmpty)
        // A fresh turn must not carry the aborted tool exchange.
        let requestCount = FakeGoProvider.recorded.count
        FakeGoProvider.append(scripts: [
            .stream([
                FakeGoProvider.chunk(content: "好的"),
                FakeGoProvider.chunk(finish: "stop"),
            ]),
        ])
        await session.submit("重來")
        #expect(session.state.phase == .idle)
        #expect(FakeGoProvider.recorded.count > requestCount)
        let messages = providerMessages(requestCount)
        for message in messages where message["role"] as? String == "assistant" {
            #expect(message["tool_calls"] == nil)
        }
        #expect(!messages.contains { $0["role"] as? String == "tool" })
    }

    // MARK: Endpoint validation & client lifecycle

    @Test func endpointWithForeignOriginIsRejected() async {
        server.reset()
        server.endpointEventValue = "https://evil.test/mcp/messages/abc"
        let client = server.makeClient()
        await #expect(throws: RuleAssistantTransportError.self) {
            _ = try await client.tools()
        }
        client.close()
    }

    @Test func endpointWithQueryIsRejected() async {
        server.reset()
        server.endpointEventValue = "/mcp/messages/abc?key=1"
        let client = server.makeClient()
        await #expect(throws: RuleAssistantTransportError.self) {
            _ = try await client.tools()
        }
        client.close()
    }

    @Test func endpointWithWrongPathIsRejected() async {
        server.reset()
        server.endpointEventValue = "/other/path"
        let client = server.makeClient()
        await #expect(throws: RuleAssistantTransportError.self) {
            _ = try await client.tools()
        }
        client.close()
    }

    @Test func initializeHandshakeAndToolsList() async throws {
        server.reset()
        let client = server.makeClient()
        let tools = try await client.tools()
        #expect(tools.contains { $0.name == "get_auto_apply_row_corrections" })
        #expect(tools.contains { $0.name == "preview_auto_apply_control_event" })
        // The MCP handshake happened in order: initialize before tools/list.
        let methods = server.rpcs.map(\.method)
        #expect(methods.first == "initialize")
        #expect(methods.contains("tools/list"))
        client.close()
    }

    @Test func closedClientDoesNotReconnect() async {
        server.reset()
        let client = server.makeClient()
        _ = try? await client.tools()
        client.close()
        await #expect(throws: RuleAssistantTransportError.self) {
            _ = try await client.tools()
        }
    }

    @Test func providerRedirectIsNotFollowed() async {
        server.reset()
        FakeGoProvider.reset(scripts: [
            .redirect(301, "https://provider.test/v1/chat/elsewhere"),
        ])
        let client = FakeGoProvider.makeClient()
        await #expect(throws: RuleAssistantTransportError.self) {
            try await client.stream(messages: [], tools: []) { _ in }
        }
        // Exactly one request: the 301 target was never queried.
        #expect(FakeGoProvider.recorded.count == 1)
    }

    @Test func closedProviderCannotStream() async {
        FakeGoProvider.reset(scripts: [.stream([FakeGoProvider.chunk(finish: "stop")])])
        let client = FakeGoProvider.makeClient()
        client.close()
        await #expect(throws: RuleAssistantTransportError.self) {
            try await client.stream(messages: [], tools: []) { _ in }
        }
        // No request ever left the client.
        #expect(FakeGoProvider.recorded.isEmpty)
    }

    @Test func non200ProviderStatusIsTransportError() async {
        FakeGoProvider.reset(scripts: [.httpError(429)])
        let client = FakeGoProvider.makeClient()
        do {
            try await client.stream(messages: [], tools: []) { _ in }
            Issue.record("expected a transport error")
        } catch let error as RuleAssistantTransportError {
            #expect(error.message.contains("429"))
        } catch {
            Issue.record("unexpected error type: \(error)")
        }
    }
}
