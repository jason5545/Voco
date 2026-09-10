import Foundation
import Testing
@testable import Voco

/// Live end-to-end smoke: real OpenCode Go + real client, fake Worker. Runs only when
/// VOCO_GO_KEY is present in the environment; never touches the real Worker.
@MainActor
@Suite(.serialized)
struct RuleAssistantLiveTests {
    nonisolated private static let goKey: String? = {
        let key = ProcessInfo.processInfo.environment["VOCO_GO_KEY"]?
            .trimmingCharacters(in: .whitespacesAndNewlines)
        return (key?.isEmpty == false) ? key : nil
    }()

    /// Skipped unless VOCO_GO_KEY is present; never touches the real Worker.
    @Test(.enabled(if: RuleAssistantLiveTests.goKey != nil))
    func liveProviderProducesParseableDraft() async throws {
        let key = try #require(Self.goKey)
        let server = FakeMCPServer.shared
        server.reset()
        let session = RuleAssistantSession(
            context: RuleAssistantContext(
                rowPk: 42,
                timestampMs: 1_700_000_000_000,
                rawTranscript: "我明天要去小振家",
                text: "我明天要去小振家",
                recordId: "11111111-2222-3333-4444-555555555555"
            ),
            providerFactory: { OpenCodeGoClient(apiKey: key) },
            mcpFactory: { server.makeClient() },
            syncNow: { RuleAssistantSyncResult(outcome: .upToDate, message: "fake", remoteSha256: nil, installedSha256: nil) }
        )
        await session.submit("「小振」其實是「小鎮」，我說的是地名")
        // The model must either propose a parseable draft or (acceptably) ask a clarifying
        // question in plain text; a hard failure means the wire contract broke.
        switch session.state.phase {
        case .draftReady:
            #expect(!session.state.drafts.isEmpty)
            #expect(session.state.drafts.allSatisfy { $0.draft.isSafeForWrite() })
        case .idle:
            #expect(session.state.transcript.contains { $0.role == "assistant" && !$0.text.isEmpty })
        default:
            Issue.record("live round ended in unexpected phase: \(session.state.phase)")
        }
        session.closeClients()
    }
}
