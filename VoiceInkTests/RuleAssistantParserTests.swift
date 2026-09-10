import Foundation
import Testing
@testable import Voco

@Suite(.serialized)
struct RuleAssistantSseParserTests {
    private func collect(_ chunks: [String], finish: Bool = true) throws -> [SseEvent] {
        var events: [SseEvent] = []
        let parser = SseEventParser { events.append($0) }
        for chunk in chunks {
            try parser.feed(chunk)
        }
        if finish {
            try parser.finish()
        }
        return events
    }

    @Test func multiLineDataAndCRLF() throws {
        let events = try collect([
            "data: first\r\ndata: second\r\n\r\n",
        ])
        #expect(events.count == 1)
        #expect(events[0].data == "first\nsecond")
    }

    @Test func splitAcrossFeedsAndUtf8() throws {
        // A multibyte character and an event boundary split across feeds.
        let events = try collect([
            "data: 中",
            "文字\n\nda",
            "ta: next\n\n",
        ])
        #expect(events.map(\.data) == ["中文字", "next"])
    }

    @Test func eventAndIdFields() throws {
        let events = try collect(["event: message\nid: 7\ndata: hello\n\n"])
        #expect(events.count == 1)
        #expect(events[0].event == "message")
        #expect(events[0].id == "7")
        #expect(events[0].data == "hello")
    }

    @Test func commentsAreIgnored() throws {
        let events = try collect([": keepalive\n\ndata: x\n\n"])
        #expect(events.map(\.data) == ["x"])
    }

    @Test func trailingPartialLineFlushesOnFinish() throws {
        let events = try collect(["data: tail-without-newline"])
        #expect(events.map(\.data) == ["tail-without-newline"])
    }

    @Test func oversizedLineThrows() throws {
        var events: [SseEvent] = []
        let parser = SseEventParser { events.append($0) }
        #expect(throws: SseParserError.self) {
            try parser.feed(String(repeating: "x", count: 300_000))
        }
    }
}

@Suite(.serialized)
struct RuleAssistantDeltaParserTests {
    private func parse(_ sseData: [String], finish: Bool = true) -> [OpenCodeDelta] {
        var deltas: [OpenCodeDelta] = []
        let parser = OpenCodeDeltaParser { deltas.append($0) }
        for data in sseData {
            parser.accept(SseEvent(event: nil, data: data))
        }
        if finish {
            parser.finish()
        }
        return deltas
    }

    @Test func reasoningOnly() {
        let deltas = parse([FakeGoProvider.chunk(reasoning: "想一下")])
        #expect(deltas.count == 1)
        #expect(deltas[0].reasoning == "想一下")
        #expect(deltas[0].content == "")
        #expect(deltas[0].rawReasoningContent == "想一下")
    }

    @Test func thinkTagSplitAcrossChunks() {
        let deltas = parse([
            FakeGoProvider.chunk(content: "前面<th"),
            FakeGoProvider.chunk(content: "ink>內心話</th"),
            FakeGoProvider.chunk(content: "ink>後面"),
        ])
        #expect(deltas.map(\.content).joined() == "前面後面")
        #expect(deltas.map(\.reasoning).joined() == "內心話")
        // Raw wire text keeps the tags for the next provider round-trip.
        #expect(deltas.map(\.rawContent).joined() == "前面<think>內心話</think>後面")
    }

    @Test func trailingPartialTagFlushesOnFinish() {
        let deltas = parse([FakeGoProvider.chunk(content: "答案<th")])
        #expect(deltas.map(\.content).joined() == "答案<th")
    }

    @Test func reasoningFieldAliasIsAccepted() {
        let json = RuleAssistantTestJSON.string([
            "choices": [["delta": ["reasoning": "想"], "finish_reason": NSNull()]],
        ])
        let deltas = parse([json])
        #expect(deltas[0].reasoning == "想")
    }

    @Test func nullFieldsAreTolerated() {
        let json = RuleAssistantTestJSON.string([
            "choices": [[
                "delta": ["content": NSNull(), "reasoning_content": NSNull()],
                "finish_reason": NSNull(),
            ]],
        ])
        let deltas = parse([json])
        #expect(deltas.count == 1)
        #expect(deltas[0].content == "")
        #expect(deltas[0].finishReason == nil)
        #expect(deltas[0].done == false)
    }

    @Test func usageOnlyChunkIsIgnored() {
        let usage = RuleAssistantTestJSON.string([
            "choices": [Any](),
            "usage": ["total_tokens": 10],
        ])
        let deltas = parse([usage])
        #expect(deltas.isEmpty)
    }

    @Test func providerErrorObjectAndString() {
        let objectError = RuleAssistantTestJSON.string([
            "error": ["message": "rate limited", "type": "rate_limit"],
        ])
        #expect(parse([objectError]).first?.error == "rate limited")

        let stringError = RuleAssistantTestJSON.string(["error": "boom"])
        #expect(parse([stringError]).first?.error == "boom")
    }

    @Test func doneMarkerProducesDoneDelta() {
        let deltas = parse(["[DONE]"])
        #expect(deltas.count == 1)
        #expect(deltas[0].done)
    }

    @Test func finishReasonMarksDone() {
        let deltas = parse([FakeGoProvider.chunk(content: "好", finish: "stop")])
        #expect(deltas[0].done)
        #expect(deltas[0].finishReason == "stop")
    }

    @Test func toolCallFragmentsKeepIndex() {
        let deltas = parse([
            FakeGoProvider.chunk(toolCalls: [
                FakeGoProvider.toolCallFragment(index: 3, id: "call_9", name: "lookup", arguments: "{"),
            ]),
        ])
        #expect(deltas[0].toolCalls.count == 1)
        #expect(deltas[0].toolCalls[0].index == 3)
        #expect(deltas[0].toolCalls[0].id == "call_9")
        #expect(deltas[0].toolCalls[0].name == "lookup")
        #expect(deltas[0].toolCalls[0].arguments == "{")
    }
}

@Suite(.serialized)
struct RuleAssistantToolAccumulatorTests {
    @Test func fragmentedArgumentsAcrossChunks() throws {
        let accumulator = ToolCallAccumulator()
        accumulator.add(ToolCallFragment(index: 0, id: "call_1", name: "lookup_auto_apply_policy", arguments: "{\"source\":"))
        accumulator.add(ToolCallFragment(index: 0, arguments: " \"小振\"}"))
        let calls = try accumulator.complete()
        #expect(calls.count == 1)
        let function = calls[0]["function"] as? [String: Any]
        #expect(function?["name"] as? String == "lookup_auto_apply_policy")
        let args = RuleAssistantTestJSON.object(function?["arguments"] as? String ?? "")
        #expect(args["source"] as? String == "小振")
    }

    @Test func multipleIndicesCompleteInOrder() throws {
        let accumulator = ToolCallAccumulator()
        accumulator.add(ToolCallFragment(index: 1, id: "b", name: "second", arguments: "{}"))
        accumulator.add(ToolCallFragment(index: 0, id: "a", name: "first", arguments: "{}"))
        let calls = try accumulator.complete()
        #expect(calls.compactMap { ($0["function"] as? [String: Any])?["name"] as? String } == ["first", "second"])
    }

    @Test func missingIdIsNotExecutable() {
        let accumulator = ToolCallAccumulator()
        accumulator.add(ToolCallFragment(index: 0, name: "lookup", arguments: "{}"))
        #expect(!accumulator.problems().isEmpty)
        #expect(throws: RuleAssistantFailure.self) {
            _ = try accumulator.complete()
        }
    }

    @Test func incompleteArgumentJsonIsNotExecutable() {
        let accumulator = ToolCallAccumulator()
        accumulator.add(ToolCallFragment(index: 0, id: "call_1", name: "lookup", arguments: "{\"a\":"))
        #expect(!accumulator.problems().isEmpty)
    }

    @Test func oversizedArgumentsAreRefused() {
        let accumulator = ToolCallAccumulator()
        accumulator.add(ToolCallFragment(
            index: 0,
            id: "call_1",
            name: "lookup",
            arguments: String(repeating: "x", count: ToolCallAccumulator.maxArgumentChars + 1)
        ))
        #expect(accumulator.problems().count == 1)
        #expect(throws: RuleAssistantFailure.self) {
            _ = try accumulator.complete()
        }
    }

    @Test func moreThanMaxCallsIsRefused() {
        let accumulator = ToolCallAccumulator()
        for index in 0..<(ToolCallAccumulator.maxToolCalls + 1) {
            accumulator.add(ToolCallFragment(index: index, id: "c\(index)", name: "t", arguments: "{}"))
        }
        #expect(!accumulator.problems().isEmpty)
    }
}
