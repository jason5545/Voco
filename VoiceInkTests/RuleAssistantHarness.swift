import Foundation
import Network
@testable import Voco

// MARK: - Shared helpers

enum RuleAssistantTestJSON {
    static func string(_ object: [String: Any]) -> String {
        let data = try! JSONSerialization.data(withJSONObject: object, options: [.sortedKeys])
        return String(data: data, encoding: .utf8)!
    }

    static func object(_ string: String) -> [String: Any] {
        let data = string.data(using: .utf8)!
        return (try! JSONSerialization.jsonObject(with: data)) as! [String: Any]
    }
}

// MARK: - Minimal loopback HTTP server

/// A tiny HTTP/1.1 server on 127.0.0.1 used to feed real SSE and JSON-RPC bytes to the real
/// clients. One request per connection; streaming responses stay open until closed.
final class FakeHTTPServer: @unchecked Sendable {
    struct Request {
        let method: String
        let path: String
        /// Header names lowercased.
        let headers: [String: String]
        let body: Data
    }

    enum Action {
        /// Full response, then the connection closes.
        case respond(status: Int, headers: [String: String], body: Data)
        /// Chunked stream: headers, then initialBody, then either close (closeAfterBody) or stay
        /// open for pushBytes until closeConnection.
        case stream(status: Int, headers: [String: String], initialBody: String?, closeAfterBody: Bool)
    }

    private let queue = DispatchQueue(label: "fake-http-server")
    private var listener: NWListener?
    private var connections: [ObjectIdentifier: NWConnection] = [:]
    private var nextConnectionId = 0
    private let handler: (Request, ObjectIdentifier) -> Action

    init(handler: @escaping (Request, ObjectIdentifier) -> Action) {
        self.handler = handler
    }

    var baseURL: URL {
        URL(string: "http://127.0.0.1:\(port)")!
    }

    private(set) var port: Int = 0

    func start() throws {
        let listener = try NWListener(using: .tcp, on: .any)
        self.listener = listener
        listener.newConnectionHandler = { [weak self] connection in
            self?.accept(connection)
        }
        let ready = DispatchSemaphore(value: 0)
        listener.stateUpdateHandler = { state in
            switch state {
            case .ready, .failed:
                ready.signal()
            default:
                break
            }
        }
        listener.start(queue: queue)
        if ready.wait(timeout: .now() + 5) == .timedOut {
            throw RuleAssistantTransportError("fake server did not start")
        }
        // listener.port reports the requested .any (0) until the ready state assigns the real one.
        guard let boundPort = listener.port, boundPort.rawValue != 0 else {
            throw RuleAssistantTransportError("fake server did not bind a port")
        }
        port = Int(boundPort.rawValue)
        listener.stateUpdateHandler = nil
    }

    func pushBytes(_ string: String, to connectionId: ObjectIdentifier) {
        queue.async { [weak self] in
            guard let connection = self?.connections[connectionId] else { return }
            connection.send(content: Self.chunkFrame(Data(string.utf8)), completion: .idempotent)
        }
    }

    func stop() {
        queue.sync {
            listener?.cancel()
            listener = nil
            for (_, connection) in connections {
                connection.cancel()
            }
            connections.removeAll()
        }
    }

    // MARK: Connection lifecycle

    private func accept(_ connection: NWConnection) {
        let id = ObjectIdentifier(connection)
        queue.async { [weak self] in
            self?.connections[id] = connection
        }
        connection.start(queue: queue)
        receive(into: Data(), connection: connection, id: id)
    }

    private func receive(into buffer: Data, connection: NWConnection, id: ObjectIdentifier) {
        var buffer = buffer
        connection.receive(minimumIncompleteLength: 1, maximumLength: 1 << 20) { [weak self] data, _, isComplete, error in
            guard let self else { return }
            if let data {
                buffer.append(data)
            }
            if self.tryHandle(buffer: buffer, connection: connection, id: id) {
                return
            }
            if isComplete || error != nil {
                self.drop(connection: connection, id: id)
                return
            }
            self.receive(into: buffer, connection: connection, id: id)
        }
    }

    /// Returns true once the request was fully read and the handler invoked.
    private func tryHandle(buffer: Data, connection: NWConnection, id: ObjectIdentifier) -> Bool {
        guard let headerRange = buffer.range(of: Data("\r\n\r\n".utf8)) else { return false }
        let headerData = buffer[buffer.startIndex..<headerRange.lowerBound]
        guard let headerText = String(data: headerData, encoding: .utf8) else { return true }
        var lines = headerText.components(separatedBy: "\r\n")
        let requestLine = lines.removeFirst()
        let parts = requestLine.split(separator: " ")
        guard parts.count >= 2 else { return true }
        var headers: [String: String] = [:]
        for line in lines {
            guard let colon = line.firstIndex(of: ":") else { continue }
            headers[String(line[..<colon]).lowercased()] = String(line[line.index(after: colon)...])
                .trimmingCharacters(in: .whitespaces)
        }
        let contentLength = Int(headers["content-length"] ?? "0") ?? 0
        let bodyStart = headerRange.upperBound
        guard buffer.distance(from: bodyStart, to: buffer.endIndex) >= contentLength else { return false }
        let body = buffer[bodyStart..<buffer.index(bodyStart, offsetBy: contentLength)]
        let request = Request(
            method: String(parts[0]),
            path: String(parts[1]),
            headers: headers,
            body: Data(body)
        )
        let action = handler(request, id)
        // NWConnection drops sends issued synchronously inside a receive completion; hop out first.
        queue.async { [weak self] in
            guard let self, self.connections[id] != nil else { return }
            switch action {
            case .respond(let status, let responseHeaders, let body):
                var response = "HTTP/1.1 \(status) \(Self.statusText(status))\r\n"
                for (name, value) in responseHeaders {
                    response += "\(name): \(value)\r\n"
                }
                response += "Content-Length: \(body.count)\r\nConnection: close\r\n\r\n"
                let payload = Data(response.utf8) + body
                connection.send(content: payload, completion: .contentProcessed { [weak self] _ in
                    self?.queue.async {
                        self?.drop(connection: connection, id: id)
                    }
                })
            case .stream(let status, let responseHeaders, let initialBody, let closeAfterBody):
                var response = "HTTP/1.1 \(status) \(Self.statusText(status))\r\n"
                for (name, value) in responseHeaders {
                    response += "\(name): \(value)\r\n"
                }
                // Chunked: URLSession streams these bodies; a bare connection-close body never flows.
                response += "Transfer-Encoding: chunked\r\n"
                response += "\r\n"
                var payload = Data(response.utf8)
                if let initialBody {
                    payload.append(Self.chunkFrame(Data(initialBody.utf8)))
                }
                if closeAfterBody {
                    payload.append(Data("0\r\n\r\n".utf8))
                    connection.send(content: payload, completion: .contentProcessed { [weak self] _ in
                        self?.queue.async {
                            self?.drop(connection: connection, id: id)
                        }
                    })
                } else {
                    connection.send(content: payload, completion: .idempotent)
                }
            }
        }
        return true
    }

    static func chunkFrame(_ data: Data) -> Data {
        Data(String(data.count, radix: 16).utf8) + Data("\r\n".utf8) + data + Data("\r\n".utf8)
    }

    private func drop(connection: NWConnection, id: ObjectIdentifier) {
        connection.cancel()
        connections.removeValue(forKey: id)
    }

    private static func statusText(_ status: Int) -> String {
        switch status {
        case 200: return "OK"
        case 202: return "Accepted"
        case 301: return "Moved Permanently"
        case 400: return "Bad Request"
        case 404: return "Not Found"
        case 429: return "Too Many Requests"
        case 500: return "Internal Server Error"
        default: return "Status"
        }
    }

    /// Closes a streaming connection (server-initiated end of body), sending the terminal chunk first.
    func closeConnection(_ connectionId: ObjectIdentifier) {
        queue.async { [weak self] in
            guard let self, let connection = self.connections[connectionId] else { return }
            connection.send(content: Data("0\r\n\r\n".utf8), completion: .contentProcessed { [weak self] _ in
                self?.queue.async {
                    guard let self, let connection = self.connections[connectionId] else { return }
                    self.drop(connection: connection, id: connectionId)
                }
            })
        }
    }
}

// MARK: - Shared server processes

/// Starts both loopback servers once per test process.
enum FakeServers {
    private static let lock = NSLock()
    private static var started = false

    static func ensureStarted() {
        lock.lock()
        defer { lock.unlock() }
        guard !started else { return }
        do {
            try FakeGoProvider.startLockedServer()
            try FakeMCPServer.shared.startServer()
            started = true
        } catch {
            fatalError("fake servers failed to start: \(error)")
        }
    }
}

// MARK: - Fake OpenCode Go provider

/// Loopback fake for the provider endpoint. Each POST consumes the next scripted response;
/// every request (headers + parsed body) is recorded. Suites using it must be serialized.
enum FakeGoProvider {
    enum Script {
        /// Full SSE payload, then the connection closes.
        case sse(String)
        /// 200 headers, then silence until the client cancels.
        case hang
        /// 3xx with a Location header; a compliant client must not follow it.
        case redirect(Int, String)
        /// Non-2xx status with an empty body.
        case httpError(Int)
    }

    struct RecordedRequest {
        let authorization: String?
        let userAgent: String?
        let sessionId: String?
        let body: [String: Any]
    }

    private static let lock = NSLock()
    private static var recordedStorage: [RecordedRequest] = []
    private static var scriptStorage: [Script] = []
    private static var server: FakeHTTPServer?

    static var recorded: [RecordedRequest] {
        lock.lock()
        defer { lock.unlock() }
        return recordedStorage
    }

    static var endpoint: URL {
        FakeServers.ensureStarted()
        return server!.baseURL.appending(path: "v1/chat/completions")
    }

    static func reset(scripts: [Script]) {
        lock.lock()
        recordedStorage = []
        scriptStorage = scripts
        lock.unlock()
    }

    static func append(scripts: [Script]) {
        lock.lock()
        scriptStorage.append(contentsOf: scripts)
        lock.unlock()
    }

    static func nextScript() -> Script {
        lock.lock()
        defer { lock.unlock() }
        return scriptStorage.isEmpty ? .httpError(500) : scriptStorage.removeFirst()
    }

    static func record(_ entry: RecordedRequest) {
        lock.lock()
        recordedStorage.append(entry)
        lock.unlock()
    }

    static func makeClient(sessionId: String = "fixed-session-id") -> OpenCodeGoClient {
        OpenCodeGoClient(
            apiKey: "test-go-key",
            sessionId: sessionId,
            endpoint: endpoint
        )
    }

    static func startLockedServer() throws {
        let http = FakeHTTPServer { request, connectionId in
            let body = (try? JSONSerialization.jsonObject(with: request.body)) as? [String: Any] ?? [:]
            record(RecordedRequest(
                authorization: request.headers["authorization"],
                userAgent: request.headers["user-agent"],
                sessionId: request.headers["x-opencode-session"],
                body: body
            ))
            switch nextScript() {
            case .sse(let payload):
                return .stream(
                    status: 200,
                    headers: ["Content-Type": "text/event-stream"],
                    initialBody: payload,
                    closeAfterBody: true
                )
            case .hang:
                return .stream(
                    status: 200,
                    headers: ["Content-Type": "text/event-stream"],
                    initialBody: nil,
                    closeAfterBody: false
                )
            case .redirect(let status, let location):
                return .respond(status: status, headers: ["Location": location], body: Data())
            case .httpError(let status):
                return .respond(status: status, headers: [:], body: Data())
            }
        }
        try http.start()
        server = http
    }

    /// One SSE data chunk for a chat completion delta.
    static func chunk(
        content: String? = nil,
        reasoning: String? = nil,
        toolCalls: [[String: Any]]? = nil,
        finish: String? = nil
    ) -> String {
        var delta: [String: Any] = [:]
        if let content { delta["content"] = content }
        if let reasoning { delta["reasoning_content"] = reasoning }
        if let toolCalls { delta["tool_calls"] = toolCalls }
        let choice: [String: Any] = ["delta": delta, "finish_reason": finish ?? NSNull()]
        return RuleAssistantTestJSON.string(["choices": [choice]])
    }

    static func toolCallFragment(
        index: Int,
        id: String? = nil,
        name: String? = nil,
        arguments: String? = nil
    ) -> [String: Any] {
        var fragment: [String: Any] = ["index": index]
        if let id { fragment["id"] = id }
        var function: [String: Any] = [:]
        if let name { function["name"] = name }
        if let arguments { function["arguments"] = arguments }
        if !function.isEmpty { fragment["function"] = function }
        return fragment
    }
}

extension FakeGoProvider.Script {
    /// Wraps SSE data chunks into a full stream terminated by [DONE].
    static func stream(_ chunks: [String]) -> FakeGoProvider.Script {
        .sse(chunks.map { "data: \($0)\n\n" }.joined() + "data: [DONE]\n\n")
    }
}

// MARK: - Fake Worker MCP server (legacy HTTP+SSE)

enum FakeMCPOutcome {
    case result([String: Any])
    case isError([String: Any])
    case rpcError(Int, String)
    /// The POST itself fails (response lost); nothing is delivered on the SSE stream.
    case httpError(Int)
}

/// Loopback fake Worker. GET /mcp/sse stays open and carries endpoint + JSON-RPC responses;
/// POST /mcp/messages/* is the 202-ack request channel. Suites using it must be serialized.
final class FakeMCPServer {
    static let shared = FakeMCPServer()

    struct RecordedRPC {
        let method: String
        let params: [String: Any]
    }

    struct RecordedCall {
        let name: String
        let args: [String: Any]
    }

    private let lock = NSLock()
    private var rpcStorage: [RecordedRPC] = []
    private var callStorage: [RecordedCall] = []
    private var http: FakeHTTPServer?
    private var sseConnectionId: ObjectIdentifier?

    var endpointEventValue = "/mcp/messages/fake"
    var toolList: [[String: Any]] = FakeMCPServer.defaultTools
    var toolHandler: (String, [String: Any]) -> FakeMCPOutcome = FakeMCPServer.defaultToolHandler
    /// Corrections returned per requested correctionRow in get_auto_apply_row_corrections.
    var rowCorrections: [[String: Any]] = []
    /// Number of GET /mcp/sse requests the fake server has seen.
    var sseHits = 0

    var baseURL: URL {
        FakeServers.ensureStarted()
        return http!.baseURL
    }

    var rpcs: [RecordedRPC] {
        lock.lock()
        defer { lock.unlock() }
        return rpcStorage
    }

    var toolCalls: [RecordedCall] {
        lock.lock()
        defer { lock.unlock() }
        return callStorage
    }

    func reset() {
        lock.lock()
        rpcStorage = []
        callStorage = []
        lock.unlock()
        endpointEventValue = "/mcp/messages/fake"
        toolList = FakeMCPServer.defaultTools
        toolHandler = FakeMCPServer.defaultToolHandler
        rowCorrections = []
    }

    func makeClient() -> WorkerMCPClient {
        WorkerMCPClient(baseURL: baseURL, syncKey: "test-sync-key")
    }

    func startServer() throws {
        let server = FakeHTTPServer { [weak self] request, connectionId in
            guard let self else { return .respond(status: 500, headers: [:], body: Data()) }
            if request.method == "GET", request.path == "/mcp/sse" {
                self.lock.lock()
                self.sseConnectionId = connectionId
                self.sseHits += 1
                let endpointValue = self.endpointEventValue
                self.lock.unlock()
                return .stream(
                    status: 200,
                    headers: [
                        "Content-Type": "text/event-stream",
                        "Cache-Control": "no-cache",
                    ],
                    initialBody: "event: endpoint\ndata: \(endpointValue)\n\n",
                    closeAfterBody: false
                )
            }
            if request.method == "POST", request.path.hasPrefix("/mcp/messages/") {
                let status = self.handleRPC(body: request.body)
                return .respond(status: status, headers: [:], body: Data())
            }
            return .respond(status: 404, headers: [:], body: Data())
        }
        try server.start()
        http = server
    }

    // MARK: RPC dispatch

    /// Returns the HTTP status for the POST ack; stream responses are pushed separately.
    private func handleRPC(body: Data) -> Int {
        guard let rpc = try? JSONSerialization.jsonObject(with: body) as? [String: Any],
              let method = rpc["method"] as? String
        else { return 400 }
        let params = rpc["params"] as? [String: Any] ?? [:]
        recordRPC(method, params: params)
        guard let rpcId = rpc["id"] as? Int else {
            // Notification (e.g. notifications/initialized): no stream response.
            return 202
        }
        switch method {
        case "initialize":
            pushMessage([
                "jsonrpc": "2.0",
                "id": rpcId,
                "result": [
                    "protocolVersion": "2024-11-05",
                    "capabilities": [String: Any](),
                    "serverInfo": ["name": "fake-worker", "version": "0"],
                ] as [String: Any],
            ])
        case "tools/list":
            pushMessage([
                "jsonrpc": "2.0",
                "id": rpcId,
                "result": ["tools": toolList],
            ])
        case "tools/call":
            let name = params["name"] as? String ?? ""
            let args = params["arguments"] as? [String: Any] ?? [:]
            recordCall(name, args: args)
            switch toolHandler(name, args) {
            case .result(let body):
                pushMessage([
                    "jsonrpc": "2.0",
                    "id": rpcId,
                    "result": [
                        "content": [["type": "text", "text": RuleAssistantTestJSON.string(body)]],
                        "isError": false,
                    ] as [String: Any],
                ])
            case .isError(let body):
                pushMessage([
                    "jsonrpc": "2.0",
                    "id": rpcId,
                    "result": [
                        "content": [["type": "text", "text": RuleAssistantTestJSON.string(body)]],
                        "isError": true,
                    ] as [String: Any],
                ])
            case .rpcError(let code, let message):
                pushMessage([
                    "jsonrpc": "2.0",
                    "id": rpcId,
                    "error": ["code": code, "message": message],
                ])
            case .httpError(let status):
                // The ack itself is lost; nothing ever arrives on the SSE stream either.
                return status
            }
        default:
            pushMessage([
                "jsonrpc": "2.0",
                "id": rpcId,
                "error": ["code": -32601, "message": "method not found"],
            ])
        }
        return 202
    }

    func pushMessage(_ json: [String: Any]) {
        lock.lock()
        let connectionId = sseConnectionId
        lock.unlock()
        guard let connectionId else { return }
        http?.pushBytes("event: message\ndata: \(RuleAssistantTestJSON.string(json))\n\n", to: connectionId)
    }

    func recordRPC(_ method: String, params: [String: Any]) {
        lock.lock()
        rpcStorage.append(RecordedRPC(method: method, params: params))
        lock.unlock()
    }

    func recordCall(_ name: String, args: [String: Any]) {
        lock.lock()
        callStorage.append(RecordedCall(name: name, args: args))
        lock.unlock()
    }

    // MARK: Default scripting

    static let writeToolNames: [String] = [
        "add_auto_apply_correction", "add_auto_apply_context_locked_rule",
        "add_auto_apply_replacement_rule", "add_auto_apply_replacement_family",
        "tombstone_auto_apply_rule", "move_auto_apply_alias_to_family",
        "merge_auto_apply_replacement_families", "replace_auto_apply_context_locked_rule",
    ]

    static let defaultTools: [[String: Any]] = {
        let readOnly = [
            "lookup_auto_apply_policy", "list_auto_apply_families", "detect_duplicate_control_event",
            "preview_auto_apply_control_event", "get_auto_apply_reconcile_status",
            "get_auto_apply_row_corrections", "get_latest_auto_apply_manifest",
            "get_auto_apply_model", "suggest_auto_apply_tombstone", "explain_auto_apply_control_event",
        ]
        return (readOnly + writeToolNames).map {
            ["name": $0, "description": "fake tool", "inputSchema": ["type": "object"]]
        }
    }()

    static let previewOK: [String: Any] = [
        "wouldPublish": true,
        "realtimeSupported": true,
        "conflicts": [Any](),
        "skipped": [Any](),
        "unsupported": [Any](),
        "baseModelSha256": String(repeating: "b", count: 64),
        "runtimeEffect": "policies-changed",
        "policiesAdded": 1,
    ]

    static let duplicateNone: [String: Any] = [
        "alreadyApplied": false,
        "duplicatePolicy": ["found": false, "count": 0],
        "duplicateEvent": ["found": false, "count": 0],
    ]

    static let lookupNone: [String: Any] = [
        "ok": true,
        "matchedPoliciesCount": 0,
        "returnedPoliciesCount": 0,
        "policies": [Any](),
    ]

    static func writePublished(sha: String) -> [String: Any] {
        [
            "runtimeEffect": "applied",
            "realtimeOverlay": ["published": true, "modelSha256": sha],
            "warnings": [Any](),
        ]
    }

    static let defaultToolHandler: (String, [String: Any]) -> FakeMCPOutcome = { name, args in
        switch name {
        case "get_auto_apply_row_corrections":
            let rows = (args["rows"] as? [[String: Any]]) ?? []
            return .result([
                "schema": "voco.row-corrections.v1",
                "rows": rows.map {
                    ["correctionRow": $0, "corrections": FakeMCPServer.shared.rowCorrections] as [String: Any]
                },
            ])
        case "preview_auto_apply_control_event":
            return .result(previewOK)
        case "lookup_auto_apply_policy":
            return .result(lookupNone)
        case "detect_duplicate_control_event":
            return .result(duplicateNone)
        default:
            return .result([:])
        }
    }
}

// MARK: - Session assembly

@MainActor
func makeRuleAssistantSession(
    server: FakeMCPServer,
    context: RuleAssistantContext = RuleAssistantContext(
        rowPk: 42,
        timestampMs: 1_700_000_000_000,
        rawTranscript: "我們去小振家",
        text: "我們去小振家",
        recordId: "11111111-2222-3333-4444-555555555555"
    ),
    sync: RuleAssistantSyncResult = RuleAssistantSyncResult(
        outcome: .installed,
        message: "ok",
        remoteSha256: String(repeating: "a", count: 64),
        installedSha256: String(repeating: "a", count: 64)
    ),
    neighbors: [RuleAssistantContext] = [],
    guardSuggester: @escaping (String) -> [String] = { _ in [] },
    onCorrections: @escaping (RuleAssistantContext, String) -> Void = { _, _ in }
) -> RuleAssistantSession {
    RuleAssistantSession(
        context: context,
        providerFactory: { FakeGoProvider.makeClient() },
        mcpFactory: { server.makeClient() },
        syncNow: { sync },
        neighborLoader: { before, after in
            Array(neighbors.prefix(before + after))
        },
        onCorrections: onCorrections,
        guardSuggester: guardSuggester
    )
}
