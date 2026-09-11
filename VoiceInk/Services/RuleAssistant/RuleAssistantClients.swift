import Foundation

// MARK: - Errors

/// Transport or protocol failure. Messages never include request bodies, headers, or provider payloads.
struct RuleAssistantTransportError: Error, Equatable, LocalizedError {
    let message: String

    init(_ message: String) {
        self.message = message
    }

    var errorDescription: String? { message }
}

/// The Worker answered a JSON-RPC request with an error object (it processed the request;
/// this is not a transport failure).
struct McpRPCError: Error, Equatable {
    let code: Int
    let serverMessage: String

    var message: String { "MCP error \(code): \(serverMessage)" }
}

/// A Worker tool answered with isError: true; body is the parsed tool text when it was JSON.
struct McpToolError: Error {
    let body: [String: Any]
    let message: String
}

enum RuleAssistantClientError: Error, Equatable {
    case toolNotAllowlisted(String)
}

// MARK: - Client protocols

protocol RuleAssistantProvider {
    func stream(
        messages: [OpenCodeMessage],
        tools: [[String: Any]],
        onDelta: (OpenCodeDelta) async throws -> Void
    ) async throws
    /// Aborts any active request; the client must not be reused afterwards.
    func close()
}

protocol RuleAssistantMCP {
    func tools() async throws -> [McpTool]
    func call(_ name: String, args: [String: Any]) async throws -> [String: Any]
    /// Non-blocking; the underlying session is torn down in the background.
    func close()
}

struct McpTool {
    let name: String
    let description: String?
    let schema: [String: Any]

    func asOpenAITool() -> [String: Any] {
        var function: [String: Any] = ["name": name]
        if let description { function["description"] = description }
        function["parameters"] = schema
        return ["type": "function", "function": function]
    }
}

// MARK: - Shared helpers

/// Incremental byte-to-String feeder for SSE streams. URLSession.AsyncBytes.lines drops empty
/// lines, which destroys SSE event boundaries, so both clients read raw bytes through this
/// instead. Handles multibyte UTF-8 sequences split across chunk boundaries.
struct SseByteFeeder {
    private var pending = Data()

    /// Appends one byte; emits at every newline (SSE is line-oriented and blank lines matter)
    /// or once chunkSize bytes are buffered.
    mutating func append(_ byte: UInt8, chunkSize: Int = 8192) -> String? {
        pending.append(byte)
        guard byte == 0x0A || pending.count >= chunkSize else { return nil }
        return takeDecodedPrefix()
    }

    /// Flushes whatever remains (lossy only for genuinely malformed trailing bytes at EOF).
    mutating func finish() -> String {
        defer { pending.removeAll() }
        return String(decoding: pending, as: UTF8.self)
    }

    private mutating func takeDecodedPrefix() -> String {
        while !pending.isEmpty {
            var count = pending.count
            while count > 0 {
                if let decoded = String(data: pending.prefix(count), encoding: .utf8) {
                    pending.removeFirst(count)
                    return decoded
                }
                count -= 1
            }
            // A valid UTF-8 sequence is at most 4 bytes; longer undecodable prefixes are malformed.
            if pending.count > 4 {
                pending.removeFirst()
            } else {
                break
            }
        }
        return ""
    }
}

/// Refuses to follow redirects for both the provider and the Worker MCP client.
final class RuleAssistantNoRedirectDelegate: NSObject, URLSessionTaskDelegate, Sendable {
    func urlSession(
        _ session: URLSession,
        task: URLSessionTask,
        willPerformHTTPRedirection response: HTTPURLResponse,
        newRequest request: URLRequest,
        completionHandler: @escaping (URLRequest?) -> Void
    ) {
        completionHandler(nil)
    }
}

/// Serializes async calls without actor reentrancy.
final class AsyncSerialGate {
    private let lock = NSLock()
    private var busy = false
    private var waiters: [CheckedContinuation<Void, Never>] = []

    func acquire() async {
        await withCheckedContinuation { continuation in
            lock.lock()
            if !busy {
                busy = true
                lock.unlock()
                continuation.resume()
            } else {
                waiters.append(continuation)
                lock.unlock()
            }
        }
    }

    func release() {
        lock.lock()
        if !waiters.isEmpty {
            let next = waiters.removeFirst()
            lock.unlock()
            next.resume()
        } else {
            busy = false
            lock.unlock()
        }
    }
}

// MARK: - OpenCode Go provider client

final class OpenCodeGoClient: RuleAssistantProvider {
    private let apiKey: String
    let sessionId: String
    private let endpoint: URL
    private let configuration: URLSessionConfiguration

    private let stateLock = NSLock()
    private var closed = false
    private var session: URLSession?

    init(
        apiKey: String,
        sessionId: String = UUID().uuidString,
        endpoint: URL = RuleAssistantConstants.endpointURL,
        configuration: URLSessionConfiguration = .ephemeral
    ) {
        self.apiKey = apiKey
        self.sessionId = sessionId
        self.endpoint = endpoint
        let copy = configuration.copy() as? URLSessionConfiguration ?? .ephemeral
        copy.timeoutIntervalForRequest = Self.readTimeout
        self.configuration = copy
    }

    func close() {
        stateLock.lock()
        closed = true
        let current = session
        session = nil
        stateLock.unlock()
        current?.invalidateAndCancel()
    }

    private var isClosed: Bool {
        stateLock.lock()
        defer { stateLock.unlock() }
        return closed
    }

    private func makeSession() -> URLSession {
        stateLock.lock()
        defer { stateLock.unlock() }
        if let session { return session }
        let created = URLSession(
            configuration: configuration,
            delegate: RuleAssistantNoRedirectDelegate(),
            delegateQueue: nil
        )
        session = created
        return created
    }

    func stream(
        messages: [OpenCodeMessage],
        tools: [[String: Any]],
        onDelta: (OpenCodeDelta) async throws -> Void
    ) async throws {
        if isClosed {
            throw RuleAssistantTransportError(String(localized: "Stopped."))
        }
        var request = URLRequest(url: endpoint)
        request.httpMethod = "POST"
        request.cachePolicy = .reloadIgnoringLocalCacheData
        request.setValue("Bearer \(apiKey)", forHTTPHeaderField: "Authorization")
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.setValue("text/event-stream", forHTTPHeaderField: "Accept")
        request.setValue(RuleAssistantConstants.userAgent, forHTTPHeaderField: "User-Agent")
        request.setValue(sessionId, forHTTPHeaderField: "x-opencode-session")
        var body: [String: Any] = [
            "model": RuleAssistantConstants.model,
            "stream": true,
            "messages": messages.map { $0.toJSON() },
        ]
        if !tools.isEmpty {
            body["tools"] = tools
            body["tool_choice"] = "auto"
        }
        request.httpBody = try JSONSerialization.data(withJSONObject: body)

        let session = makeSession()
        let bytes: URLSession.AsyncBytes
        let response: URLResponse
        do {
            (bytes, response) = try await session.bytes(for: request)
        } catch {
            if isClosed { throw RuleAssistantTransportError(String(localized: "Stopped.")) }
            throw RuleAssistantTransportError(
                String(localized: "OpenCode Go connection failed: \(type(of: error))")
            )
        }
        guard let http = response as? HTTPURLResponse else {
            throw RuleAssistantTransportError("OpenCode Go: invalid response")
        }
        guard (200...299).contains(http.statusCode) else {
            // Never surface provider response bodies: they may echo request metadata.
            throw RuleAssistantTransportError("OpenCode Go HTTP \(http.statusCode)")
        }

        var ended = false
        var failure: String?
        var pendingDeltas: [OpenCodeDelta] = []
        let deltaParser = OpenCodeDeltaParser { delta in
            pendingDeltas.append(delta)
        }
        let sseParser = SseEventParser { event in
            if event.data == "[DONE]" {
                ended = true
            } else if !ended {
                deltaParser.accept(event)
            }
        }

        let deadline = ContinuousClock.now + Self.totalTimeout
        var totalChars = 0
        do {
            var feeder = SseByteFeeder()
            for try await byte in bytes {
                if isClosed { throw RuleAssistantTransportError(String(localized: "Stopped.")) }
                totalChars += 1
                if totalChars > Self.maxStreamChars {
                    throw RuleAssistantTransportError(String(localized: "The provider stream exceeded the safety length limit."))
                }
                if ContinuousClock.now > deadline {
                    throw RuleAssistantTransportError(String(localized: "The provider stream exceeded the total time limit."))
                }
                guard let chunk = feeder.append(byte) else { continue }
                try sseParser.feed(chunk)
                let deltas = pendingDeltas
                pendingDeltas.removeAll()
                for delta in deltas {
                    if let error = delta.error {
                        if failure == nil { failure = error }
                    } else {
                        try await onDelta(delta)
                    }
                }
                if let failure { throw RuleAssistantTransportError(failure) }
                if ended { break }
            }
            let tail = feeder.finish()
            if !tail.isEmpty {
                try sseParser.feed(tail)
                let deltas = pendingDeltas
                pendingDeltas.removeAll()
                for delta in deltas {
                    if let error = delta.error {
                        if failure == nil { failure = error }
                    } else {
                        try await onDelta(delta)
                    }
                }
            }
        } catch let error as RuleAssistantTransportError {
            throw error
        } catch let error as SseParserError {
            throw RuleAssistantTransportError(error.message)
        } catch is CancellationError {
            throw RuleAssistantTransportError(String(localized: "Stopped."))
        } catch {
            if isClosed { throw RuleAssistantTransportError(String(localized: "Stopped.")) }
            throw error
        }
        try sseParser.finish()
        deltaParser.finish()
        let drained = pendingDeltas
        pendingDeltas.removeAll()
        for delta in drained {
            if let error = delta.error {
                if failure == nil { failure = error }
            } else {
                try await onDelta(delta)
            }
        }
        if let failure { throw RuleAssistantTransportError(failure) }
        if !ended {
            throw RuleAssistantTransportError(String(localized: "The provider stream ended before [DONE]."))
        }
    }

    private static let readTimeout: TimeInterval = 90
    private static let totalTimeout: Duration = .seconds(240)
    private static let maxStreamChars = 2_000_000
}

// MARK: - Worker MCP client (legacy HTTP+SSE)

/// Legacy SSE MCP client for the Worker: GET /mcp/sse yields an endpoint event, JSON-RPC requests are
/// POSTed there (HTTP 202 is only an ack) and responses arrive on the SSE stream matched by id.
/// Only the configured Worker origin is accepted; the key is never sent anywhere else.
final class WorkerMCPClient: RuleAssistantMCP {
    static let allowedTools: Set<String> = [
        "lookup_auto_apply_policy", "list_auto_apply_families", "detect_duplicate_control_event",
        "preview_auto_apply_control_event", "get_auto_apply_reconcile_status", "get_auto_apply_row_corrections",
        "get_latest_auto_apply_manifest", "get_auto_apply_model", "suggest_auto_apply_tombstone",
        "explain_auto_apply_control_event",
        "add_auto_apply_correction", "add_auto_apply_context_locked_rule",
        "add_auto_apply_replacement_rule", "add_auto_apply_replacement_family", "tombstone_auto_apply_rule",
        "move_auto_apply_alias_to_family", "merge_auto_apply_replacement_families",
        "replace_auto_apply_context_locked_rule",
    ]

    private let baseURL: URL
    private let syncKey: String
    private let configuration: URLSessionConfiguration

    private let lock = NSLock()
    private var closed = false
    private var session: URLSession?
    private var readerTask: Task<Void, Never>?
    private var eventBuffer: [SseEvent] = []
    private var readerFailure: RuleAssistantTransportError?
    private var readerEnded = false
    private var messageURL: URL?
    private var rpcId: Int64 = 0
    private var initialized = false
    private var cachedTools: [McpTool]?
    private let gate = AsyncSerialGate()

    init(
        baseURL: URL,
        syncKey: String,
        configuration: URLSessionConfiguration = .ephemeral
    ) {
        // Only the configured Worker origin is allowed; plain HTTP is for local test harnesses only.
        self.baseURL = URL(string: baseURL.absoluteString.trimmingCharacters(in: CharacterSet(charactersIn: "/")))
            ?? baseURL
        self.syncKey = syncKey
        let copy = configuration.copy() as? URLSessionConfiguration ?? .ephemeral
        copy.timeoutIntervalForRequest = Self.sseReadTimeout
        self.configuration = copy
    }

    // MARK: Public API

    func tools() async throws -> [McpTool] {
        await gate.acquire()
        defer { gate.release() }
        if isClosed { throw closedError() }
        if let cached = locked({ cachedTools }) { return cached }
        let result = try await rpcInternal(method: "tools/list", params: [:])
        let parsed: [McpTool] = result.raDictArray("tools").compactMap { item in
            guard let name = item.raString("name"), Self.allowedTools.contains(name) else { return nil }
            return McpTool(
                name: name,
                description: item.raNonBlankString("description"),
                schema: item.raDict("inputSchema") ?? ["type": "object"]
            )
        }
        locked { cachedTools = parsed }
        return parsed
    }

    func call(_ name: String, args: [String: Any]) async throws -> [String: Any] {
        guard Self.allowedTools.contains(name) else {
            throw RuleAssistantClientError.toolNotAllowlisted(name)
        }
        await gate.acquire()
        defer { gate.release() }
        let result: [String: Any]
        do {
            result = try await rpcInternal(method: "tools/call", params: ["name": name, "arguments": args])
        } catch let rpcError as McpRPCError {
            // e.g. -32602 invalid params: surface it like an isError tool result so the model (or the draft
            // gate) handles it instead of the whole turn failing.
            throw McpToolError(
                body: ["isError": true, "code": rpcError.code, "reason": rpcError.message],
                message: "Worker rejected the \(name) call (\(rpcError.code)): \(rpcError.serverMessage)"
            )
        }
        return try unwrapToolResult(result)
    }

    /// Non-blocking; the socket teardown happens with task cancellation.
    func close() {
        lock.lock()
        closed = true
        let reader = readerTask
        readerTask = nil
        let currentSession = session
        session = nil
        lock.unlock()
        reader?.cancel()
        currentSession?.invalidateAndCancel()
    }

    // MARK: Session lifecycle

    private func locked<T>(_ body: () -> T) -> T {
        lock.lock()
        defer { lock.unlock() }
        return body()
    }

    private var isClosed: Bool { locked { closed } }

    private func closedError() -> RuleAssistantTransportError {
        RuleAssistantTransportError(String(localized: "The MCP session is closed."))
    }

    private func makeSession() -> URLSession {
        locked {
            if let session { return session }
            let created = URLSession(
                configuration: configuration,
                delegate: RuleAssistantNoRedirectDelegate(),
                delegateQueue: nil
            )
            session = created
            return created
        }
    }

    private func deliver(_ event: SseEvent) {
        locked { eventBuffer.append(event) }
    }

    private func readerTerminated(_ error: RuleAssistantTransportError?) {
        locked {
            if let error {
                readerFailure = error
            } else {
                readerEnded = true
            }
        }
    }

    /// Forget a broken stream so the next call fails fast (after close) or reconnects (before close).
    private func dropSession() {
        let reader = locked { () -> Task<Void, Never>? in
            let task = readerTask
            readerTask = nil
            messageURL = nil
            initialized = false
            eventBuffer.removeAll()
            readerFailure = nil
            readerEnded = false
            return task
        }
        reader?.cancel()
    }

    private func ensureSession() async throws {
        if isClosed { throw closedError() }
        let alive = locked { messageURL != nil && initialized && readerTask != nil && !readerEnded && readerFailure == nil }
        if alive { return }

        if locked({ messageURL == nil || readerTask == nil || readerEnded || readerFailure != nil }) {
            dropSession()
            guard let scheme = baseURL.scheme?.lowercased(),
                  scheme == "https" || baseURL.host == "127.0.0.1" || baseURL.host == "localhost"
            else {
                throw RuleAssistantTransportError("The Worker URL must use https")
            }
            var request = URLRequest(url: baseURL.appendingPathComponent("mcp/sse"))
            request.httpMethod = "GET"
            request.setValue("Bearer \(syncKey)", forHTTPHeaderField: "Authorization")
            request.setValue("text/event-stream", forHTTPHeaderField: "Accept")
            request.setValue(RuleAssistantConstants.userAgent, forHTTPHeaderField: "User-Agent")
            // SSE must never be cached; without this, in some processes URLSession withholds the
            // stream response indefinitely.
            request.cachePolicy = .reloadIgnoringLocalCacheData

            let session = makeSession()
            let task = Task { [weak self] in
                guard let self else { return }
                do {
                    let (bytes, response) = try await session.bytes(for: request)
                    guard let http = response as? HTTPURLResponse else {
                        throw RuleAssistantTransportError("MCP SSE: invalid response")
                    }
                    guard http.statusCode == 200 else {
                        throw RuleAssistantTransportError("MCP SSE HTTP \(http.statusCode)")
                    }
                    let contentType = http.value(forHTTPHeaderField: "Content-Type") ?? ""
                    guard contentType.hasPrefix("text/event-stream") else {
                        throw RuleAssistantTransportError("MCP SSE content type mismatch")
                    }
                    let parser = SseEventParser { [weak self] event in
                        self?.deliver(event)
                    }
                    var feeder = SseByteFeeder()
                    for try await byte in bytes {
                        try Task.checkCancellation()
                        if let chunk = feeder.append(byte) {
                            try parser.feed(chunk)
                        }
                    }
                    let tail = feeder.finish()
                    if !tail.isEmpty {
                        try parser.feed(tail)
                    }
                    try parser.finish()
                    readerTerminated(nil)
                } catch is CancellationError {
                    // close()/dropSession() cancelled the reader; waiters are released by their own checks.
                } catch let error as RuleAssistantTransportError {
                    readerTerminated(error)
                } catch let error as SseParserError {
                    readerTerminated(RuleAssistantTransportError(error.message))
                } catch {
                    let message = locked { closed }
                        ? String(localized: "The MCP session is closed.")
                        : String(localized: "MCP SSE connection failed: \(type(of: error))")
                    readerTerminated(RuleAssistantTransportError(message))
                }
            }
            locked { readerTask = task }
            let endpoint = try await readEndpoint()
            let validated = try validateEndpoint(origin: baseURL, endpoint: endpoint)
            locked { messageURL = validated }
        }

        if !locked({ initialized }) {
            let initResult = try await rpc(method: "initialize", params: [
                "protocolVersion": "2024-11-05",
                "capabilities": [String: Any](),
                "clientInfo": [
                    "name": RuleAssistantConstants.mcpClientName,
                    "version": RuleAssistantConstants.appVersion,
                ],
            ])
            guard let protocolVersion = initResult.raNonBlankString("protocolVersion"),
                  initResult.raDict("capabilities") != nil
            else {
                throw RuleAssistantTransportError("MCP initialize response is invalid")
            }
            // JSON-RPC notification: no id, no response on the stream.
            try await post([
                "jsonrpc": "2.0",
                "method": "notifications/initialized",
                "params": [String: Any](),
            ])
            locked { initialized = true }
        }
    }

    // MARK: JSON-RPC

    private func rpcInternal(method: String, params: [String: Any]) async throws -> [String: Any] {
        try await ensureSession()
        return try await rpc(method: method, params: params)
    }

    private func rpc(method: String, params: [String: Any]) async throws -> [String: Any] {
        let id = locked { () -> Int64 in
            rpcId += 1
            return rpcId
        }
        let request: [String: Any] = [
            "jsonrpc": "2.0",
            "id": id,
            "method": method,
            "params": params,
        ]
        try await post(request)
        let envelope = try await readRpcEnvelope(id: id)
        return try Self.requireRpcResult(envelope)
    }

    private func post(_ request: [String: Any]) async throws {
        if isClosed { throw closedError() }
        guard let url = locked({ messageURL }) else {
            throw RuleAssistantTransportError("MCP session is not connected")
        }
        var urlRequest = URLRequest(url: url)
        urlRequest.httpMethod = "POST"
        urlRequest.timeoutInterval = Self.postReadTimeout
        urlRequest.setValue("Bearer \(syncKey)", forHTTPHeaderField: "Authorization")
        urlRequest.setValue("application/json", forHTTPHeaderField: "Content-Type")
        urlRequest.setValue(RuleAssistantConstants.userAgent, forHTTPHeaderField: "User-Agent")
        urlRequest.httpBody = try JSONSerialization.data(withJSONObject: request)
        do {
            let (_, response) = try await makeSession().data(for: urlRequest)
            guard let http = response as? HTTPURLResponse else {
                throw RuleAssistantTransportError("MCP POST: invalid response")
            }
            guard (200...299).contains(http.statusCode) else {
                throw RuleAssistantTransportError("MCP HTTP \(http.statusCode)")
            }
        } catch let error as RuleAssistantTransportError {
            throw error
        } catch {
            if isClosed { throw closedError() }
            throw RuleAssistantTransportError(String(localized: "MCP POST failed: \(type(of: error))"))
        }
    }

    private func readRpcEnvelope(id: Int64) async throws -> [String: Any] {
        let deadline = ContinuousClock.now + Self.rpcTimeout
        while true {
            let event = try await nextEvent(deadline: deadline, timeoutMessage: "MCP response timeout")
            if let type = event.event, type != "message" { continue }
            guard let json = RAJSON.parseObject(event.data) else { continue }
            if !(json["id"] is NSNull), json.raInt64("id") == id {
                return json
            }
        }
    }

    private func readEndpoint() async throws -> String {
        let deadline = ContinuousClock.now + Self.endpointTimeout
        while true {
            let event = try await nextEvent(deadline: deadline, timeoutMessage: "MCP endpoint timeout")
            if event.event == "endpoint" {
                return event.data.trimmingCharacters(in: .whitespacesAndNewlines)
            }
        }
    }

    /// Polls the reader buffer; wakes on close, stream end, deadline, or the next event.
    private func nextEvent(deadline: ContinuousClock.Instant, timeoutMessage: String) async throws -> SseEvent {
        while true {
            if isClosed { throw closedError() }
            enum Next {
                case event(SseEvent)
                case failed(RuleAssistantTransportError)
                case ended
                case empty
            }
            let next: Next = locked {
                if !eventBuffer.isEmpty {
                    return .event(eventBuffer.removeFirst())
                }
                if let readerFailure { return .failed(readerFailure) }
                if readerEnded { return .ended }
                return .empty
            }
            switch next {
            case .event(let event):
                return event
            case .failed(let error):
                dropSession()
                throw error
            case .ended:
                dropSession()
                if isClosed { throw closedError() }
                throw RuleAssistantTransportError("MCP SSE closed before response")
            case .empty:
                break
            }
            if ContinuousClock.now > deadline {
                dropSession()
                throw RuleAssistantTransportError(timeoutMessage)
            }
            try await Task.sleep(for: .milliseconds(Self.pollMs))
        }
    }

    private static func requireRpcResult(_ envelope: [String: Any]) throws -> [String: Any] {
        guard envelope.raString("jsonrpc") == "2.0" else {
            throw RuleAssistantTransportError("Invalid MCP JSON-RPC envelope")
        }
        if let error = envelope.raDict("error") {
            let code = Int(error.raInt64("code") ?? 0)
            let message = String((error.raString("message") ?? "").prefix(300))
            throw McpRPCError(code: code, serverMessage: message)
        }
        guard let result = envelope.raDict("result") else {
            throw RuleAssistantTransportError("MCP response has no result")
        }
        return result
    }

    // MARK: Tool result unwrapping

    private func unwrapToolResult(_ result: [String: Any]) throws -> [String: Any] {
        guard let content = result.raArray("content") else {
            throw RuleAssistantTransportError("MCP tool result has no content")
        }
        guard let first = content.first as? [String: Any], let text = first.raString("text") else {
            throw RuleAssistantTransportError("MCP tool result has no text content")
        }
        guard text.count <= Self.maxToolResponseChars else {
            throw RuleAssistantTransportError("MCP response exceeded safety limit")
        }
        let body = RAJSON.parseObject(text) ?? ["text": text]
        if result.raBool("isError") {
            throw McpToolError(body: body, message: Self.toolErrorMessage(body))
        }
        return body
    }

    private static func toolErrorMessage(_ body: [String: Any]) -> String {
        let reason = body.raNonBlankString("reason")
            ?? body.raNonBlankString("message")
            ?? body.raNonBlankString("error")
            ?? body.raNonBlankString("text")
            ?? "MCP tool failed"
        return String(reason.prefix(300))
    }

    // MARK: Endpoint validation

    private static let messagePathPattern = try! NSRegularExpression(pattern: #"^/mcp/messages/[0-9A-Za-z._-]{1,128}$"#)

    private func validateEndpoint(origin: URL, endpoint: String) throws -> URL {
        let resolved: URL? = {
            if let absolute = URL(string: endpoint), absolute.scheme != nil {
                return absolute.standardized
            }
            return URL(string: endpoint, relativeTo: origin)?.absoluteURL.standardized
        }()
        guard let resolved else {
            throw RuleAssistantTransportError("MCP endpoint is not a valid URL")
        }
        let sameOrigin = resolved.scheme?.lowercased() == origin.scheme?.lowercased()
            && resolved.host?.lowercased() == origin.host?.lowercased()
            && Self.effectivePort(resolved) == Self.effectivePort(origin)
        guard sameOrigin else {
            throw RuleAssistantTransportError("MCP endpoint origin mismatch")
        }
        guard resolved.user == nil, resolved.password == nil else {
            throw RuleAssistantTransportError("MCP endpoint must not carry credentials")
        }
        guard resolved.query == nil, resolved.fragment == nil else {
            throw RuleAssistantTransportError("MCP endpoint must not carry query parameters")
        }
        let path = resolved.path
        guard Self.messagePathPattern.firstMatch(in: path, range: NSRange(path.startIndex..., in: path)) != nil else {
            throw RuleAssistantTransportError("MCP endpoint path mismatch")
        }
        return resolved
    }

    private static func effectivePort(_ url: URL) -> Int {
        if let port = url.port { return port }
        return url.scheme?.lowercased() == "https" ? 443 : 80
    }

    // MARK: Limits

    private static let maxToolResponseChars = 128_000
    // The Worker writes a keepalive comment every 15 s; silence longer than this means the stream is dead.
    private static let sseReadTimeout: TimeInterval = 45
    private static let postReadTimeout: TimeInterval = 30
    private static let pollMs = 100
    private static let endpointTimeout: Duration = .seconds(30)
    private static let rpcTimeout: Duration = .seconds(90)
}
