import Foundation

/// High-precision tracer for measuring the hotkey-press → mini-window-visible pipeline.
/// Writes timestamped checkpoints to ~/Library/Logs/Voco/startup-trace.log.
/// Usage mirrors `ChinesePostProcessingService.debugLog()`.
///
/// Typical flow:
/// ```
/// StartupTracer.begin("hotkey_press")
/// StartupTracer.checkpoint("toggleMiniRecorder_enter")
/// StartupTracer.checkpoint("sound_played")
/// ...
/// StartupTracer.end("recording_started")
/// ```
enum StartupTracer {

    // MARK: - Session State

    /// High-precision session start time (CFAbsoluteTime, ~µs resolution)
    private(set) static var sessionStart: CFAbsoluteTime = 0
    private(set) static var sessionID: String = ""
    private static var isActive = false

    // MARK: - Log File

    private static let logURL: URL = {
        let dir = FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent("Library/Logs/Voco", isDirectory: true)
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        return dir.appendingPathComponent("startup-trace.log")
    }()

    // MARK: - Public API

    /// Start a new tracing session. Resets the timer.
    static func begin(_ label: String) {
        sessionStart = CFAbsoluteTimeGetCurrent()
        sessionID = String(format: "%04X", Int.random(in: 0...0xFFFF))
        isActive = true
        writeEntry(elapsed: 0, label: "BEGIN: \(label)")
    }

    /// Record an intermediate checkpoint with elapsed time since `begin`.
    static func checkpoint(_ label: String) {
        guard isActive else { return }
        let elapsed = (CFAbsoluteTimeGetCurrent() - sessionStart) * 1000 // ms
        writeEntry(elapsed: elapsed, label: label)
    }

    /// End the tracing session.
    static func end(_ label: String) {
        guard isActive else { return }
        let elapsed = (CFAbsoluteTimeGetCurrent() - sessionStart) * 1000
        writeEntry(elapsed: elapsed, label: "END: \(label)")
        isActive = false
    }

    // MARK: - File I/O

    private static func writeEntry(elapsed: Double, label: String) {
        let timestamp = ISO8601DateFormatter().string(from: Date())
        let entry = "[\(timestamp)] [\(sessionID)] +\(String(format: "%7.2f", elapsed))ms  \(label)\n"
        guard let data = entry.data(using: .utf8) else { return }

        if FileManager.default.fileExists(atPath: logURL.path) {
            if let handle = try? FileHandle(forWritingTo: logURL) {
                handle.seekToEndOfFile()
                handle.write(data)
                handle.closeFile()
            }
        } else {
            try? data.write(to: logURL)
        }
    }
}
