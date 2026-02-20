import Foundation
import AppKit
import os
import Darwin

enum BrowserType {
    case safari
    case arc
    case chrome
    case edge
    case firefox
    case brave
    case opera
    case vivaldi
    case orion
    case zen
    case yandex
    
    var scriptName: String {
        switch self {
        case .safari: return "safariURL"
        case .arc: return "arcURL"
        case .chrome: return "chromeURL"
        case .edge: return "edgeURL"
        case .firefox: return "firefoxURL"
        case .brave: return "braveURL"
        case .opera: return "operaURL"
        case .vivaldi: return "vivaldiURL"
        case .orion: return "orionURL"
        case .zen: return "zenURL"
        case .yandex: return "yandexURL"
        }
    }
    
    var bundleIdentifier: String {
        switch self {
        case .safari: return "com.apple.Safari"
        case .arc: return "company.thebrowser.Browser"
        case .chrome: return "com.google.Chrome"
        case .edge: return "com.microsoft.edgemac"
        case .firefox: return "org.mozilla.firefox"
        case .brave: return "com.brave.Browser"
        case .opera: return "com.operasoftware.Opera"
        case .vivaldi: return "com.vivaldi.Vivaldi"
        case .orion: return "com.kagi.kagimacOS"
        case .zen: return "app.zen-browser.zen"
        case .yandex: return "ru.yandex.desktop.yandex-browser"
        }
    }
    
    var displayName: String {
        switch self {
        case .safari: return "Safari"
        case .arc: return "Arc"
        case .chrome: return "Google Chrome"
        case .edge: return "Microsoft Edge"
        case .firefox: return "Firefox"
        case .brave: return "Brave"
        case .opera: return "Opera"
        case .vivaldi: return "Vivaldi"
        case .orion: return "Orion"
        case .zen: return "Zen Browser"
        case .yandex: return "Yandex Browser"
        }
    }
    
    static var allCases: [BrowserType] {
        [.safari, .arc, .chrome, .edge, .brave, .opera, .vivaldi, .orion, .yandex]
    }
    
    static var installedBrowsers: [BrowserType] {
        allCases.filter { browser in
            let workspace = NSWorkspace.shared
            return workspace.urlForApplication(withBundleIdentifier: browser.bundleIdentifier) != nil
        }
    }
}

enum BrowserURLError: Error {
    case scriptNotFound
    case executionFailed
    case executionTimedOut
    case browserNotRunning
    case noActiveWindow
    case noActiveTab
}

class BrowserURLService {
    static let shared = BrowserURLService()
    
    private let logger = Logger(
        subsystem: AppIdentifiers.subsystem,
        category: "browser.applescript"
    )
    
    private init() {}
    
    func getCurrentURL(from browser: BrowserType) async throws -> String {
        guard let scriptURL = Bundle.main.url(forResource: browser.scriptName, withExtension: "scpt") else {
            logger.error("❌ AppleScript file not found: \(browser.scriptName).scpt")
            throw BrowserURLError.scriptNotFound
        }
        
        logger.debug("🔍 Attempting to execute AppleScript for \(browser.displayName)")
        
        // Check if browser is running
        if !isRunning(browser) {
            logger.error("❌ Browser not running: \(browser.displayName)")
            throw BrowserURLError.browserNotRunning
        }

        do {
            logger.debug("▶️ Executing AppleScript for \(browser.displayName)")
            let output = try await executeAppleScript(scriptPath: scriptURL.path, timeout: 1.2)

            if output.isEmpty {
                logger.error("❌ Empty output from AppleScript for \(browser.displayName)")
                throw BrowserURLError.noActiveTab
            }

            // Check if output contains error messages
            if output.lowercased().contains("error") {
                logger.error("❌ AppleScript error for \(browser.displayName): \(output)")
                throw BrowserURLError.executionFailed
            }

            logger.debug("✅ Successfully retrieved URL from \(browser.displayName): \(output)")
            return output
        } catch let error as BrowserURLError {
            logger.error("❌ AppleScript failed for \(browser.displayName): \(String(describing: error))")
            throw error
        } catch {
            logger.error("❌ AppleScript execution failed for \(browser.displayName): \(error.localizedDescription)")
            throw BrowserURLError.executionFailed
        }
    }
    
    func isRunning(_ browser: BrowserType) -> Bool {
        let workspace = NSWorkspace.shared
        let runningApps = workspace.runningApplications
        let isRunning = runningApps.contains { $0.bundleIdentifier == browser.bundleIdentifier }
        logger.debug("\(browser.displayName) running status: \(isRunning)")
        return isRunning
    }

    private func executeAppleScript(scriptPath: String, timeout: TimeInterval) async throws -> String {
        try await withCheckedThrowingContinuation { continuation in
            DispatchQueue.global(qos: .userInitiated).async {
                let task = Process()
                task.executableURL = URL(fileURLWithPath: "/usr/bin/osascript")
                task.arguments = [scriptPath]

                let pipe = Pipe()
                task.standardOutput = pipe
                task.standardError = pipe

                let exitSemaphore = DispatchSemaphore(value: 0)
                task.terminationHandler = { _ in
                    exitSemaphore.signal()
                }

                do {
                    try task.run()
                } catch {
                    continuation.resume(throwing: BrowserURLError.executionFailed)
                    return
                }

                let timedOut = exitSemaphore.wait(timeout: .now() + timeout) == .timedOut
                if timedOut {
                    if task.isRunning {
                        task.terminate()
                        _ = exitSemaphore.wait(timeout: .now() + 0.15)
                        if task.isRunning {
                            kill(task.processIdentifier, SIGKILL)
                            _ = exitSemaphore.wait(timeout: .now() + 0.15)
                        }
                    }
                    continuation.resume(throwing: BrowserURLError.executionTimedOut)
                    return
                }

                let outputData = pipe.fileHandleForReading.readDataToEndOfFile()
                guard let output = String(data: outputData, encoding: .utf8)?
                    .trimmingCharacters(in: .whitespacesAndNewlines) else {
                    continuation.resume(throwing: BrowserURLError.executionFailed)
                    return
                }

                continuation.resume(returning: output)
            }
        }
    }
}
