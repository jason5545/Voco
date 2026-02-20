import Foundation
import AppKit
import os

class ActiveWindowService: ObservableObject {
    static let shared = ActiveWindowService()
    @Published var currentApplication: NSRunningApplication?
    private var enhancementService: AIEnhancementService?
    private let browserURLService = BrowserURLService.shared
    private var whisperState: WhisperState?
    
    private let logger = Logger(
        subsystem: AppIdentifiers.subsystem,
        category: "browser.detection"
    )
    
    private init() {}
    
    func configure(with enhancementService: AIEnhancementService) {
        self.enhancementService = enhancementService
    }
    
    func configureWhisperState(_ whisperState: WhisperState) {
        self.whisperState = whisperState
    }
    
    func applyConfiguration(powerModeId: UUID? = nil) async {
        let startTime = Date()

        if let powerModeId = powerModeId,
           let config = PowerModeManager.shared.getConfiguration(with: powerModeId) {
            await MainActor.run {
                PowerModeManager.shared.setActiveConfiguration(config)
            }
            await PowerModeSessionManager.shared.beginSession(with: config)
            let duration = Date().timeIntervalSince(startTime)
            if duration > 0.4 {
                logger.notice("⏱️ applyConfiguration (direct id) took \(String(format: "%.3f", duration))s")
            }
            return
        }

        guard let frontmostApp = NSWorkspace.shared.frontmostApplication,
              let bundleIdentifier = frontmostApp.bundleIdentifier else {
            return
        }

        await MainActor.run {
            currentApplication = frontmostApp
        }

        var configToApply: PowerModeConfig?

        let shouldResolveBrowserURL = PowerModeManager.shared.hasEnabledURLConfigurations
        if shouldResolveBrowserURL,
           let browserType = BrowserType.allCases.first(where: { $0.bundleIdentifier == bundleIdentifier }) {
            let browserLookupStart = Date()
            do {
                let currentURL = try await browserURLService.getCurrentURL(from: browserType)
                if let config = PowerModeManager.shared.getConfigurationForURL(currentURL) {
                    configToApply = config
                }
                let browserLookupDuration = Date().timeIntervalSince(browserLookupStart)
                if browserLookupDuration > 0.4 {
                    logger.notice("⏱️ Browser URL lookup for \(browserType.displayName) took \(String(format: "%.3f", browserLookupDuration))s")
                }
            } catch {
                logger.error("❌ Failed to get URL from \(browserType.displayName): \(error.localizedDescription)")
            }
        }

        if configToApply == nil {
            configToApply = PowerModeManager.shared.getConfigurationForApp(bundleIdentifier)
        }

        if configToApply == nil {
            configToApply = PowerModeManager.shared.getDefaultConfiguration()
        }

        if let config = configToApply {
            await MainActor.run {
                PowerModeManager.shared.setActiveConfiguration(config)
            }
            await PowerModeSessionManager.shared.beginSession(with: config)
        }

        let totalDuration = Date().timeIntervalSince(startTime)
        if totalDuration > 0.4 {
            logger.notice("⏱️ applyConfiguration total took \(String(format: "%.3f", totalDuration))s")
        }
    }
}
