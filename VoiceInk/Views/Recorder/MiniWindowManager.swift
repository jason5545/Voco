import SwiftUI
import AppKit

@MainActor
class MiniWindowManager {
    private var windowController: NSWindowController?
    private var panel: MiniRecorderPanel?

    private let makeView: () -> AnyView

    init(
        engine: VoiceInkEngine,
        recorder: Recorder,
        assistantSession: AssistantSession,
        onRecordButtonTapped: @escaping () -> Void,
        onCloseTapped: @escaping () -> Void,
        onAssistantFollowUp: @escaping (String) -> Void
    ) {
        self.makeView = {
            AnyView(
                MiniRecorderView(
                    stateProvider: engine,
                    recorder: recorder,
                    assistantSession: assistantSession,
                    onRecordButtonTapped: onRecordButtonTapped,
                    onCloseTapped: onCloseTapped,
                    onAssistantFollowUp: onAssistantFollowUp
                )
            )
        }
    }

    func show() {
        StartupTracer.checkpoint("MiniWindowManager.show_enter")
        if panel == nil { initializeWindow() }
        panel?.show()
        StartupTracer.checkpoint("MiniWindowManager.show_panel_visible")
    }

    func hide() {
        panel?.orderOut(nil)
    }

    func destroyWindow() {
        deinitializeWindow()
    }

    private func initializeWindow() {
        StartupTracer.checkpoint("initializeWindow_enter")
        deinitializeWindow()
        let metrics = MiniRecorderPanel.calculateWindowMetrics()
        StartupTracer.checkpoint("initializeWindow_metrics_calculated")
        let newPanel = MiniRecorderPanel(contentRect: metrics)
        StartupTracer.checkpoint("initializeWindow_panel_created")
        let view = makeView()
        StartupTracer.checkpoint("initializeWindow_swiftui_view_created")
        let hostingController = NSHostingController(rootView: view)
        StartupTracer.checkpoint("initializeWindow_hosting_controller_created")
        newPanel.contentView = hostingController.view
        panel = newPanel
        windowController = NSWindowController(window: newPanel)
        StartupTracer.checkpoint("initializeWindow_windowController_created")
    }

    private func deinitializeWindow() {
        panel?.orderOut(nil)
        windowController?.close()
        windowController = nil
        panel = nil
    }
}
