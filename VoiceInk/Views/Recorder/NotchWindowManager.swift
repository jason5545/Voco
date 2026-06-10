import SwiftUI
import AppKit

@MainActor
class NotchWindowManager {
    private var windowController: NSWindowController?
    private var panel: NotchRecorderPanel?

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
                NotchRecorderView(
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
        StartupTracer.checkpoint("NotchWindowManager.show_enter")
        if panel == nil { initializeWindow() }
        panel?.show()
        StartupTracer.checkpoint("NotchWindowManager.show_panel_visible")
    }

    func hide() {
        panel?.orderOut(nil)
    }

    func destroyWindow() {
        deinitializeWindow()
    }

    private func initializeWindow() {
        StartupTracer.checkpoint("NotchWindowManager.initializeWindow_enter")
        deinitializeWindow()
        let metrics = NotchRecorderPanel.calculateWindowMetrics()
        StartupTracer.checkpoint("NotchWindowManager.metrics_calculated")
        let newPanel = NotchRecorderPanel(contentRect: metrics.frame)
        StartupTracer.checkpoint("NotchWindowManager.panel_created")
        let view = makeView()
        StartupTracer.checkpoint("NotchWindowManager.swiftui_view_created")
        let hostingController = NotchRecorderHostingController(rootView: view)
        StartupTracer.checkpoint("NotchWindowManager.hosting_controller_created")
        newPanel.contentView = hostingController.view
        panel = newPanel
        windowController = NSWindowController(window: newPanel)
        StartupTracer.checkpoint("NotchWindowManager.windowController_created")
    }

    private func deinitializeWindow() {
        panel?.orderOut(nil)
        windowController?.close()
        windowController = nil
        panel = nil
    }

}
