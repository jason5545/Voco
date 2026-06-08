import SwiftUI
import AppKit

class MiniRecorderPanel: NSPanel {
    var allowsKeyboardInput = false

    override var canBecomeKey: Bool { allowsKeyboardInput }
    override var canBecomeMain: Bool { false }
    
    init(contentRect: NSRect) {
        super.init(
            contentRect: contentRect,
            styleMask: [.nonactivatingPanel, .fullSizeContentView],
            backing: .buffered,
            defer: false
        )
        configurePanel()
    }
    
    private func configurePanel() {
        isFloatingPanel = true
        level = .floating
        hidesOnDeactivate = false
        collectionBehavior = [.canJoinAllSpaces, .fullScreenAuxiliary]
        isMovable = true
        isMovableByWindowBackground = true
        backgroundColor = .clear
        isOpaque = false
        hasShadow = false
        titlebarAppearsTransparent = true
        titleVisibility = .hidden
        standardWindowButton(.closeButton)?.isHidden = true
    }
    
    static func calculateWindowMetrics(width requestedWidth: CGFloat? = nil, height requestedHeight: CGFloat? = nil) -> NSRect {
        guard let screen = NSScreen.main else {
            return NSRect(x: 0, y: 0, width: requestedWidth ?? 300, height: requestedHeight ?? 120)
        }

        // Fixed window size — large enough to accommodate live transcript content
        let width: CGFloat = requestedWidth ?? 300
        let height: CGFloat = requestedHeight ?? 120
        let padding: CGFloat = 24

        let visibleFrame = screen.visibleFrame
        let centerX = visibleFrame.midX
        let xPosition = centerX - (width / 2)
        let yPosition = visibleFrame.minY + padding

        return NSRect(
            x: xPosition,
            y: yPosition,
            width: width,
            height: height
        )
    }

    func show() {
        let metrics = MiniRecorderPanel.calculateWindowMetrics()
        setFrame(metrics, display: true)
        orderFrontRegardless()
    }
    
    func hide(completion: @escaping () -> Void) {
        completion()
    }
}
