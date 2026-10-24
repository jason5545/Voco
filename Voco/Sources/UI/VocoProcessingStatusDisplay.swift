import SwiftUI

struct ProcessingStatusDisplay: View {
    enum Mode {
        case transcribing
        case enhancing
        case editing
    }

    let mode: Mode
    let color: Color

    private var label: LocalizedStringKey {
        switch mode {
        case .transcribing: return "Transcribing"
        case .enhancing: return "Enhancing"
        case .editing: return "Editing"
        }
    }

    private var animationSpeed: Double {
        switch mode {
        case .transcribing: return 0.18
        case .enhancing: return 0.22
        case .editing: return 0.22
        }
    }

    var body: some View {
        VStack(spacing: 4) {
            Text(label)
                .foregroundColor(color)
                .font(.system(size: 11, weight: .medium))
                .lineLimit(1)
                .minimumScaleFactor(0.5)

            ProgressAnimation(color: color, animationSpeed: animationSpeed)
        }
        .frame(height: 28)
    }
}
