import SwiftUI

enum RecorderCandidateReviewLayout {
    static let miniWidth: CGFloat = 460
    static let miniHeight: CGFloat = 320
    static let notchSideExpansion: CGFloat = 230
    static let notchPanelHeight: CGFloat = 320
    static let notchWindowHeight: CGFloat = 380
}

// MARK: - Shared Popover State

enum ActivePopoverState {
    case none
    case enhancement
    case power
}

// MARK: - Icon Toggle Button

struct RecorderToggleButton: View {
    let isEnabled: Bool
    let icon: String
    let disabled: Bool
    let action: () -> Void

    init(isEnabled: Bool, icon: String, disabled: Bool = false, action: @escaping () -> Void) {
        self.isEnabled = isEnabled
        self.icon = icon
        self.disabled = disabled
        self.action = action
    }

    private var isEmoji: Bool {
        !icon.contains(".") && !icon.contains("-") && icon.unicodeScalars.contains { !$0.isASCII }
    }

    var body: some View {
        Button(action: action) {
            Group {
                if isEmoji {
                    Text(icon).font(.system(size: 14))
                } else {
                    Image(systemName: icon).font(.system(size: 13))
                }
            }
            .foregroundColor(disabled ? .white.opacity(0.3) : (isEnabled ? .white : .white.opacity(0.6)))
        }
        .buttonStyle(PlainButtonStyle())
        .disabled(disabled)
    }
}

// MARK: - Record Button

struct RecorderRecordButton: View {
    let isRecording: Bool
    let isProcessing: Bool
    let action: () -> Void

    var body: some View {
        Button(action: action) {
            ZStack {
                Circle()
                    .fill(buttonColor)
                    .frame(width: 25, height: 25)

                if isProcessing {
                    ProcessingIndicator(color: .white).frame(width: 16, height: 16)
                } else if isRecording {
                    RoundedRectangle(cornerRadius: 3).fill(Color.white).frame(width: 9, height: 9)
                } else {
                    Circle().fill(Color.white).frame(width: 9, height: 9)
                }
            }
        }
        .buttonStyle(PlainButtonStyle())
        .disabled(isProcessing)
    }

    private var buttonColor: Color {
        if isProcessing { return Color(red: 0.4, green: 0.4, blue: 0.45) }
        if isRecording  { return .red }
        return Color(red: 0.3, green: 0.3, blue: 0.35)
    }
}

// MARK: - Processing Indicator

struct ProcessingIndicator: View {
    @State private var rotation: Double = 0
    let color: Color

    var body: some View {
        Circle()
            .trim(from: 0.1, to: 0.9)
            .stroke(color, lineWidth: 1.7)
            .frame(width: 14, height: 14)
            .rotationEffect(.degrees(rotation))
            .onAppear {
                withAnimation(.linear(duration: 1).repeatForever(autoreverses: false)) {
                    rotation = 360
                }
            }
    }
}

// MARK: - Progress Dot Animation

struct ProgressAnimation: View {
    let color: Color
    let animationSpeed: Double

    private let dotCount = 5
    private let dotSize: CGFloat = 3
    private let dotSpacing: CGFloat = 2

    @State private var currentDot = 0
    @State private var timer: Timer?

    init(color: Color = .white, animationSpeed: Double = 0.3) {
        self.color = color
        self.animationSpeed = animationSpeed
    }

    var body: some View {
        HStack(spacing: dotSpacing) {
            ForEach(0..<dotCount, id: \.self) { index in
                RoundedRectangle(cornerRadius: dotSize / 2)
                    .fill(color.opacity(index <= currentDot ? 0.85 : 0.25))
                    .frame(width: dotSize, height: dotSize)
            }
        }
        .onAppear { startAnimation() }
        .onDisappear {
            timer?.invalidate()
            timer = nil
        }
    }

    private func startAnimation() {
        timer?.invalidate()
        currentDot = 0
        timer = Timer.scheduledTimer(withTimeInterval: animationSpeed, repeats: true) { _ in
            currentDot = (currentDot + 1) % (dotCount + 2)
            if currentDot > dotCount { currentDot = -1 }
        }
    }
}

// MARK: - Enhancement Prompt Button

struct RecorderPromptButton: View {
    @EnvironmentObject private var enhancementService: AIEnhancementService
    @Binding var activePopover: ActivePopoverState
    let buttonSize: CGFloat
    let padding: EdgeInsets

    @State private var isHoveringButton: Bool = false
    @State private var isHoveringPopover: Bool = false
    @State private var dismissWorkItem: DispatchWorkItem?

    init(activePopover: Binding<ActivePopoverState>, buttonSize: CGFloat = 28, padding: EdgeInsets = EdgeInsets(top: 0, leading: 7, bottom: 0, trailing: 0)) {
        self._activePopover = activePopover
        self.buttonSize = buttonSize
        self.padding = padding
    }

    var body: some View {
        RecorderToggleButton(
            isEnabled: enhancementService.isEnhancementEnabled,
            icon: enhancementService.activePrompt?.icon ?? enhancementService.allPrompts.first(where: { $0.id == PredefinedPrompts.defaultPromptId })?.icon ?? "checkmark.seal.fill",
            disabled: false
        ) {
            if enhancementService.isEnhancementEnabled {
                activePopover = activePopover == .enhancement ? .none : .enhancement
            } else {
                enhancementService.isEnhancementEnabled = true
            }
        }
        .frame(width: buttonSize)
        .padding(padding)
        .onHover {
            isHoveringButton = $0
            syncPopoverVisibility()
        }
        .popover(isPresented: .constant(activePopover == .enhancement), arrowEdge: .bottom) {
            EnhancementPromptPopover()
                .environmentObject(enhancementService)
                .onHover {
                    isHoveringPopover = $0
                    syncPopoverVisibility()
                }
        }
    }

    private func syncPopoverVisibility() {
        if isHoveringButton || isHoveringPopover {
            dismissWorkItem?.cancel()
            dismissWorkItem = nil
            activePopover = .enhancement
        } else {
            dismissWorkItem?.cancel()
            let work = DispatchWorkItem { [activePopoverBinding = $activePopover] in
                if activePopoverBinding.wrappedValue == .enhancement {
                    activePopoverBinding.wrappedValue = .none
                }
            }
            dismissWorkItem = work
            DispatchQueue.main.asyncAfter(deadline: .now() + 0.25, execute: work)
        }
    }
}

// MARK: - Power Mode Button

struct RecorderPowerModeButton: View {
    @ObservedObject private var powerModeManager = PowerModeManager.shared
    @Binding var activePopover: ActivePopoverState
    let buttonSize: CGFloat
    let padding: EdgeInsets

    @State private var isHoveringButton: Bool = false
    @State private var isHoveringPopover: Bool = false
    @State private var dismissWorkItem: DispatchWorkItem?

    init(activePopover: Binding<ActivePopoverState>, buttonSize: CGFloat = 28, padding: EdgeInsets = EdgeInsets(top: 0, leading: 0, bottom: 0, trailing: 7)) {
        self._activePopover = activePopover
        self.buttonSize = buttonSize
        self.padding = padding
    }

    var body: some View {
        RecorderToggleButton(
            isEnabled: !powerModeManager.enabledConfigurations.isEmpty,
            icon: powerModeManager.enabledConfigurations.isEmpty ? "✨" : (powerModeManager.currentActiveConfiguration?.emoji ?? "✨"),
            disabled: powerModeManager.enabledConfigurations.isEmpty
        ) {
            activePopover = activePopover == .power ? .none : .power
        }
        .frame(width: buttonSize)
        .padding(padding)
        .onHover {
            isHoveringButton = $0
            syncPopoverVisibility()
        }
        .popover(isPresented: .constant(activePopover == .power), arrowEdge: .bottom) {
            PowerModePopover()
                .onHover {
                    isHoveringPopover = $0
                    syncPopoverVisibility()
                }
        }
    }

    private func syncPopoverVisibility() {
        if isHoveringButton || isHoveringPopover {
            dismissWorkItem?.cancel()
            dismissWorkItem = nil
            activePopover = .power
        } else {
            dismissWorkItem?.cancel()
            let work = DispatchWorkItem { [activePopoverBinding = $activePopover] in
                if activePopoverBinding.wrappedValue == .power {
                    activePopoverBinding.wrappedValue = .none
                }
            }
            dismissWorkItem = work
            DispatchQueue.main.asyncAfter(deadline: .now() + 0.25, execute: work)
        }
    }
}

// MARK: - Live Transcript View

struct LiveTranscriptView: View {
    let text: String

    var body: some View {
        ScrollViewReader { proxy in
            ScrollView(.vertical, showsIndicators: false) {
                Text(text)
                    .font(.system(size: 12))
                    .foregroundColor(.white.opacity(0.8))
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .padding(.horizontal, 16)
                    .padding(.vertical, 6)
                    .id("bottom")
            }
            .frame(height: 56)
            .mask(
                LinearGradient(
                    stops: [
                        .init(color: .clear, location: 0.0),
                        .init(color: .black, location: 0.18),
                        .init(color: .black, location: 1.0)
                    ],
                    startPoint: .top,
                    endPoint: .bottom
                )
            )
            .onChange(of: text) {
                proxy.scrollTo("bottom", anchor: .bottom)
            }
        }
        .transaction { $0.disablesAnimations = true }
    }
}

// MARK: - Recorder Status Display

struct RecorderStatusDisplay: View {
    let currentState: RecordingState
    let audioMeter: AudioMeter
    let menuBarHeight: CGFloat?
    let isEditMode: Bool

    init(currentState: RecordingState, audioMeter: AudioMeter, menuBarHeight: CGFloat? = nil, isEditMode: Bool = false) {
        self.currentState = currentState
        self.audioMeter = audioMeter
        self.menuBarHeight = menuBarHeight
        self.isEditMode = isEditMode
    }

    var body: some View {
        Group {
            if currentState == .enhancing {
                ProcessingStatusDisplay(
                    mode: isEditMode ? .editing : .enhancing,
                    color: isEditMode ? .orange : .white
                )
                .transition(.opacity)
            } else if currentState == .transcribing {
                ProcessingStatusDisplay(mode: .transcribing, color: isEditMode ? .orange : .white)
                    .transition(.opacity)
            } else if currentState == .recording {
                AudioVisualizer(
                    audioMeter: audioMeter,
                    color: isEditMode ? .orange : .white,
                    isActive: currentState == .recording
                )
                .scaleEffect(y: menuBarHeight != nil ? min(1.0, (menuBarHeight! - 8) / 25) : 1.0, anchor: .center)
                .transition(.opacity)
            } else {
                StaticVisualizer(color: isEditMode ? .orange : .white)
                    .scaleEffect(y: menuBarHeight != nil ? min(1.0, (menuBarHeight! - 8) / 25) : 1.0, anchor: .center)
                    .transition(.opacity)
            }
        }
        .animation(.easeInOut(duration: 0.2), value: currentState)
    }
}

// MARK: - Dictionary Confirmation View (Edit Mode)
struct DictionaryConfirmationView: View {
    let original: String
    let replacement: String
    let onConfirm: () -> Void
    let onDismiss: () -> Void

    var body: some View {
        HStack(spacing: 8) {
            Text("\(original) → \(replacement)")
                .font(.system(size: 11, weight: .medium))
                .foregroundColor(.white)
                .lineLimit(1)
                .truncationMode(.middle)

            Button(action: onConfirm) {
                Image(systemName: "checkmark.circle.fill")
                    .foregroundColor(.green)
                    .font(.system(size: 16))
            }
            .buttonStyle(PlainButtonStyle())

            Button(action: onDismiss) {
                Image(systemName: "xmark.circle.fill")
                    .foregroundColor(.gray)
                    .font(.system(size: 16))
            }
            .buttonStyle(PlainButtonStyle())
        }
        .padding(.horizontal, 12)
    }
}

// MARK: - Candidate Review View

struct CandidateReviewView: View {
    let review: VocoCandidateReview
    let onSelect: (String) -> Void
    let onInteraction: () -> Void
    let onDismiss: () -> Void

    @State private var typedCandidate = ""
    @FocusState private var isTypedCandidateFocused: Bool

    var body: some View {
        VStack(spacing: 0) {
            header

            if !review.displayReviewSignals.isEmpty {
                signalStrip
                    .padding(.top, 7)
            }

            ScrollView(.vertical, showsIndicators: true) {
                LazyVStack(spacing: 7) {
                    ForEach(Array(review.candidates.prefix(5).enumerated()), id: \.offset) { index, candidate in
                        candidateButton(index: index, candidate: candidate)
                    }
                }
                .padding(.vertical, 9)
            }
            .frame(maxHeight: .infinity)
            .onHover { hovering in
                if hovering { onInteraction() }
            }

            Divider()
                .background(Color.white.opacity(0.14))

            typedRescueRow
                .padding(.top, 9)
        }
        .padding(.horizontal, 14)
        .padding(.top, 11)
        .padding(.bottom, 12)
    }

    private var header: some View {
        HStack(spacing: 8) {
            Image(systemName: "exclamationmark.triangle.fill")
                .font(.system(size: 12, weight: .semibold))
                .foregroundColor(.yellow)

            VStack(alignment: .leading, spacing: 1) {
                Text("Review needed")
                    .font(.system(size: 12, weight: .semibold))
                    .foregroundColor(.white.opacity(0.92))

                Text("\(confidenceText) confidence")
                    .font(.system(size: 10, weight: .medium))
                    .foregroundColor(.white.opacity(0.54))
            }

            Spacer(minLength: 0)

            Button(action: onDismiss) {
                Image(systemName: "xmark.circle.fill")
                    .font(.system(size: 15, weight: .medium))
                    .foregroundColor(.white.opacity(0.45))
            }
            .buttonStyle(.plain)
            .keyboardShortcut(.cancelAction)
        }
    }

    private var signalStrip: some View {
        ScrollView(.horizontal, showsIndicators: false) {
            HStack(spacing: 5) {
                ForEach(review.displayReviewSignals, id: \.self) { signal in
                    Text(signal)
                        .font(.system(size: 9, weight: .semibold))
                        .foregroundColor(.white.opacity(0.72))
                        .lineLimit(1)
                        .padding(.horizontal, 7)
                        .padding(.vertical, 3)
                        .background(
                            Capsule()
                                .fill(Color.white.opacity(0.1))
                        )
                }
            }
        }
    }

    private func candidateButton(index: Int, candidate: String) -> some View {
        let shortcut = review.keyboardShortcutForCandidate(at: index) ?? "\(index + 1)"
        return Button {
            onSelect(candidate)
        } label: {
            HStack(alignment: .top, spacing: 9) {
                Text(shortcut)
                    .font(.system(size: 11, weight: .bold))
                    .foregroundColor(.black.opacity(0.76))
                    .frame(width: 22, height: 22)
                    .background(Circle().fill(Color.white.opacity(index == 0 ? 0.92 : 0.62)))
                    .padding(.top, 1)

                VStack(alignment: .leading, spacing: 4) {
                    HStack(spacing: 6) {
                        Text(review.labelForCandidate(at: index))
                            .font(.system(size: 10, weight: .bold))
                            .foregroundColor(.white.opacity(index == 0 ? 0.86 : 0.62))
                            .lineLimit(1)

                        if let source = review.sourceDisplayNameForCandidate(at: index) {
                            Text(source)
                                .font(.system(size: 10, weight: .medium))
                                .foregroundColor(.white.opacity(0.46))
                                .lineLimit(1)
                        }
                    }

                    Text(candidate)
                        .font(.system(size: 12, weight: index == 0 ? .semibold : .regular))
                        .foregroundColor(.white.opacity(index == 0 ? 0.96 : 0.82))
                        .lineLimit(3)
                        .fixedSize(horizontal: false, vertical: true)
                        .multilineTextAlignment(.leading)
                }

                Spacer(minLength: 0)
            }
            .padding(.horizontal, 10)
            .padding(.vertical, 8)
            .frame(maxWidth: .infinity, alignment: .leading)
            .background(
                RoundedRectangle(cornerRadius: 8, style: .continuous)
                    .fill(Color.white.opacity(index == 0 ? 0.16 : 0.08))
            )
            .overlay(
                RoundedRectangle(cornerRadius: 8, style: .continuous)
                    .stroke(Color.white.opacity(index == 0 ? 0.2 : 0.08), lineWidth: 1)
            )
        }
        .buttonStyle(.plain)
        .keyboardShortcut(KeyEquivalent(Character(shortcut)), modifiers: [])
    }

    private var confidenceText: String {
        "\(Int((review.confidenceScore * 100).rounded()))%"
    }

    private var trimmedTypedCandidate: String {
        typedCandidate.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    private var typedRescueRow: some View {
        HStack(alignment: .bottom, spacing: 8) {
            TextField("Type correction", text: $typedCandidate, axis: .vertical)
                .textFieldStyle(.plain)
                .font(.system(size: 12, weight: .medium))
                .foregroundColor(.white.opacity(0.9))
                .lineLimit(1...3)
                .focused($isTypedCandidateFocused)
                .padding(.horizontal, 10)
                .padding(.vertical, 8)
                .background(
                    RoundedRectangle(cornerRadius: 8, style: .continuous)
                        .fill(Color.white.opacity(isTypedCandidateFocused ? 0.14 : 0.09))
                )
                .overlay(
                    RoundedRectangle(cornerRadius: 8, style: .continuous)
                        .stroke(Color.white.opacity(isTypedCandidateFocused ? 0.28 : 0.09), lineWidth: 1)
                )
                .onSubmit(submitTypedCandidate)
                .onTapGesture {
                    onInteraction()
                    isTypedCandidateFocused = true
                }
                .onChange(of: typedCandidate) { _, newValue in
                    if VocoCandidateReview.shouldRefreshTimeout(forTypedCandidate: newValue) {
                        onInteraction()
                    }
                }

            Button(action: submitTypedCandidate) {
                Image(systemName: "arrow.turn.down.left")
                    .font(.system(size: 12, weight: .bold))
                    .foregroundColor(trimmedTypedCandidate.isEmpty ? .white.opacity(0.25) : .black.opacity(0.78))
                    .frame(width: 30, height: 30)
                    .background(
                        Circle()
                            .fill(trimmedTypedCandidate.isEmpty ? Color.white.opacity(0.08) : Color.white.opacity(0.86))
                    )
            }
            .buttonStyle(.plain)
            .disabled(trimmedTypedCandidate.isEmpty)
            .help("Use typed correction")
        }
    }

    private func submitTypedCandidate() {
        let candidate = trimmedTypedCandidate
        guard !candidate.isEmpty else { return }
        onSelect(candidate)
    }
}
