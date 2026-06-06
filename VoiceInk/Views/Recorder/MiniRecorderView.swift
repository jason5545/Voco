import SwiftUI

struct MiniRecorderView<S: RecorderStateProvider & ObservableObject>: View {
    @ObservedObject var stateProvider: S
    @ObservedObject var recorder: Recorder
    @EnvironmentObject var windowManager: MiniWindowManager
    @EnvironmentObject private var enhancementService: AIEnhancementService
    @AppStorage("showLiveTextPreview") private var showLiveTextPreview = false

    @State private var activePopover: ActivePopoverState = .none

    // MARK: - Layout Constants

    private let controlBarHeight: CGFloat = 40
    private let compactWidth: CGFloat = 184
    private let expandedWidth: CGFloat = 300
    private let candidateReviewHeight: CGFloat = 150
    private let compactCornerRadius: CGFloat = 20
    private let expandedCornerRadius: CGFloat = 14

    // true when live transcript is streaming in during recording
    private var hasLiveTranscript: Bool {
        showLiveTextPreview
            && stateProvider.recordingState == .recording
            && !stateProvider.partialTranscript.isEmpty
    }

    private var controlBar: some View {
        HStack(spacing: 0) {
            RecorderPromptButton(
                activePopover: $activePopover,
                buttonSize: 22,
                padding: EdgeInsets()
            )
            .padding(.leading, 12)

            Spacer(minLength: 0)

            RecorderStatusDisplay(
                currentState: stateProvider.recordingState,
                audioMeter: recorder.audioMeter,
                isEditMode: stateProvider.isEditMode
            )

            Spacer(minLength: 0)

            RecorderPowerModeButton(
                activePopover: $activePopover,
                buttonSize: 22,
                padding: EdgeInsets()
            )
            .padding(.trailing, 12)
        }
        .frame(height: controlBarHeight)
    }

    private var contentWidth: CGFloat {
        if stateProvider.pendingCandidateReview != nil { return expandedWidth }
        return hasLiveTranscript ? expandedWidth : compactWidth
    }

    private var transcriptSection: some View {
        VStack(spacing: 0) {
            if hasLiveTranscript {
                LiveTranscriptView(text: stateProvider.partialTranscript)
                Divider().background(Color.white.opacity(0.15))
            }
        }
    }

    var body: some View {
        if windowManager.isVisible {
            VStack(spacing: 0) {
                if let review = stateProvider.pendingCandidateReview {
                    CandidateReviewView(
                        review: review,
                        onSelect: { stateProvider.selectCandidateReview(candidate: $0) },
                        onInteraction: { stateProvider.keepCandidateReviewAlive() },
                        onDismiss: { stateProvider.dismissCandidateReview() }
                    )
                    .frame(height: candidateReviewHeight)
                } else if let entry = stateProvider.pendingDictionaryEntry {
                    DictionaryConfirmationView(
                        original: entry.original,
                        replacement: entry.replacement,
                        onConfirm: { stateProvider.confirmDictionaryEntry() },
                        onDismiss: { stateProvider.dismissDictionaryEntry() }
                    )
                    .frame(height: controlBarHeight)
                } else {
                    transcriptSection
                    controlBar
                }
            }
            .frame(width: contentWidth)
            .background(Color.black)
            .clipShape(RoundedRectangle(cornerRadius: contentWidth == expandedWidth ? expandedCornerRadius : compactCornerRadius, style: .continuous))
            .animation(.easeInOut(duration: 0.3), value: hasLiveTranscript)
            .animation(.easeInOut(duration: 0.2), value: stateProvider.pendingCandidateReview?.id)
            .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .bottom)
        }
    }
}
