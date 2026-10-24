import Foundation

extension WordReplacement {
    static let sourceUser = "user"
    static let sourceEditMode = "editMode"
    static let sourceCorrectionFeedback = "correctionFeedback"
    static let learningReviewThreshold = 3

    var isLearningCandidate: Bool {
        !isEnabled && (
            source == Self.sourceEditMode ||
            source == Self.sourceCorrectionFeedback
        )
    }

    var sourceDisplayName: String {
        switch source {
        case Self.sourceUser:
            return "User"
        case Self.sourceEditMode:
            return "Edit Mode"
        case Self.sourceCorrectionFeedback:
            return "Feedback"
        default:
            return source
        }
    }

    var learningProgressLabel: String? {
        guard isLearningCandidate else { return nil }
        let current = max(1, min(hitCount, Self.learningReviewThreshold))
        return "\(current)/\(Self.learningReviewThreshold)"
    }

    func approveLearningCandidate() {
        isEnabled = true
        source = Self.sourceUser
        lastSeenDate = Date()
    }
}
