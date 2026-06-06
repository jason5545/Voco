import Foundation
import SwiftData

@Model
final class WordReplacement {
    static let sourceUser = "user"
    static let sourceEditMode = "editMode"
    static let sourceCorrectionFeedback = "correctionFeedback"
    static let learningPromotionThreshold = 3

    var id: UUID = UUID()
    var originalText: String = ""
    var replacementText: String = ""
    var dateAdded: Date = Date()
    var isEnabled: Bool = true
    var source: String = "user"
    var hitCount: Int = 1
    var lastSeenDate: Date = Date()

    init(originalText: String, replacementText: String, dateAdded: Date = Date(), isEnabled: Bool = true, source: String = "user") {
        self.originalText = originalText
        self.replacementText = replacementText
        self.dateAdded = dateAdded
        self.isEnabled = isEnabled
        self.source = source
        self.hitCount = 1
        self.lastSeenDate = dateAdded
    }

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
        let current = max(1, min(hitCount, Self.learningPromotionThreshold))
        return "\(current)/\(Self.learningPromotionThreshold)"
    }

    func approveLearningCandidate() {
        isEnabled = true
        source = Self.sourceUser
        lastSeenDate = Date()
    }
}
