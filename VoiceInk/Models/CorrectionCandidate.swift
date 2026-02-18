import Foundation
import SwiftData

@Model
final class CorrectionCandidate {
    var id: UUID = UUID()
    var originalText: String = ""
    var correctedText: String = ""
    var count: Int = 1
    var lastSeen: Date = Date()
    var isPromoted: Bool = false

    init(originalText: String, correctedText: String, count: Int = 1, lastSeen: Date = Date(), isPromoted: Bool = false) {
        self.originalText = originalText
        self.correctedText = correctedText
        self.count = count
        self.lastSeen = lastSeen
        self.isPromoted = isPromoted
    }
}
