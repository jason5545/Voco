import Foundation

enum RetranscriptionChangeCategory: String, Codable, Equatable {
    case unchanged
    case minorChange
    case meaningfulChange

    var displayName: String {
        switch self {
        case .unchanged:
            return "Unchanged"
        case .minorChange:
            return "Minor"
        case .meaningfulChange:
            return "Meaningful"
        }
    }
}

struct RetranscriptionAnalysis: Codable, Equatable {
    let editDistance: Int
    let changeRatio: Double
    let confidenceDelta: Double?
    let changeCategory: RetranscriptionChangeCategory
}

enum RetranscriptionAnalyticsService {
    static func analyze(
        sourceText: String,
        retranscribedText: String,
        sourceConfidenceScore: Double?,
        retranscribedConfidenceScore: Double?
    ) -> RetranscriptionAnalysis {
        let normalizedSource = normalizedForComparison(sourceText)
        let normalizedNew = normalizedForComparison(retranscribedText)
        let distance = levenshteinDistance(Array(normalizedSource), Array(normalizedNew))
        let denominator = max(normalizedSource.count, normalizedNew.count, 1)
        let ratio = Double(distance) / Double(denominator)
        let delta: Double? = {
            guard let sourceConfidenceScore,
                  let retranscribedConfidenceScore
            else {
                return nil
            }
            return retranscribedConfidenceScore - sourceConfidenceScore
        }()

        return RetranscriptionAnalysis(
            editDistance: distance,
            changeRatio: ratio,
            confidenceDelta: delta,
            changeCategory: changeCategory(distance: distance, ratio: ratio)
        )
    }

    private static func normalizedForComparison(_ text: String) -> String {
        OpenCCConverter.shared.convert(text)
            .lowercased()
            .filter { $0.isLetter || $0.isNumber }
    }

    private static func changeCategory(distance: Int, ratio: Double) -> RetranscriptionChangeCategory {
        if distance == 0 { return .unchanged }
        if ratio < 0.12 { return .minorChange }
        return .meaningfulChange
    }

    private static func levenshteinDistance(_ lhs: [Character], _ rhs: [Character]) -> Int {
        guard !lhs.isEmpty else { return rhs.count }
        guard !rhs.isEmpty else { return lhs.count }

        var previous = Array(0...rhs.count)
        for (i, leftChar) in lhs.enumerated() {
            var current = [i + 1]
            current.reserveCapacity(rhs.count + 1)

            for (j, rightChar) in rhs.enumerated() {
                let substitutionCost = leftChar == rightChar ? 0 : 1
                let insertion = current[j] + 1
                let deletion = previous[j + 1] + 1
                let substitution = previous[j] + substitutionCost
                current.append(min(insertion, deletion, substitution))
            }

            previous = current
        }

        return previous[rhs.count]
    }
}
