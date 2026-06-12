import Foundation

enum PinyinSimilarity {
    static func score(_ lhs: String, _ rhs: String, database: PinyinDatabase = .shared) -> Double {
        let left = pinyinString(lhs, database: database)
        let right = pinyinString(rhs, database: database)
        guard !left.isEmpty, !right.isEmpty else { return 0 }
        return sequenceRatio(Array(left), Array(right))
    }

    private static func pinyinString(_ text: String, database: PinyinDatabase) -> String {
        var parts: [String] = []
        for character in text where character.isCJK {
            if let pinyins = database.charToPinyin[character], let first = pinyins.first {
                parts.append(PinyinDatabase.stripTone(first))
            }
        }
        return parts.joined(separator: " ")
    }

    private static func sequenceRatio(_ lhs: [Character], _ rhs: [Character]) -> Double {
        let lhsCount = lhs.count
        let rhsCount = rhs.count
        guard lhsCount > 0 || rhsCount > 0 else { return 1.0 }

        var dp = [[Int]](
            repeating: [Int](repeating: 0, count: rhsCount + 1),
            count: lhsCount + 1
        )
        for lhsIndex in 1...lhsCount {
            for rhsIndex in 1...rhsCount {
                if lhs[lhsIndex - 1] == rhs[rhsIndex - 1] {
                    dp[lhsIndex][rhsIndex] = dp[lhsIndex - 1][rhsIndex - 1] + 1
                } else {
                    dp[lhsIndex][rhsIndex] = max(
                        dp[lhsIndex - 1][rhsIndex],
                        dp[lhsIndex][rhsIndex - 1]
                    )
                }
            }
        }

        let lcsLength = dp[lhsCount][rhsCount]
        return Double(2 * lcsLength) / Double(lhsCount + rhsCount)
    }
}
