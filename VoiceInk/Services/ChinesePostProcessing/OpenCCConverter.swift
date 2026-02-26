import Foundation
import OpenCC

extension ChineseConverter: @unchecked Sendable {}

/// OpenCC s2twp converter wrapper
/// Converts Simplified Chinese to Traditional Chinese (Taiwan) with phrase conversion
class OpenCCConverter {
    static let shared = OpenCCConverter()

    private let converter: ChineseConverter

    /// Supplementary phrase mappings not covered by OpenCC s2twp dictionaries
    private static let supplementaryMappings: [(String, String)] = [
        ("優盤", "隨身碟"),  // TWPhrases only has U盤→隨身碟, not 優盤
        ("拍制", "拍製"),    // STPhrases doesn't have this compound
        ("賬", "帳"),        // TWVariants missing: 賬→帳 (Taiwan standard for 帳號/帳戶/帳目 etc.)
    ]

    private init() {
        converter = try! ChineseConverter(options: [.traditionalize, .twStandard, .twIdiom])
    }

    /// Convert Simplified Chinese to Traditional Chinese (Taiwan with phrases)
    func convert(_ text: String) -> String {
        var result = converter.convert(text)

        for (from, to) in Self.supplementaryMappings {
            result = result.replacingOccurrences(of: from, with: to)
        }

        return result
    }
}
