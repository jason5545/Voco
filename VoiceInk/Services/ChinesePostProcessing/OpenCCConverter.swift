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

    /// Revert OpenCC 里→裡 false conversions in Japanese name patterns.
    /// 裡 is not a valid Japanese kanji; names always use 里(さと/り).
    private static let japaneseNameReversions: [(String, String)] = [
        ("裡沙", "里沙"),  // りさ (Risa) — e.g. 織部里沙 (LiSA)
        ("裡奈", "里奈"),  // りな (Rina)
        ("裡美", "里美"),  // さとみ (Satomi)
        ("裡香", "里香"),  // りか (Rika)
        ("裡穗", "里穗"),  // りほ (Riho)
        ("裡菜", "里菜"),  // りな (Rina)
        ("裡帆", "里帆"),  // りほ (Riho)
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

        for (from, to) in Self.japaneseNameReversions {
            result = result.replacingOccurrences(of: from, with: to)
        }

        return result
    }
}
