import Foundation

/// Pinyin-based correction for common speech recognition errors
/// Reference: xvoice/src/pinyin.py lines 37-79
class PinyinCorrector {
    static let shared = PinyinCorrector()

    /// Direct correction table: words that are commonly confused in speech recognition
    /// Key = wrong recognition, Value = correct word
    private let directCorrections: [String: String] = [
        "耳度": "額度",
        "變色": "辨識",       // biànsè vs biànshí
        "邊視": "辨識",       // biānshì vs biànshí
        "邊是": "辨識",
        "變是": "辨識",
        "便是": "辨識",
        // Place name errors
        "北頭": "北投",       // běitóu vs běitóu (homophone)
        "去永所": "區公所",
        "去公所": "區公所",
        "曲公所": "區公所",
        // Common errors from history analysis
        "大圓模型": "大語言模型",
        "大援模型": "大語言模型",
        "大宇": "大語言",     // dàyǔ vs dàyǔyán
        "病史": "辨識",       // bìngshǐ vs biànshí
        "日劇": "日誌",       // rìjù vs rìzhì
        "成事": "程式",       // chéngshì vs chéngshì (homophone)
        "專欄": "專案",       // zhuānlán vs zhuānàn
        "雨停": "語音",       // yǔtíng vs yǔyīn
        "靜動人": "漸凍人",   // jìngdòngrén vs jiàndòngrén
        "近動人": "漸凍人",
        "單字": "單指",       // dānzì vs dānzhǐ
        "克爾勃": "克勞德",   // Claude
        "配頌": "配送",
        "鬥號": "逗號",
        "斗號": "逗號",
        // From log analysis (2026-01)
        "精子座": "金子做",   // jīngzǐzuò vs jīnzizuò
        "精子做": "金子做",
        "城市": "程式",       // chéngshì vs chéngshì (homophone)
        "城市嘛": "程式碼",   // chéngshìma vs chéngshìmǎ
        "推測": "推送",       // tuīcè vs tuīsòng
        "魔女飛行": "模擬飛行", // mónǚfēixíng vs mónǐfēixíng
        // From database learning (2024-12)
        "雲酸": "運算",
        "端雲酸": "端運算",
        "雲端雲": "雲端運",
        "雲端雲酸": "雲端運算",
        "我哪好": "我很好",
        "哪好": "很好",
        // From user feedback (2026-02)
        "轉入": "轉錄",       // zhuǎnrù vs zhuǎnlù
    ]

    /// Sorted keys by length (longest first) to avoid substring conflicts
    private let sortedKeys: [String]

    private init() {
        sortedKeys = directCorrections.keys.sorted { $0.count > $1.count }
    }

    /// Apply direct corrections to the input text
    /// - Parameter text: Input text to correct
    /// - Returns: Corrected text and list of corrections made
    func correct(_ text: String) -> (text: String, corrections: [(original: String, corrected: String)]) {
        var result = text
        var corrections: [(original: String, corrected: String)] = []

        for key in sortedKeys {
            guard let replacement = directCorrections[key] else { continue }
            if result.contains(key) {
                corrections.append((original: key, corrected: replacement))
                result = result.replacingOccurrences(of: key, with: replacement)
            }
        }

        return (text: result, corrections: corrections)
    }
}
