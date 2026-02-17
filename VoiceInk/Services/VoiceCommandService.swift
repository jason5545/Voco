import Foundation

enum VoiceCommand: String, CaseIterable {
    case deleteAll = "全部刪除"

    func execute() {
        switch self {
        case .deleteAll:
            CursorPaster.selectAllAndDelete()
        }
    }
}

class VoiceCommandService {
    static let shared = VoiceCommandService()

    private init() {}

    /// Trailing punctuation that Whisper may append to short utterances
    private static let trailingPunctuation = CharacterSet(charactersIn: "。，！？、；：.!?,;:")

    func detectCommand(in text: String) -> VoiceCommand? {
        let cleaned = text
            .trimmingCharacters(in: .whitespacesAndNewlines)
            .trimmingCharacters(in: Self.trailingPunctuation)
        return VoiceCommand.allCases.first { $0.rawValue == cleaned }
    }
}
