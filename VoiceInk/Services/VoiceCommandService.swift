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

    func detectCommand(in text: String) -> VoiceCommand? {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        return VoiceCommand.allCases.first { $0.rawValue == trimmed }
    }
}
