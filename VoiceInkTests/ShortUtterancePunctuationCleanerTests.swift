import Testing
@testable import Voco

struct ShortUtterancePunctuationCleanerTests {
    @Test func removesTerminalSentencePunctuationFromFourContentCharacters() {
        #expect(ShortUtterancePunctuationCleaner.removeTerminalSentencePunctuation(from: "復康巴士。") == "復康巴士")
        #expect(ShortUtterancePunctuationCleaner.removeTerminalSentencePunctuation(from: "紅會廣場？") == "紅會廣場")
    }

    @Test func preservesTerminalSentencePunctuationAfterFourContentCharacters() {
        #expect(ShortUtterancePunctuationCleaner.removeTerminalSentencePunctuation(from: "我要去廣場。") == "我要去廣場。")
        #expect(ShortUtterancePunctuationCleaner.removeTerminalSentencePunctuation(from: "GitHub.") == "GitHub.")
    }

    @Test func preservesTrailingWhitespaceWhenCleaning() {
        #expect(ShortUtterancePunctuationCleaner.removeTerminalSentencePunctuation(from: "復康巴士。 ") == "復康巴士 ")
    }
}
