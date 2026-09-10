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

    @Test func stripsSentencePunctuationFromStandalonePhoneNumbers() {
        #expect(ShortUtterancePunctuationCleaner.removeTerminalSentencePunctuation(from: "0809080650。") == "0809080650")
        #expect(ShortUtterancePunctuationCleaner.removeTerminalSentencePunctuation(from: "0912345678？") == "0912345678")
        #expect(ShortUtterancePunctuationCleaner.removeTerminalSentencePunctuation(from: "+886912345678.") == "+886912345678")
        #expect(ShortUtterancePunctuationCleaner.removeTerminalSentencePunctuation(from: "02-2345-6789。") == "02-2345-6789")
        #expect(ShortUtterancePunctuationCleaner.removeTerminalSentencePunctuation(from: "0809，080，650。") == "0809080650")
        #expect(ShortUtterancePunctuationCleaner.removeTerminalSentencePunctuation(from: " 0809080650。 ") == " 0809080650 ")
    }

    @Test func leavesNonPhoneDigitStringsAlone() {
        // Too short to be a phone number; also not short enough for the 4-character rule.
        #expect(ShortUtterancePunctuationCleaner.removeTerminalSentencePunctuation(from: "123456。") == "123456。")
        // Decimals and thousands separators are not flattened.
        #expect(ShortUtterancePunctuationCleaner.removeTerminalSentencePunctuation(from: "3.14159265。") == "3.14159265。")
        #expect(ShortUtterancePunctuationCleaner.removeTerminalSentencePunctuation(from: "1,234,567,890。") == "1,234,567,890。")
        // Anything with letters or CJK is a sentence, not a phone number.
        #expect(ShortUtterancePunctuationCleaner.removeTerminalSentencePunctuation(from: "電話是0809080650。") == "電話是0809080650。")
        #expect(ShortUtterancePunctuationCleaner.removeTerminalSentencePunctuation(from: "0809080650元。") == "0809080650元。")
        #expect(ShortUtterancePunctuationCleaner.standalonePhoneNumber(in: "0809080650") == nil)
    }

    @Test func preservesTrailingWhitespaceWhenCleaning() {
        #expect(ShortUtterancePunctuationCleaner.removeTerminalSentencePunctuation(from: "復康巴士。 ") == "復康巴士 ")
    }
}
