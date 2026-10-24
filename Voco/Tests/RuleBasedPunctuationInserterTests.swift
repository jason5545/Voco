import Testing
@testable import Voco

struct RuleBasedPunctuationInserterTests {
    @Test func fallbackLeavesFourContentCharactersUnpunctuated() {
        #expect(RuleBasedPunctuationInserter.insert(into: "紅會廣場") == "紅會廣場")
        #expect(RuleBasedPunctuationInserter.insert(into: "宏匯廣場") == "宏匯廣場")
    }

    @Test func fallbackAddsSentenceEndPunctuationAfterFourContentCharacters() {
        #expect(RuleBasedPunctuationInserter.insert(into: "我要去廣場") == "我要去廣場。")
    }

    @Test func fallbackDoesNotSplitHuoZheShi() {
        let output = RuleBasedPunctuationInserter.insert(into: "叫我們看一下或者是你可以先跑一次")

        #expect(output == "叫我們看一下或者是你可以先跑一次。")
        #expect(!output.contains("或者，是"))
    }

    @Test func fallbackDoesNotBreakAfterDeOrLongCJKRuns() {
        let output = RuleBasedPunctuationInserter.insert(into: "其實我的重點是綜上所述的話我留這個手機的意義是不是就不大了")

        #expect(output == "其實我的重點是綜上所述的話，我留這個手機的意義是不是就不大了？")
        #expect(!output.contains("的，話"))
        #expect(!output.contains("的，意義"))
    }

    @Test func fallbackDoesNotBreakAfterErShiOrDe() {
        let output = RuleBasedPunctuationInserter.insert(into: "這不是宗教的東西而是語音辨識後處理造成的問題")

        #expect(output == "這不是宗教的東西而是語音辨識後處理造成的問題。")
        #expect(!output.contains("而是，語音"))
        #expect(!output.contains("造成的，問題"))
    }

    @Test func fallbackKeepsPronounBreaksConservative() {
        let possessive = RuleBasedPunctuationInserter.insert(into: "我想確認一下我的設定")
        let objectPronoun = RuleBasedPunctuationInserter.insert(into: "麻煩你等一下叫我們確認")

        #expect(possessive == "我想確認一下我的設定。")
        #expect(objectPronoun == "麻煩你等一下叫我們確認。")
    }
}
