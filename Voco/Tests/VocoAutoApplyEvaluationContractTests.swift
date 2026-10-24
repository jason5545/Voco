import Foundation
import Testing
@testable import Voco

/// Cross-implementation test for the auto-apply evaluation contract.
///
/// Loads the same fixture as the Python side
/// (VocoReplayLab/docs/auto-apply-evaluation-contract.fixture.json) and
/// verifies that the Swift runtime implementation produces identical results.
///
/// The fixture is embedded as a string constant so this test does not depend
/// on the ReplayLab repo being present at test time.
struct VocoAutoApplyEvaluationContractTests {

    private var service: VocoAutoApplyModelService {
        VocoAutoApplyModelService(
            modelURL: URL(fileURLWithPath: "/dev/null/contract-test.json"),
            defaults: try! temporaryDefaults()
        )
    }

    @Test func strictTextKey() throws {
        for case_ in try fixture().strictTextKey {
            #expect(VocoAutoApplyModelService.strictTextKey(case_.input) == case_.expected,
                    "\(case_.input)")
        }
    }

    @Test func containsAsciiToken() throws {
        for case_ in try fixture().containsAsciiToken {
            #expect(service.containsAsciiTokenPublic(case_.input) == case_.expected,
                    "\(case_.input)")
        }
    }

    @Test func contextContainsToken() throws {
        for case_ in try fixture().contextContainsToken {
            #expect(service.contextContainsToken(text: case_.text, token: case_.token) == case_.expected,
                    "text=\(case_.text) token=\(case_.token)")
        }
    }

    @Test func replacementMatches() throws {
        for case_ in try fixture().replacementMatches {
            #expect(service.replacementMatchesPublic(text: case_.text, source: case_.source) == case_.expected,
                    "text=\(case_.text) source=\(case_.source)")
        }
    }

    @Test func cjkUnsafeContinuationBoundaryMatches() throws {
        #expect(service.replacementMatchesPublic(
            text: "剛剛有提到尖銳成。",
            source: "尖銳成",
            sourceBoundaryMode: VocoAutoApplyModelService.cjkUnsafeContinuationBoundaryMode
        ) == true)
        #expect(service.replacementMatchesPublic(
            text: "這個意見很尖銳成分很高",
            source: "尖銳成",
            sourceBoundaryMode: VocoAutoApplyModelService.cjkUnsafeContinuationBoundaryMode
        ) == false)
        #expect(service.replacementMatchesPublic(
            text: "這個講法很尖銳眼光也很準",
            source: "尖銳眼",
            sourceBoundaryMode: VocoAutoApplyModelService.cjkUnsafeContinuationBoundaryMode
        ) == false)
    }

    @Test func textIsActionCommand() throws {
        for case_ in try fixture().textIsActionCommand {
            #expect(service.textIsActionCommand(text: case_.input, actionCommandSurfaces: []) == case_.expected,
                    "\(case_.input)")
        }
    }

    // MARK: - Fixture

    private struct Fixture: Decodable {
        let contractVersion: Int
        let strictTextKey: [StrictTextKeyCase]
        let containsAsciiToken: [ContainsAsciiTokenCase]
        let contextContainsToken: [ContextContainsTokenCase]
        let replacementMatches: [ReplacementMatchesCase]
        let textIsActionCommand: [TextIsActionCommandCase]
    }
    private struct StrictTextKeyCase: Decodable { let input: String; let expected: String }
    private struct ContainsAsciiTokenCase: Decodable { let input: String; let expected: Bool }
    private struct ContextContainsTokenCase: Decodable { let text: String; let token: String; let expected: Bool }
    private struct ReplacementMatchesCase: Decodable { let text: String; let source: String; let expected: Bool }
    private struct TextIsActionCommandCase: Decodable { let input: String; let expected: Bool }

    private func fixture() throws -> Fixture {
        try JSONDecoder().decode(Fixture.self, from: Data(fixtureJSON.utf8))
    }

    private func temporaryDefaults() throws -> UserDefaults {
        let suiteName = "VocoAutoApplyEvaluationContractTests-\(UUID().uuidString)"
        guard let suite = UserDefaults(suiteName: suiteName) else {
            throw FixtureError.invalidDefaults
        }
        return suite
    }

    private enum FixtureError: Error { case invalidDefaults }
}

private let fixtureJSON = """
{
  "contractVersion": 1,
  "description": "Cross-implementation test fixture for auto-apply evaluation contract.",
  "strictTextKey": [
    {"input": "A 三二零的 autopilot 都會被跳掉，而 A 三五零不會。", "expected": "a 三二零的 autopilot 都會被跳掉,而 a 三五零不會。"},
    {"input": "全部刪除", "expected": "全部刪除"},
    {"input": "全部删除", "expected": "全部删除"},
    {"input": "  hello   world  ", "expected": "hello world"},
    {"input": "GitHub", "expected": "github"},
    {"input": "git", "expected": "git"},
    {"input": "ＡＢＣ", "expected": "abc"},
    {"input": "a b c", "expected": "a b c"},
    {"input": "C++ 編譯器", "expected": "c++ 編譯器"},
    {"input": "", "expected": ""}
  ],
  "containsAsciiToken": [
    {"input": "A 三二零的 autopilot", "expected": true},
    {"input": "全部刪除", "expected": false},
    {"input": "GitHub", "expected": true},
    {"input": "git", "expected": true},
    {"input": "ＡＢＣ", "expected": false},
    {"input": "a b c", "expected": true},
    {"input": "C++ 編譯器", "expected": true},
    {"input": "三二零", "expected": false},
    {"input": "", "expected": false}
  ],
  "contextContainsToken": [
    {"text": "我在用 GitHub 寫程式", "token": "GitHub", "expected": true},
    {"text": "我在用 git 寫程式", "token": "git", "expected": true},
    {"text": "github", "token": "git", "expected": false},
    {"text": "我在用 github 寫程式", "token": "git", "expected": false},
    {"text": "c p p 是一種語言", "token": "cpp", "expected": true},
    {"text": "c_p_p 是一種語言", "token": "cpp", "expected": true},
    {"text": "cancel 這個詞", "token": "can", "expected": false},
    {"text": "can 你幫我", "token": "can", "expected": true},
    {"text": "我在講吉他", "token": "吉他", "expected": true},
    {"text": "明德路", "token": "明德", "expected": true},
    {"text": "明德捷運站", "token": "明德", "expected": true},
    {"text": "施明德", "token": "明德", "expected": true},
    {"text": "不知道", "token": "明德", "expected": false}
  ],
  "replacementMatches": [
    {"text": "github", "source": "git", "expected": false},
    {"text": "github", "source": "github", "expected": true},
    {"text": "cancel", "source": "can", "expected": false},
    {"text": "cancel", "source": "cancel", "expected": true},
    {"text": "全部刪除", "source": "全部刪除", "expected": true},
    {"text": "三二零", "source": "三二零", "expected": true}
  ],
  "textIsActionCommand": [
    {"input": "全部刪除", "expected": true},
    {"input": "全部删除", "expected": true},
    {"input": "全部刪除。", "expected": true},
    {"input": "全部刪除！", "expected": true},
    {"input": " 全部刪除 ", "expected": true},
    {"input": "我要全部刪除", "expected": false},
    {"input": "刪除", "expected": false},
    {"input": "全部", "expected": false},
    {"input": "", "expected": false}
  ]
}
"""
