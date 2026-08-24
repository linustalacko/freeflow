import Foundation

@main
struct FreeFlowTests {
    static func main() {
        AppContextServiceTests.run()
        DictationProfileTests.run()
        ModelConfigurationTests.run()
        ShortcutCoreTests.run()
        SemanticVersionTests.run()
        LLMCooldownManagerTests.run()
        SpokenFormattingTests.run()
        TranscriptFastPathTests.run()
        TranscriptTextCoreTests.run()
        VocabularyLearnerTests.run()
        print("FreeFlowTests passed")
    }
}
