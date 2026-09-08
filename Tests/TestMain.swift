import Foundation

@main
struct FreeFlowTests {
    static func main() async {
        ActivityJournalTests.run()
        await JournalLifecycleTests.run()
        await RawCaptureStoreTests.run()
        AppContextServiceTests.run()
        DictationProfileTests.run()
        ModelConfigurationTests.run()
        RealtimeTranscriptStateTests.run()
        ShortcutCoreTests.run()
        SemanticVersionTests.run()
        LLMCooldownManagerTests.run()
        SpokenFormattingTests.run()
        TranscriptFastPathTests.run()
        TranscriptTextCoreTests.run()
        VocabularyLearnerTests.run()
        VoiceWritingTests.run()
        VoiceWritingTargetTests.run()
        await VoiceWritingRequestTests.run()
        print("FreeFlowTests passed")
    }
}
