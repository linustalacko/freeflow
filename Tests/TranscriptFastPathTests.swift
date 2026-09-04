import Foundation

enum TranscriptFastPathTests {
    static func run() {
        testFillersAndPunctuationAreHandled()
        testDictatedListsNowTakeTheFastPath()
        testAmbiguousFormattingStillBailsToTheModel()
        testInterpretationStillBailsToTheModel()
        testProfileShapesTerminalPunctuation()
        testEmailGreetingBailsOnlyInMailApps()
        testShortUtterancesHonorUserOptions()
    }

    private static let chat = DictationProfileResolver.profile(for: .chat)
    private static let email = DictationProfileResolver.profile(for: .email)
    private static let terminal = DictationProfileResolver.profile(for: .terminal)
    private static let document = DictationProfileResolver.profile(for: .document)
    private static let unknown = DictationProfileResolver.profile(for: .unknown)

    private static func testShortUtterancesHonorUserOptions() {
        TestSupport.expectEqual(TranscriptFastPath.cleanedIfAlreadyClean(
            "Thanks", maxWords: 60, vocabulary: "", profile: chat,
            outputLanguage: "French"), nil)
        TestSupport.expectEqual(TranscriptFastPath.cleanedIfAlreadyClean(
            "Thanks", maxWords: 60, vocabulary: "", profile: chat,
            customSystemPrompt: "Use a formal salutation."), nil)
        TestSupport.expectEqual(TranscriptFastPath.cleanedIfAlreadyClean(
            "acme", maxWords: 60, vocabulary: "ACME", profile: chat), nil)
        TestSupport.expectEqual(TranscriptFastPath.cleanedIfAlreadyClean(
            "Thanks", maxWords: 0, vocabulary: "", profile: chat), nil)
        expectClean("sounds good", profile: chat, "Sounds good")
        expectClean("hello comma friend", profile: chat, "Hello, friend")
        expectBails("no actually Wednesday", profile: chat)
    }

    private static func testFillersAndPunctuationAreHandled() {
        expectClean("um so the deploy is done", profile: unknown, "So the deploy is done.")
        expectClean("the gap is 5 mm wide", profile: document, "The gap is 5 mm wide.")
        expectClean("take it to the ER", profile: document, "Take it to the ER.")
        expectClean("mhm that could work", profile: chat, "Mhm that could work")
        expectClean("hmm that seems unlikely", profile: chat, "Hmm that seems unlikely")
        expectClean("5 mm", profile: chat, "5 mm")
        // A dictated "comma" becomes a comma. Checked against a document target
        // so the assertion is about punctuation conversion, not the casual rule.
        expectClean("hi dana comma thanks for the update", profile: document, "Hi dana, thanks for the update.")
    }

    private static func testDictatedListsNowTakeTheFastPath() {
        // Previously every one of these bailed to the LLM purely because the
        // word "bullet" appeared.
        expectClean(
            "bullet point ship the release bullet point update the docs",
            profile: chat,
            "- Ship the release\n- Update the docs"
        )
        expectClean(
            "thanks for the note new paragraph I'll look tomorrow",
            profile: unknown,
            "Thanks for the note\n\nI'll look tomorrow."
        )
    }

    private static func testAmbiguousFormattingStillBailsToTheModel() {
        expectBails("we are opening a new line of business", profile: chat)
        expectBails("I need a quote for the new laptop", profile: chat)
        // Talking *about* a bullet is prose, and now takes the fast path instead
        // of costing a round trip just because the word appeared.
        expectClean(
            "add a bullet about the rollback plan and another about cleanup",
            profile: chat,
            "Add a bullet about the rollback plan and another about cleanup"
        )
    }

    private static func testInterpretationStillBailsToTheModel() {
        // Self-corrections, restarts, and repeated sentences need the model.
        expectBails("let's meet Thursday no actually Wednesday", profile: unknown)
        expectBails("is there anything is there any way to fix this", profile: unknown)
        expectBails("send it over. send it over.", profile: unknown)
        // Dictated syntax this layer does not convert.
        expectBails("the flag is dash dash fix", profile: unknown)
        expectBails("call it user underscore id", profile: unknown)
    }

    private static func testProfileShapesTerminalPunctuation() {
        // A shell command gets no trailing full stop.
        expectClean("git status", profile: terminal, "git status")
        // A short chat line still gets its opening capital, but no closing period.
        expectClean("sounds good to me", profile: chat, "Sounds good to me")
        // Prose everywhere else still gets normal sentence punctuation.
        expectClean("the deploy finished", profile: unknown, "The deploy finished.")
    }

    private static func testEmailGreetingBailsOnlyInMailApps() {
        // In a mail client the model lays out salutation + blank line + body.
        expectBails("hi Dana thanks for sending that over", profile: email)
        // In a chat box the same words are just a message.
        TestSupport.expect(
            TranscriptFastPath.cleanedIfAlreadyClean(
                "hi Dana thanks for sending that over",
                maxWords: 60,
                vocabulary: "",
                profile: chat
            ) != nil,
            "a greeting in a chat app should take the fast path"
        )
    }

    private static func expectClean(
        _ input: String,
        profile: DictationProfile,
        _ expected: String,
        file: StaticString = #filePath,
        line: UInt = #line
    ) {
        let actual = TranscriptFastPath.cleanedIfAlreadyClean(
            input, maxWords: 60, vocabulary: "", profile: profile
        )
        TestSupport.expect(
            actual == expected,
            "For \"\(input)\" expected \"\(expected)\", got \(actual.map { "\"\($0)\"" } ?? "nil (bailed to model)")",
            file: file,
            line: line
        )
    }

    private static func expectBails(
        _ input: String,
        profile: DictationProfile,
        file: StaticString = #filePath,
        line: UInt = #line
    ) {
        let actual = TranscriptFastPath.cleanedIfAlreadyClean(
            input, maxWords: 60, vocabulary: "", profile: profile
        )
        TestSupport.expect(
            actual == nil,
            "Expected \"\(input)\" to bail to the model, got \"\(actual ?? "")\"",
            file: file,
            line: line
        )
    }
}
