import Foundation

enum SpokenFormattingTests {
    static func run() {
        testBulletLists()
        testNumberedLists()
        testListsNeedRealIntent()
        testLineAndParagraphBreaks()
        testBreakWordsAsNouns()
        testQuotes()
        testVerbatimTargetsBailOut()
        testPlainTextIsUntouched()
    }

    private static let chat = DictationProfileResolver.profile(for: .chat)
    private static let email = DictationProfileResolver.profile(for: .email)
    private static let terminal = DictationProfileResolver.profile(for: .terminal)

    private static func testBulletLists() {
        expectText(
            "bullet point ship the release bullet point update the docs",
            profile: chat,
            "- Ship the release\n- Update the docs"
        )
        // Anything before the first marker becomes a lead-in line.
        expectText(
            "here's the plan bullet point fix the login bug bullet point ship it",
            profile: chat,
            "Here's the plan:\n- Fix the login bug\n- Ship it"
        )
        // Mail clients render "- " literally, so a middot reads better.
        expectText(
            "bullet point one thing bullet point another thing",
            profile: email,
            "\u{2022} One thing\n\u{2022} Another thing"
        )
        // An intro alone splits the items on commas.
        expectText(
            "bullet points fix the login bug, update the docs, ship the release",
            profile: chat,
            "- Fix the login bug\n- Update the docs\n- Ship the release"
        )
    }

    private static func testNumberedLists() {
        expectText(
            "numbered list write the spec, get review, merge",
            profile: chat,
            "1. Write the spec\n2. Get review\n3. Merge"
        )
    }

    private static func testListsNeedRealIntent() {
        // The canonical false positive from the cleanup prompt: one mention of
        // "bullet", used as a noun. Not a list — and not ambiguous either, so
        // the sentence passes through untouched rather than costing a round trip.
        expectUnformatted(
            "add a bullet about the rollback plan and another about feature flag cleanup",
            profile: chat
        )
        expectUnformatted("I added a bullet to the deck", profile: chat)
    }

    private static func testLineAndParagraphBreaks() {
        expectText(
            "thanks for the update new paragraph I'll take a look tomorrow",
            profile: email,
            "Thanks for the update\n\nI'll take a look tomorrow"
        )
        expectText(
            "first item new line second item",
            profile: chat,
            "First item\nSecond item"
        )
    }

    private static func testBreakWordsAsNouns() {
        // "a new line of business" is not a line break.
        expectAmbiguous("we are opening a new line of business", profile: chat)
        expectAmbiguous("that would be another new paragraph entirely", profile: chat)
    }

    private static func testQuotes() {
        expectText(
            "he said quote this is fine unquote and left",
            profile: chat,
            "he said \u{201C}this is fine\u{201D} and left"
        )
        expectText(
            "the sign said quote no entry end quote",
            profile: chat,
            "the sign said \u{201C}no entry\u{201D}"
        )
        // Half a pair is ordinary prose.
        expectAmbiguous("I need a quote for the new laptop", profile: chat)
    }

    private static func testVerbatimTargetsBailOut() {
        // In a terminal these words are probably literal arguments.
        expectAmbiguous("git commit dash m bullet point fix", profile: terminal)
        // ...but a plain command passes straight through.
        expectText("git status", profile: terminal, "git status")
    }

    private static func testPlainTextIsUntouched() {
        let input = "Let's ship the release on Wednesday."
        let result = SpokenFormatting.apply(input, profile: chat)
        TestSupport.expect(result.confident, "plain prose should be confident")
        TestSupport.expect(!result.didFormat, "plain prose should not be reformatted")
        TestSupport.expectEqual(result.text, input)
    }

    private static func expectText(
        _ input: String,
        profile: DictationProfile,
        _ expected: String,
        file: StaticString = #filePath,
        line: UInt = #line
    ) {
        let result = SpokenFormatting.apply(input, profile: profile)
        TestSupport.expect(
            result.confident,
            "Expected a confident result for \"\(input)\"",
            file: file,
            line: line
        )
        TestSupport.expect(
            result.text == expected,
            "For \"\(input)\" expected \"\(expected)\", got \"\(result.text)\"",
            file: file,
            line: line
        )
    }

    /// Confident, but no rule fired — the text is prose and passes through.
    private static func expectUnformatted(
        _ input: String,
        profile: DictationProfile,
        file: StaticString = #filePath,
        line: UInt = #line
    ) {
        let result = SpokenFormatting.apply(input, profile: profile)
        TestSupport.expect(
            result.confident && !result.didFormat && result.text == input,
            "Expected \"\(input)\" to pass through unformatted, got \"\(result.text)\" (confident: \(result.confident))",
            file: file,
            line: line
        )
    }

    private static func expectAmbiguous(
        _ input: String,
        profile: DictationProfile,
        file: StaticString = #filePath,
        line: UInt = #line
    ) {
        let result = SpokenFormatting.apply(input, profile: profile)
        TestSupport.expect(
            !result.confident,
            "Expected \"\(input)\" to be ambiguous, got \"\(result.text)\"",
            file: file,
            line: line
        )
    }
}
