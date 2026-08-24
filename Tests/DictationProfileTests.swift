import Foundation

enum DictationProfileTests {
    static func run() {
        testBundleIdentifierMatching()
        testBrowserTitleDisambiguation()
        testAppNameFallback()
        testUnknownPreservesLegacyBehavior()
        testProfileRules()
    }

    private static func testBundleIdentifierMatching() {
        expectTarget("com.apple.mail", .email)
        expectTarget("com.superhuman.electron", .email)
        expectTarget("com.microsoft.Outlook", .email)
        expectTarget("com.tinyspeck.slackmacgap", .chat)
        expectTarget("com.hnc.Discord", .chat)
        expectTarget("com.apple.dt.Xcode", .code)
        expectTarget("com.todesktop.230313mzl4w4u92", .code)   // Cursor
        expectTarget("com.googlecode.iterm2", .terminal)
        expectTarget("com.mitchellh.ghostty", .terminal)
        expectTarget("md.obsidian", .document)
        expectTarget("notion.id", .document)
        expectTarget("com.raycast.macos", .searchField)
    }

    private static func testBrowserTitleDisambiguation() {
        // A browser is whatever tab is open — the bundle id alone says nothing.
        expectTarget("com.google.Chrome", .email, windowTitle: "Inbox (12) - me@example.com - Gmail")
        expectTarget("com.google.Chrome", .chat, windowTitle: "Slack | general | Acme")
        expectTarget("com.google.Chrome", .document, windowTitle: "Design doc - Google Docs")
        // An uninformative title must not be guessed at.
        expectTarget("com.apple.Safari", .unknown, windowTitle: "example.com")
        expectTarget("company.thebrowser.Browser", .unknown, windowTitle: "")
    }

    private static func testAppNameFallback() {
        // Unknown bundle id (a fork or a beta), recognizable name.
        expectTarget("com.example.unknownfork", .email, appName: "Mail")
        expectTarget("com.example.unknownfork", .terminal, appName: "Ghostty")
        expectTarget("com.example.unknownfork", .chat, appName: "Slack")
        // Name matching stays narrow: a substring hit would mis-profile far more
        // often than it helps.
        expectTarget("com.example.unknownfork", .unknown, appName: "Mailbox Designer")
        expectTarget("com.example.unknownfork", .unknown, appName: "Discord Bot Studio")
    }

    private static func testUnknownPreservesLegacyBehavior() {
        // Before profiles existed the fast path always assumed an email target
        // and added no style hint. An unrecognized app must behave the same.
        let profile = DictationProfileResolver.profile(for: .unknown)
        TestSupport.expect(profile.wantsEmailLayout, "unknown target should keep email-layout behavior")
        TestSupport.expect(profile.styleHint.isEmpty, "unknown target should add no style hint")
        TestSupport.expect(!profile.preservesVerbatim, "unknown target should not be verbatim")
    }

    private static func testProfileRules() {
        let terminal = DictationProfileResolver.profile(for: .terminal)
        TestSupport.expect(terminal.preservesVerbatim, "terminal should preserve verbatim")
        TestSupport.expect(!terminal.wantsEmailLayout, "terminal should not want email layout")

        let email = DictationProfileResolver.profile(for: .email)
        TestSupport.expect(email.wantsEmailLayout, "email should want email layout")
        TestSupport.expect(!email.isCasual, "email should not be casual")
        TestSupport.expect(!email.allowsMarkdown, "email clients render a dash literally")
        TestSupport.expectEqual(email.bulletMarker, "\u{2022} ")

        let chat = DictationProfileResolver.profile(for: .chat)
        TestSupport.expect(chat.isCasual, "chat should be casual")
        TestSupport.expect(chat.allowsMarkdown, "chat should allow markdown bullets")
        TestSupport.expectEqual(chat.bulletMarker, "- ")

        // Every non-unknown target must carry a hint, or the prompt injection
        // silently does nothing.
        for target in [DictationTarget.email, .chat, .code, .terminal, .document, .searchField] {
            TestSupport.expect(
                !DictationProfileResolver.profile(for: target).styleHint.isEmpty,
                "\(target.rawValue) should have a style hint"
            )
        }
    }

    private static func expectTarget(
        _ bundleIdentifier: String,
        _ expected: DictationTarget,
        appName: String? = nil,
        windowTitle: String? = nil,
        file: StaticString = #filePath,
        line: UInt = #line
    ) {
        let actual = DictationProfileResolver.target(
            bundleIdentifier: bundleIdentifier,
            appName: appName,
            windowTitle: windowTitle
        )
        TestSupport.expect(
            actual == expected,
            "Expected \(expected.rawValue) for \(bundleIdentifier) (name: \(appName ?? "nil"), title: \(windowTitle ?? "nil")), got \(actual.rawValue)",
            file: file,
            line: line
        )
    }
}
