import Foundation

/// Where the dictation is about to land, decided from the frontmost app.
///
/// This is the deterministic half of "context": knowing you're in Slack rather
/// than Mail decides salutation layout, whether markdown bullets are legal, and
/// whether trailing periods belong — and it costs nothing. The LLM context call
/// stays for the things a bundle identifier genuinely can't know (who you're
/// replying to, what the thread is about).
enum DictationTarget: String {
    case email
    case chat
    case code
    case terminal
    case document
    case searchField
    case unknown
}

/// Formatting rules for a target, consumed by both the deterministic fast path
/// and the LLM prompt so the two agree on what the output should look like.
struct DictationProfile {
    let target: DictationTarget

    /// Bullets may be emitted as markdown (`- item`). False in apps that show
    /// the dash literally, where a middot reads better.
    var allowsMarkdown: Bool

    /// A spoken greeting means "lay this out as an email": salutation line,
    /// blank line, body.
    var wantsEmailLayout: Bool

    /// Chat-style: a single short line doesn't need a closing period.
    var isCasual: Bool

    /// Terminal/code: don't reflow, don't sentence-case, don't add end
    /// punctuation — the text is likely a command or an identifier.
    var preservesVerbatim: Bool

    /// One line appended to the cleanup system prompt. Empty for `.unknown`, so
    /// an unrecognized app behaves exactly as it did before profiles existed.
    var styleHint: String

    /// The bullet marker to use when the speaker dictates a list.
    var bulletMarker: String { allowsMarkdown ? "- " : "• " }
}

enum DictationProfileResolver {
    /// Bundle-identifier fragments per target. Matched as substrings against the
    /// lowercased bundle id so `com.apple.mail`, `com.apple.MobileMail`, and
    /// forks all land on the same profile.
    private static let bundleFragments: [(DictationTarget, [String])] = [
        (.email, [
            "com.apple.mail", "com.microsoft.outlook", "com.readdle.smartemail",
            "com.superhuman", "com.airmailapp", "com.postbox-inc", "org.mozilla.thunderbird",
            "com.missiveapp", "com.frontapp", "com.hey.app", "com.bloomtechnologies.spark",
            "com.mimestream.mimestream", "ch.protonmail",
        ]),
        (.chat, [
            "com.tinyspeck.slackmacgap", "com.hnc.discord", "com.apple.ichat",
            "com.apple.messages", "net.whatsapp.whatsapp", "ru.keepcoder.telegram",
            "com.telegram", "org.whispersystems.signal-desktop", "com.microsoft.teams",
            "com.linear", "zoom.us", "com.google.chat", "im.riot", "com.beeper",
        ]),
        (.code, [
            "com.apple.dt.xcode", "com.microsoft.vscode", "com.visualstudio.code",
            "com.todesktop",           // Cursor
            "dev.zed.zed", "com.jetbrains", "com.sublimetext", "com.github.atom",
            "com.panic.nova", "com.figma",
        ]),
        (.terminal, [
            "com.apple.terminal", "com.googlecode.iterm2", "dev.warp.warp-stable",
            "com.mitchellh.ghostty", "co.zeit.hyper", "net.kovidgoyal.kitty",
            "io.alacritty", "com.tabby",
        ]),
        (.document, [
            "notion.id", "md.obsidian", "net.shinyfrog.bear", "com.apple.notes",
            "com.microsoft.word", "com.apple.iwork.pages", "com.literatureandlatte.scrivener",
            "com.ulyssesapp", "com.agiletortoise.drafts", "com.craft", "com.linear-app",
            "com.atlassian", "com.evernote",
        ]),
        (.searchField, [
            "com.raycast.macos", "com.runningwithcrayons.alfred", "com.apple.spotlight",
            "com.blacktree.quicksilver",
        ]),
    ]

    private static let browserBundles: [String] = [
        "com.apple.safari", "com.google.chrome", "org.mozilla.firefox",
        "com.microsoft.edgemac", "company.thebrowser.browser", "com.brave.browser",
        "com.operasoftware.opera", "com.vivaldi.vivaldi", "ai.perplexity.comet",
    ]

    /// Window-title fragments that identify a web app inside a browser. The title
    /// is the only signal available — a browser's bundle id says nothing about
    /// whether the tab is Gmail or GitHub.
    private static let browserTitleFragments: [(DictationTarget, [String])] = [
        (.email, ["gmail", "inbox", "outlook", "proton mail", "fastmail", "zoho mail", "roundcube"]),
        (.chat, ["slack", "discord", "whatsapp", "telegram", "messenger", "teams", "chatgpt", "claude"]),
        (.document, ["google docs", "notion", "confluence", "linear", "jira", "github", "hackmd", "coda"]),
    ]

    static func profile(bundleIdentifier: String?, appName: String?, windowTitle: String? = nil) -> DictationProfile {
        profile(for: target(bundleIdentifier: bundleIdentifier, appName: appName, windowTitle: windowTitle))
    }

    static func target(bundleIdentifier: String?, appName: String?, windowTitle: String? = nil) -> DictationTarget {
        let bundle = (bundleIdentifier ?? "").lowercased()
        let name = (appName ?? "").lowercased()

        if !bundle.isEmpty {
            for (target, fragments) in bundleFragments where fragments.contains(where: { bundle.contains($0) }) {
                return target
            }
        }

        // A browser is whatever tab is open. Fall through to app-name matching
        // when the title is uninformative rather than guessing `.document`.
        if browserBundles.contains(where: { bundle.contains($0) }) {
            let title = (windowTitle ?? "").lowercased()
            if !title.isEmpty {
                for (target, fragments) in browserTitleFragments where fragments.contains(where: { title.contains($0) }) {
                    return target
                }
            }
            return .unknown
        }

        // Name matching is the fallback for apps whose bundle id we don't know
        // (forks, betas, renamed builds). Deliberately narrow: a substring hit on
        // a common word would mis-profile far more often than it helps.
        if !name.isEmpty {
            if ["mail", "outlook", "superhuman", "spark", "missive", "front"].contains(where: { name == $0 || name.hasPrefix($0 + " ") }) {
                return .email
            }
            if ["slack", "discord", "messages", "telegram", "signal", "whatsapp"].contains(where: { name == $0 }) {
                return .chat
            }
            if ["terminal", "iterm", "iterm2", "warp", "ghostty", "kitty", "alacritty"].contains(where: { name == $0 }) {
                return .terminal
            }
            if ["xcode", "cursor", "zed", "code", "visual studio code"].contains(where: { name == $0 }) {
                return .code
            }
        }

        return .unknown
    }

    static func profile(for target: DictationTarget) -> DictationProfile {
        switch target {
        case .email:
            DictationProfile(
                target: target,
                allowsMarkdown: false,
                wantsEmailLayout: true,
                isCasual: false,
                preservesVerbatim: false,
                styleHint: "The text is going into an email. If a greeting was spoken, put it on its own first line followed by a blank line. If a closing was spoken, put it in its own final paragraph. Never invent a greeting or closing that was not spoken."
            )
        case .chat:
            DictationProfile(
                target: target,
                allowsMarkdown: true,
                wantsEmailLayout: false,
                isCasual: true,
                preservesVerbatim: false,
                styleHint: "The text is going into a chat message. Keep it casual and conversational. Never add a greeting or a sign-off."
            )
        case .code:
            DictationProfile(
                target: target,
                allowsMarkdown: true,
                wantsEmailLayout: false,
                isCasual: true,
                preservesVerbatim: true,
                styleHint: "The text is going into a code editor. Preserve identifiers, paths, flags, and casing exactly. Do not add end punctuation to something that is not a sentence."
            )
        case .terminal:
            DictationProfile(
                target: target,
                allowsMarkdown: false,
                wantsEmailLayout: false,
                isCasual: true,
                preservesVerbatim: true,
                styleHint: "The text is going into a terminal. Treat it as a shell command: preserve flags, paths, and casing exactly, and add no trailing punctuation."
            )
        case .document:
            DictationProfile(
                target: target,
                allowsMarkdown: true,
                wantsEmailLayout: false,
                isCasual: false,
                preservesVerbatim: false,
                styleHint: "The text is going into a document or note. Full sentences and normal prose punctuation."
            )
        case .searchField:
            DictationProfile(
                target: target,
                allowsMarkdown: false,
                wantsEmailLayout: false,
                isCasual: true,
                preservesVerbatim: true,
                styleHint: "The text is going into a search or launcher field. Return the query only, with no trailing punctuation."
            )
        case .unknown:
            DictationProfile(
                target: target,
                allowsMarkdown: true,
                wantsEmailLayout: true,
                isCasual: false,
                preservesVerbatim: false,
                styleHint: ""
            )
        }
    }
}
