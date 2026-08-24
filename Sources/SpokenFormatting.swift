import Foundation

/// Turns dictated formatting words into real formatting.
///
/// "bullet point ship the release bullet point update the docs" becomes two
/// bullets; "new paragraph" becomes a paragraph break; "quote … unquote" becomes
/// quotation marks. These were previously bail-outs in `TranscriptFastPath` —
/// every dictation containing them paid a full LLM round trip and came back with
/// the model's guess at layout. Deterministic beats that on both counts.
///
/// The engine is deliberately conservative: every rule that could fire on
/// ordinary prose ("a new line of business", "I need a quote") returns
/// `confident == false` instead of guessing, and the caller falls back to the
/// model. Silence is cheap; a wrongly-bulleted email is not.
enum SpokenFormatting {
    struct Result {
        let text: String
        /// False when a formatting word was present but its reading was
        /// ambiguous — the caller should use the LLM instead of this output.
        let confident: Bool
        /// True when at least one formatting rule actually fired.
        let didFormat: Bool
        /// True only when the output is a bulleted or numbered list. List items
        /// are not sentences, so the caller must not append a closing full stop —
        /// unlike a paragraph break, which leaves ordinary prose behind.
        let producedList: Bool

        static func unchanged(_ text: String) -> Result {
            Result(text: text, confident: true, didFormat: false, producedList: false)
        }

        static func ambiguous(_ text: String) -> Result {
            Result(text: text, confident: false, didFormat: false, producedList: false)
        }
    }

    /// Phrases that announce a list before its items.
    private static let listIntros: [String] = [
        "bullet list", "bulleted list", "bullet points", "in bullet points",
        "as bullet points", "as bullets", "as a list", "in a list",
    ]

    private static let numberedIntros: [String] = [
        "numbered list", "number list", "as a numbered list", "ordered list",
    ]

    /// Per-item markers. Longest first so "next bullet point" doesn't match the
    /// shorter "bullet point" at the wrong offset.
    private static let itemMarkers: [String] = [
        "next bullet point", "next bullet", "new bullet point", "new bullet",
        "bullet point", "next point", "bullet",
    ]

    /// Words that make a following "new line"/"new paragraph" a noun phrase
    /// rather than a command ("a new line of business", "another new paragraph").
    private static let nounDeterminers: Set<String> = [
        "a", "an", "the", "this", "that", "these", "those", "each", "every",
        "another", "one", "of", "any", "some", "no", "my", "your", "our", "their",
    ]

    /// Applies every rule that fires unambiguously.
    static func apply(_ raw: String, profile: DictationProfile) -> Result {
        var text = raw
        var didFormat = false
        var producedList = false

        // Verbatim targets (terminal, launcher) want their words, not layout.
        guard !profile.preservesVerbatim else {
            return containsAnyFormattingWord(text) ? .ambiguous(text) : .unchanged(text)
        }

        switch quoteRewrite(text) {
        case .ambiguous: return .ambiguous(raw)
        case .rewritten(let updated):
            text = updated
            didFormat = true
        case .none: break
        }

        switch breakRewrite(text) {
        case .ambiguous: return .ambiguous(raw)
        case .rewritten(let updated):
            text = updated
            didFormat = true
        case .none: break
        }

        switch listRewrite(text, profile: profile) {
        case .ambiguous: return .ambiguous(raw)
        case .rewritten(let updated):
            text = updated
            didFormat = true
            producedList = true
        case .none: break
        }

        return Result(text: text, confident: true, didFormat: didFormat, producedList: producedList)
    }

    private enum Rewrite {
        case none
        case ambiguous
        case rewritten(String)
    }

    /// True if any word this engine reasons about appears at all — used to bail
    /// on verbatim targets, where formatting words are probably literal content.
    static func containsAnyFormattingWord(_ text: String) -> Bool {
        let padded = " " + normalizedCores(text).joined(separator: " ") + " "
        let all = itemMarkers + listIntros + numberedIntros
            + ["new line", "newline", "new paragraph", "quote", "unquote", "end quote"]
        return all.contains { padded.contains(" " + $0 + " ") }
    }

    // MARK: - Lists

    private static func listRewrite(_ text: String, profile: DictationProfile) -> Rewrite {
        let cores = normalizedCores(text)
        let padded = " " + cores.joined(separator: " ") + " "

        let numbered = numberedIntros.first { padded.contains(" " + $0 + " ") }
        let bulletIntro = listIntros.first { padded.contains(" " + $0 + " ") }

        // Count standalone markers. "bullet point" contains "bullet", so count the
        // longest match at each position rather than summing every phrase.
        //
        // An intro's own words are not items: "bullet points: a, b, c" contains
        // the marker "bullet" only because the intro does, and counting it would
        // swallow "points" into the first item.
        var markerHits = markerOccurrences(in: cores)
        if let intro = numbered ?? bulletIntro {
            let introWords = intro.split(separator: " ").map(String.init)
            if let introStart = firstIndex(of: introWords, in: cores) {
                let introEnd = introStart + introWords.count
                markerHits = markerHits.filter { $0.start >= introEnd || $0.start < introStart }
            }
        }

        // A lone marker with no intro is someone talking *about* a bullet, not
        // dictating one — "add a bullet about the rollback plan". That is not a
        // list and not ambiguous; it is ordinary prose, left alone.
        let wantsList = numbered != nil || bulletIntro != nil || markerHits.count >= 2
        guard wantsList else { return .none }

        let items: [String]
        if markerHits.isEmpty {
            // Intro but no per-item markers: the items are whatever follows the
            // intro, separated by commas.
            guard let split = itemsAfterIntro(text, intro: numbered ?? bulletIntro ?? "") else {
                return .ambiguous
            }
            items = split
        } else {
            guard let split = itemsAtMarkers(text, hits: markerHits, intro: numbered ?? bulletIntro) else {
                return .ambiguous
            }
            items = split
        }

        guard items.count >= 2 else { return .ambiguous }

        let cleanedItems = items.map { item -> String in
            var value = item.trimmingCharacters(in: .whitespacesAndNewlines)
            while let last = value.last, ",;:".contains(last) {
                value.removeLast()
                value = value.trimmingCharacters(in: .whitespacesAndNewlines)
            }
            guard let first = value.first else { return value }
            return String(first).uppercased() + value.dropFirst()
        }.filter { !$0.isEmpty }

        guard cleanedItems.count >= 2 else { return .ambiguous }

        let lines: [String]
        if numbered != nil {
            lines = cleanedItems.enumerated().map { "\($0.offset + 1). \($0.element)" }
        } else {
            lines = cleanedItems.map { profile.bulletMarker + $0 }
        }

        // Anything said before the intro/first marker stays as a lead-in line.
        let preamble = leadIn(text, hits: markerHits, intro: numbered ?? bulletIntro)
        let body = lines.joined(separator: "\n")
        guard let preamble, !preamble.isEmpty else { return .rewritten(body) }
        return .rewritten(preamble + "\n" + body)
    }

    /// Index ranges (in word offsets) where a list marker starts, longest match wins.
    private static func markerOccurrences(in cores: [String]) -> [(start: Int, length: Int)] {
        var hits: [(start: Int, length: Int)] = []
        var index = 0
        while index < cores.count {
            var matched = false
            for marker in itemMarkers {
                let words = marker.split(separator: " ").map(String.init)
                guard index + words.count <= cores.count else { continue }
                if Array(cores[index..<(index + words.count)]) == words {
                    hits.append((start: index, length: words.count))
                    index += words.count
                    matched = true
                    break
                }
            }
            if !matched { index += 1 }
        }
        return hits
    }

    private static func itemsAtMarkers(_ text: String, hits: [(start: Int, length: Int)], intro: String?) -> [String]? {
        let words = text.split(whereSeparator: { $0.isWhitespace }).map(String.init)
        guard let firstHit = hits.first, firstHit.start + firstHit.length <= words.count else { return nil }
        var items: [String] = []
        for (offset, hit) in hits.enumerated() {
            let start = hit.start + hit.length
            let end = offset + 1 < hits.count ? hits[offset + 1].start : words.count
            guard start <= end, end <= words.count else { return nil }
            items.append(words[start..<end].joined(separator: " "))
        }
        return items
    }

    private static func itemsAfterIntro(_ text: String, intro: String) -> [String]? {
        let cores = normalizedCores(text)
        let introWords = intro.split(separator: " ").map(String.init)
        guard let start = firstIndex(of: introWords, in: cores) else { return nil }
        let words = text.split(whereSeparator: { $0.isWhitespace }).map(String.init)
        let itemStart = start + introWords.count
        guard itemStart < words.count else { return nil }

        let tail = words[itemStart...].joined(separator: " ")
            .trimmingCharacters(in: CharacterSet(charactersIn: " :,-–"))
        // Split on commas, treating a trailing "and" as a separator too.
        let parts = tail
            .components(separatedBy: ",")
            .flatMap { part -> [String] in
                let trimmed = part.trimmingCharacters(in: .whitespaces)
                guard let range = trimmed.range(of: " and ", options: [.caseInsensitive]) else { return [trimmed] }
                return [String(trimmed[..<range.lowerBound]), String(trimmed[range.upperBound...])]
            }
            .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
            .filter { !$0.isEmpty }
        return parts
    }

    private static func leadIn(_ text: String, hits: [(start: Int, length: Int)], intro: String?) -> String? {
        let cores = normalizedCores(text)
        let words = text.split(whereSeparator: { $0.isWhitespace }).map(String.init)
        var cut: Int?
        if let intro {
            let introWords = intro.split(separator: " ").map(String.init)
            cut = firstIndex(of: introWords, in: cores)
        }
        if let first = hits.first {
            cut = min(cut ?? first.start, first.start)
        }
        guard let cut, cut > 0, cut <= words.count else { return nil }
        var lead = words[0..<cut].joined(separator: " ").trimmingCharacters(in: .whitespacesAndNewlines)
        while let last = lead.last, ",;:-".contains(last) {
            lead.removeLast()
            lead = lead.trimmingCharacters(in: .whitespacesAndNewlines)
        }
        guard !lead.isEmpty else { return nil }
        if let first = lead.first, first.isLowercase {
            lead = String(first).uppercased() + lead.dropFirst()
        }
        if let last = lead.last, !".!?:".contains(last) {
            lead += ":"
        }
        return lead
    }

    // MARK: - Line and paragraph breaks

    private static func breakRewrite(_ text: String) -> Rewrite {
        let cores = normalizedCores(text)
        let words = text.split(whereSeparator: { $0.isWhitespace }).map(String.init)
        guard cores.count == words.count else { return .none }

        let phrases: [(words: [String], replacement: String)] = [
            (["new", "paragraph"], "\n\n"),
            (["new", "line"], "\n"),
            (["newline"], "\n"),
        ]

        var out: [String] = []
        var index = 0
        var didBreak = false
        while index < cores.count {
            var matched = false
            for phrase in phrases {
                guard index + phrase.words.count <= cores.count,
                      Array(cores[index..<(index + phrase.words.count)]) == phrase.words else { continue }

                // "a new line of business" — a determiner before, or "of" after,
                // means the speaker used it as a noun.
                let before = index > 0 ? cores[index - 1] : ""
                let afterIndex = index + phrase.words.count
                let after = afterIndex < cores.count ? cores[afterIndex] : ""
                if nounDeterminers.contains(before) || after == "of" {
                    return .ambiguous
                }
                out.append(phrase.replacement)
                index += phrase.words.count
                didBreak = true
                matched = true
                break
            }
            if !matched {
                out.append(words[index])
                index += 1
            }
        }
        guard didBreak else { return .none }

        // Stitch: a break absorbs the space around it, and a preceding word that
        // ended mid-sentence gets its punctuation left alone.
        var rendered = ""
        for token in out {
            if token == "\n" || token == "\n\n" {
                rendered = rendered.trimmingCharacters(in: .whitespaces)
                while let last = rendered.last, ",;:".contains(last) { rendered.removeLast() }
                rendered += token
            } else {
                if !rendered.isEmpty && !rendered.hasSuffix("\n") { rendered += " " }
                rendered += token
            }
        }
        let cleaned = rendered.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !cleaned.isEmpty else { return .ambiguous }
        return .rewritten(capitalizeAfterBreaks(cleaned))
    }

    private static func capitalizeAfterBreaks(_ text: String) -> String {
        var result = ""
        var atLineStart = true
        for character in text {
            if atLineStart, character.isLetter {
                result.append(contentsOf: String(character).uppercased())
                atLineStart = false
            } else {
                result.append(character)
                if !character.isWhitespace { atLineStart = false }
            }
            if character == "\n" { atLineStart = true }
        }
        return result
    }

    // MARK: - Quotes

    private static func quoteRewrite(_ text: String) -> Rewrite {
        let cores = normalizedCores(text)
        let words = text.split(whereSeparator: { $0.isWhitespace }).map(String.init)
        guard cores.count == words.count else { return .none }

        let openIndex = cores.firstIndex(of: "quote")
        let closeIndex = cores.firstIndex { $0 == "unquote" } ?? closingQuoteIndex(cores)

        guard let openIndex else {
            // "unquote" with no "quote" is almost certainly a misfire.
            return closeIndex == nil ? .none : .ambiguous
        }
        guard let closeIndex, closeIndex > openIndex + 1 else {
            // "I need a quote for that" — one half of the pair, so it's prose.
            return .ambiguous
        }

        let closeLength = (cores[closeIndex] == "unquote") ? 1 : 2  // "end quote"
        let closeStart = (closeLength == 2) ? closeIndex - 1 : closeIndex
        guard closeStart > openIndex else { return .ambiguous }

        var out: [String] = []
        out.append(contentsOf: words[0..<openIndex])
        var quoted = words[(openIndex + 1)..<closeStart].joined(separator: " ")
        quoted = quoted.trimmingCharacters(in: .whitespacesAndNewlines)
        while let last = quoted.last, ",".contains(last) { quoted.removeLast() }
        guard !quoted.isEmpty else { return .ambiguous }
        out.append("\u{201C}" + quoted + "\u{201D}")
        let afterClose = closeIndex + 1
        if afterClose < words.count {
            out.append(contentsOf: words[afterClose...])
        }
        return .rewritten(out.joined(separator: " "))
    }

    /// Index of the "quote" in a trailing "end quote", or nil.
    private static func closingQuoteIndex(_ cores: [String]) -> Int? {
        for index in 1..<max(cores.count, 1) where cores[index] == "quote" && cores[index - 1] == "end" {
            return index
        }
        return nil
    }

    // MARK: - Shared helpers

    private static func normalizedCores(_ text: String) -> [String] {
        text.split(whereSeparator: { $0.isWhitespace })
            .map { $0.lowercased().trimmingCharacters(in: CharacterSet(charactersIn: ".,!?;:\u{2019}'\"")) }
    }

    private static func firstIndex(of needle: [String], in haystack: [String]) -> Int? {
        guard !needle.isEmpty, haystack.count >= needle.count else { return nil }
        for start in 0...(haystack.count - needle.count) where Array(haystack[start..<(start + needle.count)]) == needle {
            return start
        }
        return nil
    }
}
