import Foundation

/// Deterministic cleanup that lets most dictations skip the LLM round-trip.
///
/// Modern STT (Parakeet, Whisper) already punctuates and capitalises, so on real
/// transcripts the model's job is mostly: strip "um/uh", collapse a stuttered
/// word, convert a dictated "comma", tidy capitalisation. Those are mechanical —
/// this does them in ~0ms. Measured against the LLM's own output on real run-log
/// transcripts: identical words on every transcript it accepts, and it *bails* to
/// the model for anything that needs interpretation:
///   • self-corrections / restarts ("no actually", "I mean", "wait", a phrase that
///     restarts: "is there anything is there any way")
///   • dictated formatting (new paragraph, bullets, brackets, quotes, underscore…)
///   • an email-style greeting (the model lays out salutation + blank line)
///   • a vocabulary term present with the wrong casing/spelling
///   • more than `maxWords` words (`clean_transcript_fast_path_max_words`; 0 = off)
/// The caller also skips this path when a custom system prompt or output
/// language is configured (the user wants the model's judgement then).
enum TranscriptFastPath {
    static let defaultMaxWords = 60
    static let maxWordsDefaultsKey = "clean_transcript_fast_path_max_words"

    /// Removed outright.
    private static let fillerWords: Set<String> = [
        "um", "uh", "uhm", "umm", "uhh", "erm", "er", "ah", "eh", "hmm", "hm", "mhm", "mm",
    ]

    /// Anything the model has to *interpret* → bail. Matched on lowercased,
    /// punctuation-stripped word sequences.
    private static let bailPhrases: [String] = [
        "no actually", "actually no", "i mean", "scratch that", "never mind", "nevermind",
        "wait", "sorry", "not that", "let me rephrase", "start over", "strike that", "correction",
        "new paragraph", "new line", "newline", "bullet", "bullets", "bullet point", "bullet points",
        "numbered list", "bullet list", "open paren", "close paren", "open bracket", "close bracket",
        "quote", "unquote", "colon", "semicolon", "underscore", "dash dash", "all caps", "slash", "hyphen",
        "period",  // "trial period" vs a dictated full stop — let the model decide
    ]

    private static let greetings: [String] = ["hi", "hey", "hello", "dear", "good morning", "good afternoon", "good evening"]

    /// Dictated punctuation converted deterministically (word → mark).
    private static let punctuationWords: [(String, String)] = [
        ("comma", ","), ("full stop", "."), ("question mark", "?"),
        ("exclamation mark", "!"), ("exclamation point", "!"),
    ]

    private static let sentenceEnders: Set<Character> = [".", "!", "?"]
    private static let clauseMarks: Set<Character> = [",", ".", "!", "?", ";", ":"]

    /// Returns the cleaned text to paste, or nil if the transcript should go
    /// through LLM post-processing.
    /// Bundle identifiers / app names for which a leading greeting means "lay
    /// this out as an email" (the model puts the salutation on its own line).
    static func looksLikeEmailTarget(bundleIdentifier: String?, appName: String?) -> Bool {
        let hay = ((bundleIdentifier ?? "") + " " + (appName ?? "")).lowercased()
        return ["mail", "outlook", "gmail", "spark", "superhuman", "airmail", "postbox", "thunderbird", "hey.com", "front", "missive"]
            .contains { hay.contains($0) }
    }

    static func cleanedIfAlreadyClean(_ raw: String, maxWords: Int, vocabulary: String, emailTarget: Bool = true) -> String? {
        guard maxWords > 0 else { return nil }
        // STT emits mid-sentence line breaks; the model would join them.
        var text = raw
            .components(separatedBy: .newlines)
            .map { $0.trimmingCharacters(in: .whitespaces) }
            .filter { !$0.isEmpty }
            .joined(separator: " ")
            .trimmingCharacters(in: .whitespacesAndNewlines)
        guard !text.isEmpty else { return nil }
        guard !text.contains(where: { "\"([{".contains($0) }) else { return nil }

        let words = text.split(whereSeparator: { $0.isWhitespace }).map(String.init)
        guard words.count <= maxWords else { return nil }
        let cores = words.map(core)
        let joined = " " + cores.joined(separator: " ") + " "
        for phrase in bailPhrases where joined.contains(" " + phrase + " ") {
            return nil
        }
        // A greeting only needs the model's email layout when we're in a mail app;
        // "Hello, hello, what is the time?" in a chat box is just text.
        if emailTarget {
            for g in greetings where joined.hasPrefix(" " + g + " ") {
                return nil
            }
        }
        if hasFalseStart(words: words, cores: cores) { return nil }
        if hasRepeatedSentence(text) { return nil }
        if vocabularyMismatch(text: text, joinedCores: joined, vocabulary: vocabulary) { return nil }

        // --- transform ---
        for (word, mark) in punctuationWords {
            text = replaceWord(word, with: mark, in: text)
        }
        var out: [String] = []
        for token in text.split(whereSeparator: { $0.isWhitespace }).map(String.init) {
            if fillerWords.contains(core(token)) {
                // "Um." at the end of a sentence: keep the sentence-ending mark.
                if let last = token.last, sentenceEnders.contains(last),
                   let prev = out.last, let prevLast = prev.last, !sentenceEnders.contains(prevLast) {
                    out[out.count - 1] = prev + String(last)
                }
                continue
            }
            out.append(token)
        }
        text = out.joined(separator: " ")
        text = tidyPunctuation(text)
        guard !text.isEmpty else { return nil }
        text = capitalizeSentences(text)
        if let last = text.last, !sentenceEnders.contains(last) {
            text += "."
        }
        return text
    }

    // MARK: - Detection

    static func core(_ token: String) -> String {
        token.lowercased().trimmingCharacters(in: CharacterSet(charactersIn: ".,!?;:"))
    }

    /// "the the" (unpunctuated adjacent repeat) or a 2–3 word phrase that
    /// restarts within a couple of words ("is there anything is there any way").
    static func hasFalseStart(words: [String], cores: [String]) -> Bool {
        if cores.count >= 2 {
            for i in 0..<(cores.count - 1) {
                guard !cores[i].isEmpty, cores[i] == cores[i + 1], let last = words[i].last else { continue }
                if !clauseMarks.contains(last) { return true }
            }
        }
        for n in 2...3 where cores.count > n {
            for i in 0..<(cores.count - n) {
                let gram = Array(cores[i..<(i + n)])
                guard gram.allSatisfy({ !$0.isEmpty }) else { continue }
                var j = i + 1
                while j <= min(cores.count - n, i + n + 1) {
                    if Array(cores[j..<(j + n)]) == gram { return true }
                    j += 1
                }
            }
        }
        return false
    }

    /// "Can you give me a prompt? Can you give me a prompt?" — the model
    /// de-duplicates; we don't guess.
    static func hasRepeatedSentence(_ text: String) -> Bool {
        let sentences = text
            .split(whereSeparator: { sentenceEnders.contains($0) })
            .map { core(String($0)).trimmingCharacters(in: .whitespaces) }
            .filter { !$0.isEmpty }
        guard sentences.count >= 2 else { return false }
        for i in 0..<(sentences.count - 1) where sentences[i] == sentences[i + 1] {
            return true
        }
        return false
    }

    static func vocabularyMismatch(text: String, joinedCores: String, vocabulary: String) -> Bool {
        let padded = " " + text + " "
        for term in vocabularyTerms(vocabulary) {
            let lowerTerm = term.lowercased()
            guard joinedCores.contains(" " + lowerTerm + " ") else { continue }
            let present = [" ", ".", ",", "?", "!"].contains { padded.contains(" " + term + $0) }
            if !present { return true }
        }
        return false
    }

    static func vocabularyTerms(_ vocabulary: String) -> [String] {
        vocabulary
            .components(separatedBy: CharacterSet(charactersIn: ",\n"))
            .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
            .filter { !$0.isEmpty }
    }

    // MARK: - Transform helpers

    /// Case-insensitive whole-word replacement of a dictated punctuation word,
    /// absorbing the space before it ("hi Dana comma" → "hi Dana,").
    private static func replaceWord(_ word: String, with mark: String, in text: String) -> String {
        let pattern = "\\s*\\b" + NSRegularExpression.escapedPattern(for: word) + "\\b"
        guard let regex = try? NSRegularExpression(pattern: pattern, options: [.caseInsensitive]) else { return text }
        let range = NSRange(text.startIndex..<text.endIndex, in: text)
        return regex.stringByReplacingMatches(in: text, options: [], range: range, withTemplate: mark)
    }

    private static func tidyPunctuation(_ input: String) -> String {
        var text = input
        let rules: [(String, String)] = [
            ("\\s+([,.!?;:])", "$1"),          // no space before marks
            ("[,;:]([.!?])", "$1"),             // ",." → "."
            ("([.!?])[,;:]", "$1"),             // ".," → "."
            ("^[,;:.]\\s*", ""),                // leading orphan mark (after removing "Um,")
            ("\\s{2,}", " "),
        ]
        for (pattern, template) in rules {
            guard let regex = try? NSRegularExpression(pattern: pattern, options: []) else { continue }
            let range = NSRange(text.startIndex..<text.endIndex, in: text)
            text = regex.stringByReplacingMatches(in: text, options: [], range: range, withTemplate: template)
        }
        return text.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    /// Upper-case the first letter, and the first letter after a sentence end
    /// that is followed by whitespace ("thingsattempted.md" is left alone).
    private static func capitalizeSentences(_ input: String) -> String {
        var result = ""
        var pendingCapital = true
        var sawEnder = false
        for ch in input {
            if ch.isWhitespace {
                if sawEnder { pendingCapital = true }
                result.append(ch)
                continue
            }
            if pendingCapital, ch.isLetter {
                result.append(contentsOf: String(ch).uppercased())
                pendingCapital = false
            } else {
                result.append(ch)
                if ch.isLetter || ch.isNumber { pendingCapital = false }
            }
            sawEnder = sentenceEnders.contains(ch)
        }
        return result
    }
}
