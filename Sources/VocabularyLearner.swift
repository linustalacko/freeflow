import Foundation

/// Mines the corrections you make to pasted text for words the recognizer keeps
/// getting wrong.
///
/// `InlineEditCaptureService` already watches the field you dictated into and
/// records what you changed. Nothing consumed those edits before: fixing
/// "Aysha" by hand five times taught the pipeline nothing. This turns a repeated
/// single-word fix into a vocabulary term, which then biases both the STT prompt
/// and the cleanup model — so the sixth time, it is spelled right.
///
/// Conservative by construction: only 1:1 word substitutions count, both sides
/// must be alphabetic, and a term needs `minimumOccurrences` independent
/// sightings before it is proposed. A one-off rewording never becomes vocabulary.
enum VocabularyLearner {
    static let minimumOccurrences = 2

    struct Candidate: Equatable {
        let term: String
        /// The misrecognitions that produced this term, for the settings UI.
        let heardAs: [String]
        let occurrences: Int
    }

    /// Words too common to be anyone's custom vocabulary. A correction landing on
    /// one of these is ordinary editing, not a spelling the pipeline should learn.
    private static let stopWords: Set<String> = [
        "the", "a", "an", "and", "or", "but", "if", "then", "this", "that", "these",
        "those", "is", "are", "was", "were", "be", "been", "being", "have", "has",
        "had", "do", "does", "did", "will", "would", "can", "could", "should",
        "may", "might", "must", "shall", "to", "of", "in", "on", "at", "by", "for",
        "with", "from", "about", "into", "over", "after", "before", "it", "its",
        "he", "him", "his", "she", "her", "hers", "they", "them", "theirs",
        "we", "us", "ours", "you", "your", "yours", "our", "their", "mine",
        "my", "me", "i", "not", "no", "yes", "so", "as", "just", "very", "really",
        "there", "here", "what", "when", "where", "who", "why", "how", "all",
        "some", "any", "one", "two", "three", "now", "also", "than", "too", "up",
        "out", "off", "down", "get", "got", "make", "made", "like", "want", "need",
        "think", "know", "see", "go", "going", "let", "lets", "well", "okay", "ok",
    ]

    /// Extracts candidate terms from `(original, corrected)` pairs, newest first
    /// in the returned order of ties. `existingVocabulary` terms are skipped.
    static func candidates(
        from pairs: [(original: String, corrected: String)],
        existingVocabulary: String,
        minimumOccurrences: Int = VocabularyLearner.minimumOccurrences
    ) -> [Candidate] {
        let existing = Set(
            TranscriptFastPath.vocabularyTerms(existingVocabulary).map { $0.lowercased() }
        )

        var occurrences: [String: (term: String, heardAs: [String], count: Int)] = [:]

        for pair in pairs {
            for substitution in substitutions(original: pair.original, corrected: pair.corrected) {
                guard isLearnable(from: substitution.from, to: substitution.to) else { continue }
                let key = substitution.to.lowercased()
                guard !existing.contains(key) else { continue }
                var entry = occurrences[key] ?? (term: substitution.to, heardAs: [], count: 0)
                entry.count += 1
                if !entry.heardAs.contains(where: { $0.caseInsensitiveCompare(substitution.from) == .orderedSame }) {
                    entry.heardAs.append(substitution.from)
                }
                occurrences[key] = entry
            }
        }

        return occurrences.values
            .filter { $0.count >= minimumOccurrences }
            .map { Candidate(term: $0.term, heardAs: $0.heardAs, occurrences: $0.count) }
            .sorted {
                $0.occurrences == $1.occurrences
                    ? $0.term.localizedCaseInsensitiveCompare($1.term) == .orderedAscending
                    : $0.occurrences > $1.occurrences
            }
    }

    /// A substitution is learnable when it looks like a spelling the recognizer
    /// missed rather than a rewrite: both sides alphabetic, the target not a
    /// stop word, and the two similar enough to be the same intended word.
    static func isLearnable(from source: String, to target: String) -> Bool {
        guard target.count >= 3, source.count >= 2 else { return false }
        guard target.allSatisfy({ $0.isLetter || $0 == "-" || $0 == "'" }) else { return false }
        guard source.allSatisfy({ $0.isLetter || $0 == "-" || $0 == "'" }) else { return false }
        guard !stopWords.contains(target.lowercased()) else { return false }
        guard source != target else { return false }

        // Same word, different case ("aysha" → "Aysha") is exactly what we want.
        if source.lowercased() == target.lowercased() { return true }

        // Otherwise require phonetic-ish proximity so "tomorrow" → "Wednesday"
        // (a genuine rewrite) is not mistaken for a misrecognition.
        let distance = editDistance(source.lowercased(), target.lowercased())
        let allowed = max(2, min(source.count, target.count) / 3)
        return distance <= allowed
    }

    /// Word-level 1:1 substitutions between two texts, via an LCS alignment.
    /// Insertions and deletions are ignored — only a word swapped for exactly one
    /// other word tells us about spelling.
    static func substitutions(original: String, corrected: String) -> [(from: String, to: String)] {
        let source = words(original)
        let target = words(corrected)
        guard !source.isEmpty, !target.isEmpty else { return [] }
        // Wholesale rewrites carry no spelling signal and would produce noise.
        guard abs(source.count - target.count) <= 2 else { return [] }

        // Align case-insensitively so a pure re-capitalization doesn't look like
        // a deletion plus an insertion and desynchronize everything after it.
        let common = longestCommonSubsequence(source.map { $0.lowercased() }, target.map { $0.lowercased() })

        var results: [(from: String, to: String)] = []
        var sourceIndex = 0
        var targetIndex = 0
        for anchor in common + [(source.count, target.count)] {
            let sourceGap = anchor.0 - sourceIndex
            let targetGap = anchor.1 - targetIndex
            if sourceGap == 1 && targetGap == 1 {
                results.append((from: source[sourceIndex], to: target[targetIndex]))
            }
            // The alignment matched these two words, but only case-insensitively:
            // "groq" → "Groq" is exactly the fix we most want to learn.
            if anchor.0 < source.count, anchor.1 < target.count,
               source[anchor.0] != target[anchor.1],
               !isSentenceInitial(index: anchor.1, in: target) {
                results.append((from: source[anchor.0], to: target[anchor.1]))
            }
            sourceIndex = anchor.0 + 1
            targetIndex = anchor.1 + 1
        }
        return results
    }

    /// A capital at the start of a sentence says nothing about how the word is
    /// spelled — only that a sentence began there.
    private static func isSentenceInitial(index: Int, in words: [String]) -> Bool {
        guard index > 0 else { return true }
        guard let last = words[index - 1].last else { return false }
        return ".!?".contains(last)
    }

    // MARK: - Helpers

    private static func words(_ text: String) -> [String] {
        text.split(whereSeparator: { $0.isWhitespace })
            .map { $0.trimmingCharacters(in: CharacterSet(charactersIn: ".,!?;:\u{201C}\u{201D}\"()")) }
            .filter { !$0.isEmpty }
    }

    /// Index pairs of matched words, in order.
    private static func longestCommonSubsequence(_ source: [String], _ target: [String]) -> [(Int, Int)] {
        guard !source.isEmpty, !target.isEmpty else { return [] }
        var table = Array(repeating: Array(repeating: 0, count: target.count + 1), count: source.count + 1)
        for i in stride(from: source.count - 1, through: 0, by: -1) {
            for j in stride(from: target.count - 1, through: 0, by: -1) {
                table[i][j] = source[i] == target[j]
                    ? table[i + 1][j + 1] + 1
                    : max(table[i + 1][j], table[i][j + 1])
            }
        }
        var pairs: [(Int, Int)] = []
        var i = 0
        var j = 0
        while i < source.count && j < target.count {
            if source[i] == target[j] {
                pairs.append((i, j))
                i += 1
                j += 1
            } else if table[i + 1][j] >= table[i][j + 1] {
                i += 1
            } else {
                j += 1
            }
        }
        return pairs
    }

    static func editDistance(_ source: String, _ target: String) -> Int {
        let a = Array(source)
        let b = Array(target)
        guard !a.isEmpty else { return b.count }
        guard !b.isEmpty else { return a.count }
        var previous = Array(0...b.count)
        var current = Array(repeating: 0, count: b.count + 1)
        for i in 1...a.count {
            current[0] = i
            for j in 1...b.count {
                let cost = a[i - 1] == b[j - 1] ? 0 : 1
                current[j] = min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + cost)
            }
            previous = current
        }
        return previous[b.count]
    }
}
