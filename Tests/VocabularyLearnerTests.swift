import Foundation

enum VocabularyLearnerTests {
    static func run() {
        testRepeatedCorrectionBecomesATerm()
        testSingleCorrectionIsNotEnough()
        testCaseOnlyFixesAreLearned()
        testRewritesAreNotLearned()
        testStopWordsAndExistingTermsAreSkipped()
        testSubstitutionExtraction()
    }

    private static func testRepeatedCorrectionBecomesATerm() {
        let pairs = [
            (original: "I'll ask Aisha about the deploy.", corrected: "I'll ask Aysha about the deploy."),
            (original: "Aisha is reviewing it now.", corrected: "Aysha is reviewing it now."),
        ]
        let candidates = VocabularyLearner.candidates(from: pairs, existingVocabulary: "")
        TestSupport.expectEqual(candidates.count, 1)
        TestSupport.expectEqual(candidates.first?.term, "Aysha")
        TestSupport.expectEqual(candidates.first?.occurrences, 2)
        TestSupport.expectEqual(candidates.first?.heardAs, ["Aisha"])
    }

    private static func testSingleCorrectionIsNotEnough() {
        let pairs = [
            (original: "I'll ask Aisha about the deploy.", corrected: "I'll ask Aysha about the deploy.")
        ]
        TestSupport.expect(
            VocabularyLearner.candidates(from: pairs, existingVocabulary: "").isEmpty,
            "one correction should not create a vocabulary term"
        )
    }

    private static func testCaseOnlyFixesAreLearned() {
        // "groq" → "Groq" twice is exactly the signal we want.
        let pairs = [
            (original: "running it on groq now", corrected: "running it on Groq now"),
            (original: "the groq endpoint is slow", corrected: "the Groq endpoint is slow"),
        ]
        let candidates = VocabularyLearner.candidates(from: pairs, existingVocabulary: "")
        TestSupport.expectEqual(candidates.first?.term, "Groq")
    }

    private static func testRewritesAreNotLearned() {
        // A genuine change of meaning, not a misrecognition.
        let pairs = [
            (original: "let's meet Thursday", corrected: "let's meet Wednesday"),
            (original: "we ship Thursday", corrected: "we ship Wednesday"),
        ]
        TestSupport.expect(
            VocabularyLearner.candidates(from: pairs, existingVocabulary: "").isEmpty,
            "a distant word swap is a rewrite, not a spelling fix"
        )
        TestSupport.expect(
            !VocabularyLearner.isLearnable(from: "Thursday", to: "Wednesday"),
            "Thursday→Wednesday should not be learnable"
        )
        TestSupport.expect(
            VocabularyLearner.isLearnable(from: "Aisha", to: "Aysha"),
            "Aisha→Aysha should be learnable"
        )
    }

    private static func testStopWordsAndExistingTermsAreSkipped() {
        let stopWordPairs = [
            (original: "send it to them", corrected: "send it to him"),
            (original: "give it to them", corrected: "give it to him"),
        ]
        TestSupport.expect(
            VocabularyLearner.candidates(from: stopWordPairs, existingVocabulary: "").isEmpty,
            "common words should never become vocabulary"
        )

        let pairs = [
            (original: "ping Aisha", corrected: "ping Aysha"),
            (original: "Aisha shipped it", corrected: "Aysha shipped it"),
        ]
        TestSupport.expect(
            VocabularyLearner.candidates(from: pairs, existingVocabulary: "Aysha, Parakeet").isEmpty,
            "terms already in the vocabulary should not be re-suggested"
        )
    }

    private static func testSubstitutionExtraction() {
        let subs = VocabularyLearner.substitutions(
            original: "deploy the parakeet server",
            corrected: "deploy the Parakeet server"
        )
        TestSupport.expectEqual(subs.count, 1)
        TestSupport.expectEqual(subs.first?.from, "parakeet")
        TestSupport.expectEqual(subs.first?.to, "Parakeet")

        // A wholesale rewrite carries no per-word spelling signal.
        TestSupport.expect(
            VocabularyLearner.substitutions(
                original: "yes",
                corrected: "Actually let me get back to you on that tomorrow morning"
            ).isEmpty,
            "wholesale rewrites should yield no substitutions"
        )
    }
}
