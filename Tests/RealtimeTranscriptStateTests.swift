import Foundation

enum RealtimeTranscriptStateTests {
    static func run() {
        var state = RealtimeTranscriptState()
        state.complete("Keep", itemID: "part1")
        state.beginCommit()
        // This was already in flight when the user released the shortcut.
        state.complete("every", itemID: "part2")
        TestSupport.expectEqual(state.readyTranscript, nil)
        state.acknowledgeCommit(itemID: "final")
        state.complete("single", itemID: "part3")
        TestSupport.expectEqual(state.readyTranscript, nil)
        state.appendDelta("word")
        TestSupport.expectEqual(state.readyTranscript, nil)
        state.complete("word.", itemID: "final")
        TestSupport.expectEqual(state.readyTranscript, "Keep every single word.")

        var emptyTail = RealtimeTranscriptState()
        emptyTail.complete("All settled.", itemID: "part")
        emptyTail.beginCommit()
        emptyTail.acknowledgeCommit(itemID: "final")
        emptyTail.complete("", itemID: "final")
        TestSupport.expectEqual(emptyTail.readyTranscript, "All settled.")

        var malformed = RealtimeTranscriptState()
        malformed.beginCommit()
        malformed.acknowledgeCommit(itemID: nil)
        malformed.complete("Uncorrelated", itemID: nil)
        TestSupport.expectEqual(malformed.readyTranscript, nil)

        // Older local servers reuse an item ID for successive settled chunks.
        // Preserve those chunks, but require completion after the commit ack.
        var legacy = RealtimeTranscriptState()
        legacy.complete("First", itemID: "same")
        legacy.beginCommit()
        legacy.complete("second", itemID: "same")
        legacy.acknowledgeCommit(itemID: "same")
        TestSupport.expectEqual(legacy.readyTranscript, nil)
        legacy.complete("third.", itemID: "same")
        TestSupport.expectEqual(legacy.readyTranscript, "First second third.")
    }
}
