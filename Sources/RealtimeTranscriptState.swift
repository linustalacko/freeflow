import Foundation

/// Protocol state, separate from the socket so event ordering is testable.
/// A completion received after key-up may still be an earlier settled chunk.
/// Only the item acknowledged by the server can finish the explicit commit.
struct RealtimeTranscriptState {
    private(set) var finalText = ""
    private(set) var partialText = ""
    private(set) var commitSent = false
    private(set) var committedItemID: String?
    private(set) var postCommitCompleted = false

    mutating func beginCommit() {
        commitSent = true
    }

    mutating func acknowledgeCommit(itemID: String?) {
        guard commitSent, committedItemID == nil,
              let itemID, !itemID.isEmpty else { return }
        committedItemID = itemID
    }

    mutating func appendDelta(_ delta: String) {
        partialText += delta
    }

    mutating func complete(_ transcript: String, itemID: String?) {
        let trimmed = transcript.trimmingCharacters(in: .whitespacesAndNewlines)
        if !trimmed.isEmpty {
            if !finalText.isEmpty { finalText += " " }
            finalText += trimmed
        }
        partialText = ""
        if let committedItemID, itemID == committedItemID {
            postCommitCompleted = true
        }
    }

    var readyTranscript: String? {
        guard commitSent, postCommitCompleted else { return nil }
        return finalText
    }
}
