import Foundation

enum JournalPolicy {
    static let defaultInterval = 180.0
    static let minimumInterval = 60.0
    static let maximumInterval = 900.0
    static let idleLimit = 120.0
    static let maximumImageDimension = 1920
    static let maximumInputCharacters = 2400
    static let maximumInputBytes = 1200
    static let contextTokens = 2048
    static let outputTokens = 160
    static let model = "qwen2.5:0.5b"

    static func activityWindow(interval: TimeInterval) -> TimeInterval {
        // Include the whole sampling interval, including timer coalescing. A
        // two-minute cutoff with a three-minute timer loses activity between checks.
        max(idleLimit, interval + min(15, interval * 0.1))
    }

    static func imageSize(width: Int, height: Int) -> (width: Int, height: Int) {
        guard width > 0, height > 0 else { return (1, 1) }
        let scale = min(1, Double(maximumImageDimension) / Double(max(width, height)))
        return (max(1, Int(Double(width) * scale)), max(1, Int(Double(height) * scale)))
    }

    // UTF-8 bytes also bound worst-case token counts for code and non-Latin text.
    static func boundedText(_ text: String, bytes limit: Int = maximumInputBytes) -> String {
        var result = ""
        var bytes = 0
        for character in text {
            let count = String(character).utf8.count
            guard bytes + count <= limit else { break }
            result.append(character)
            bytes += count
        }
        return result
    }
}

/// Session consent is intentionally never restored from preferences.
struct JournalSession {
    private(set) var enabled = false
    private(set) var generation = UUID()
    private(set) var awaitingFirstApp = false

    mutating func setEnabled(_ value: Bool) {
        enabled = value
        generation = UUID()
        awaitingFirstApp = value
    }

    mutating func invalidate() { generation = UUID() }

    func accepts(_ token: UUID) -> Bool { enabled && generation == token }

    func canCapture(busy: Bool, suspended: Bool, idleSeconds: Double, activityWindow: Double = JournalPolicy.idleLimit) -> Bool {
        enabled && !busy && !suspended && idleSeconds.isFinite && idleSeconds >= 0 &&
            (awaitingFirstApp || idleSeconds < activityWindow)
    }

    func shouldRetryOnActivation(isEligible: Bool) -> Bool { enabled && awaitingFirstApp && isEligible }
    mutating func didBeginCapture() { awaitingFirstApp = false }
}

/// Only hashes survive between samples. Failed summaries are retried.
struct JournalDeduplicator {
    private var image: String?
    private var text: String?
    func hasImage(_ value: String) -> Bool { image == value }
    func hasText(_ value: String) -> Bool { text == value }
    mutating func remember(image: String, text: String) { self.image = image; self.text = text }
    mutating func reset() { image = nil; text = nil }
}
