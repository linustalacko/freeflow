import Foundation

enum JournalPolicy {
    static let defaultInterval = 180.0
    static let minimumInterval = 60.0
    static let maximumInterval = 900.0
    static let idleLimit = 120.0
    static let maximumImageDimension = 1280
    static let maximumInputCharacters = 2400
    static let maximumInputBytes = 1200
    static let contextTokens = 2048
    static let outputTokens = 160
    static let model = "qwen2.5:0.5b"

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

    mutating func setEnabled(_ value: Bool) {
        enabled = value
        generation = UUID()
    }

    mutating func invalidate() { generation = UUID() }

    func accepts(_ token: UUID) -> Bool { enabled && generation == token }

    func canCapture(busy: Bool, suspended: Bool, idleSeconds: Double) -> Bool {
        enabled && !busy && !suspended && idleSeconds.isFinite && idleSeconds >= 0 && idleSeconds < JournalPolicy.idleLimit
    }
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
