import Foundation

struct JournalEntry: Codable, Identifiable {
    var id = UUID()
    var start: Date
    var end: Date
    var seconds: Double = 0
    var app: String
    var category = "Unclassified"
    var summary = "Awaiting local interpretation of visible activity."
    var confidence = "unclassified"
    var sampleCount = 0
    var goalID: UUID?
    var edited = false
}

struct JournalGoal: Codable, Identifiable {
    var id = UUID()
    var day: Date
    var text: String
    var minutes: Double
}

struct JournalArchive: Codable {
    var entries: [JournalEntry] = []
    var goals: [JournalGoal] = []
}

enum JournalCore {
    static func captureInterval(_ seconds: Double) -> Double {
        seconds.isFinite && seconds > 0 ? min(JournalPolicy.maximumInterval, max(JournalPolicy.minimumInterval, seconds.rounded())) : JournalPolicy.defaultInterval
    }

    static func commitHistory(_ entries: [JournalEntry], day: Date, calendar: Calendar = .current) -> String {
        let time = DateFormatter()
        time.locale = Locale(identifier: "en_US_POSIX")
        time.timeZone = calendar.timeZone
        time.dateFormat = "h:mm"
        func period(_ date: Date) -> String { calendar.component(.hour, from: date) < 12 ? "am" : "pm" }
        return entries.filter { calendar.isDate($0.start, inSameDayAs: day) && $0.end >= $0.start }
            .sorted { $0.start < $1.start }
            .compactMap { entry in
                let summary = entry.summary.split(whereSeparator: { $0.isWhitespace }).joined(separator: " ")
                guard !summary.isEmpty else { return nil }
                let startPeriod = period(entry.start) == period(entry.end) ? "" : period(entry.start)
                return "\(time.string(from: entry.start))\(startPeriod)-\(time.string(from: entry.end))\(period(entry.end)): \(summary)"
            }.joined(separator: "\n")
    }

    // Long scheduling gaps are unknown time, never attributed to the previous app.
    static func creditedSeconds(from: Date, to: Date, idle: Double) -> Double {
        let elapsed = to.timeIntervalSince(from)
        guard elapsed > 0, elapsed <= 35, idle.isFinite, idle >= 0 else { return 0 }
        return max(0, elapsed - max(0, idle - 120))
    }

    static func redact(_ text: String) -> String {
        var result = text
        let patterns = [
            #"(?i)\b(?:https?://|www\.)\S+"#,
            #"(?i)\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b"#,
            #"(?:/Users/|/home/)[^\s]+"#,
            #"(?i)\b(?:sk-|ghp_|github_pat_|xox[baprs]-)[A-Za-z0-9_-]+"#,
            #"(?i)\b(?:api[_ -]?key|password|secret|authorization|token)\s*[:=]\s*\S+"#,
            #"\b[A-Za-z0-9_+/=-]{40,}\b"#
        ]
        for pattern in patterns {
            result = result.replacingOccurrences(of: pattern, with: "[redacted]", options: .regularExpression)
        }
        return result
    }

    static func excluded(bundleID: String, name: String, custom: String) -> Bool {
        let builtins = ["1password", "bitwarden", "keychain", "lastpass", "keepass", "dashlane", "loginwindow"]
        let terms = builtins + custom.split(separator: ",").map { $0.trimmingCharacters(in: .whitespacesAndNewlines).lowercased() }.filter { !$0.isEmpty }
        let target = "\(bundleID) \(name)".lowercased()
        return terms.contains { target.contains($0) }
    }

    static func evidence(_ text: String) -> String {
        // Defense in depth for common prompt-injection markers. This does not make
        // model interpretation authoritative; all entries remain user-correctable.
        let pattern = #"(?i)(ignore\s+(all\s+|previous\s+|prior\s+)*(instructions|rules|prompts)|system\s*(message|prompt|override)\s*:|<\|(?:im_start|system|assistant)|\[INST\]|output\s+(the\s+)?(password|secret|api.?key))"#
        return redact(text.split(separator: "\n", omittingEmptySubsequences: false).map { line in
            line.range(of: pattern, options: .regularExpression) == nil ? String(line) : "[instruction-like text omitted]"
        }.joined(separator: "\n"))
    }

    struct Interpretation: Decodable {
        var category: String
        var summary: String
        var confidence: String
        var goalID: String?
        var continuesPrevious: Bool? = nil
    }

    static func interpretation(_ data: Data) throws -> Interpretation {
        let value = try JSONDecoder().decode(Interpretation.self, from: data)
        guard !value.summary.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty,
              value.summary.count <= 900, !value.category.isEmpty, value.category.count <= 80,
              ["low", "medium", "high"].contains(value.confidence) else {
            throw CocoaError(.coderReadCorrupt)
        }
        return Interpretation(category: redact(value.category), summary: redact(value.summary), confidence: value.confidence, goalID: value.goalID, continuesPrevious: value.continuesPrevious)
    }
}

struct JournalDiskStore {
    let directory: URL
    var url: URL { directory.appendingPathComponent("journal.json") }

    func load() throws -> JournalArchive {
        guard FileManager.default.fileExists(atPath: url.path) else { return JournalArchive() }
        return try JSONDecoder().decode(JournalArchive.self, from: Data(contentsOf: url))
    }

    func save(_ archive: JournalArchive) throws {
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true, attributes: [.posixPermissions: 0o700])
        try FileManager.default.setAttributes([.posixPermissions: 0o700], ofItemAtPath: directory.path)
        // Create the temporary file private from its first byte, then atomically replace.
        let temporary = directory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: temporary) }
        guard FileManager.default.createFile(atPath: temporary.path, contents: try JSONEncoder().encode(archive), attributes: [.posixPermissions: 0o600]) else {
            throw CocoaError(.fileWriteUnknown)
        }
        if FileManager.default.fileExists(atPath: url.path) {
            _ = try FileManager.default.replaceItemAt(url, withItemAt: temporary)
        } else {
            try FileManager.default.moveItem(at: temporary, to: url)
        }
    }
}
