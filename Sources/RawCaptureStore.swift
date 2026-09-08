import Foundation

struct RawRect: Codable {
    var x: Double
    var y: Double
    var width: Double
    var height: Double
    init(_ rect: CGRect) { x = rect.origin.x; y = rect.origin.y; width = rect.width; height = rect.height }
}

struct RawOCRLine: Codable {
    var text: String
    var confidence: Float
    var boundingBox: RawRect
}

struct RawOCR: Codable {
    var engine = "Apple Vision"
    var languageCorrection = false
    var status = "pending"
    var text = ""
    var lines: [RawOCRLine] = []
    var error: String?
}

struct RawObservation: Codable, Identifiable {
    var schemaVersion = 1
    var id: UUID
    var requestedAt: Date
    var capturedAt: Date
    var localTimestamp: String
    var timeZoneIdentifier: String
    var utcOffsetSeconds: Int
    var intervalSeconds: Double
    var idleSeconds: Double?
    var appName: String
    var bundleIdentifier: String?
    var processID: Int32
    var executablePath: String?
    var windowID: UInt32
    var windowTitle: String
    var windowBounds: RawRect
    var imageWidth: Int
    var imageHeight: Int
    var osVersion: String
    var documentURL: String?
    var focusedElementRole: String?
    var frontmostAtCompletion: Bool
    var ocr = RawOCR()
}

struct RawInference: Codable {
    var model = JournalLocalModel.model
    var status: String
    var completedAt: Date?
    var summary: String?
    var category: String?
    var confidence: String?
    var error: String?
    var rawOCRCharacterCount: Int?
    var modelInputCharacterLimit = 18000
    var modelInputTruncated: Bool?
    // Optional so older text-only inference exports still decode.
    var inputMode: String?
}

struct RawCaptureIndex: Codable, Identifiable {
    var id: UUID
    var capturedAt: Date
    var day: String
    var appName: String
    var relativePath: String
    var summary = ""
    var inferenceStatus = "pending"
}

enum RawCaptureJSON {
    static func encoder() -> JSONEncoder {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        encoder.dateEncodingStrategy = .custom { date, encoder in
            var container = encoder.singleValueContainer()
            try container.encode(timestamp(date, timeZone: TimeZone(secondsFromGMT: 0)!))
        }
        return encoder
    }
    static func decoder() -> JSONDecoder {
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .custom { decoder in
            let container = try decoder.singleValueContainer()
            let value = try container.decode(String.self)
            let formatter = ISO8601DateFormatter()
            formatter.formatOptions = [.withInternetDateTime, .withFractionalSeconds]
            guard let date = formatter.date(from: value) else { throw CocoaError(.coderReadCorrupt) }
            return date
        }
        return decoder
    }
    static func timestamp(_ date: Date, timeZone: TimeZone) -> String {
        let formatter = ISO8601DateFormatter()
        formatter.formatOptions = [.withInternetDateTime, .withFractionalSeconds]
        formatter.timeZone = timeZone
        return formatter.string(from: date)
    }
    static func day(_ date: Date) -> String {
        let formatter = DateFormatter()
        formatter.locale = Locale(identifier: "en_US_POSIX")
        formatter.timeZone = .current
        formatter.dateFormat = "yyyy-MM-dd"
        return formatter.string(from: date)
    }
    static func write<T: Encodable>(_ value: T, to url: URL) throws {
        try encoder().encode(value).write(to: url, options: .atomic)
        try FileManager.default.setAttributes([.posixPermissions: 0o600], ofItemAtPath: url.path)
    }
}

actor RawCaptureStore {
    let root: URL
    init(root: URL) { self.root = root }

    func list(day: String? = nil) throws -> [RawCaptureIndex] {
        guard FileManager.default.fileExists(atPath: root.path) else { return [] }
        let days: [URL]
        if let day {
            guard day.range(of: #"^\d{4}-\d{2}-\d{2}$"#, options: .regularExpression) != nil else { throw CocoaError(.fileReadInvalidFileName) }
            let directory = root.appendingPathComponent(day)
            guard FileManager.default.fileExists(atPath: directory.path) else { return [] }
            days = [directory]
        } else {
            days = try FileManager.default.contentsOfDirectory(at: root, includingPropertiesForKeys: nil, options: .skipsHiddenFiles)
        }
        var result: [RawCaptureIndex] = []
        for directory in days where day == nil || directory.lastPathComponent == day {
            for folder in try FileManager.default.contentsOfDirectory(at: directory, includingPropertiesForKeys: nil, options: .skipsHiddenFiles) {
                let index = try RawCaptureJSON.decoder().decode(RawCaptureIndex.self, from: Data(contentsOf: folder.appendingPathComponent("index.json")))
                result.append(index)
            }
        }
        return result.sorted { $0.capturedAt < $1.capturedAt }
    }

    func save(_ record: RawObservation, model: String = JournalLocalModel.model) throws -> RawCaptureIndex {
        try save(capturedAt: record.capturedAt, appName: record.appName, id: record.id, model: model)
    }

    func save(capturedAt: Date, appName: String, id: UUID = UUID(), model: String = JournalLocalModel.model) throws -> RawCaptureIndex {
        let day = RawCaptureJSON.day(capturedAt)
        let directory = root.appendingPathComponent(day)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true, attributes: [.posixPermissions: 0o700])
        try FileManager.default.setAttributes([.posixPermissions: 0o700], ofItemAtPath: root.path)
        if let values = try? root.resourceValues(forKeys: [.volumeAvailableCapacityForImportantUsageKey]),
           let available = values.volumeAvailableCapacityForImportantUsage,
           available < 512 * 1024 * 1024 { throw CocoaError(.fileWriteOutOfSpace) }
        let staging = directory.appendingPathComponent("." + id.uuidString)
        let final = directory.appendingPathComponent(id.uuidString)
        try FileManager.default.createDirectory(at: staging, withIntermediateDirectories: false, attributes: [.posixPermissions: 0o700])
        defer { try? FileManager.default.removeItem(at: staging) }
        let index = RawCaptureIndex(id: id, capturedAt: capturedAt, day: day, appName: appName, relativePath: "\(day)/\(id.uuidString)")
        try RawCaptureJSON.write(RawInference(model: model, status: "pending"), to: staging.appendingPathComponent("inference.json"))
        try RawCaptureJSON.write(index, to: staging.appendingPathComponent("index.json"))
        try FileManager.default.moveItem(at: staging, to: final)
        return index
    }

    func folder(_ index: RawCaptureIndex) throws -> URL {
        // Never trust a path read from an editable local JSON file.
        guard index.relativePath == "\(index.day)/\(index.id.uuidString)",
              index.day.range(of: #"^\d{4}-\d{2}-\d{2}$"#, options: .regularExpression) != nil else { throw CocoaError(.fileReadInvalidFileName) }
        return root.appendingPathComponent(index.relativePath)
    }

    func saveInference(_ inference: RawInference, index: RawCaptureIndex) throws -> RawCaptureIndex {
        let directory = try folder(index)
        try RawCaptureJSON.write(inference, to: directory.appendingPathComponent("inference.json"))
        var updated = index
        updated.inferenceStatus = inference.status
        updated.summary = inference.summary ?? ""
        try RawCaptureJSON.write(updated, to: directory.appendingPathComponent("index.json"))
        return updated
    }

    // Remove legacy source material, including OCR embedded inside observation.json.
    // Summary/index files and unrelated user files are preserved.
    func removeSourceMaterial() throws {
        guard FileManager.default.fileExists(atPath: root.path) else { return }
        guard let files = FileManager.default.enumerator(at: root, includingPropertiesForKeys: [.isRegularFileKey], options: []) else { return }
        let sourceNames: Set<String> = ["screenshot.png", "ocr.txt", "observation.json"]
        for case let file as URL in files where sourceNames.contains(file.lastPathComponent) {
            if try file.resourceValues(forKeys: [.isRegularFileKey]).isRegularFile == true {
                try FileManager.default.removeItem(at: file)
            }
        }
    }

    func summaryText(day: Date, timeZone: TimeZone = .current) throws -> String {
        var calendar = Calendar(identifier: .gregorian)
        calendar.timeZone = timeZone
        let time = DateFormatter()
        time.locale = Locale(identifier: "en_US_POSIX")
        time.timeZone = timeZone
        time.dateFormat = "h:mma"
        let offset = DateFormatter()
        offset.locale = Locale(identifier: "en_US_POSIX")
        offset.timeZone = timeZone
        offset.dateFormat = "XXXXX"
        let entries = try list(on: day, timeZone: timeZone).filter {
            calendar.isDate($0.capturedAt, inSameDayAs: day) && $0.inferenceStatus == "complete" &&
            !$0.summary.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
        }.map { item in
            let summary = item.summary.split(whereSeparator: { $0.isWhitespace }).joined(separator: " ")
            let zone = offset.string(from: item.capturedAt)
            return "\(time.string(from: item.capturedAt).lowercased()) \(zone) · \(item.appName): \(summary)"
        }.joined(separator: "\n")
        guard !entries.isEmpty else { return "" }
        return Self.copyPrompt + "\n\n" + entries
    }

    /// Read only nearby day folders, even after crossing timezones. History size
    /// does not increase the memory or I/O needed to show or copy a single day.
    func list(on day: Date, timeZone: TimeZone = .current) throws -> [RawCaptureIndex] {
        var calendar = Calendar(identifier: .gregorian)
        calendar.timeZone = timeZone
        let formatter = DateFormatter()
        formatter.locale = Locale(identifier: "en_US_POSIX")
        formatter.timeZone = timeZone
        formatter.dateFormat = "yyyy-MM-dd"
        var result: [RawCaptureIndex] = []
        for offset in -2...2 {
            let nearby = calendar.date(byAdding: .day, value: offset, to: day)!
            result += try list(day: formatter.string(from: nearby)).filter { calendar.isDate($0.capturedAt, inSameDayAs: day) }
        }
        return result.sorted { $0.capturedAt < $1.capturedAt }
    }

    static let copyPrompt = """
    Summarize the work topics below in short, easy-to-read bullet points, grouped by time. These are occasional screen observations, not proof of completed actions or continuous work. Gaps can mean the journal was stopped, the computer was idle, or the screen was unchanged. Do not infer exact hours worked from the timestamps.
    """
}
