import Foundation
import AppKit

enum RawCaptureStoreTests {
    static func run() async {
        let temporary = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: temporary) }
        do {
            let root = temporary.appendingPathComponent("raw")
            let store = RawCaptureStore(root: root)
            let stamp = Date(timeIntervalSince1970: 1_789_000_000)
            var record = RawObservation(id: UUID(), requestedAt: stamp, capturedAt: stamp,
                localTimestamp: "2026-09-08T09:32:00.000+10:00", timeZoneIdentifier: "Australia/Melbourne",
                utcOffsetSeconds: 36000, intervalSeconds: 7, idleSeconds: 125,
                appName: "Synthetic Editor", bundleIdentifier: "test.synthetic", processID: 123,
                executablePath: "/Applications/Synthetic Editor.app/Contents/MacOS/Editor", windowID: 42,
                windowTitle: "Synthetic notes", windowBounds: RawRect(CGRect(x: 10, y: 20, width: 800, height: 600)),
                imageWidth: 1600, imageHeight: 1200, osVersion: "Synthetic OS", documentURL: "file:///synthetic/notes.txt",
                focusedElementRole: "AXTextArea", frontmostAtCompletion: true)
            let literal = "PRIVATE_SYNTHETIC_OCR_123"
            record.ocr = RawOCR(status: "complete", text: literal)
            let index = try await store.save(record)
            let folder = try await store.folder(index)
            let storedNames = try FileManager.default.contentsOfDirectory(atPath: folder.path)
            TestSupport.expectEqual(Set(storedNames), Set(["index.json", "inference.json"]))
            for name in storedNames {
                let contents = try String(contentsOf: folder.appendingPathComponent(name), encoding: .utf8)
                TestSupport.expectEqual(contents.contains(literal), false)
                TestSupport.expectEqual(contents.contains(record.windowTitle), false)
            }
            let summary = "Reviewed export timing and repeated final frames."
            _ = try await store.saveInference(RawInference(status: "complete", summary: summary), index: index)
            // Simulate old source files and verify migration preserves summaries and unrelated files.
            for name in ["screenshot.png", "ocr.txt", "observation.json"] {
                try Data(literal.utf8).write(to: folder.appendingPathComponent(name))
            }
            try Data("keep".utf8).write(to: folder.appendingPathComponent("unrelated.txt"))
            try await store.removeSourceMaterial()
            try await store.removeSourceMaterial() // Idempotent on later launches.
            for name in ["screenshot.png", "ocr.txt", "observation.json"] {
                TestSupport.expectEqual(FileManager.default.fileExists(atPath: folder.appendingPathComponent(name).path), false)
            }
            TestSupport.expectEqual(FileManager.default.fileExists(atPath: folder.appendingPathComponent("unrelated.txt").path), true)
            let zone = TimeZone(identifier: "Australia/Melbourne")!
            let copied = try await store.summaryText(day: stamp, timeZone: zone)
            TestSupport.expectEqual(copied.contains(summary), true)
            TestSupport.expectEqual(copied.contains("Synthetic Editor"), true)
            TestSupport.expectEqual(copied.contains("+10:00"), true)
            TestSupport.expectEqual(copied.contains(literal), false)
            TestSupport.expectEqual(copied.contains("Synthetic notes"), false)
            TestSupport.expectEqual(copied.contains("RAW"), false)
            TestSupport.expectEqual(copied.hasPrefix(RawCaptureStore.copyPrompt + "\n\n"), true)
            TestSupport.expectEqual(copied.components(separatedBy: "Synthetic Editor:").count, 2)
            let empty = try await store.summaryText(day: stamp.addingTimeInterval(86400), timeZone: zone)
            TestSupport.expectEqual(empty, "")
            _ = try await store.saveInference(RawInference(status: "failed", summary: "Not a completed summary"), index: index)
            let failed = try await store.summaryText(day: stamp, timeZone: zone)
            TestSupport.expectEqual(failed, "")
            // A distant broken history entry must not be read while opening/copying this day.
            let distant = root.appendingPathComponent("2001-01-01/broken-entry")
            try FileManager.default.createDirectory(at: distant, withIntermediateDirectories: true)
            try Data("invalid synthetic index".utf8).write(to: distant.appendingPathComponent("index.json"))
            let nearby = try await store.list(on: stamp, timeZone: zone)
            TestSupport.expectEqual(nearby.count, 1)
            let distantIgnored = try await store.summaryText(day: stamp, timeZone: zone)
            TestSupport.expectEqual(distantIgnored, "")
            // Test pasteboard round-trip without reading or replacing the user's clipboard.
            let pasteboard = NSPasteboard.withUniqueName()
            defer { pasteboard.releaseGlobally() }
            TestSupport.expectEqual(pasteboard.setString(copied, forType: .string), true)
            TestSupport.expectEqual(pasteboard.string(forType: .string), copied)
            let mode = try FileManager.default.attributesOfItem(atPath: folder.path)[.posixPermissions] as? NSNumber
            TestSupport.expectEqual(mode?.intValue, 0o700)
            var malicious = index
            malicious.relativePath = "../../outside"
            var rejected = false
            do { _ = try await store.folder(malicious) } catch { rejected = true }
            TestSupport.expectEqual(rejected, true)
            print("Summary-only persistence, migration and clipboard round-trip passed")
        } catch { fatalError("Synthetic summary export test failed: \(error)") }
    }
}
