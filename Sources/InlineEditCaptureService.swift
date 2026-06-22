import Foundation
import ApplicationServices
import AppKit
import os

private let captureLog = OSLog(subsystem: "com.zachlatta.freeflow", category: "InlineEditCapture")

/// Passively captures the edits you make to dictated text *in the app you typed
/// into*, so you don't have to correct anything by hand in the Run Log.
///
/// Flow:
///   1. Right after FreeFlow pastes, we find the focused field and (for
///      Electron/Chromium apps) force it to expose its Accessibility tree.
///   2. We then POLL that field for ~1 minute. The moment its text contains our
///      inserted span we lock in the surrounding prefix/suffix; after that, every
///      change to the span is remembered as the latest candidate correction.
///   3. We emit the candidate when you start the next dictation, OR when the field
///      clears (you sent the message — common in chat apps), OR when the poll
///      window ends. Polling-before-send is what makes this work in Claude /
///      Telegram / browsers, where the field empties on send before any
///      "next dictation" re-read could fire.
///
/// Conservative: silent unless confident, skips secure fields, never emits a
/// change that touches the surrounding text. Writes a human-readable trace to
/// `~/Library/Application Support/FreeFlow Dev/inline-capture.log` so capture
/// behaviour is diagnosable (os_log default-level messages don't reliably surface
/// for ad-hoc/dev-signed apps). All calls run on the main thread.
final class InlineEditCaptureService {
    private final class Pending {
        let itemID: UUID
        let element: AXUIElement
        let inserted: String
        let started: Date
        var prefix: String?     // set once we first see `inserted` in the field
        var suffix: String?
        var best: String?       // latest edited span that differs from `inserted`
        init(itemID: UUID, element: AXUIElement, inserted: String, started: Date) {
            self.itemID = itemID
            self.element = element
            self.inserted = inserted
            self.started = started
        }
    }

    private var pending: Pending?
    private var pollTimer: Timer?

    private let pollInterval: TimeInterval = 1.5
    private let establishGiveUp: TimeInterval = 12   // field never exposed our text
    private let maxDuration: TimeInterval = 60       // overall capture window

    /// Invoked on the main thread with (historyItemID, correctedText) when a
    /// confident inline edit is detected.
    var onCorrection: ((UUID, String) -> Void)?

    // MARK: - Public entry points

    /// Call right after FreeFlow pastes `inserted` for history item `itemID`.
    func noteInsertion(itemID: UUID, inserted: String) {
        flush()                 // emit any still-pending previous edit first
        stopPolling()
        pending = nil

        guard inserted.trimmingCharacters(in: .whitespacesAndNewlines).count >= 2 else {
            debugLog("noteInsertion: skipped — inserted text too short")
            return
        }
        guard let element = focusedElement() else {
            debugLog("noteInsertion: skipped — no focused Accessibility element")
            return
        }
        if isSecureField(element) {
            debugLog("noteInsertion: skipped — secure (password) field")
            return
        }
        // Electron/Chromium/browsers build their AX tree lazily; nudge it on.
        forceAccessibility(for: element)

        let p = Pending(itemID: itemID, element: element, inserted: inserted, started: Date())
        let role = string(element, kAXRoleAttribute as CFString) ?? "?"
        if let value = string(element, kAXValueAttribute as CFString), !value.isEmpty {
            tryEstablish(p, value: value)
            if p.prefix != nil {
                debugLog("noteInsertion: ARMED for \(itemID.uuidString.prefix(8)) (role=\(role), \(value.count) chars) — watching for edits")
            } else {
                debugLog("noteInsertion: field has text but not our paste yet (role=\(role)) — will keep polling")
            }
        } else {
            debugLog("noteInsertion: field exposes no AX value yet (role=\(role)) — polling in case the tree fills in")
        }
        pending = p
        startPolling()
    }

    /// Emit the best pending correction (called when the next dictation starts).
    func flush() {
        guard let p = pending else { return }
        debugLog("flush: emitting on next-dictation")
        emit(p, reason: "next-dictation")
        stopPolling()
        pending = nil
    }

    // MARK: - Polling

    private func startPolling() {
        stopPolling()
        let t = Timer.scheduledTimer(withTimeInterval: pollInterval, repeats: true) { [weak self] _ in
            self?.poll()
        }
        RunLoop.main.add(t, forMode: .common)
        pollTimer = t
    }

    private func stopPolling() {
        pollTimer?.invalidate()
        pollTimer = nil
    }

    private func poll() {
        guard let p = pending else { stopPolling(); return }
        let elapsed = Date().timeIntervalSince(p.started)
        let value = string(p.element, kAXValueAttribute as CFString)

        if p.prefix == nil {
            // Still trying to lock onto our pasted text.
            if let value, !value.isEmpty {
                tryEstablish(p, value: value)
                if p.prefix != nil { debugLog("poll: locked onto pasted text (\(value.count) chars)") }
            }
            if p.prefix == nil && elapsed > establishGiveUp {
                debugLog("poll: gave up — field never exposed our pasted text (Electron/web field not AX-readable). elapsed=\(Int(elapsed))s")
                pending = nil
                stopPolling()
            }
            return
        }

        // Established: watch the span between prefix and suffix.
        guard let prefix = p.prefix, let suffix = p.suffix else { return }
        if value == nil || !(value!.hasPrefix(prefix) && value!.hasSuffix(suffix)
                             && value!.count >= prefix.count + suffix.count) {
            // Field cleared or changed out from under us — almost always "sent".
            debugLog("poll: field cleared/changed (sent?) — finalizing. best=\(p.best != nil ? "yes" : "none")")
            emit(p, reason: "field-cleared")
            pending = nil
            stopPolling()
            return
        }

        let v = value!
        let start = v.index(v.startIndex, offsetBy: prefix.count)
        let end = v.index(v.endIndex, offsetBy: -suffix.count)
        let span = String(v[start..<end])
        if span != p.inserted, isPlausibleCorrection(span, inserted: p.inserted) {
            if span != p.best {
                p.best = span
                debugLog("poll: candidate updated -> \(span.prefix(60))")
            }
        }

        if elapsed > maxDuration {
            debugLog("poll: window elapsed — finalizing. best=\(p.best != nil ? "yes" : "none")")
            emit(p, reason: "timeout")
            pending = nil
            stopPolling()
        }
    }

    // MARK: - Establish / emit

    /// Lock onto where our pasted text sits in the field, recording the
    /// unchanging text before and after it.
    private func tryEstablish(_ p: Pending, value: String) {
        guard let range = value.range(of: p.inserted, options: .backwards) else { return }
        p.prefix = String(value[..<range.lowerBound])
        p.suffix = String(value[range.upperBound...])
    }

    private func emit(_ p: Pending, reason: String) {
        guard let best = p.best else { return }
        guard isPlausibleCorrection(best, inserted: p.inserted) else {
            debugLog("emit(\(reason)): rejected — change empty/identical/too large")
            return
        }
        debugLog("emit(\(reason)): CAPTURED correction for \(p.itemID.uuidString.prefix(8))")
        onCorrection?(p.itemID, best)
    }

    private func isPlausibleCorrection(_ candidate: String, inserted: String) -> Bool {
        let trimmed = candidate.trimmingCharacters(in: .whitespacesAndNewlines)
        return !trimmed.isEmpty
            && candidate != inserted
            && candidate.count <= max(80, inserted.count * 4)
    }

    // MARK: - Accessibility helpers (same style as AppContextService)

    private func focusedElement() -> AXUIElement? {
        let systemWide = AXUIElementCreateSystemWide()
        return element(from: systemWide, attribute: kAXFocusedUIElementAttribute as CFString)
    }

    /// Force Electron/Chromium/Catalyst apps to build their AX tree so the
    /// focused text field exposes a value. Harmless for native apps.
    private func forceAccessibility(for element: AXUIElement) {
        var pid: pid_t = 0
        guard AXUIElementGetPid(element, &pid) == .success, pid > 0 else { return }
        let app = AXUIElementCreateApplication(pid)
        AXUIElementSetAttributeValue(app, "AXManualAccessibility" as CFString, kCFBooleanTrue)
    }

    private func element(from element: AXUIElement, attribute: CFString) -> AXUIElement? {
        var value: CFTypeRef?
        guard AXUIElementCopyAttributeValue(element, attribute, &value) == .success,
              let raw = value,
              CFGetTypeID(raw) == AXUIElementGetTypeID() else {
            return nil
        }
        return unsafeBitCast(raw, to: AXUIElement.self)
    }

    private func string(_ element: AXUIElement, _ attribute: CFString) -> String? {
        var value: CFTypeRef?
        guard AXUIElementCopyAttributeValue(element, attribute, &value) == .success else {
            return nil
        }
        return value as? String
    }

    private func isSecureField(_ element: AXUIElement) -> Bool {
        for attribute in [kAXRoleAttribute, kAXSubroleAttribute] {
            if let role = string(element, attribute as CFString),
               role.lowercased().contains("secure") {
                return true
            }
        }
        return false
    }

    // MARK: - Debug log (readable trace; os_log doesn't surface for dev builds)

    private static let logURL: URL = {
        let base = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask).first!
        return base.appendingPathComponent("\(AppName.displayName)/inline-capture.log")
    }()

    private func debugLog(_ message: String) {
        os_log(.default, log: captureLog, "%{public}@", message)
        let line = "\(Self.ts()) \(message)\n"
        guard let data = line.data(using: .utf8) else { return }
        let url = Self.logURL
        if let fh = try? FileHandle(forWritingTo: url) {
            defer { try? fh.close() }
            fh.seekToEndOfFile()
            fh.write(data)
        } else {
            try? FileManager.default.createDirectory(
                at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
            try? line.write(to: url, atomically: true, encoding: .utf8)
        }
    }

    private static func ts() -> String {
        let f = DateFormatter()
        f.dateFormat = "HH:mm:ss"
        return f.string(from: Date())
    }
}
