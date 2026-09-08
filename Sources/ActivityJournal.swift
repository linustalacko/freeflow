import AppKit
import SwiftUI

private enum JournalSample {
    case captured(Date, String, String, String)
    case skipped(String)
}

@MainActor
final class ActivityJournal: ObservableObject {
    static let shared = ActivityJournal()
    @Published private var session = JournalSession()
    var enabled: Bool { session.enabled }
    @Published private(set) var records: [RawCaptureIndex] = []
    @Published private(set) var status = "Stopped"
    @Published private(set) var storageError: String?
    @Published private(set) var exporting = false
    @Published private(set) var captureInterval = JournalCore.captureInterval(UserDefaults.standard.double(forKey: "journal_lightweight_interval"))
    @Published var excludedApps = UserDefaults.standard.string(forKey: "journal_excluded_apps") ?? "" {
        didSet { UserDefaults.standard.set(excludedApps, forKey: "journal_excluded_apps") }
    }
    private let root: URL
    private let store: RawCaptureStore
    private let runtime = JournalModelRuntime()
    private var selectedDay = Date()
    private var viewGeneration = UUID()
    private var timer: Timer?
    private var suspended = false
    private var observers: [NSObjectProtocol] = []
    private var captureTask: Task<Void, Never>?
    private var deduplicator = JournalDeduplicator()
    private var window: NSWindow?
    private var prepared = false

    private init() {
        root = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
            .appendingPathComponent(AppName.displayName).appendingPathComponent("GitForWorkRaw")
        store = RawCaptureStore(root: root)
    }

    func start() {
        guard !enabled, storageError == nil else { return }
        guard CGPreflightScreenCaptureAccess() else { status = "Screen Recording permission required"; showWindow(); return }
        guard JournalModelRuntime.executable != nil else { status = JournalModelRuntime.Failure.notInstalled.localizedDescription; showWindow(); return }
        session.setEnabled(true)
        deduplicator.reset()
        suspended = false
        let center = NSWorkspace.shared.notificationCenter
        for name in [NSWorkspace.willSleepNotification, NSWorkspace.screensDidSleepNotification, NSWorkspace.sessionDidResignActiveNotification] {
            observers.append(center.addObserver(forName: name, object: nil, queue: .main) { [weak self] _ in
                Task { @MainActor in
                    guard let self, self.enabled else { return }
                    self.suspended = true
                    self.session.invalidate()
                    self.captureTask?.cancel()
                    self.runtime.stop()
                    self.status = "Asleep · no screenshots"
                }
            })
        }
        for name in [NSWorkspace.didWakeNotification, NSWorkspace.screensDidWakeNotification, NSWorkspace.sessionDidBecomeActiveNotification] {
            observers.append(center.addObserver(forName: name, object: nil, queue: .main) { [weak self] _ in
                Task { @MainActor in self?.suspended = false }
            })
        }
        status = "Journal running"
        configureTimer()
        capture()
    }

    func stop() {
        session.setEnabled(false)
        timer?.invalidate()
        timer = nil
        for observer in observers { NSWorkspace.shared.notificationCenter.removeObserver(observer) }
        observers.removeAll()
        captureTask?.cancel()
        runtime.stop()
        deduplicator.reset()
        status = "Stopped"
    }

    func shutdown() { stop() }

    func showDay(_ day: Date) {
        selectedDay = day
        let token = UUID()
        viewGeneration = token
        Task {
            do {
                let items = try await store.list(on: day)
                if viewGeneration == token { records = items }
            } catch { storageError = "Could not read summaries. Existing data has been preserved." }
        }
    }

    func setCaptureInterval(_ seconds: Double) {
        captureInterval = JournalCore.captureInterval(seconds)
        UserDefaults.standard.set(captureInterval, forKey: "journal_lightweight_interval")
        configureTimer()
    }

    private func configureTimer() {
        timer?.invalidate()
        timer = nil
        guard enabled, storageError == nil else { return }
        timer = Timer.scheduledTimer(withTimeInterval: captureInterval, repeats: true) { [weak self] _ in
            Task { @MainActor in self?.capture() }
        }
        timer?.tolerance = min(15, captureInterval * 0.1)
    }

    func requestCapturePermission() {
        guard !CGPreflightScreenCaptureAccess() else { return }
        let key = "journal_screen_permission_requested"
        if !UserDefaults.standard.bool(forKey: key) {
            UserDefaults.standard.set(true, forKey: key)
            _ = CGRequestScreenCaptureAccess()
        } else { openPermissions() }
    }

    func openPermissions() {
        NSWorkspace.shared.open(URL(string: "x-apple.systempreferences:com.apple.preference.security?Privacy_ScreenCapture")!)
    }

    private func publish(_ index: RawCaptureIndex) {
        guard Calendar.current.isDate(index.capturedAt, inSameDayAs: selectedDay) else { return }
        if let position = records.firstIndex(where: { $0.id == index.id }) { records[position] = index }
        else { records.append(index) }
    }

    private func failStorage() {
        stop()
        storageError = "Journal stopped. Check disk space and file access, then restart the app. Summaries are preserved."
    }

    private func capture() {
        guard enabled, storageError == nil else { return }
        let idle = JournalCapture.idleSeconds()
        guard session.canCapture(busy: captureTask != nil, suspended: suspended, idleSeconds: idle) else {
            if captureTask == nil { status = suspended ? "Asleep · no screenshots" : "Idle · no screenshots" }
            return
        }
        if let state = CGSessionCopyCurrentDictionary() as? [String: Any], state["CGSSessionScreenIsLocked"] as? Bool == true {
            status = "Locked · no screenshots"; return
        }
        guard CGPreflightScreenCaptureAccess() else { stop(); status = "Screen Recording permission required"; return }
        guard let app = NSWorkspace.shared.frontmostApplication, app.bundleIdentifier != Bundle.main.bundleIdentifier,
              !JournalCore.excluded(bundleID: app.bundleIdentifier ?? "", name: app.localizedName ?? "", custom: excludedApps) else {
            status = "Excluded app · no screenshots"; return
        }
        let token = session.generation
        let pid = app.processIdentifier
        let name = app.localizedName ?? "Unknown app"
        let previous = deduplicator
        status = "Reading screen"
        captureTask = Task(priority: .utility) {
            var pending: RawCaptureIndex?
            do {
                if !prepared {
                    try await store.removeSourceMaterial()
                    for item in try await store.list(on: Date()) where item.inferenceStatus == "pending" {
                        _ = try await store.saveInference(RawInference(status: "interrupted", error: "Previous session ended."), index: item)
                    }
                    prepared = true
                }
                try Task.checkCancellation()
                // The image lives only in this child task and is released before model loading.
                let worker = Task.detached(priority: .utility) { () -> JournalSample in
                    guard let capture = try await JournalCapture.captureWindow(pid: pid) else { return .skipped("Window unavailable or private · skipped") }
                    let context = "\(pid):\(capture.windowID)"
                    let imageHash = JournalCapture.imageFingerprint(capture.image, context: context)
                    guard !previous.hasImage(imageHash) else { return .skipped("Unchanged screen · skipped") }
                    try Task.checkCancellation()
                    let text = try JournalCapture.recognize(image: capture.image)
                    try Task.checkCancellation()
                    guard !text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else { return .skipped("No readable text · skipped") }
                    let textHash = JournalCapture.textFingerprint(text, context: context)
                    guard !previous.hasText(textHash) else { return .skipped("Unchanged text · skipped") }
                    return .captured(capture.capturedAt, imageHash, textHash, "Window: \(String(capture.title.prefix(160)))\n\(text)")
                }
                let sample = try await withTaskCancellationHandler(operation: { try await worker.value }, onCancel: { worker.cancel() })
                guard session.accepts(token) else { throw CancellationError() }
                if case let .captured(capturedAt, imageHash, textHash, evidence) = sample {
                    // A focus change during capture discards the sample instead of describing the wrong app.
                    guard NSWorkspace.shared.frontmostApplication?.processIdentifier == pid else {
                        status = "App changed · skipped"
                        captureTask = nil
                        return
                    }
                    let index = try await store.save(capturedAt: capturedAt, appName: name)
                    pending = index
                    guard session.accepts(token) else { throw CancellationError() }
                    publish(index)
                    status = "Summarizing locally"
                    try await runtime.start()
                    let result = try await JournalLocalModel.summarize(app: name, observations: evidence)
                    guard session.accepts(token) else { throw CancellationError() }
                    let inference = RawInference(status: "complete", completedAt: Date(), summary: result.summary,
                                                 category: result.category, confidence: result.confidence,
                                                 modelInputCharacterLimit: JournalPolicy.maximumInputCharacters, inputMode: "ocr")
                    publish(try await store.saveInference(inference, index: index))
                    pending = nil
                    deduplicator.remember(image: imageHash, text: textHash)
                    status = "Journal running · model released between summaries"
                } else if case let .skipped(reason) = sample { status = reason }
            } catch {
                let cancelled = Task.isCancelled || !session.accepts(token)
                if let index = pending {
                    do {
                        let inference = RawInference(status: cancelled ? "interrupted" : "failed", completedAt: Date(),
                                                     error: cancelled ? "Journal stopped; source discarded." : "Local summary unavailable; source discarded.")
                        publish(try await store.saveInference(inference, index: index))
                    } catch { failStorage() }
                }
                if !cancelled, enabled {
                    if let failure = error as? JournalModelRuntime.Failure { status = failure.localizedDescription }
                    else { status = "Summary unavailable; retrying at the next interval" }
                }
            }
            await runtime.finish()
            captureTask = nil
        }
    }

    func revealCapture(_ index: RawCaptureIndex) {
        Task { if let folder = try? await store.folder(index) { NSWorkspace.shared.activateFileViewerSelecting([folder]) } }
    }

    func revealStorage() { NSWorkspace.shared.open(root) }

    func export(day: Date? = nil) {
        guard !exporting else { return }
        exporting = true
        let selected = day ?? Date()
        Task {
            defer { exporting = false }
            do {
                let text = try await store.summaryText(day: selected)
                guard !text.isEmpty else { status = "No completed summaries for this day"; return }
                NSPasteboard.general.clearContents()
                guard NSPasteboard.general.setString(text, forType: .string) else { throw CocoaError(.fileWriteUnknown) }
                status = "Copied summaries to clipboard"
            } catch { status = "Could not copy summaries. Please try again." }
        }
    }

    func showWindow() {
        if window == nil {
            let panel = NSWindow(contentRect: NSRect(x: 0, y: 0, width: 620, height: 650), styleMask: [.titled, .closable, .miniaturizable, .resizable], backing: .buffered, defer: false)
            panel.title = "Activity Journal"
            panel.contentView = NSHostingView(rootView: ActivityJournalView(journal: self))
            panel.minSize = NSSize(width: 500, height: 420)
            panel.isReleasedWhenClosed = false
            panel.center()
            window = panel
        }
        window?.makeKeyAndOrderFront(nil)
        NSApp.activate(ignoringOtherApps: true)
    }
}
