import Foundation

/// Own a local-only model process for one summary, never a login service.
@MainActor
final class JournalModelRuntime {
    private var process: Process?
    private let executableURL: URL?
    private let arguments: [String]
    private let healthCheck: (() async -> Bool)?
    var isRunning: Bool { process?.isRunning == true }

    convenience init() { self.init(executable: Self.executable) }

    init(executable: URL?, arguments: [String] = ["serve"], healthCheck: (() async -> Bool)? = nil) {
        self.executableURL = executable
        self.arguments = arguments
        self.healthCheck = healthCheck
    }

    enum Failure: LocalizedError {
        case notInstalled, portBusy, unavailable
        var errorDescription: String? {
            switch self {
            case .notInstalled: return "Install the optional journal model first; see docs/ACTIVITY_JOURNAL.md."
            case .portBusy: return "The journal model port is in use. Stop the old journal service before starting."
            case .unavailable: return "Local journal model unavailable. Check the optional model installation."
            }
        }
    }

    static var executable: URL? {
        ["/opt/homebrew/bin/ollama", "/usr/local/bin/ollama", "/Applications/Ollama.app/Contents/Resources/ollama"]
            .first { FileManager.default.isExecutableFile(atPath: $0) }.map { URL(fileURLWithPath: $0) }
    }

    static func environment(home: String) -> [String: String] {
        // Do not inherit provider credentials, proxies, logging or cloud configuration.
        ["HOME": home, "PATH": "/usr/bin:/bin:/usr/sbin:/sbin", "OLLAMA_HOST": "127.0.0.1:11436",
         "OLLAMA_NO_CLOUD": "1", "OLLAMA_MAX_LOADED_MODELS": "1", "OLLAMA_NUM_PARALLEL": "1",
         "OLLAMA_KEEP_ALIVE": "0", "OLLAMA_CONTEXT_LENGTH": String(JournalPolicy.contextTokens),
         "OLLAMA_FLASH_ATTENTION": "1", "OLLAMA_KV_CACHE_TYPE": "q8_0", "OLLAMA_MAX_QUEUE": "1"]
    }

    func start() async throws {
        if let child = process, !child.isRunning { process = nil }
        guard process == nil else { throw Failure.unavailable }
        guard let executable = executableURL else { throw Failure.notInstalled }
        guard !(await responds()) else { throw Failure.portBusy }
        try Task.checkCancellation()
        let child = Process()
        child.executableURL = executable
        child.arguments = arguments
        child.environment = Self.environment(home: FileManager.default.homeDirectoryForCurrentUser.path)
        child.standardInput = FileHandle.nullDevice
        child.standardOutput = FileHandle.nullDevice
        child.standardError = FileHandle.nullDevice
        try child.run()
        process = child
        for _ in 0..<30 {
            try Task.checkCancellation()
            guard child.isRunning else { throw Failure.unavailable }
            if await responds() {
                guard child.isRunning else { throw Failure.unavailable }
                return
            }
            try await Task.sleep(nanoseconds: 100_000_000)
        }
        throw Failure.unavailable
    }

    private func responds() async -> Bool {
        if let healthCheck { return await healthCheck() }
        let session = JournalLocalModel.session(timeout: 0.25)
        defer { session.invalidateAndCancel() }
        let url = JournalLocalModel.endpoint.deletingLastPathComponent().appendingPathComponent("version")
        guard let (_, response) = try? await session.data(from: url) else { return false }
        return (response as? HTTPURLResponse)?.statusCode == 200
    }

    func stop() {
        if let process, process.isRunning { process.terminate() }
    }

    func finish() async {
        guard let child = process else { return }
        if child.isRunning { child.terminate() }
        // The next sample cannot start while the previous owner still holds the port.
        await Task { @MainActor in
            for _ in 0..<50 where child.isRunning {
                try? await Task.sleep(nanoseconds: 100_000_000)
            }
        }.value
        if !child.isRunning { process = nil }
    }
}
