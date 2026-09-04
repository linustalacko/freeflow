import Foundation

enum RealtimeTranscriptionError: LocalizedError {
    case invalidBaseURL(String)
    case notConnected
    case serverError(code: String, message: String)
    case closedBeforeFinal
    case finalTimedOut
    case alreadyCommitted

    var errorDescription: String? {
        switch self {
        case .invalidBaseURL(let url): return "Cannot derive a WebSocket URL from \(url)"
        case .notConnected: return "Realtime transcription socket is not connected"
        case .serverError(let code, let message): return "Realtime server error [\(code)]: \(message)"
        case .closedBeforeFinal: return "Realtime socket closed before emitting the final transcript"
        case .finalTimedOut: return "Realtime transcription timed out; retrying the recorded audio"
        case .alreadyCommitted: return "Realtime audio has already been committed"
        }
    }
}

final class RealtimeTranscriptionService {
    struct Configuration {
        let baseURL: String
        let apiKey: String
        let model: String
        let language: String?
    }

    private let config: Configuration
    private let session: URLSession
    private let finalTimeout: TimeInterval
    private var task: URLSessionWebSocketTask?
    private var receiveTask: Task<Void, Never>?

    private let stateQueue = DispatchQueue(label: "com.zachlatta.freeflow.realtime.state")
    private var transcriptState = RealtimeTranscriptState()
    private var finalContinuation: CheckedContinuation<String, Error>?
    private var finalDeadline: DispatchWorkItem?
    private var closed: Bool = false
    private var terminalError: Error?

    /// Published on the main queue as partial transcript updates. The service
    /// concatenates all `completed` events and currently-streaming `delta`
    /// events — useful for a live overlay readout.
    var onPartialUpdate: ((String) -> Void)?

    init(config: Configuration, session: URLSession = .shared, finalTimeout: TimeInterval = 8) {
        self.config = config
        self.session = session
        self.finalTimeout = finalTimeout.isFinite && finalTimeout > 0 ? finalTimeout : 8
    }

    // MARK: Lifecycle

    func start() throws {
        guard let wsURL = Self.deriveWebSocketURL(
            baseURL: config.baseURL,
            model: config.model,
            language: config.language
        ) else {
            throw RealtimeTranscriptionError.invalidBaseURL(config.baseURL)
        }

        var request = URLRequest(url: wsURL)
        if !config.apiKey.isEmpty {
            request.setValue("Bearer \(config.apiKey)", forHTTPHeaderField: "Authorization")
        }

        let task = session.webSocketTask(with: request)
        stateQueue.sync {
            self.task = task
        }
        task.resume()

        receiveTask = Task { [weak self] in
            await self?.receiveLoop()
        }

        sendSessionUpdate()
    }

    /// Closing the socket also releases any in-flight receive. Safe to repeat.
    func cancel() {
        fail(CancellationError())
    }

    // MARK: Producer

    /// Append 16-bit little-endian PCM samples. The caller owns rate matching
    /// (the service declares 24 kHz mono in `session.update`, matching the
    /// OpenAI Realtime default).
    func appendPCM16(_ data: Data) {
        let currentTask: URLSessionWebSocketTask? = stateQueue.sync {
            closed || transcriptState.commitSent ? nil : task
        }
        guard let currentTask, !data.isEmpty else { return }
        let audioB64 = data.base64EncodedString()
        let message: [String: Any] = [
            "type": "input_audio_buffer.append",
            "audio": audioB64,
        ]
        send(message, over: currentTask)
    }

    /// Signal end-of-input and await the acknowledged item's final transcript.
    /// A silent peer must release the caller to its file-upload fallback.
    func commitAndAwaitFinal() async throws -> String {
        return try await withTaskCancellationHandler {
            try Task.checkCancellation()
            return try await withCheckedThrowingContinuation { continuation in
                var currentTask: URLSessionWebSocketTask?
                var immediateError: Error?
                stateQueue.sync {
                    if let terminalError { immediateError = terminalError; return }
                    guard !closed else {
                        immediateError = RealtimeTranscriptionError.closedBeforeFinal; return
                    }
                    guard let task else {
                        immediateError = RealtimeTranscriptionError.notConnected; return
                    }
                    guard !transcriptState.commitSent else {
                        immediateError = RealtimeTranscriptionError.alreadyCommitted; return
                    }
                    transcriptState.beginCommit()
                    finalContinuation = continuation
                    currentTask = task
                    let deadline = DispatchWorkItem { [weak self] in
                        self?.fail(RealtimeTranscriptionError.finalTimedOut)
                    }
                    finalDeadline = deadline
                    DispatchQueue.global(qos: .userInitiated).asyncAfter(
                        deadline: .now() + finalTimeout, execute: deadline)
                }
                if let immediateError {
                    continuation.resume(throwing: immediateError)
                } else if let currentTask {
                    send(["type": "input_audio_buffer.commit"], over: currentTask)
                }
            }
        } onCancel: {
            self.cancel()
        }
    }

    // MARK: Receive loop

    private func receiveLoop() async {
        while !Task.isCancelled {
            let currentTask: URLSessionWebSocketTask? = stateQueue.sync {
                task
            }
            guard let currentTask else { break }
            do {
                let message = try await currentTask.receive()
                switch message {
                case .string(let text):
                    handleServerEvent(text)
                case .data(let data):
                    if let text = String(data: data, encoding: .utf8) {
                        handleServerEvent(text)
                    }
                @unknown default:
                    break
                }
            } catch {
                finishWithClose()
                return
            }
        }
        finishWithClose()
    }

    private func finishWithClose() {
        fail(RealtimeTranscriptionError.closedBeforeFinal)
    }

    private func fail(_ error: Error) {
        var continuation: CheckedContinuation<String, Error>?
        let currentTask: URLSessionWebSocketTask? = stateQueue.sync {
            guard !closed else { return nil }
            closed = true
            terminalError = error
            finalDeadline?.cancel()
            finalDeadline = nil
            continuation = finalContinuation
            finalContinuation = nil
            let currentTask = task
            task = nil
            return currentTask
        }
        continuation?.resume(throwing: error)
        currentTask?.cancel(with: .normalClosure, reason: nil)
    }

    private func handleServerEvent(_ text: String) {
        guard let data = text.data(using: .utf8),
              let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let eventType = json["type"] as? String else {
            return
        }

        guard stateQueue.sync(execute: { !closed }) else { return }

        switch eventType {
        case "conversation.item.input_audio_transcription.delta":
            if let delta = json["delta"] as? String, !delta.isEmpty {
                appendDelta(delta)
            }
            resumeIfReadyAfterCommit()
        case "conversation.item.input_audio_transcription.completed":
            if let transcript = json["transcript"] as? String {
                commitSegment(transcript, itemID: json["item_id"] as? String)
            }
        case "input_audio_buffer.committed":
            stateQueue.sync {
                transcriptState.acknowledgeCommit(itemID: json["item_id"] as? String)
            }
            resumeIfReadyAfterCommit()
        case "error":
            let errObj = json["error"] as? [String: Any]
            let code = errObj?["code"] as? String ?? "unknown"
            let message = errObj?["message"] as? String ?? "unknown realtime error"
            let error = RealtimeTranscriptionError.serverError(code: code, message: message)
            fail(error)
        default:
            resumeIfReadyAfterCommit()
            break
        }
    }

    private func appendDelta(_ delta: String) {
        let snapshot: String = stateQueue.sync {
            transcriptState.appendDelta(delta)
            return transcriptState.finalText + transcriptState.partialText
        }
        reportPartial(snapshot)
    }

    private func commitSegment(_ transcript: String, itemID: String?) {
        let snapshot: String = stateQueue.sync {
            transcriptState.complete(transcript, itemID: itemID)
            return transcriptState.finalText
        }
        reportPartial(snapshot)
        resumeIfReadyAfterCommit()
    }

    private func reportPartial(_ text: String) {
        guard let handler = onPartialUpdate else { return }
        DispatchQueue.main.async {
            handler(text)
        }
    }

    // MARK: Send helpers

    private func send(_ payload: [String: Any], over task: URLSessionWebSocketTask) {
        guard let data = try? JSONSerialization.data(withJSONObject: payload),
              let text = String(data: data, encoding: .utf8) else {
            return
        }
        task.send(.string(text)) { [weak self] error in
            if let error {
                self?.fail(error)
            }
        }
    }

    private func sendSessionUpdate() {
        let currentTask: URLSessionWebSocketTask? = stateQueue.sync {
            task
        }
        guard let currentTask else { return }
        var transcription: [String: Any] = [:]
        let model = config.model.trimmingCharacters(in: .whitespacesAndNewlines)
        if !model.isEmpty {
            transcription["model"] = model
        }
        if let language = config.language, !language.isEmpty {
            transcription["language"] = language
        }
        let session: [String: Any] = [
            "type": "transcription",
            "audio": [
                "input": [
                    "format": [
                        "type": "audio/pcm",
                        "rate": 24_000,
                    ],
                    "transcription": transcription,
                    "turn_detection": NSNull(),
                ],
            ],
        ]
        send(["type": "session.update", "session": session], over: currentTask)
    }

    // MARK: URL derivation

    /// Turn `https://host[/prefix]` or `http://host[/prefix]` into
    /// `wss://host[/prefix]/realtime`, reusing a trailing `/v1` prefix when
    /// the configured base URL already includes it.
    static func deriveWebSocketURL(
        baseURL: String,
        model: String,
        language: String?
    ) -> URL? {
        let trimmed = baseURL.trimmingCharacters(in: .whitespacesAndNewlines)
        guard var components = URLComponents(string: trimmed) else { return nil }

        switch components.scheme?.lowercased() {
        case "http": components.scheme = "ws"
        case "https": components.scheme = "wss"
        case "ws", "wss": break
        default: return nil
        }

        var path = components.path
        if path.hasSuffix("/") { path.removeLast() }
        if path.hasSuffix("/v1") {
            path += "/realtime"
        } else {
            path += "/v1/realtime"
        }
        components.path = path

        var queryItems = components.queryItems ?? []
        if !queryItems.contains(where: { $0.name == "intent" }) {
            queryItems.append(URLQueryItem(name: "intent", value: "transcription"))
        }
        components.queryItems = queryItems.isEmpty ? nil : queryItems
        return components.url
    }

    private func resumeIfReadyAfterCommit() {
        var pendingResume: (CheckedContinuation<String, Error>, String)?
        stateQueue.sync {
            guard let cont = finalContinuation,
                  let finalText = transcriptState.readyTranscript, !closed else {
                return
            }
            finalContinuation = nil
            closed = true
            finalDeadline?.cancel()
            finalDeadline = nil
            pendingResume = (cont, finalText)
        }
        if let (cont, text) = pendingResume {
            let currentTask: URLSessionWebSocketTask? = stateQueue.sync {
                task
            }
            currentTask?.cancel(with: .normalClosure, reason: nil)
            cont.resume(returning: text)
        }
    }

}
