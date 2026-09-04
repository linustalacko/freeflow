import Foundation

/// Compiled into a temporary executable by test_realtime_client.py.
@main
struct RealtimeClientProbe {
    static func main() async throws {
        let baseURL = CommandLine.arguments[1]
        let scenario = CommandLine.arguments[2]
        let service = RealtimeTranscriptionService(config: .init(
            baseURL: baseURL, apiKey: "", model: "synthetic", language: nil), finalTimeout: 0.4)
        try service.start()
        defer { service.cancel() }
        let start = Date()
        let result = Task { try await service.commitAndAwaitFinal() }
        if scenario == "cancel" {
            try await Task.sleep(nanoseconds: 40_000_000)
            result.cancel()
        }
        if scenario == "duplicate" {
            try await Task.sleep(nanoseconds: 40_000_000)
            do {
                _ = try await service.commitAndAwaitFinal()
                fatalError("second commit must not replace the first continuation")
            } catch RealtimeTranscriptionError.alreadyCommitted { }
        }
        do {
            let text = try await result.value
            precondition(scenario == "ordered" || scenario == "duplicate")
            precondition(text == "Keep every word.", "truncated or duplicated transcript")
        } catch RealtimeTranscriptionError.finalTimedOut {
            precondition(scenario == "silent" || scenario == "malformed")
            precondition(Date().timeIntervalSince(start) < 1.5)
        } catch is CancellationError {
            precondition(scenario == "cancel")
            precondition(Date().timeIntervalSince(start) < 0.4)
        } catch RealtimeTranscriptionError.closedBeforeFinal {
            precondition(scenario == "closed")
        }
        print("PASS \(scenario)")
    }
}
