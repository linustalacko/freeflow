import Foundation

/// File replay through the production realtime client, without microphone,
/// app settings, cleanup, clipboard, or permission access. JSON travels over
/// pipes; the benchmark parent scores the text without logging it.
@main
struct BenchmarkReplay {
    struct Request: Decodable {
        let pcmPath: String
        let baseURL: String
        let model: String
        let apiKey: String
        let language: String
        let timeout: Double
    }

    struct Result: Encodable {
        var text: String?
        var error: String?
        var total_s: Double?
        var after_audio_s: Double?
        var replay_s: Double?
    }

    static func now() -> Double { ProcessInfo.processInfo.systemUptime }

    static func replay(_ request: Request) async throws -> Result {
        let pcm = try Data(contentsOf: URL(fileURLWithPath: request.pcmPath))
        guard !pcm.isEmpty, pcm.count % 2 == 0 else {
            return Result(error: "invalid_pcm")
        }
        let service = RealtimeTranscriptionService(config: .init(
            baseURL: request.baseURL, apiKey: request.apiKey,
            model: request.model, language: request.language), finalTimeout: request.timeout)
        defer { service.cancel() }
        let started = now()
        try service.start()
        // Each 20 ms chunk becomes available only after its audio interval.
        // Absolute deadlines avoid accumulating per-chunk sleep overhead.
        let audioStart = now()
        for offset in stride(from: 0, to: pcm.count, by: 960) {
            let end = min(offset + 960, pcm.count)
            let deadline = audioStart + Double(end) / 48_000
            let remaining = deadline - now()
            if remaining > 0 {
                try await Task.sleep(nanoseconds: UInt64(remaining * 1_000_000_000))
            }
            service.appendPCM16(pcm.subdata(in: offset..<end))
        }
        let commit = now()
        let text = try await service.commitAndAwaitFinal()
        let final = now()
        return Result(text: text, total_s: final - started,
                      after_audio_s: final - commit, replay_s: commit - audioStart)
    }

    static func main() async {
        let result: Result
        do {
            let request = try JSONDecoder().decode(Request.self,
                from: FileHandle.standardInput.readDataToEndOfFile())
            result = try await replay(request)
        } catch RealtimeTranscriptionError.finalTimedOut {
            result = Result(error: "final_timeout")
        } catch {
            // Provider errors can contain content, credentials, or private URLs.
            result = Result(error: "replay_failed")
        }
        if let data = try? JSONEncoder().encode(result) {
            FileHandle.standardOutput.write(data)
        }
    }
}
