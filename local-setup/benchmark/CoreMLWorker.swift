import CoreML
import FluidAudio
import Foundation

/// Optional benchmark worker. Audio and results travel over parent-owned pipes;
/// callers must score in memory and never forward stdout to a console or file.
@main
struct CoreMLWorker {
    struct Request: Decodable { let pcm_f32_base64: String }
    struct Token: Encodable {
        let text: String
        let start: Double
        let end: Double
    }
    struct Response: Encodable {
        var ready: Bool?
        var text: String?
        var tokens: [Token]?
        var model_s: Double?
        var error: String?
    }

    static func emit(_ response: Response) {
        guard var data = try? JSONEncoder().encode(response) else { return }
        data.append(10)
        FileHandle.standardOutput.write(data)
    }

    static func main() async {
        guard CommandLine.arguments.count == 2 else {
            emit(Response(error: "model_directory_required"))
            return
        }
        let directory = URL(fileURLWithPath: CommandLine.arguments[1], isDirectory: true)
        do {
            let models = try await AsrModels.downloadAndLoad(
                to: directory, version: .v2, encoderPrecision: .int8,
                encoderComputeUnits: .cpuAndNeuralEngine)
            let manager = AsrManager(config: ASRConfig(
                tdtConfig: TdtConfig(blankId: AsrModelVersion.v2.blankId),
                parallelChunkConcurrency: 1))
            try await manager.loadModels(models)
            emit(Response(ready: true))
            while let line = readLine() {
                do {
                    let request = try JSONDecoder().decode(Request.self, from: Data(line.utf8))
                    guard let data = Data(base64Encoded: request.pcm_f32_base64), data.count % 4 == 0 else {
                        emit(Response(error: "invalid_pcm"))
                        continue
                    }
                    let samples: [Float] = data.withUnsafeBytes { raw in
                        stride(from: 0, to: raw.count, by: 4).map { offset in
                            Float(bitPattern: UInt32(littleEndian: raw.loadUnaligned(fromByteOffset: offset, as: UInt32.self)))
                        }
                    }
                    guard !samples.isEmpty, samples.allSatisfy({ $0.isFinite }) else {
                        emit(Response(error: "invalid_pcm"))
                        continue
                    }
                    let start = ProcessInfo.processInfo.systemUptime
                    var state = try TdtDecoderState()
                    let result = try await manager.transcribe(samples, decoderState: &state)
                    let elapsed = ProcessInfo.processInfo.systemUptime - start
                    emit(Response(text: result.text,
                                  tokens: result.tokenTimings?.map { Token(text: $0.token, start: $0.startTime, end: $0.endTime) },
                                  model_s: elapsed))
                } catch {
                    emit(Response(error: "coreml_transcription_failed"))
                }
            }
        } catch {
            emit(Response(error: "coreml_setup_failed"))
        }
    }
}
