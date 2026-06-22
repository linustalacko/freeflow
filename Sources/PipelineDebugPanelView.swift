import SwiftUI

struct PipelineDebugPanelView: View {
    @EnvironmentObject var appState: AppState

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            header
            Divider()

            ScrollView {
                VStack(alignment: .leading, spacing: 16) {
                    PipelineDebugContentView(
                        statusMessage: appState.debugStatusMessage,
                        postProcessingStatus: appState.lastPostProcessingStatus,
                        contextSummary: appState.lastContextSummary,
                        contextScreenshotStatus: appState.lastContextScreenshotStatus,
                        contextScreenshotDataURL: appState.lastContextScreenshotDataURL,
                        rawTranscript: appState.lastRawTranscript,
                        postProcessedTranscript: appState.lastPostProcessedTranscript,
                        postProcessingPrompt: appState.lastPostProcessingPrompt
                    )

                    if let latest = appState.pipelineHistory.first {
                        Divider()
                        PipelineCorrectionEditor(item: latest)
                    }

                    if appState.lastContextSummary.isEmpty && appState.lastRawTranscript.isEmpty {
                        Text("Run a dictation pass to populate debug output.")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }
                }
            }
        }
        .padding(16)
        .frame(width: 620, height: 640, alignment: .topLeading)
    }

    private var header: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack(alignment: .firstTextBaseline) {
                VStack(alignment: .leading, spacing: 2) {
                    Text("Pipeline Debug")
                        .font(.title3)
                    Text("Live data for the transcription + post-processing pipeline.")
                        .font(.body)
                        .foregroundStyle(.secondary)
                }
                Spacer()
                Button("Export Test Case…") {
                    exportTestCase()
                }
                .font(.body)
                .disabled(appState.pipelineHistory.first == nil)
            }
        }
    }

    private func exportTestCase() {
        guard let item = appState.pipelineHistory.first else { return }
        TestCaseExporter.exportWithSavePanel(
            item: item,
            audioDirURL: AppState.audioStorageDirectory()
        )
    }
}

/// Lets you fix the latest post-processed output and save it as a gold training
/// label (`correctedTranscript`) — the signal that lets a fine-tuned model beat
/// the current one instead of just cloning it.
struct PipelineCorrectionEditor: View {
    @EnvironmentObject var appState: AppState
    let item: PipelineHistoryItem
    @State private var text: String = ""
    @State private var justSaved = false

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack(spacing: 8) {
                Text("Corrected output")
                    .font(.body.bold())
                if let corrected = item.correctedTranscript, !corrected.isEmpty {
                    Text("gold label saved")
                        .font(.caption)
                        .foregroundStyle(.green)
                }
            }
            Text("Fix the latest output to what it should have been — saved as a training label for fine-tuning.")
                .font(.caption)
                .foregroundStyle(.secondary)
            TextEditor(text: $text)
                .font(.system(size: 15, weight: .regular, design: .monospaced))
                .frame(height: 90)
                .padding(6)
                .background(Color(nsColor: .textBackgroundColor))
                .overlay(RoundedRectangle(cornerRadius: 6).stroke(Color.secondary.opacity(0.2)))
            HStack(spacing: 8) {
                Button("Save correction") {
                    appState.saveCorrection(text, for: item)
                    justSaved = true
                }
                .font(.body)
                .disabled(text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
                if justSaved {
                    Text("Saved \u{2713}")
                        .font(.caption)
                        .foregroundStyle(.green)
                }
            }
        }
        .onAppear(perform: reset)
        .onChange(of: item.id) { _ in reset() }
    }

    private func reset() {
        text = item.correctedTranscript ?? item.postProcessedTranscript
        justSaved = false
    }
}
