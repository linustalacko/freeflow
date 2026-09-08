import SwiftUI

struct ActivityJournalView: View {
    @ObservedObject var journal: ActivityJournal
    @State private var day = Date()
    @State private var settings = false
    @State private var captureSeconds = JournalPolicy.defaultInterval
    @FocusState private var editingCaptureInterval: Bool
    private var isToday: Bool { Calendar.current.isDateInToday(day) }

    var body: some View {
        VStack(spacing: 0) {
            HStack {
                Image(systemName: "clock.arrow.circlepath").font(.title2).accessibilityHidden(true)
                Text("Activity Journal").font(.headline)
                Spacer()
                Button(journal.enabled ? "Stop Journal" : "Start Journal") {
                    if journal.enabled { journal.stop() } else { journal.start() }
                }.accessibilityIdentifier("journal.toggle")
                Button { settings.toggle() } label: { Image(systemName: "gearshape") }
                    .buttonStyle(.plain).help("Settings")
                    .popover(isPresented: $settings, arrowEdge: .bottom) { preferences }
            }.padding(20)
            HStack {
                Button { moveDay(-1) } label: { Image(systemName: "chevron.left") }.help("Previous day")
                Text(isToday ? "Today" : day.formatted(date: .abbreviated, time: .omitted)).font(.headline)
                Button { moveDay(1) } label: { Image(systemName: "chevron.right") }.disabled(isToday).help("Next day")
                Spacer()
                Text("\(journal.records.count) entries").foregroundStyle(.secondary)
                Button(journal.exporting ? "Copying…" : "Copy all") { journal.export(day: day) }
                    .disabled(journal.exporting).help("Copy this day’s LLM summaries with app names and local times")
            }.buttonStyle(.plain).padding(.horizontal, 20).padding(.bottom, 16)
            ScrollView {
                LazyVStack(alignment: .leading, spacing: 0) {
                    if journal.records.isEmpty {
                        VStack(spacing: 8) {
                            Image(systemName: "camera").font(.system(size: 30)).foregroundStyle(.tertiary)
                            Text("No summaries yet").font(.headline)
                            Text(journal.enabled ? "Checks for activity every \(Int(journal.captureInterval / 60)) minutes." : "Start the journal when you want to track your work.").foregroundStyle(.secondary)
                        }.frame(maxWidth: .infinity).padding(.vertical, 75)
                    }
                    ForEach(journal.records.sorted { $0.capturedAt > $1.capturedAt }) { item in
                        Button { journal.revealCapture(item) } label: {
                            VStack(alignment: .leading, spacing: 7) {
                                HStack {
                                    Text(item.capturedAt, style: .time).monospacedDigit()
                                    Text("· \(item.appName)")
                                    Spacer()
                                    Image(systemName: "folder")
                                }.font(.caption).foregroundStyle(.secondary)
                                Text(item.summary.isEmpty ? (item.inferenceStatus == "pending" && journal.enabled ? "Summarizing…" : "No completed summary.") : item.summary)
                                    .multilineTextAlignment(.leading).frame(maxWidth: .infinity, alignment: .leading)
                            }.padding(.vertical, 14).contentShape(Rectangle())
                        }.buttonStyle(.plain).help("Open saved summary")
                        Divider()
                    }
                }.padding(.horizontal, 20)
            }
            Divider()
            HStack(spacing: 16) {
                if let checked = journal.lastCheckAt {
                    HStack(spacing: 4) { Text("Last check"); Text(checked.formatted(date: .omitted, time: .standard)).accessibilityIdentifier("journal.lastCheck") }
                }
                if journal.enabled, let next = journal.nextCheckAt {
                    HStack(spacing: 4) { Text("Next check"); Text(next.formatted(date: .omitted, time: .standard)).accessibilityIdentifier("journal.nextCheck") }
                }
                Spacer()
            }.font(.caption).foregroundStyle(.secondary).padding(.horizontal, 12).padding(.top, 10)
            HStack {
                Text(journal.storageError ?? journal.status).lineLimit(2).accessibilityIdentifier("journal.status")
                if journal.status == "Screen Recording permission required" {
                    Button("Allow…") { journal.requestCapturePermission() }
                }
                Spacer()
                Label("On this Mac", systemImage: "lock").fixedSize()
            }.font(.caption).foregroundStyle(.secondary).padding(12)
            if let message = journal.actionMessage {
                Text(message).font(.caption).foregroundStyle(.secondary).padding(.bottom, 10)
            }
        }
        .onAppear { journal.showDay(day) }
        .onChange(of: day) { value in journal.showDay(value) }
    }

    private func moveDay(_ offset: Int) { day = Calendar.current.date(byAdding: .day, value: offset, to: day) ?? day }

    private var preferences: some View {
        VStack(alignment: .leading, spacing: 14) {
            Text("Starts stopped each time you open FreeFlow. Idle and unchanged screens are skipped; the model is released after each summary.").foregroundStyle(.secondary)
            HStack {
                Text("Check every")
                TextField("Seconds", value: $captureSeconds, format: .number).frame(width: 50)
                    .focused($editingCaptureInterval).onSubmit { saveInterval() }
                Text("seconds")
            }
            Text("60–900 seconds. Longer intervals use less energy.").foregroundStyle(.secondary)
            Divider()
            TextField("Excluded apps, separated by commas", text: $journal.excludedApps)
            Text("Only summaries, app names, times and processing status are saved. Screenshots are processed in memory, then discarded. No OCR text is stored.").foregroundStyle(.secondary)
            Button("Open summaries folder") { journal.revealStorage() }
            Button("Screen Recording permission…") { journal.openPermissions() }
        }.font(.callout).padding(20).frame(width: 340)
            .onAppear { captureSeconds = journal.captureInterval }
            .onChange(of: editingCaptureInterval) { focused in if !focused { saveInterval() } }
            .onDisappear { saveInterval() }
    }
    private func saveInterval() { journal.setCaptureInterval(captureSeconds); captureSeconds = journal.captureInterval }
}
