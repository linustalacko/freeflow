import Foundation

enum JournalLifecycleTests {
    @MainActor static func run() async {
        var session = JournalSession()
        TestSupport.expectEqual(session.enabled, false)
        TestSupport.expectEqual(session.canCapture(busy: false, suspended: false, idleSeconds: 0), false)
        session.setEnabled(true)
        let first = session.generation
        TestSupport.expectEqual(session.canCapture(busy: false, suspended: false, idleSeconds: 0), true)
        TestSupport.expectEqual(session.shouldRetryOnActivation(isEligible: false), false)
        TestSupport.expectEqual(session.shouldRetryOnActivation(isEligible: true), true)
        // Explicit Start should work even after a long reading/idle period.
        TestSupport.expectEqual(session.canCapture(busy: false, suspended: false, idleSeconds: 600), true)
        TestSupport.expectEqual(session.canCapture(busy: true, suspended: false, idleSeconds: 600), false)
        TestSupport.expectEqual(session.canCapture(busy: false, suspended: true, idleSeconds: 600), false)
        TestSupport.expectEqual(session.canCapture(busy: false, suspended: false, idleSeconds: .infinity), false)
        session.didBeginCapture()
        TestSupport.expectEqual(session.shouldRetryOnActivation(isEligible: true), false)
        TestSupport.expectEqual(session.canCapture(busy: true, suspended: false, idleSeconds: 0), false)
        TestSupport.expectEqual(session.canCapture(busy: false, suspended: true, idleSeconds: 0), false)
        for idle in [120.0, 3600, .infinity, .nan, -1] {
            TestSupport.expectEqual(session.canCapture(busy: false, suspended: false, idleSeconds: idle), false)
        }
        // Input in the first minute of a three-minute interval must not disappear.
        let window = JournalPolicy.activityWindow(interval: 180)
        TestSupport.expectEqual(session.canCapture(busy: false, suspended: false, idleSeconds: 150, activityWindow: window), true)
        TestSupport.expectEqual(session.canCapture(busy: false, suspended: false, idleSeconds: 196, activityWindow: window), false)
        session.setEnabled(false)
        TestSupport.expectEqual(session.shouldRetryOnActivation(isEligible: true), false)
        TestSupport.expectEqual(session.accepts(first), false)
        session.setEnabled(true)
        TestSupport.expectEqual(session.accepts(first), false)
        let resumed = session.generation
        session.invalidate()
        TestSupport.expectEqual(session.accepts(resumed), false)

        var duplicates = JournalDeduplicator()
        TestSupport.expectEqual(duplicates.hasImage("synthetic-frame"), false)
        duplicates.remember(image: "synthetic-frame", text: "synthetic-text")
        TestSupport.expectEqual(duplicates.hasImage("synthetic-frame"), true)
        TestSupport.expectEqual(duplicates.hasText("synthetic-text"), true)
        TestSupport.expectEqual(duplicates.hasText("different-task"), false)
        duplicates.reset()
        TestSupport.expectEqual(duplicates.hasImage("synthetic-frame"), false)

        // A harmless subprocess and fake readiness replace Ollama/network completely.
        var probes = 0
        let runtime = JournalModelRuntime(executable: URL(fileURLWithPath: "/bin/sleep"), arguments: ["30"]) {
            probes += 1
            return probes > 1
        }
        TestSupport.expectEqual(runtime.isRunning, false)
        do { try await runtime.start() } catch { fatalError("Synthetic runtime did not start") }
        TestSupport.expectEqual(runtime.isRunning, true)
        runtime.stop()
        await runtime.finish()
        TestSupport.expectEqual(runtime.isRunning, false)

        let occupied = JournalModelRuntime(executable: URL(fileURLWithPath: "/bin/sleep"), arguments: ["30"], healthCheck: { true })
        do { try await occupied.start(); fatalError("Must not use an unowned model server") }
        catch { TestSupport.expectEqual(occupied.isRunning, false) }

        let slow = JournalModelRuntime(executable: URL(fileURLWithPath: "/bin/sleep"), arguments: ["30"], healthCheck: { false })
        let startup = Task { try await slow.start() }
        for _ in 0..<50 where !slow.isRunning { try? await Task.sleep(nanoseconds: 10_000_000) }
        TestSupport.expectEqual(slow.isRunning, true)
        startup.cancel()
        _ = try? await startup.value
        // Cleanup must still wait for and release the owned process in a cancelled task.
        let cleanup = Task { await slow.finish() }
        cleanup.cancel()
        await cleanup.value
        TestSupport.expectEqual(slow.isRunning, false)
        print("Journal opt-in, idle/busy gates, cancellation and runtime cleanup passed")
    }
}
