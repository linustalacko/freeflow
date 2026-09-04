import AppKit
import WebKit
import ApplicationServices

@main
final class WritingSmoke: NSObject, NSApplicationDelegate, WKNavigationDelegate {
    let web = WKWebView()
    var window: NSWindow!
    let previousApp = NSWorkspace.shared.frontmostApplication
    var finished = false

    static func main() {
        let harness = WritingSmoke()
        let app = NSApplication.shared
        app.delegate = harness
        app.setActivationPolicy(.regular)
        app.run()
    }

    func applicationDidFinishLaunching(_ notification: Notification) {
        guard AXIsProcessTrusted() else { finish(false, "Accessibility unavailable; no prompt was requested"); return }
        let menu = NSMenu()
        let edit = NSMenuItem(title: "Edit", action: nil, keyEquivalent: "")
        edit.submenu = NSMenu(title: "Edit")
        edit.submenu?.addItem(withTitle: "Paste", action: #selector(NSText.paste(_:)), keyEquivalent: "v")
        menu.addItem(edit)
        NSApp.mainMenu = menu
        window = NSWindow(contentRect: NSRect(x: 120, y: 120, width: 760, height: 520), styleMask: [.titled, .closable], backing: .buffered, defer: false)
        window.title = "FreeFlow writing smoke test — synthetic email"
        window.contentView = web
        window.makeKeyAndOrderFront(nil)
        NSApp.activate(ignoringOtherApps: true)
        web.navigationDelegate = self
        web.loadHTMLString("""
        <!doctype html><html><body style="font:18px system-ui;padding:30px">
        <aside>Unrelated sidebar content must not be captured.</aside>
        <main><h1>Project check-in</h1><p>From: Avery</p><p>Can we meet Thursday?</p>
        <div id="reply" role="textbox" aria-label="Message Body" contenteditable="true" aria-multiline="true" style="min-height:150px;border:1px solid gray"></div>
        <button>Send</button></main>
        <script>document.getElementById('reply').focus()</script></body></html>
        """, baseURL: URL(string: "https://mail.google.com/mail/u/0/#inbox/1234567890abcdef"))
        DispatchQueue.main.asyncAfter(deadline: .now() + 15) { self.finish(false, "Timed out") }
    }

    func webView(_ webView: WKWebView, didFinish navigation: WKNavigation!) {
        window.makeKeyAndOrderFront(nil)
        NSApp.activate(ignoringOtherApps: true)
        window.makeFirstResponder(web)
        web.evaluateJavaScript("document.getElementById('reply').focus()") { _, _ in
            DispatchQueue.main.asyncAfter(deadline: .now() + 0.4) { self.checkDraft() }
        }
    }

    var probe: Process?
    var outputBuffer = ""
    func checkDraft() {
        guard !finished else { return }
        let process = Process()
        probe = process
        process.executableURL = URL(fileURLWithPath: CommandLine.arguments[1])
        process.arguments = [String(getpid())]
        let output = Pipe()
        process.standardOutput = output
        output.fileHandleForReading.readabilityHandler = { handle in
            let data = handle.availableData
            guard !data.isEmpty else { handle.readabilityHandler = nil; return }
            DispatchQueue.main.async {
                self.outputBuffer += String(decoding: data, as: UTF8.self)
                while let newline = self.outputBuffer.firstIndex(of: "\n") {
                    let line = String(self.outputBuffer[..<newline])
                    self.outputBuffer.removeSubrange(...newline)
                    print(line)
                }
            }
        }
        process.terminationHandler = { child in
            DispatchQueue.main.asyncAfter(deadline: .now() + 0.1) { self.finish(child.terminationStatus == 0, "Native browser smoke test finished") }
        }
        do { try process.run() } catch { finish(false, "Probe launch failed") }
    }

    func finish(_ passed: Bool, _ message: String) {
        guard !finished else { return }
        finished = true
        if probe?.isRunning == true { probe?.terminate() }
        print(passed ? "PASS:" : "FAIL:", message)
        previousApp?.activate(options: [])
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.1) { exit(passed ? 0 : 1) }
    }
}
