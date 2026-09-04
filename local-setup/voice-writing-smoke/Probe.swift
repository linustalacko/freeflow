import AppKit
import ApplicationServices

@main
struct Probe {
    static func main() {
        let owner = pid_t(CommandLine.arguments[1])!
        var access = VoiceWritingTargetService.AccessibilityAccess.live
        let frontmost = access.frontmostPID
        access.frontmostPID = { frontmost() == owner ? owner : nil }
        let service = VoiceWritingTargetService(access: access)
        do {
            access.prepareTextAccessibility(owner)
            var snapshot = service.snapshot()
            for _ in 0..<10 where snapshot == nil {
                Thread.sleep(forTimeInterval: 0.15)
                snapshot = service.snapshot()
            }
            guard snapshot != nil else {
                print("FAIL: synthetic browser does not expose a writable text selection"); exit(1)
            }
            let start = Date()
            let draft = try service.prepare(.draftReply)
            guard draft.thread.contains("Can we meet Thursday?"), !draft.thread.contains("Unrelated sidebar") else { print("FAIL: context scope"); exit(1) }
            try service.selectReplacement(for: draft, text: "Friday works. ")
            print("TIMING: draft capture and validation \(Int(Date().timeIntervalSince(start) * 1000))ms")
            guard paste("Friday works. ", into: owner) else { exit(1) }
            let edit = try service.prepare(.editLast)
            try service.selectReplacement(for: edit, text: "Friday afternoon works. ")
            guard paste("Friday afternoon works. ", into: owner) else { exit(1) }
            let undo = try service.prepare(.editLast)
            guard undo.anchor.undoText == "Friday works. " else { print("FAIL: undo text"); exit(1) }
            try service.selectReplacement(for: undo, text: undo.anchor.undoText!)
            guard paste(undo.anchor.undoText!, into: owner),
                  let final = service.snapshot(), WritingTextAnchor.sameFieldText(final.value, "Friday works. ") else {
                print("FAIL: final undo value"); exit(1)
            }
            print("PASS: Native browser context capture, exact selection, Cmd-V, follow-up edit, undo and clipboard restoration")
        } catch { print("FAIL:", error.localizedDescription); exit(1) }
    }
    // Clipboard data is kept only in memory, never printed or saved. Events are
    // addressed to the synthetic fixture's PID, not another foreground app.
    private static func paste(_ text: String, into pid: pid_t) -> Bool {
        guard NSWorkspace.shared.frontmostApplication?.processIdentifier == pid else { return false }
        let clipboard = NSPasteboard.general
        let saved = (clipboard.pasteboardItems ?? []).map { item in
            item.types.compactMap { type -> (NSPasteboard.PasteboardType, Data)? in
                item.data(forType: type).map { (type, $0) }
            }
        }
        clipboard.clearContents()
        clipboard.setString(text, forType: .string)
        let ownedCount = clipboard.changeCount
        defer {
            if clipboard.changeCount == ownedCount {
                clipboard.clearContents()
                let restored = saved.map { entries -> NSPasteboardItem in
                    let item = NSPasteboardItem()
                    for (type, data) in entries { item.setData(data, forType: type) }
                    return item
                }
                if !restored.isEmpty { clipboard.writeObjects(restored) }
                let items = clipboard.pasteboardItems ?? []
                let matches = items.count == saved.count && items.enumerated().allSatisfy { index, item in
                    index < saved.count && saved[index].allSatisfy { type, data in item.data(forType: type) == data }
                }
                if !matches { print("FAIL: clipboard restoration"); exit(1) }
            } else {
                print("FAIL: clipboard changed during the smoke test; kept the newer clipboard contents")
                exit(1)
            }
        }
        let source = CGEventSource(stateID: .hidSystemState)
        for down in [true, false] {
            let event = CGEvent(keyboardEventSource: source, virtualKey: 9, keyDown: down)
            event?.flags = .maskCommand
            event?.postToPid(pid)
        }
        Thread.sleep(forTimeInterval: 0.3)
        return true
    }

}
