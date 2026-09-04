import AppKit
import ApplicationServices

enum VoiceWritingTargetTests {
    static func run() {
        do {
            let fixture = Fixture()
            let service = fixture.service()
            let reply = try service.prepare(.draftReply)
            TestSupport.expectEqual(reply.thread, "From: Avery\nProject check-in\nCan we meet Thursday?")
            TestSupport.expectEqual(reply.anchor.text, "")
            try service.selectReplacement(for: reply, text: "Friday works. ")
            fixture.paste("Friday works. ")
            let edit = try service.prepare(.editLast)
            TestSupport.expectEqual(edit.anchor.text, "Friday works. ")
            try service.selectReplacement(for: edit, text: "Friday afternoon works. ")
            fixture.paste("Friday afternoon works. ")
            let undo = try service.prepare(.editLast)
            TestSupport.expectEqual(undo.anchor.undoText, "Friday works. ")
            try service.selectReplacement(for: undo, text: undo.anchor.undoText!)
            fixture.paste(undo.anchor.undoText!)
            TestSupport.expectEqual(fixture.value, "Friday works. ")
        } catch { fatalError("Synthetic writing round trip failed: \(error)") }

        for change in ["focus", "text", "caret", "navigation", "permission"] {
            let fixture = Fixture()
            let service = fixture.service()
            let session = try! service.prepare(.draftReply)
            switch change {
            case "focus": fixture.focused = fixture.sidebar
            case "text": fixture.value = "User started typing"
            case "caret": fixture.selection = NSRange(location: 1, length: 0)
            case "navigation": fixture.pageURL = "https://mail.google.com/mail/u/0/#inbox/different0000000"
            default: fixture.trusted = false
            }
            expectFailure { try service.selectReplacement(for: session, text: "Do not insert") }
            TestSupport.expectEqual(fixture.selectCalls, 0)
        }

        do {
            let fixture = Fixture()
            let service = fixture.service()
            service.rememberInsertion("Hello. ", into: service.snapshot())
            fixture.paste("Hello. ")
            fixture.pageURL = "https://mail.google.com/mail/u/0/#inbox/another000000000"
            expectFailure { _ = try service.prepare(.editLast) }
        }
        for invalid in ["inbox", "no-main", "recipient", "selection", "oversized", "readonly", "spoofed-site", "malformed-focus"] {
            let fixture = Fixture()
            switch invalid {
            case "inbox": fixture.pageURL = "https://mail.google.com/mail/u/0/#inbox"
            case "no-main": fixture.set(fixture.main, kAXSubroleAttribute, "AXGroup" as CFString)
            case "recipient": fixture.set(fixture.body, kAXRoleAttribute, kAXTextFieldRole as CFString)
            case "selection": fixture.value = "Existing draft"; fixture.selection = NSRange(location: 0, length: 5)
            case "oversized": fixture.set(fixture.message, kAXValueAttribute, String(repeating: "x", count: 20_001) as CFString)
            case "readonly": fixture.selectable = false
            case "spoofed-site": fixture.pageURL = "https://mail.google.com.example/mail/u/0/#inbox/1234567890abcdef"
            default: fixture.malformedFocus = true
            }
            expectFailure { _ = try fixture.service().prepare(.draftReply) }
            TestSupport.expectEqual(fixture.selectCalls, 0)
        }
        let fixture = Fixture()
        let service = fixture.service()
        let session = try! service.prepare(.draftReply)
        fixture.selectable = false
        expectFailure { try service.selectReplacement(for: session, text: "No paste") }
        TestSupport.expectEqual(fixture.selectCalls, 0)
    }

    private static func expectFailure(_ action: () throws -> Void) {
        do { try action(); fatalError("Expected the unsafe writing operation to fail") }
        catch is VoiceWritingError { }
        catch { fatalError("Unexpected error: \(error)") }
    }

    /// All elements and content are invented; no real Accessibility requests.
    private final class Fixture {
        let app = AXUIElementCreateApplication(100_001)
        let web = AXUIElementCreateApplication(100_002)
        let main = AXUIElementCreateApplication(100_003)
        let body = AXUIElementCreateApplication(100_004)
        let sidebar = AXUIElementCreateApplication(100_005)
        let message = AXUIElementCreateApplication(100_006)
        var focused: AXUIElement
        var trusted = true
        var selectable = true
        var malformedFocus = false
        var selectCalls = 0
        var value = ""
        var selection = NSRange(location: 0, length: 0)
        var pageURL = "https://mail.google.com/mail/u/0/#inbox/1234567890abcdef"
        var nodes: [(AXUIElement, [String: CFTypeRef])] = []

        init() {
            focused = body
            let sender = AXUIElementCreateApplication(100_007)
            let subject = AXUIElementCreateApplication(100_008)
            let send = AXUIElementCreateApplication(100_009)
            let hidden = AXUIElementCreateApplication(100_010)
            nodes = [
                (app, [:]),
                (web, [kAXRoleAttribute: "AXWebArea" as CFString, kAXChildrenAttribute: [sidebar, main] as CFArray]),
                (main, [kAXRoleAttribute: kAXGroupRole as CFString, kAXSubroleAttribute: "AXLandmarkMain" as CFString,
                        kAXParentAttribute: web, kAXChildrenAttribute: [sender, subject, message, body, send, hidden] as CFArray]),
                (body, [kAXRoleAttribute: kAXTextAreaRole as CFString, kAXParentAttribute: main]),
                (sidebar, [kAXRoleAttribute: kAXStaticTextRole as CFString, kAXValueAttribute: "Private sidebar content" as CFString]),
                (sender, [kAXRoleAttribute: kAXStaticTextRole as CFString, kAXValueAttribute: "From: Avery" as CFString]),
                (subject, [kAXRoleAttribute: kAXStaticTextRole as CFString, kAXValueAttribute: "Project check-in" as CFString]),
                (message, [kAXRoleAttribute: kAXStaticTextRole as CFString, kAXValueAttribute: "Can we meet Thursday?" as CFString]),
                (send, [kAXRoleAttribute: kAXButtonRole as CFString, kAXValueAttribute: "Send" as CFString]),
                (hidden, [kAXRoleAttribute: kAXStaticTextRole as CFString, kAXValueAttribute: "Hidden text" as CFString, "AXHidden": kCFBooleanTrue])
            ]
        }

        func set(_ element: AXUIElement, _ name: String, _ value: CFTypeRef) {
            let index = nodes.firstIndex { CFEqual($0.0, element) }!
            nodes[index].1[name] = value
        }

        func paste(_ text: String) {
            value = (value as NSString).replacingCharacters(in: selection, with: text)
            selection = NSRange(location: selection.location + text.utf16.count, length: 0)
        }

        func service() -> VoiceWritingTargetService {
            VoiceWritingTargetService(access: .init(
                frontmostPID: { self.trusted ? 100_001 : nil },
                copy: { element, name in
                    if CFEqual(element, self.app), name == kAXFocusedUIElementAttribute {
                        return self.malformedFocus ? "invalid" as CFString : self.focused
                    }
                    if CFEqual(element, self.web), name == kAXURLAttribute { return self.pageURL as CFString }
                    if CFEqual(element, self.body) {
                        if name == kAXValueAttribute { return self.value as CFString }
                        if name == kAXSelectedTextRangeAttribute {
                            var range = CFRange(location: self.selection.location, length: self.selection.length)
                            return AXValueCreate(.cfRange, &range)
                        }
                    }
                    return self.nodes.first { CFEqual($0.0, element) }?.1[name]
                },
                canSelect: { _ in self.selectable },
                select: { _, range in self.selectCalls += 1; self.selection = range; return true }
            ))
        }
    }
}
