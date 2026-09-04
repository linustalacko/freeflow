import AppKit
import ApplicationServices

enum VoiceWritingError: LocalizedError {
    case unavailableField, changedTarget, noLastInsertion, noUndo, gmailRequired, threadUnavailable

    var errorDescription: String? {
        switch self {
        case .unavailableField: return "Click an editable text field first. This app must support Accessibility text and selection."
        case .changedTarget: return "The text or destination changed. Nothing was replaced. Click the original field and try again."
        case .noLastInsertion: return "Dictate into this field first. Edit Last works while that text is unchanged."
        case .noUndo: return "There is no voice edit to undo in this field."
        case .gmailRequired: return "Open a Gmail conversation, click Reply, then click inside the reply body."
        case .threadUnavailable: return "Gmail's thread text is unavailable or too large. Expand the message you want to answer and collapse older messages, then try again."
        }
    }
}

/// Main-thread, ephemeral handles. No clipboard reads, screenshots, browser
/// scripting, persistence, or Accessibility permission prompts.
final class VoiceWritingTargetService {
    struct AccessibilityAccess {
        var frontmostPID: () -> pid_t?
        var copy: (AXUIElement, String) -> CFTypeRef?
        var canSelect: (AXUIElement) -> Bool
        var select: (AXUIElement, NSRange) -> Bool
        var prepareTextAccessibility: (pid_t) -> Void = { _ in }

        static let live = Self(
            frontmostPID: {
                guard AXIsProcessTrusted(), let app = NSWorkspace.shared.frontmostApplication else { return nil }
                AXUIElementSetMessagingTimeout(AXUIElementCreateApplication(app.processIdentifier), 0.08)
                return app.processIdentifier
            },
            copy: { element, name in
                var value: CFTypeRef?
                guard AXUIElementCopyAttributeValue(element, name as CFString, &value) == .success else { return nil }
                return value
            },
            canSelect: { element in
                var settable = DarwinBoolean(false)
                return AXUIElementIsAttributeSettable(element, kAXSelectedTextRangeAttribute as CFString, &settable) == .success && settable.boolValue
            },
            select: { element, range in
                var cfRange = CFRange(location: range.location, length: range.length)
                guard let value = AXValueCreate(.cfRange, &cfRange) else { return false }
                return AXUIElementSetAttributeValue(element, kAXSelectedTextRangeAttribute as CFString, value) == .success
            },
            prepareTextAccessibility: { pid in
                guard AXIsProcessTrusted(), NSWorkspace.shared.frontmostApplication?.processIdentifier == pid else { return }
                let element = AXUIElementCreateApplication(pid)
                AXUIElementSetMessagingTimeout(element, 0.08)
                // Chromium/Electron expose text lazily. This enables the app's
                // AX tree, as our inline correction capture already does; it
                // does not request or change any macOS permission.
                AXUIElementSetAttributeValue(element, "AXManualAccessibility" as CFString, kCFBooleanTrue)
            }
        )
    }

    private let access: AccessibilityAccess
    init(access: AccessibilityAccess = .live) { self.access = access }

    struct Snapshot {
        let element: AXUIElement
        let pid: pid_t
        let value: String
        let selection: NSRange
    }

    struct Session {
        let id = UUID()
        let action: VoiceWritingAction
        let snapshot: Snapshot
        let anchor: WritingTextAnchor
        let thread: String
        let documentURL: String?
    }

    private var lastInsertion: (snapshot: Snapshot, anchor: WritingTextAnchor, documentURL: String?)?

    func snapshot() -> Snapshot? {
        guard let pid = access.frontmostPID() else { return nil }
        let application = AXUIElementCreateApplication(pid)
        guard let focused = elementAttribute(application, kAXFocusedUIElementAttribute),
              string(focused, kAXSubroleAttribute) != kAXSecureTextFieldSubrole,
              let value = string(focused, kAXValueAttribute), value.utf16.count <= 200_000,
              let rangeValue = attribute(focused, kAXSelectedTextRangeAttribute),
              CFGetTypeID(rangeValue) == AXValueGetTypeID() else { return nil }
        var range = CFRange()
        guard AXValueGetValue(rangeValue as! AXValue, .cfRange, &range),
              Range(NSRange(location: range.location, length: range.length), in: value) != nil else { return nil }
        guard access.canSelect(focused) else { return nil }
        return Snapshot(element: focused, pid: pid, value: value,
                        selection: NSRange(location: range.location, length: range.length))
    }

    func rememberInsertion(_ text: String, into snapshot: Snapshot?) {
        guard let snapshot, let anchor = WritingTextAnchor.inserting(text, into: snapshot.value, at: snapshot.selection) else {
            lastInsertion = nil
            return
        }
        lastInsertion = (snapshot, anchor, documentURL(snapshot.element))
    }

    func prepare(_ action: VoiceWritingAction) throws -> Session {
        guard let pid = access.frontmostPID() else { throw VoiceWritingError.unavailableField }
        access.prepareTextAccessibility(pid)
        guard let current = snapshot(), current.pid == pid else { throw VoiceWritingError.unavailableField }
        switch action {
        case .editLast:
            guard let previous = lastInsertion,
                  sameField(current, previous.snapshot), WritingTextAnchor.sameFieldText(current.value, previous.anchor.value),
                  documentURL(current.element) == previous.documentURL else {
                throw VoiceWritingError.noLastInsertion
            }
            return Session(action: action, snapshot: current, anchor: previous.anchor, thread: "", documentURL: documentURL(current.element))
        case .draftReply:
            // Do not overwrite a draft or an automatic signature. Insert at the
            // captured caret; follow-up editing can then refine just our reply.
            guard current.selection.length == 0,
                  string(current.element, kAXRoleAttribute) == kAXTextAreaRole,
                  let anchor = WritingTextAnchor.inserting("", into: current.value, at: current.selection) else {
                throw VoiceWritingError.gmailRequired
            }
            let context = try gmailThread(containing: current.element)
            guard !context.text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
                throw VoiceWritingError.threadUnavailable
            }
            return Session(action: action, snapshot: current, anchor: anchor, thread: context.text, documentURL: context.url)
        }
    }

    /// Check immediately before selecting/pasting, including cursor movement and
    /// SPA navigation. Never activate another application or select all its text.
    func selectReplacement(for session: Session, text: String) throws {
        guard let current = snapshot(), sameField(current, session.snapshot),
              current.value.utf16.elementsEqual(session.snapshot.value.utf16),
              current.selection == session.snapshot.selection,
              documentURL(current.element) == session.documentURL,
              let next = session.anchor.replacing(with: text, currentValue: current.value) else {
            throw VoiceWritingError.changedTarget
        }
        guard access.select(current.element, session.anchor.range),
              let selected = snapshot(), sameField(selected, current), selected.value.utf16.elementsEqual(current.value.utf16),
              selected.selection == session.anchor.range else { throw VoiceWritingError.changedTarget }
        lastInsertion = (current, next, session.documentURL)
    }

    private func sameField(_ lhs: Snapshot, _ rhs: Snapshot) -> Bool {
        lhs.pid == rhs.pid && CFEqual(lhs.element, rhs.element)
    }

    private func attribute(_ element: AXUIElement, _ name: String) -> CFTypeRef? {
        access.copy(element, name)
    }

    private func elementAttribute(_ element: AXUIElement, _ name: String) -> AXUIElement? {
        guard let value = attribute(element, name), CFGetTypeID(value) == AXUIElementGetTypeID() else { return nil }
        return (value as! AXUIElement)
    }

    private func string(_ element: AXUIElement, _ name: String) -> String? {
        attribute(element, name) as? String
    }

    private func url(_ element: AXUIElement) -> String? {
        if let url = attribute(element, kAXURLAttribute) as? URL { return url.absoluteString }
        return string(element, kAXURLAttribute)
    }

    private func ancestors(_ element: AXUIElement) -> [AXUIElement] {
        var result: [AXUIElement] = [element]
        let deadline = Date().addingTimeInterval(0.3)
        while result.count < 32, Date() < deadline,
              let parent = elementAttribute(result.last!, kAXParentAttribute),
              !result.contains(where: { CFEqual($0, parent) }) {
            result.append(parent)
            if string(parent, kAXRoleAttribute) == "AXWebArea" { break }
        }
        return result
    }

    private func documentURL(_ element: AXUIElement) -> String? {
        ancestors(element).compactMap { url($0) }.first
    }

    private func gmailThread(containing composer: AXUIElement) throws -> (text: String, url: String) {
        let parents = ancestors(composer)
        guard let webArea = parents.first(where: { string($0, kAXRoleAttribute) == "AXWebArea" }),
              let pageURL = url(webArea), GmailWritingPolicy.isConversationURL(pageURL),
              let main = parents.first(where: { string($0, kAXSubroleAttribute) == "AXLandmarkMain" }) else { throw VoiceWritingError.gmailRequired }

        // Scope to the main conversation containing this composer, excluding the
        // composer itself and controls. Never walk other tabs or Gmail's sidebar.
        let deadline = Date().addingTimeInterval(0.5)
        var stack = [main]
        var visited: [AXUIElement] = []
        var pieces: [String] = []
        var count = 0
        while let element = stack.popLast() {
            guard Date() < deadline, visited.count < 1_200 else { throw VoiceWritingError.threadUnavailable }
            if CFEqual(element, composer) || visited.contains(where: { CFEqual($0, element) }) { continue }
            visited.append(element)
            if (attribute(element, "AXHidden") as? Bool) == true { continue }
            let role = string(element, kAXRoleAttribute) ?? ""
            if ["AXTextArea", "AXTextField", "AXButton", "AXMenu", "AXToolbar"].contains(role) { continue }
            if role == "AXStaticText", let text = string(element, kAXValueAttribute), !text.isEmpty {
                count += text.utf16.count
                guard count <= 20_000 else { throw VoiceWritingError.threadUnavailable }
                pieces.append(text)
            } else if let children = attribute(element, kAXChildrenAttribute) as? [AXUIElement] {
                stack.append(contentsOf: children.reversed())
            }
        }
        return (pieces.joined(separator: "\n"), pageURL)
    }
}
