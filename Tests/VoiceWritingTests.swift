import Foundation

enum VoiceWritingTests {
    static func run() {
        let original = "Same. 👩🏽‍💻 Same. End"
        let range = (original as NSString).range(of: "Same.", options: .backwards)
        let anchor = WritingTextAnchor.inserting("Thursday. ", into: original, at: range)!
        TestSupport.expectEqual(anchor.value, "Same. 👩🏽‍💻 Thursday.  End")
        let edited = anchor.replacing(with: "Friday. ", currentValue: anchor.value)!
        TestSupport.expectEqual(edited.value, "Same. 👩🏽‍💻 Friday.  End")
        TestSupport.expectEqual(edited.undoText, "Thursday. ")
        TestSupport.expectEqual(edited.replacing(with: edited.undoText!, currentValue: edited.value)?.value, anchor.value)
        TestSupport.expectEqual(anchor.replacing(with: "Wrong", currentValue: anchor.value + "x"), nil)
        TestSupport.expectEqual(WritingTextAnchor.inserting("x", into: "👋", at: NSRange(location: 1, length: 0)), nil)
        TestSupport.expectEqual(WritingTextAnchor.inserting("x", into: "abc", at: NSRange(location: 5, length: 0)), nil)
        let composed = WritingTextAnchor.inserting("x", into: "é", at: NSRange(location: 1, length: 0))!
        TestSupport.expectEqual(composed.replacing(with: "y", currentValue: "e\u{301}x"), nil)
        let browser = WritingTextAnchor.inserting("Friday. ", into: "", at: NSRange(location: 0, length: 0))!
        TestSupport.expectEqual(browser.replacing(with: "Thursday. ", currentValue: "Friday.\u{A0}")?.value, "Thursday. ")
        TestSupport.expect(!WritingTextAnchor.sameFieldText("a\nb", "a\r\nb"), "Line-ending changes alter UTF-16 offsets")
        for command in ["Undo.", "undo that", "  Undo the last edit! "] {
            TestSupport.expect(WritingTextAnchor.isUndo(command), "Explicit undo should use the local path")
        }
        TestSupport.expect(!WritingTextAnchor.isUndo("don't undo that"), "Do not interpret a negation as undo")
        TestSupport.expect(!WritingTextAnchor.isUndo("mention undo in the reply"), "Do not interpret prose as undo")
        TestSupport.expectEqual(WritingTextAnchor.pasteText("Hello."), "Hello. ")
        TestSupport.expectEqual(WritingTextAnchor.pasteText("Hello. "), "Hello. ")
        TestSupport.expect(GmailWritingPolicy.isConversationURL("https://mail.google.com/mail/u/0/#inbox/1234567890abcdef"), "Accept Gmail conversation")
        for url in ["https://mail.google.com.evil.example/mail/u/0/#inbox/1234567890abcdef",
                    "http://mail.google.com/mail/u/0/#inbox/1234567890abcdef", "https://mail.google.com/mail/u/0/#inbox",
                    "https://example.com/mail/u/0/#inbox/1234567890abcdef"] {
            TestSupport.expect(!GmailWritingPolicy.isConversationURL(url), "Reject non-conversation URL")
        }
        shortcuts()
    }

    private static func shortcuts() {
        let edit = ShortcutBinding(keyCode: 14, keyDisplay: "E", modifiers: [.control, .option], kind: .key, preset: nil,
                                   exactModifierKeyCodes: [59, 58])
        let reply = ShortcutBinding(keyCode: 15, keyDisplay: "R", modifiers: [.control, .option], kind: .key, preset: nil,
                                    exactModifierKeyCodes: [59, 58])
        let config = ShortcutConfiguration(hold: .defaultHold, toggle: .defaultToggle, editLast: edit, draftReply: reply,
                                           permittedAdditionalExactMatchModifiers: .shift)
        var state = ShortcutInputState()
        func feed(_ event: ShortcutInputEvent) -> ShortcutMatchResult {
            let result = ShortcutMatcher.reduce(state: state, event: event, configuration: config)
            state = result.state
            return result
        }
        _ = feed(.modifierSnapshot([59, 58]))
        TestSupport.expectEqual(feed(.keyChanged(keyCode: 14, isDown: true, isRepeat: false)).emittedEvents, [.writingTriggered(.editLast)])
        TestSupport.expect(state.hasPressedShortcutInputs(configuration: config), "Paste must wait for writing chord release")
        TestSupport.expectEqual(feed(.keyChanged(keyCode: 14, isDown: true, isRepeat: true)).emittedEvents, [])
        _ = feed(.modifierSnapshot([59, 58, 56]))
        TestSupport.expectEqual(feed(.modifierSnapshot([59, 58])).emittedEvents, [])
        TestSupport.expectEqual(feed(.keyChanged(keyCode: 14, isDown: false, isRepeat: false)).emittedEvents, [])
        TestSupport.expectEqual(feed(.keyChanged(keyCode: 15, isDown: true, isRepeat: false)).emittedEvents, [.writingTriggered(.draftReply)])
        _ = feed(.backendReset)
        TestSupport.expect(!state.hasPressedShortcutInputs(configuration: config), "Reset must release writing inputs")
        _ = feed(.modifierSnapshot([59, 58, 56]))
        TestSupport.expectEqual(feed(.keyChanged(keyCode: 14, isDown: true, isRepeat: false)).emittedEvents, [])
        TestSupport.expectEqual(feed(.keyChanged(keyCode: 14, isDown: false, isRepeat: false)).consumeDecision, .passthrough)
        let controller = DictationShortcutSessionController()
        TestSupport.expectEqual(controller.handle(event: .writingTriggered(.editLast), isTranscribing: false), nil)
        TestSupport.expectEqual(controller.activeMode, nil)
        TestSupport.expect(!edit.overlapsWritingShortcut(reply), "Different writing keys can coexist")
        TestSupport.expect(edit.overlapsWritingShortcut(.init(keyCode: 14, keyDisplay: "E", modifiers: .control, kind: .key, preset: nil)), "Reject a less specific chord on the same key")
        TestSupport.expect(edit.overlapsWritingShortcut(ShortcutPreset.rightOption.binding), "Modifier-only shortcut could fire before the writing key")
        TestSupport.expect(!edit.overlapsWritingShortcut(ShortcutPreset.fnKey.binding), "Distinct modifier inputs can coexist")
        let controlE = ShortcutBinding(keyCode: 14, keyDisplay: "E", modifiers: .control, kind: .key, preset: nil, exactModifierKeyCodes: [59])
        let conflicting = ShortcutConfiguration(hold: controlE, toggle: .disabled, editLast: edit,
                                               permittedAdditionalExactMatchModifiers: .option)
        TestSupport.expect(conflicting.editLast.isDisabled, "An Edit Mode modifier change must not create overlapping actions")
    }
}
