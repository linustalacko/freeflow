#!/usr/bin/env python3
"""Parity tests for detclean.py against its Swift twins.

`detclean.py` mirrors Sources/TranscriptFastPath.swift, SpokenFormatting.swift,
and DictationProfile.swift so the router's deterministic path produces exactly
what the app's own fast path would have. The GOLDEN table below is the Swift
implementation's actual output for each case — regenerate it by running the same
inputs through `TranscriptFastPath.cleanedIfAlreadyClean` if the Swift changes.

Run: python3 local-setup/test_detclean.py
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from detclean import det_clean, finalize_text, profile_for, profile_from_system_prompt, spoken_format

# (target, input) -> Swift output; None means "bail to the model".
GOLDEN = [
    ("document", "the gap is 5 mm wide", "The gap is 5 mm wide."),
    ("document", "take it to the ER", "Take it to the ER."),
    ("chat", "mhm that could work", "Mhm that could work"),
    ("chat", "hmm that seems unlikely", "Hmm that seems unlikely"),
    ("chat", "5 mm", "5 mm"),
    ("chat", "bullet point ship the release bullet point update the docs",
     "- Ship the release\n- Update the docs"),
    ("email", "bullet point one thing bullet point another thing",
     "• One thing\n• Another thing"),
    ("chat", "bullet points fix the login bug, update the docs, ship the release",
     "- Fix the login bug\n- Update the docs\n- Ship the release"),
    ("chat", "numbered list write the spec, get review, merge",
     "1. Write the spec\n2. Get review\n3. Merge"),
    ("chat", "here's the plan bullet point fix the login bug bullet point ship it",
     "Here's the plan:\n- Fix the login bug\n- Ship it"),
    ("unknown", "thanks for the note new paragraph I'll look tomorrow",
     "Thanks for the note\n\nI'll look tomorrow."),
    ("terminal", "git status", "git status"),
    ("chat", "sounds good to me", "Sounds good to me"),
    ("unknown", "the deploy finished", "The deploy finished."),
    ("unknown", "um so the deploy is done", "So the deploy is done."),
    ("chat", "add a bullet about the rollback plan and another about cleanup",
     "Add a bullet about the rollback plan and another about cleanup"),
    ("chat", "he said quote this is fine unquote and left",
     "He said “this is fine” and left"),
    ("chat", "we are opening a new line of business", None),
    ("chat", "I need a quote for the new laptop", None),
    ("unknown", "let's meet Thursday no actually Wednesday", None),
    ("email", "hi Dana thanks for sending that over", None),
    ("terminal", "git commit dash m bullet point fix", None),
]


class DetCleanParityTests(unittest.TestCase):
    def test_matches_swift_golden_output(self):
        for target, text, expected in GOLDEN:
            with self.subTest(target=target, text=text):
                self.assertEqual(det_clean(text, profile=profile_for(target)), expected)

    def test_fragment_plus_finalize_equals_whole_text_cleanup(self):
        """The router cleans settled chunks as fragments and finalizes the assembled
        whole; on a single piece that must be exactly the whole-text result."""
        for target, text, expected in GOLDEN:
            profile = profile_for(target)
            frag = det_clean(text, profile=profile, fragment=True)
            with self.subTest(target=target, text=text):
                if expected is None:
                    self.assertIsNone(frag, "fragment mode keeps every bail rule")
                else:
                    self.assertEqual(finalize_text(frag, profile), expected)

    def test_fragment_mode_leaves_casing_and_closing_to_the_whole(self):
        unknown = profile_for("unknown")
        self.assertEqual(det_clean("so the deploy is done", profile=unknown, fragment=True), "so the deploy is done")
        self.assertEqual(det_clean("um so the deploy is done", profile=unknown, fragment=True), "so the deploy is done")
        self.assertEqual(det_clean("um", profile=unknown, fragment=True), "", "an all-filler fragment is simply empty")
        self.assertEqual(det_clean("first bit comma then more", profile=unknown, fragment=True), "first bit, then more")
        self.assertEqual(
            finalize_text("hold up the pendant should be an action. or you should see her", unknown),
            "Hold up the pendant should be an action. Or you should see her.")
        self.assertEqual(finalize_text("sounds good", profile_for("chat")), "Sounds good")
        self.assertEqual(finalize_text("git status", profile_for("terminal")), "git status")
        self.assertEqual(finalize_text("Plan:\n- ship it\n- update docs", profile_for("chat")),
                         "Plan:\n- Ship it\n- Update docs", "a list never gets a closing full stop")
        self.assertEqual(finalize_text("", unknown), "")

    def test_profile_recovered_from_destination_line(self):
        """The app appends a `Destination:` line to its default system prompt;
        that is how the router learns which target to format for."""
        chat_prompt = (
            "You are a literal dictation cleanup layer\n\nDestination:\n"
            "- The text is going into a chat message. Keep it casual and conversational."
        )
        self.assertTrue(profile_from_system_prompt(chat_prompt)["is_casual"])

        email_prompt = (
            "You are a literal dictation cleanup layer\n\nDestination:\n"
            "- The text is going into an email. If a greeting was spoken, put it on its own first line."
        )
        self.assertTrue(profile_from_system_prompt(email_prompt)["wants_email_layout"])

    def test_missing_destination_line_matches_swift_unknown(self):
        """No hint must behave exactly as the pipeline did before profiles existed."""
        profile = profile_from_system_prompt("You are a literal dictation cleanup layer")
        self.assertEqual(profile, profile_for("unknown"))
        self.assertTrue(profile["wants_email_layout"])
        self.assertFalse(profile["preserves_verbatim"])

    def test_verbatim_targets_never_reformat(self):
        terminal = profile_for("terminal")
        text, confident, produced_list = spoken_format("bullet point one bullet point two", terminal)
        self.assertFalse(confident, "formatting words in a terminal are literal content")
        self.assertFalse(produced_list)

    def test_ambiguous_readings_bail_rather_than_guess(self):
        chat = profile_for("chat")
        for text in ("we are opening a new line of business",
                     "that would be another new paragraph entirely",
                     "I need a quote for the new laptop"):
            with self.subTest(text=text):
                self.assertFalse(spoken_format(text, chat)[1])


if __name__ == "__main__":
    unittest.main(verbosity=2)
