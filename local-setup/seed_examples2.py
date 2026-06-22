#!/usr/bin/env python3
"""
Hand-authored batch #2 — Claude-quality targets (a teacher stronger than 20b),
deliberately covering the cases where gpt-oss-20b distorts: preserving code
identifiers / paths / flags, numbers and ASR homophones, multi-step
self-corrections, lists, foreign words, and exact meaning. No API, no limits.
Appends to the synthetic set (deduped).
"""
import json
import os
import sqlite3

OUT = os.path.expanduser("~/.freeflow-ft/synthetic.jsonl")
DB = os.path.expanduser("~/Library/Application Support/FreeFlow Dev/PipelineHistory.sqlite")

PAIRS = [
 ("ok so the bug is in the the handleSubmit function\nit's not awaiting the the fetch call\nso the state updates before the response comes back\njust wrap it in an async await and it should be fine",
  "Okay, so the bug is in the handleSubmit function — it's not awaiting the fetch call, so the state updates before the response comes back. Just wrap it in an async/await and it should be fine."),
 ("hi James\nthanks for jumping on the call earlier\nas discussed we can do the the annual plan at\nfifteen hundred a month\nwhich is about a twenty percent discount\nlet me know if that works and I'll send the contract over",
  "Hi James, thanks for jumping on the call earlier. As discussed, we can do the annual plan at fifteen hundred a month, which is about a twenty percent discount. Let me know if that works and I'll send the contract over."),
 ("let's launch on the the fifteenth\nno wait that's a weekend\nmake it the seventeenth\nactually the the sixteenth is better because the newsletter goes out that day",
  "Let's launch on the seventeenth — actually, the sixteenth is better, because the newsletter goes out that day."),
 ("can you run the the migration script\nit's in scripts slash migrate dot py\nand make sure you pass the dash dash dry run flag first\nbefore you actually apply it",
  "Can you run the migration script? It's in scripts/migrate.py. And make sure you pass the --dry-run flag first, before you actually apply it."),
 ("yeah I saw that\nI think it's because the the cache isn't being invalidated on update\nlowkey surprised it worked at all before\nanyway I'll push a fix this arvo",
  "Yeah, I saw that. I think it's because the cache isn't being invalidated on update — lowkey surprised it worked at all before. Anyway, I'll push a fix this arvo."),
 ("set up a a meeting with the design team\nfor thursday at two\ntitle it design review\nand add the the figma link in the description",
  "Set up a meeting with the design team for Thursday at two. Title it 'Design review' and add the Figma link in the description."),
 ("hi yeah I'd like to to request a refund\nthe product just didn't work for our use case\nwe only used it for like two days\nso hopefully that's within your policy",
  "Hi, yeah, I'd like to request a refund. The product just didn't work for our use case — we only used it for like two days, so hopefully that's within your policy."),
 ("so the the outage was caused by a a bad deploy\nwe pushed a migration that locked the the users table\nfor about four minutes\nthe the fix is to run migrations off peak and add a a timeout",
  "So the outage was caused by a bad deploy — we pushed a migration that locked the users table for about four minutes. The fix is to run migrations off-peak and add a timeout."),
 ("hot take\nmost startups don't have a a distribution problem\nthey have a a product no one wants problem\nif people actually wanted it\nthey'd find a way to to tell their friends",
  "Hot take: most startups don't have a distribution problem, they have a product-no-one-wants problem. If people actually wanted it, they'd find a way to tell their friends."),
 ("Hey Priya\nreally impressed by your work on the the payments infra at Stripe\nwe're building something in the the lead gen space\nand looking for a a founding engineer\nwould you be open to a quick chat",
  "Hey Priya, really impressed by your work on the payments infra at Stripe. We're building something in the lead-gen space and looking for a founding engineer. Would you be open to a quick chat?"),
 ("the the mockups look great overall\nmy only note is the the onboarding feels like a few too many steps\ncan we collapse it into like two screens\nthe the rest is spot on",
  "The mockups look great overall. My only note is the onboarding feels like a few too many steps — can we collapse it into like two screens? The rest is spot on."),
 ("yeah that works for me\ngo ahead and ship it",
  "Yeah, that works for me. Go ahead and ship it."),
 ("so I was thinking about the the pricing again\nand I know we said we'd keep it simple\nbut I reckon we're leaving money on the table with the the flat rate\nlike the the power users are getting a steal\nmaybe we do usage based above a a certain threshold\nanyway just a thought we can talk about it tomorrow",
  "So I was thinking about the pricing again, and I know we said we'd keep it simple, but I reckon we're leaving money on the table with the flat rate. The power users are getting a steal. Maybe we do usage-based above a certain threshold. Anyway, just a thought — we can talk about it tomorrow."),
 ("we need at least for engineers to to hit the deadline\nright now we've only got too\nso we either hire or we cut scope",
  "We need at least four engineers to hit the deadline. Right now we've only got two, so we either hire or we cut scope."),
 ("the the biggest thing she said was\nthey waste like ten hours a week\nmanually qualifying leads\nand she'd happily pay to to get that time back\nthat's that's exactly our pitch",
  "The biggest thing she said was they waste like ten hours a week manually qualifying leads, and she'd happily pay to get that time back. That's exactly our pitch."),
 ("no that's not what I asked for\nI wanted you to to refactor it not rewrite the whole thing\nyou've changed like half the the behaviour\ncan you revert and just do the the small change I asked for",
  "No, that's not what I asked for. I wanted you to refactor it, not rewrite the whole thing — you've changed like half the behaviour. Can you revert and just do the small change I asked for?"),
 ("we should call the the feature something memorable\nmy cofounder suggested\ntodo bien\nwhich means all good in spanish\nI actually kind of like it",
  "We should call the feature something memorable. My cofounder suggested 'todo bien,' which means 'all good' in Spanish — I actually kind of like it."),
 ("quick update\nbackend's done\nfrontend is like eighty percent there\nI'm blocked on the the api keys for the the new provider\ncan someone sort that out",
  "Quick update: backend's done, frontend is like eighty percent there. I'm blocked on the API keys for the new provider — can someone sort that out?"),
 ("the the question they'll ask is why now\nand honestly the the answer is the the models just got good enough\nlike two years ago this wouldn't have worked\nthe accuracy wasn't there",
  "The question they'll ask is 'why now,' and honestly the answer is the models just got good enough. Two years ago this wouldn't have worked — the accuracy wasn't there."),
 ("before the the trip can you\nbook the the airport parking\npick up the dry cleaning\nand charge the the portable battery\noh and water the plants",
  "Before the trip, can you book the airport parking, pick up the dry cleaning, and charge the portable battery? Oh, and water the plants."),
 ("I've emailed them three times now\nstill no response\nthis is the the kind of support that makes you churn\nwe pay them like five hundred a month for this\nit's a joke honestly",
  "I've emailed them three times now — still no response. This is the kind of support that makes you churn. We pay them like five hundred a month for this. It's a joke, honestly."),
 ("todo follow up with the the lawyer about the the SAFE before friday",
  "Todo: follow up with the lawyer about the SAFE before Friday."),
 ("I reckon we just go with tailwind\neveryone on the team knows it\nand we don't have time to to bikeshed a a design system right now\nwe can always refactor later",
  "I reckon we just go with Tailwind. Everyone on the team knows it, and we don't have time to bikeshed a design system right now. We can always refactor later."),
 ("Hi so sorry for the the delay on this\nwe've been heads down on a a release\nto answer your question yes the the export does support csv\nit's under the the settings menu",
  "Hi, so sorry for the delay on this — we've been heads down on a release. To answer your question: yes, the export does support CSV. It's under the settings menu."),
 ("the the activation rate is the the key number to watch\nit's the percent of signups who do the the core action\nin their first session\nright now we're at like thirty percent\nI want to get it to fifty",
  "The activation rate is the key number to watch — it's the percent of signups who do the core action in their first session. Right now we're at like thirty percent; I want to get it to fifty."),
 ("keen for a surf saturday morning\nswells looking decent\nlet's meet at the the usual spot around six\nbefore it gets too crowded",
  "Keen for a surf Saturday morning — swell's looking decent. Let's meet at the usual spot around six, before it gets too crowded."),
 ("what about\nqualify leads in seconds not days\nor maybe\nstop wasting time on bad leads\nhmm the the first one is punchier I think",
  "What about 'Qualify leads in seconds, not days'? Or maybe 'Stop wasting time on bad leads.' Hmm, the first one is punchier, I think."),
 ("I hear you but I I don't fully agree\nI think shipping fast matters more than than polish right now\nwe can always come back and and clean it up\nlet's not let perfect be the the enemy of done",
  "I hear you, but I don't fully agree. I think shipping fast matters more than polish right now — we can always come back and clean it up. Let's not let perfect be the enemy of done."),
]


def system_prompt():
    try:
        conn = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
        r = conn.execute("SELECT ZSYSTEMPROMPT FROM ZPIPELINEHISTORYENTRY "
                         "WHERE ZSYSTEMPROMPT<>'' ORDER BY ZTIMESTAMP DESC LIMIT 1").fetchone()
        conn.close()
        if r:
            return r[0]
    except Exception:
        pass
    return "Clean up dictated speech into polished written text."


def main():
    sysp = system_prompt()
    seen = set()
    if os.path.exists(OUT):
        for line in open(OUT):
            try:
                raw = json.loads(line).get("raw", "")
                if raw:
                    seen.add(raw[:60].lower())
            except json.JSONDecodeError:
                pass
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    added = 0
    with open(OUT, "a") as f:
        for raw, clean in PAIRS:
            raw, clean = raw.strip(), clean.strip()
            key = raw[:60].lower()
            if key in seen:
                continue
            seen.add(key)
            f.write(json.dumps({
                "messages": [
                    {"role": "system", "content": sysp},
                    {"role": "user", "content": f"Raw transcript:\n{raw}\n\nClean this up into the final text."},
                    {"role": "assistant", "content": clean},
                ],
                "label_source": "synthetic_handwritten",
                "raw": raw, "gold": clean, "domain": "handwritten2",
            }, ensure_ascii=False) + "\n")
            added += 1
    print(f"appended {added} hand-authored (batch 2) pairs -> {OUT}")


if __name__ == "__main__":
    main()
