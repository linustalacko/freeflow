#!/usr/bin/env python3
"""
Hand-authored synthetic dictation-cleanup pairs — no API, no rate limits.

Written directly to match the user's REAL Whisper texture: already capitalised +
punctuated, with mid-sentence hard line breaks, run-ons, trailing fillers,
self-corrections, the odd ASR slip / repeated word, casual+profane founder/dev
content, Australian spelling. CLEAN = line breaks joined, run-ons split, fillers
dropped, slips fixed, meaning + tone preserved.

Appends to the synthetic set (deduped). Re-runnable.
"""
import json
import os
import sqlite3

OUT = os.path.expanduser("~/.freeflow-ft/synthetic.jsonl")
DB = os.path.expanduser("~/Library/Application Support/FreeFlow Dev/PipelineHistory.sqlite")

PAIRS = [
 ("Hey so the the refactor you did on the auth flow\nis mostly good but you left a couple of\nconsole logs in there\nand the error handling is to working only on the happy path um\ncan you sort that out",
  "Hey, so the refactor you did on the auth flow is mostly good, but you left a couple of console logs in there, and the error handling only works on the happy path. Can you sort that out?"),
 ("Hi Sarah\njust wanted to reach out because I saw you're scaling the the support team\nand we built something that might help with ticket triage\nwould you be open to a quick chat next week",
  "Hi Sarah, just wanted to reach out because I saw you're scaling the support team and we built something that might help with ticket triage. Would you be open to a quick chat next week?"),
 ("honestly the the meeting with that fund was a bit of a waste of time\nthey kept asking about TAM like mate we're pre revenue\nthe whole thing felt like a box ticking exercise\nI reckon we just focus on the angels who actually get it",
  "Honestly, the meeting with that fund was a bit of a waste of time. They kept asking about TAM — mate, we're pre-revenue. The whole thing felt like a box-ticking exercise. I reckon we just focus on the angels who actually get it."),
 ("This PR adds the retry logic to the email sender\nit wraps the the send call in a backoff\nand logs each attempt\nalso bumped the timeout to thirty seconds because\nGroq was occasionally slow",
  "This PR adds retry logic to the email sender. It wraps the send call in a backoff and logs each attempt. I also bumped the timeout to thirty seconds because Groq was occasionally slow."),
 ("yeah nah I'm keen for this arvo\njust finishing up some work\nmight be like twenty minutes late though\nis that alright",
  "Yeah, nah, I'm keen for this arvo. Just finishing up some work. I might be like twenty minutes late though — is that alright?"),
 ("so the new feature should let users tag a lead as qualified\nor actually no\nlet's make it three states\nqualified not qualified and needs review\nthat way the model has more signal",
  "So the new feature should let users tag a lead as qualified — actually, no, let's make it three states: qualified, not qualified, and needs review. That way the model has more signal."),
 ("ok so out of the the two hundred emails we sent\nabout forty bounced\nand we got like five replies\nthree of them were just unsubscribe requests\nso the list quality is pretty dodgy honestly",
  "Okay, so out of the two hundred emails we sent, about forty bounced and we got like five replies. Three of them were just unsubscribe requests, so the list quality is pretty dodgy, honestly."),
 ("remind me to to follow up with the YC group partner about the demo day slot um tomorrow morning",
  "Remind me to follow up with the YC group partner about the Demo Day slot tomorrow morning."),
 ("this message sounds way too salesy\nlike no one talks like this\ncan we make it sound more human\nmaybe just lead with the problem they actually have\nand cut all the the corporate fluff",
  "This message sounds way too salesy — no one talks like this. Can we make it sound more human? Maybe just lead with the problem they actually have and cut all the corporate fluff."),
 ("I think we should just use SQLite for now\nthere's no need to spin up a whole postgres instance\nfor like a few thousand rows\nwe can always migrate later if we need to",
  "I think we should just use SQLite for now. There's no need to spin up a whole Postgres instance for a few thousand rows. We can always migrate later if we need to."),
 ("Hi there thanks for flagging this\nwe've identified the the issue it was a caching bug\non our end\nit should be resolved now\nlet us know if you see it again",
  "Hi there, thanks for flagging this. We've identified the issue — it was a caching bug on our end. It should be resolved now. Let us know if you see it again."),
 ("yesterday I finished the the export script\ntoday I'm working on the eval harness\nno blockers really\noh actually one thing the groq key is rate limited so heads up",
  "Yesterday I finished the export script. Today I'm working on the eval harness. No blockers, really. Oh, actually, one thing — the Groq key is rate-limited, so heads up."),
 ("spent the the weekend fine tuning a tiny model on my laptop\nto replace a cloud api\nturns out for narrow tasks you really don't need a massive model\nwild how far you can get with like a few hundred examples",
  "Spent the weekend fine-tuning a tiny model on my laptop to replace a cloud API. Turns out for narrow tasks you really don't need a massive model. Wild how far you can get with a few hundred examples."),
 ("Hi Mark thanks for the the note\nhappy to share more detail\nour current MRR is around eight K\ngrowing roughly twenty percent month on month\nI'll send over the deck and the data room link separately",
  "Hi Mark, thanks for the note. Happy to share more detail. Our current MRR is around eight K, growing roughly twenty percent month on month. I'll send over the deck and the data-room link separately."),
 ("idea for later\nwhat if we let people record a voice note\nand the the app automatically turns it into a structured lead summary\nname company pain point next step\ncould be a killer feature",
  "Idea for later: what if we let people record a voice note and the app automatically turns it into a structured lead summary — name, company, pain point, next step? Could be a killer feature."),
 ("look I hear you on the pricing\nbut I really don't think we should go freemium yet\nwe'll just attract a bunch of tyre kickers\nand the support load will kill us\ncan we at least trial it with a a paid tier first",
  "Look, I hear you on the pricing, but I really don't think we should go freemium yet. We'll just attract a bunch of tyre-kickers and the support load will kill us. Can we at least trial it with a paid tier first?"),
 ("to reproduce this\nfirst you log in as a new user\nthen you go to the the leads page\nand if you click export before the table loads\nthe whole thing crashes\nhappens every time on safari",
  "To reproduce this: first you log in as a new user, then you go to the leads page, and if you click export before the table loads, the whole thing crashes. It happens every time on Safari."),
 ("the new dashboard looks clean but\nthe the contrast on the secondary text is too low\nI can barely read it\nalso the primary button kind of gets lost\nmaybe make it the accent colour",
  "The new dashboard looks clean, but the contrast on the secondary text is too low — I can barely read it. Also, the primary button kind of gets lost; maybe make it the accent colour."),
 ("so the prospect is a series A fintech\nthey've got about fifty people\nthe main pain is they're drowning in inbound\nand they can't qualify fast enough\nbudget wasn't really discussed but they seemed keen",
  "So the prospect is a Series A fintech. They've got about fifty people. The main pain is they're drowning in inbound and they can't qualify fast enough. Budget wasn't really discussed, but they seemed keen."),
 ("really liked this candidate\nstrong on the systems design\na bit shaky on the the frontend stuff\nbut honestly that's fine for the role\nI'd move them to the final round",
  "Really liked this candidate. Strong on the systems design, a bit shaky on the frontend stuff, but honestly that's fine for the role. I'd move them to the final round."),
 ("ok priorities for this week\nnumber one ship the the lead scoring v2\nnumber two fix the onboarding drop off\nand if we have time look into the the latency on the api\nbut that's a nice to have",
  "Okay, priorities for this week. Number one: ship the lead scoring v2. Number two: fix the onboarding drop-off. And if we have time, look into the latency on the API — but that's a nice-to-have."),
 ("hey mum just letting you know\nI'll be over for dinner on sunday\ncan I bring anything\nalso is dad still keen to to watch the footy after",
  "Hey Mum, just letting you know I'll be over for dinner on Sunday. Can I bring anything? Also, is Dad still keen to watch the footy after?"),
 ("this is the third time this month their api has gone down\nand we get zero heads up\nit's costing us actual customers\nwe need to seriously look at moving off them\nor at least having a fallback",
  "This is the third time this month their API has gone down, and we get zero heads-up. It's costing us actual customers. We need to seriously look at moving off them, or at least having a fallback."),
 ("thinking about names for the the qualification feature\nmaybe like\nsmart sort\nor lead lens\nor qualify ai\nhonestly none of them are quite right yet\nlet's sleep on it",
  "Thinking about names for the qualification feature. Maybe Smart Sort, or Lead Lens, or Qualify AI. Honestly, none of them are quite right yet. Let's sleep on it."),
 ("action items from the the sync\nLinus to finalise the pricing page\nDen to follow up with the two warm leads\nand we agreed to push the launch to to the end of the month",
  "Action items from the sync: Linus to finalise the pricing page, Den to follow up with the two warm leads, and we agreed to push the launch to the end of the month."),
 ("yeah I'm I'm down for that\nfriday works better for me than thursday though\nif that's cool with everyone",
  "Yeah, I'm down for that. Friday works better for me than Thursday though, if that's cool with everyone."),
 ("so the the conversion went up from like two percent to three point five\nafter we changed the onboarding\nwhich doesn't sound like much\nbut that's almost double\nso it's actually a massive win",
  "So the conversion went up from like two percent to three point five after we changed the onboarding. Which doesn't sound like much, but that's almost double — so it's actually a massive win."),
 ("hey sorry something's come up\ncan we push our call from\nthree to to like four thirty\nstill same day just a bit later",
  "Hey, sorry, something's come up. Can we push our call from three to like four thirty? Still the same day, just a bit later."),
 ("can you grab\nmilk eggs some bread\noh and coffee we're nearly out\nand if they have it some of that the the sourdough from the bakery section",
  "Can you grab milk, eggs, some bread? Oh, and coffee — we're nearly out. And if they have it, some of that sourdough from the bakery section."),
 ("ok so I'll be there around six\nthe the place is just off the main road\npark out the back there's usually heaps of spots\ntext me when you're close",
  "Okay, so I'll be there around six. The place is just off the main road. Park out the back — there's usually heaps of spots. Text me when you're close."),
 ("the summary you generated is good\nbut you got the the client's name wrong\nit's Aysha not Aisha\ncan you fix that and regenerate\nthe rest is fine",
  "The summary you generated is good, but you got the client's name wrong — it's Aysha, not Aisha. Can you fix that and regenerate? The rest is fine."),
 ("so I've been thinking a lot about the the positioning\nright now we're kind of saying we do lead qualification\nbut that's like everyone says that\nI think the the real wedge is the speed\nwe qualify in seconds not days\nand we should lean into that hard\nin all the messaging and the the landing page um yeah",
  "So I've been thinking a lot about the positioning. Right now we're kind of saying we do lead qualification, but everyone says that. I think the real wedge is the speed — we qualify in seconds, not days. We should lean into that hard, in all the messaging and the landing page."),
 ("let's not over engineer the the queue\nwe don't need kafka\na simple redis list will do for our volume\nwe can revisit when we're processing millions of jobs\nwhich is a good problem to have",
  "Let's not over-engineer the queue. We don't need Kafka — a simple Redis list will do for our volume. We can revisit when we're processing millions of jobs, which is a good problem to have."),
 ("Hi yeah so the the feature you're asking about\nis on the roadmap for q3\nactually sorry q2\nwe pulled it forward\nso you should see it in the next couple of months",
  "Hi, yeah, so the feature you're asking about is on the roadmap for Q2 — sorry, we pulled it forward. So you should see it in the next couple of months."),
 ("note to self stop saying just on every sales call it sounds weak",
  "Note to self: stop saying 'just' on every sales call — it sounds weak."),
 ("the the export is producing empty files\nwhen there's no data\ninstead of just skipping\nit's a small thing but it crashes the downstream script\nso worth fixing",
  "The export is producing empty files when there's no data, instead of just skipping. It's a small thing, but it crashes the downstream script, so it's worth fixing."),
 ("Hey Tom\nsaw your post about the the hiring sprint\ncongrats on the the round by the way\nwe help teams like yours qualify inbound faster\nworth a quick look maybe",
  "Hey Tom, saw your post about the hiring sprint — congrats on the round, by the way. We help teams like yours qualify inbound faster. Worth a quick look, maybe?"),
 ("honestly this whole integration has been a nightmare\ntheir docs are shit\nhalf the endpoints are undocumented\nand support takes like three days to reply\nfucking hate building on other people's platforms sometimes",
  "Honestly, this whole integration has been a nightmare. Their docs are shit, half the endpoints are undocumented, and support takes like three days to reply. I hate building on other people's platforms sometimes."),
 ("for the the launch we need\nthe landing page done\nthe demo video recorded\nand at least like ten beta users lined up\nI reckon we're a week or two out honestly",
  "For the launch we need the landing page done, the demo video recorded, and at least ten beta users lined up. I reckon we're a week or two out, honestly."),
 ("the call went well\nthey're definitely feeling the the pain\nthey're currently doing it all in a spreadsheet\nwhich is exactly who we're built for\nnext step is a demo with the the wider team",
  "The call went well. They're definitely feeling the pain — they're currently doing it all in a spreadsheet, which is exactly who we're built for. Next step is a demo with the wider team."),
 ("can you organise the the colour palette into a proper system\nright now it's all over the place\nwe've got like four different blues\nstandardise it and document the the behaviour for hover states",
  "Can you organise the colour palette into a proper system? Right now it's all over the place — we've got like four different blues. Standardise it and document the behaviour for hover states."),
 ("can't make tomorrow sorry can we do thursday instead",
  "Can't make tomorrow, sorry. Can we do Thursday instead?"),
 ("churn ticked up a bit last month\nfrom like three percent to four\nnot panic territory yet\nbut we should figure out why before it becomes a a trend",
  "Churn ticked up a bit last month, from like three percent to four. Not panic territory yet, but we should figure out why before it becomes a trend."),
 ("the the new linter is way too aggressive\nit's flagging stuff that's totally fine\nand it's slowing down the the whole CI\ncan we dial back the rules to just the important ones",
  "The new linter is way too aggressive — it's flagging stuff that's totally fine, and it's slowing down the whole CI. Can we dial back the rules to just the important ones?"),
 ("been reflecting on the the last quarter\nwe shipped a lot which is good\nbut I think we got a bit distracted\nby features that no one really asked for\nnext quarter I want us to be way more disciplined\ntalk to users first then build\nnot the other way around um yeah that's the the main takeaway",
  "Been reflecting on the last quarter. We shipped a lot, which is good, but I think we got a bit distracted by features that no one really asked for. Next quarter I want us to be way more disciplined — talk to users first, then build, not the other way around. That's the main takeaway."),
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
                "raw": raw, "gold": clean, "domain": "handwritten",
            }, ensure_ascii=False) + "\n")
            added += 1
    print(f"appended {added} hand-authored pairs -> {OUT}")


if __name__ == "__main__":
    main()
