# Automaton Attack Bot — Claude Code Handoff

Computer-vision bot that plays the **Automaton Attack** typing minigame in
Dota 2's Dark Carnival: Midnight Run event (Chapter 3, Engine Car).
A working prototype exists and is validated offline against recorded
gameplay. This doc captures everything learned so far so development can
continue without re-deriving it.

## Current state

`typebot.py` works end-to-end in offline mode against recorded clips:

- Clip 1 (single-word phase): 11/11 words detected in order, 0 errors.
- Clip 2 (multi-word phase, overlapping words): all words detected and
  correctly matched, including OCR-mangled reads
  (`METEORHARIMER` → Meteor Hammer at 0.88, `MITHRILHAMPEIER` → Mithril
  Hammer at 0.86, `FRITAALBEAST` → Primal Beast at 0.78).
- Voiceline screenshot: `BANEOFYOUREXISTENCE` → "Bane of your existence."
  at 1.00 via the phrase corpus (12.5 ms match time).

**Not yet tested:** live screen capture, actual keystroke delivery into
Dota, higher difficulty levels.

## Game mechanics (observed + wiki-confirmed)

- 60-second timer. Words fly in from the sides, arc up, descend toward
  center; unfinished words disappear at center. No fail state, just the
  timer.
- Starts one word at a time; later multiple simultaneous words that can
  overlap on screen.
- Word bank: hero names, item names, ability names, Dota terms, and
  voice lines. Voicelines score more (100 vs 10 observed).
- Score multiplier increases per completed word; **one failed word resets
  the multiplier to 1** — accuracy beats speed.
- **Spaces and punctuation can be skipped when typing** (Liquipedia).
  Lowercase input is accepted; the bot types letters-only lowercase.
- Multiple difficulty levels: words get longer/more obscure/faster.
- Set Dota to English or words may appear localized.
- The player does not target a word; typed input just applies (confirmed
  by user — "just type whatever is needed").
- On screen, each word renders twice: small yellow-green target text above
  a larger desaturated-white input tracker (yellow highlight on the next
  letter). Detection keys off the yellow target only; the tracker is
  redundant for a bot.
- Valve patched a pause-typing exploit in this minigame (July 24, 2026),
  i.e. they actively watch it. Consider human-plausible pacing/scores.

## Screen geometry (1920x1080 recordings)

- Minigame panel: screen x 481–1443, y 57–1022 (`PANEL` in typebot.py).
  Everything outside is static Dota UI (verified with a per-pixel
  variance map over the full clip).
- HUD boxes inside the panel use the SAME yellow-green as target words
  and must be masked (panel-local coords in `HUD_BOXES`): score ~(40..220,
  0..110), timer ~(390..560, 0..110), high score ~(720..940, 0..110).
  Do NOT mask a whole top band — words legitimately fly through y≈114.

## Color measurements (HSV, OpenCV ranges)

- Yellow target words: H 39–44, S 68–119, V 165–196. Detection range used:
  `(35,50,140)–(50,150,230)` — validated across both clips.
- White input tracker: S≈4, V≈230 (ignored by the bot).

## Pipeline (implemented in typebot.py)

1. Capture frame (offline: cv2.VideoCapture; live: mss, ~3–5 scans/sec).
2. HSV in-range mask for yellow → zero out HUD boxes.
3. Dilate with a (5,25) kernel to merge letters into word blobs →
   connectedComponentsWithStats.
4. Blob filters: area>800, w>50, 12<h<45.
5. Per-blob OCR: tesseract `--psm 7`, whitelist A–Z, 3x cubic upscale,
   inverted mask (black-on-white).
6. Matching, three tiers:
   - Core vocab (`vocab.txt`, 1492 entries) via difflib SequenceMatcher
     with length prefilter + quick_ratio gates. Accept ≥0.62.
   - Phrase corpus (`phrases.txt`, 39,357 voicelines) via rapidfuzz
     `process.extractOne(fuzz.ratio)`. Accept ≥0.80 (huge corpus → a
     wrong steal types an entire wrong phrase).
   - Fallback: unmatched clean OCR of ≥8 letters is typed verbatim.
   All matching is done on space/punctuation-stripped uppercase keys.
7. Confirmation: matches ≥0.85 type immediately; anything lower
   (including all fallbacks) must repeat in 2 consecutive scans.
   Rationale: OCR errors were observed to VARY frame-to-frame
   (`METEORHAKIMER` vs `METEORHARIMER`), so exact repetition implies a
   correct read. Tesseract confidence is NOT reliable here — it returned
   0 on a correctly-read long phrase — do not gate on it.
8. Dedup: typed (name, position) suppressed for 3 s within 120 px.
   In live play a typed word disappears, so re-detection later = missed
   keystrokes or a genuinely new spawn; retyping is correct recovery.
9. Typing: letters-only lowercase, pydirectinput, 10–30 ms jitter per key.

## Data sources

- `vocab.txt`: generated from **odota/dotaconstants** (GitHub, raw
  fetch of build/heroes.json, items.json, abilities.json → localized
  names, recipes and tooltip fragments filtered out) + a custom-phrases
  section at the bottom for manual additions. Includes patch 7.41 items
  (Shawl, Chasm Stone, Splintmail, Wizard Hat, Consecrated Wraps,
  Crella's Crozier, Essence Distiller, Specialist's Array, Hydra's
  Breath) and 7.41 neutral items. Cornucopia/Eternal Shroud removed from
  the game — keep removed items out of the vocab so they can't steal
  fuzzy matches.
- `phrases.txt`: generated from **mdiller/dotabase**
  (`dotabase/dotabase.db.sql`, sqlite dump of game VPK data; `responses`
  table, 91,182 rows). Filter applied: single-line length 8–48 chars,
  charset `[A-Za-z ',.!?-]`, deduped on letters-only uppercase key.
  Verified: "Bane of your existence." present verbatim (bane_spawn_02).
- **Liquipedia** hero response pages (e.g.
  https://liquipedia.net/dota2/Alchemist/Responses) hold the same data
  transcribed per hero and may include newer event lines. Liquipedia
  rate-limited (429) all fetches from the sandbox environment this
  prototype was built in; from a normal residential connection it should
  be scrapeable (respect their API guidelines: descriptive User-Agent,
  throttled requests, prefer api.php). Worth a scraper if the minigame
  uses Dark Carnival event dialogue not yet in dotabase.

## Known gaps / next steps

1. **Live validation**: confirm HSV range holds on raw framebuffer
   capture (clips went through recording compression). If detection is
   silent live, grab an mss screenshot during play and retune
   `HSV_LO/HSV_HI` from measured letter pixels.
2. **Input delivery**: pydirectinput should work for Dota's panorama UI;
   if keystrokes drop, fall back to raw SendInput via ctypes. Test
   whether the game drops keystrokes sent faster than human speed —
   determines minimum pacing.
3. **Resolution independence**: PANEL/HUD_BOXES are hard-coded for
   1920x1080. Either scale by resolution or auto-locate the panel (the
   bright center platform / panel border are easy anchors).
4. **Difficulty levels**: only the base level has been recorded. Higher
   levels = faster/longer/more obscure words; scan rate (currently
   ~3–5/sec limited by tesseract) may need optimizing — batch blobs into
   one tesseract call, or template-match against pre-rendered strings in
   the game font (the word set is closed, so template matching could
   replace OCR entirely and run at 60 fps).
5. **Phrase corpus currency**: dotabase may lag new event voicelines.
   The fallback tier covers gaps; it logs what it typed, so recurring
   unmatched phrases should be appended to vocab.txt's custom section.
6. **Multiplier protection**: consider a "safe mode" that skips
   fallback-tier typing entirely when the multiplier is high (score OCR
   is available at the HUD box if needed).
7. **Score plausibility**: Valve watches this minigame's leaderboard
   (see pause-exploit patch). Deliberate imperfection (occasional
   skipped word, human-ish WPM cap) is worth building in.

## Repo layout suggestion

```
automaton-attack-bot/
├── typebot.py        # current prototype (capture/detect/match/type)
├── vocab.txt         # core names (regenerate: scripts/build_vocab.py)
├── phrases.txt       # voiceline corpus (regenerate: scripts/build_phrases.py)
├── HANDOFF.md        # this file
└── clips/            # test recordings (offline regression via --video)
```

Suggested first Claude Code tasks: split build scripts out of the chat
history (vocab gen from dotaconstants, phrase gen from dotabase), add a
pytest that runs `--video` against the clips and asserts the expected
word sequences, then do the live-capture validation pass.

## Test commands

```
pip install opencv-python numpy pytesseract rapidfuzz mss pydirectinput
python typebot.py --video clips/clip1.mp4          # offline regression
python typebot.py --video clips/clip2.mp4 --debug  # per-blob OCR detail
python typebot.py                                  # live capture, dry run
python typebot.py --live                           # sends keystrokes
```
