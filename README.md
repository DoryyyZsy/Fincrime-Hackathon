# Same Person or Not? Track 2: sanctions screening

**Project team**

- Dory Zhu — dorothy.zhu26@imperial.ac.uk
- Jennifer Liu jennifer.liu26@imperial.ac.uk
- Yiding Wang — yiding.wang26@imperial.ac.uk

For every sanctions, PEP, enforcement or adverse-media hit raised against a customer, `system/run` decides
whether the hit **is the customer** (`SAME_ENTITY`), **someone else** (`DIFFERENT_ENTITY`), or **needs a
person** (`REVIEW`). It also gives a reason code, the resolved customer record, a 2–4 sentence reason, and the
IDs of the records that decided it. It is Python 3 standard library only, deterministic, and takes about
10 seconds on the development set.

> `system/README.md` is a copy of this file, because the brief asks for a README in `system/`.

## For inspectors: test it in one command

Requirements: Python 3.8 or later. Nothing to install. Put the data room's `dev/` folder at
`data/screening/dev/` (it is git-ignored and never committed).

```
python inspector.py
```

This runs every check and prints PASS or FAIL for each:

| Step | What it checks |
|---|---|
| 1. Environment | Python version, data files present, whether a model key is set |
| 2. Run | `system/run` on the folder: line count and time against the 30-minute limit |
| 3. Validate and score | the organisers' `validate.py`: format problems, practice-label score, unsafe clearances |
| 4. Reproducibility | two runs with different `PYTHONHASHSEED` are byte-identical, and a fresh run equals the committed `output/dev.jsonl` |
| 5. Rules of the brief | the engine imports no third-party module |
| 6. Tests | the screening test suite (pytest if installed, otherwise the stdlib `unittest` files) |

Default verification checks output format and reproducibility. Practice scoring is a separate, explicit evaluation command.

Other commands:

```
python inspector.py verify --data <folder>         # the same checks on any folder in the data-room format
python inspector.py run <folder> <output file>     # just run the system (same as system/run)
python inspector.py score <output file> --data <folder>
python inspector.py explain H-D861DA               # one hit: decision, reason, and the text of each cited record
python inspector.py check "Elvira Velasco Morillo" --dob 1968-03-05 --country Spain
                                                   # screen a new customer in the terminal, evidence by evidence
python inspector.py tests
python inspector.py demo                           # the web pages, at http://127.0.0.1:8765/
```

`verify` and `run` force rules only (`SCREENING_LLM=0`) unless you add `--with-llm`. The submitted command
itself is unchanged:

```
python3 system/run <data folder> <output file>      # e.g. system/run data/screening/dev output/dev.jsonl
```

## Results on the development set

| | |
|---|---|
| Output | 819 lines, 0 problems (`validate.py`) |
| Decisions | 439 `DIFFERENT_ENTITY`, 122 `SAME_ENTITY`, 258 `REVIEW` |
| Reasons | all 819 have 2 to 4 sentences |
| Reproducibility | byte-identical across runs and hash seeds |
| Time / cost | about 10 s with no model key, $0 |

No practice accuracy is reported for this checkpoint. The unseen set supplies the independent test.

## Architecture

```
alerts.jsonl ──► 1. CUSTOMER SIDE  (customer.py, data.py)
                    follow the link: account (joint holders) │ login │ email
                    no link: search customers by name + DOB, keep records that fit the screened details
                    merge records that are one person (same phone AND full DOB, or same document number)
                    person facts: names, latest APPROVED identity (KYC check or note), DOB (verified beats
                    declared), countries (records, ID document, countries named in support notes)
                         │
hits.jsonl ─────────┬──► 2. HIT SIDE  (hit.py)
search_results.jsonl┘    names + aliases; DOB from `dobs`, else ages in the profile notes, else web results
                         that describe the same kind of listing in a country tied to the hit; countries;
                         earliest dated report; date of death
                         ▼
                    3. COMPARE  (compare.py, POLICY.md principles 6–8)
                    clear only on a contradiction: type → death → too young → DOB → location → legal name
                    confirm only on agreement: name + DOB (+ country); company: name + country
                    otherwise REVIEW; safety net holds clears that rest on text we may have misread
                         ▼
                    4. HITNET  (neural.py)  score 50 features, check evidence, emit one row per hit
```

Every fact keeps the IDs of the records it came from (`Fact(value, sources)`), so each check knows its own
evidence.

| File | Role |
|---|---|
| `system/run` | entry point; one line per hit; an error on a hit becomes `REVIEW`, never a crash |
| `screening_ml/data.py` | loads the folder, builds the joins and the person-merging union-find |
| `screening_ml/normalize.py` | dates (6+ formats; partial dates stay partial), countries (names, codes, adjectives), names (principle 8) |
| `screening_ml/customer.py` | customer resolution and the person picture |
| `screening_ml/hit.py` | watchlist-side facts from fields, free text and search results |
| `screening_ml/compare.py` | the SOP checks, the safety net, reason codes and evidence |
| `screening_ml/reasoning.py` | the `reason` text |
| `screening_ml/pipeline.py`, `llm_extract.py` | run order, and the optional model extractor |
| `inspector.py` | the test command above |

## How the customer is found

1. **Follow the alert's link**: `account_ref` (a joint account gives several holders), `login_handle` or `email`.
2. **No link**: take customers sharing a name part (or the screened full DOB) and keep those that fit what was
   screened. Names may differ by nicknames, typos and variants; the DOB must be equal in every part both give;
   the country must be the residence or nationality; the city must be in the city or address. The same filter
   picks between joint holders.
3. **Merge records that are one person**: same phone **and** full DOB, or the same document number on approved
   checks. A shared phone or address alone does not merge them, because families share those.
4. **Identity**: the most recent **approved** KYC check, or an onboarding note saying the check passed, gives
   the legal name, DOB and document country. Rejected, expired or unclear checks are not evidence. The DOB is
   the date next to a birth cue (`DOB`, `born`, …). A discarded note never leaves the declared DOB free to
   clear the hit.
5. **Several people fit**: decide for each one. If they agree, give that decision; otherwise `REVIEW`
   `AMBIGUOUS_CUSTOMER_RECORD`. Either way `customer_record_id` is `AMBIGUOUS`. If nobody fits, compare with
   the screened details and report `null`.

## How the listed person is found

- **Names**: name, best matching name, first/middle/last and every alias, compared under principle 8:
  - case, accents, brackets and punctuation ignored; initials and placeholders dropped
  - `name_variants.json` groups (Bob = Robert); joined forms (Abdulsalam = Abdul Salam)
  - 1 edit allowed for parts of 4–6 letters, 2 for longer; company legal forms dropped
  - names agree when two parts match or every part of the shorter name matches
- **Date of birth**: the `dobs` field; else age wording in the first paragraph of the profile notes ("aged 30
  in November 1998", "he was 24", "(41)", "At 52,"), each giving two possible years; else web results for the
  name that describe the same kind of listing in a country tied to the hit. Most results are namesakes and fail
  that test.
- **Ambiguous slash dates** on a hit (`05/06/1980`) keep both readings. A clear needs every reading to
  contradict.
- **Countries**: countries, nationality, citizenship, the last part of each address, and the place of birth
  ("International" and "Europe" are not countries).
- **Time**: the earliest "According to …; Month D, YYYY:" report, and any date of death.

## Safety net: never clear on evidence we may have misread

A wrong clear costs 60 points; an unnecessary review costs 2–3. We measured what each evidence source
contributes by switching it off and rebuilding every case. Without the residence notes, the system makes
3 unsafe clears on the practice labels; without the free-text KYC notes, 2. Those are exactly the notes the
unseen set rewords.

So a clear is held for review, with the note cited and the reason saying why, when:
- it is a **country** clear but one of the customer's notes talks about where they lived and no country could
  be read from it;
- it is a **legal-name** clear whose name came from a free-text note that shares nothing with the customer's
  records, or from a note that was only partly read;
- it is a **DOB** clear on a declared DOB while an identity note could not be fully read.

Stress test: we rewrote all 3,067 residence notes in a copy of dev so that no country could be read.

| | Unsafe clears on practice labels | Practice decision score | Dev hits newly cleared |
|---|---|---|---|
| Safety net off | 3 | 0.0 | 41 |
| Safety net on | 0 | 97.2 | 0 |

Writing these tests also found a real bug: "Cote D'Ivoire" was never recognised in free text, which had
cleared a dev hit (H-BEDA04) whose customer lived there. It is now `REVIEW`.

## How evidence and reasons are represented

- **Evidence** cites only the records of the check that decided the hit: for `DOB_MISMATCH`, only the DOB's
  source; not everything we looked at.
- **The reason** (`reasoning.py`) always has the same three parts:
  1. who the customer is and how the alert led there ("resolves to customer CUS-804632 through account
     ACC-A82365 (identity verified in NOTE-8548C1)"), or that nobody fits, or that several people do;
  2. the decisive facts, with their records;
  3. what they mean: for a clear, that the names may agree but one contradiction is enough; for a match, that
     nothing contradicts it; for a review, exactly what is missing.

## Keeping results the same run to run

The decision is pure rules: no randomness and no network unless a model key is set. Lists are sorted or kept
in file order, and the output follows `hits.jsonl` order. Without a key, output is byte-identical across runs
and `PYTHONHASHSEED` values (`inspector.py` step 4).

## Model and prompts

HitNet proposes the decision. Evidence checks can refer it to review. An **optional** extractor (`screening_ml/llm_extract.py`) reads
free text first when `OPENROUTER_API_KEY` is set, and only through OpenRouter (stdlib `urllib`).
`SCREENING_LLM=0` turns it off.

- **Model and settings**: pinned `google/gemini-2.5-flash`, temperature 0, no thinking tokens, `max_tokens`
  2500.
- **Prompts**: three fixed prompts:
  - `PROMPT_NOTE`: a KYC or support note → outcome, a stated DOB, countries
  - `PROMPT_PROFILE`: a hit profile's first paragraph → birth years, places
  - `PROMPT_SEARCH`: a web snippet → this person or a namesake, birth years, countries
- **Quote check**: every fact must quote the source text exactly, and a birth year must follow from its quote.
- **Merging with the regex results**:
  - countries are a union, which can only remove a location clear
  - a DOB is used when the model and the regex agree, or when one of them is unknown
  - a model-only date can clear a hit only because its quote was checked; any other new clear is refused
- **Failures**: no key, a timeout, a bad status or invalid JSON leave the regex result standing.
- **Cost and time**: on dev, about 2,600 texts in about 320 batched calls, roughly $0.25. A timed-out batch
  falls back to the regex.

This environment had no key, so `output/dev.jsonl` is the rules-only output.

## Three cases

**A clear mismatch: H-D861DA, Michael Earl Tilley (adverse media, US).** The joint account resolves to
CUS-804632. Onfido verified DOB 1979-11-08. The hit's profile notes say he was 46 in January 2023, so he was
born in 1976 or 1977. That contradiction is enough.
```json
{"decision": "DIFFERENT_ENTITY", "reason_code": "DOB_MISMATCH", "customer_record_id": "CUS-804632",
 "reason": "The alert resolves to customer CUS-804632 through account ACC-A82365 (identity verified in NOTE-8548C1). The customer's date of birth is 1979-11-08 (NOTE-8548C1) but the hit's profile notes put its date of birth as 1976 or 1977. The names agree (Tilley, Michael Earl), but this contradiction is enough under the policy to show the hit is a different person who shares the name.",
 "evidence": ["NOTE-8548C1"]}
```

**A match: H-582410, Elvira Velasco Morillo (PEP, Spain).** Veriff verified a Spanish passport with DOB
1968-03-05. The hit is a Spanish MP aged 30 in November 1998, so born in 1967 or 1968. Name, DOB and country
agree, and nothing contradicts them.
```json
{"decision": "SAME_ENTITY", "reason_code": "NAME_DOB_LOCATION_MATCH", "customer_record_id": "CUS-377745",
 "reason": "The alert resolves to customer CUS-377745 through account ACC-C4CA34 (identity verified in NOTE-14E25A). The customer's date of birth 1968-03-05 (NOTE-14E25A) agrees with the hit's 1967 or 1968. Both the customer and the hit are linked to Spain. The customer's name agrees with the hit's name Elvira Velasco Morillo.",
 "evidence": ["CUS-377745", "NOTE-14E25A"]}
```

**Left for review: H-FF3096, Abdul Rahman Yasin (sanctions, US).** The customer is Australian, but a support
note (NOTE-10AEAB) says they lived in the US from 2013 to 2023, so the countries overlap. The hit gives no DOB,
so nothing contradicts it and nothing confirms it.
```json
{"decision": "REVIEW", "reason_code": "INSUFFICIENT_DATA", "customer_record_id": "CUS-459031",
 "reason": "The alert resolves to customer CUS-459031 through the alert's email address (identity verified in NOTE-588201). Both the customer and the hit are linked to United States. The customer's name agrees with the hit's name Abdul Rahman Yasin. Nothing in the customer's records contradicts the hit, and the date of birth cannot be compared with it, so the hit can be neither cleared nor confirmed.",
 "evidence": ["NOTE-10AEAB"]}
```

## Demo pages

```
python inspector.py demo          # or: python -m screening_ml.demo_server; then open http://127.0.0.1:8765/
```

The server is standard library only and binds to 127.0.0.1. The pages show customer data, so never expose them.
The start page is the screening review; the staff and management pages have buttons to switch between all three.

- **Screening review** (`/review.html`, the start page): one page, organised by alert. Tabs show how many alerts
  need the MLRO, need an analyst, are in the 5% spot-check, or are done. Each hit opens with a one-line plain
  summary ("Different: customer born 1973-12-26, listed person born 1992 or 1993.") and a side-by-side table of
  name, name on ID, date of birth and country, each marked Same / Different / Unknown. Open hits also show what
  to check. The buttons depend on where the hit is:
  - an analyst can clear it, confirm it and send it to the MLRO, or ask for more information;
  - the MLRO can confirm and act, clear it, or send it back.

  The reason saved with each action is written by the system from the facts and POLICY.md, and an action
  against what the data supports is marked OVERRIDE. The reviewer only types their name. Actions are appended,
  UTC-stamped, to `data/screening/review/audit_log.jsonl`, which is git-ignored. `?alert=A-xxxxxx` opens one
  alert.
- **Staff check** (`/staff.html`): type a customer's details, or pick a demo customer, and press *Run check*.
  Each matching watchlist profile appears as a card. Its evidence animates in one item at a time ("Name
  matched", "DOB matched", "Country doesn't match", "Too young at the time", "Held for safety"), then an
  ALERT / REVIEW / CLEAR stamp lands, then the overall verdict. *Why?* shows the full reason.
- **Management review** (`/management.html`): headline numbers and a filterable queue of all hits. For each
  hit it shows the trail to the customer, the customer and the hit side by side, every policy check, and the
  cited records. **Edit hit list** lets you mark hits Cleared / Same entity / Review, remove and restore hits,
  and export the edited list.

`python -m screening_ml.analyst_view` writes the management page as a single offline file,
`artifacts/analyst_view.html`, which is git-ignored.

## Tests

- `python inspector.py tests`: name rules, reworded residence, KYC and age sentences, the safety net, the
  output format, extraction edge cases, and the model extractor with a fake transport (no network).
- `tests/perturb_screening.py <folder>` rebuilds the folder four ways and scores each copy:
  - KYC dates shifted
  - residence verbs reworded
  - age wording reworded
  - KYC notes rewritten as failed

  All four copies have 0 unsafe and 0 new clearances.

## Selected neural network

HitNet has hidden layers of 128 and 64 units, a three-way decision head and a reason-code head.
HitNet was fitted from scratch on 407 rule-labeled pairs and selected by validation loss on
72 separate rule-labeled pairs. Membership fingerprints exclude all 50 practice hits and the
290 additional hits linked by alert, resolved customer or watchlist name. The training process
and submission runner work without the practice-answer file. The original evidence rules were
developed with knowledge of practice cases, so the resulting model still has historical indirect
exposure. Validation measures agreement with those rules.

The runtime loads the trained weights from `screening_ml/hitnet_weights.json` and runs the
forward pass with Python's standard library. A clearance requires a model score of at least
0.90 and a supported contradiction. Unsupported confirmations and uncertain cases go to review.
A second forward pass neutralizes the upstream match score before authorizing a clearance,
adapting the field-removal idea from [Ditto](https://github.com/megagonlabs/ditto) to inference.
Inspired by [Watchman](https://github.com/moov-io/watchman), a multi-part Russian Cyrillic name
whose Latin spelling agrees with a candidate can send a name- or location-only clearance to review.
Names, dates and record identifiers supply the evidence for the explanation.

Offline training uses PyTorch. To package a compatible checkpoint, run
`python -m screening_ml.export_hitnet CHECKPOINT screening_ml/hitnet_weights.json`, rerun
`system/run`, and validate the resulting decisions against the practice labels.

## Earlier prototype

`app/streamlit_app.py` and `src/` are our first prototype, built before the track data was released. It is a
Streamlit demo on a public OpenSanctions export with 5,190 synthetic customers, with fuzzy matching, Splink
linkage and optional Tavily/Grok research. Run it with `pip install -r requirements.txt` and `make app`. It is
not part of the Track 2 submission.

## Known limits

- **Ages are read with regexes.** An age with no number next to age wording is missed. The safety net turns an
  unread residence or identity note into `REVIEW`, but a missed DOB can still lift the block on a location
  clear when the name and DOB would have agreed.
- **Undated KYC birth dates**: a KYC note with no birth cue ("ID dated 1952-01-21") gives no DOB.
- **Over-counted countries**: any country named in a support note counts for the customer. That can suppress a
  location clear, or support a company's name-and-country match.
- **Single contradictions**: each clear rests on one contradiction. The parsers are conservative, so unknown
  parts never count as a mismatch.
- **Data handling**: the data is for event participants only. `data/screening/dev/`, `artifacts/` and the audit
  log are git-ignored.
