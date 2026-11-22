# Same Person or Not? Track 2: sanctions screening

Authors: Dory Zhu, Jennifer Liu, Yiding Wang

For every sanctions, PEP, enforcement or adverse-media hit raised against a customer, `system/run` decides
whether the hit **is the customer** (`SAME_ENTITY`), **someone else** (`DIFFERENT_ENTITY`), or **needs a
person** (`REVIEW`). It also gives a reason code, the resolved customer record, a 3–4 sentence reason, and the
IDs of the records that decided it. It is Python 3 standard library only, deterministic, and takes about
10 seconds on the development set. A Fellegi–Sunter match score ranks the review queue and does not change those decisions.

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

Expected on the dev set: **11/11 checks passed**. The validator reports 0 problems and 50/50 practice hits
with 0 unsafe clearances; the test suite reports 115 passed.

Other commands:

```
python inspector.py verify --data <folder>         # the same checks on any folder in the data-room format
python inspector.py run <folder> <output file>     # just run the system (same as system/run)
python inspector.py score <output file> --data <folder>
python inspector.py explain H-D861DA               # one hit: decision, reason, and the text of each cited record
python inspector.py check "Elvira Velasco Morillo" --dob 1968-03-05 --country Spain
                                                   # screen a new customer in the terminal, evidence by evidence
python inspector.py tests
python -m unittest tests/test_linkage.py           # the match score, with no dev data
python -m screening_ml.eval_linkage --data data/screening/dev
                                                   # how the score ranks the practice labels
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
| Practice labels | 50/50 right decisions, **0 unsafe clearances**, decision score 100, reason code 100%, customer record 100% |
| Decisions | 466 `DIFFERENT_ENTITY` (57% cleared automatically), 135 `SAME_ENTITY`, 218 `REVIEW` |
| Reasons | all 819 have 3 or 4 sentences |
| Reproducibility | byte-identical across runs and hash seeds |
| Time / cost | about 10 s with no model key, $0 |

Caveat: we looked at the practice labels while building, so this score is optimistic. The unseen set is the
real test.

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
                    4. MATCH SCORE  (linkage.py)
                    Fellegi–Sunter probability the hit is the customer, fitted by EM on the unlabelled hits
                    ranks the REVIEW queue; the rules keep the decision
                         ▼
                    5. OUTPUT  (pipeline.py, reasoning.py)  one JSON line per hit
                    the probability is written beside it, in `<output>.scores.jsonl`
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
| `screening_ml/linkage.py` | Fellegi–Sunter match probability; ranks reviews and does not decide |
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
and `PYTHONHASHSEED` values (`inspector.py` step 4). The match score uses fixed start values, a fixed
iteration cap and a fixed summation order, so the sidecar is byte-identical across those runs too. It is not
part of the decision line unless `SCREENING_MATCH_SCORE=1`.

## Probabilistic match score

The rules still decide every hit. `screening_ml/linkage.py` estimates the probability that the customer and
the hit are the same person, in the Fellegi–Sunter way, and uses it only to rank and annotate the review queue.

Each comparison is a pattern of levels:

- **Name**: exact, strong, partial, weak, disagree, or missing. The tier uses Jaro–Winkler, the share of name
  parts that match, a Soundex code, and whether the agreeing name is an alias of the listed person.
- **Rarity**: rare or common, from how often the matched part appears in the customer file. A part seen on at
  most `max(3, n/500)` customers is rare. A name that does not agree is its own level, so rarity is not
  applied to a mismatch.
- **Date of birth**: full date, year and month, year only, within one year (the two years an age in a note can
  mean), conflict, or missing. One agreeing reading wins. Every reading has to conflict before the level is a
  conflict, which covers an ambiguous slash date. A missing part of a date stays unknown.
- **Country** and **entity type**. The files do not record gender, so type is person versus company.
- **Document**: exact when an approved document number is quoted on the hit, otherwise missing. Absence is
  unknown, not a contradiction.
- **Time**: impossible when the rules already found the customer could not be the listed person (too young, or
  dead before screening).

For each level, m is how often it appears when the two records are the same entity, and u is how often it
appears when they are not. Expectation-maximisation estimates m, u and the match prior from the unlabelled
hits in the folder being scored. The 50 practice labels are not used. The start values, a cap of 80
iterations, and the order of the sums are fixed. A step stops when the largest parameter change is under
`1e-10`, or when the log-likelihood has stopped moving and the parameters are already still at the printed
precision. If one level flips meaning (a one-year date gap turning up more often for different people, for
example), only that level is put back on the boundary and the other levels keep the values just estimated.
Putting the whole field back to its start values made the fit cycle, so it used all 20 steps and never
settled. The same folder therefore gives the same scores under any `PYTHONHASHSEED`. The module is Python
standard library only.

The score does not move a decision. `guarded_decision` refuses a `DIFFERENT_ENTITY` the rules did not already
return, and refuses a `SAME_ENTITY` the rules did not already return. An unsafe clearance costs 60 and a
review costs 2 or 3, so clearing would have to be under 5% likely to be unsafe before it was cheaper than a
review. The score is unsupervised, and a review hit is one the rules could not contradict, so promotion stays
off (`PROMOTE = False`). Turning that flag on would still leave the two refusals in place. The review queue
is ordered by the probability, highest chance of being the same person first.

`validate.py` checks the decision fields and ignores anything else, but an extra field would make a fresh run
differ from the committed `output/dev.jsonl`. The probability is written next to the output:

- `<output>.scores.jsonl` — one line per hit: probability, log2 weight, comparison levels, field weights, and
  the review rank
- `<output>.scores.meta.json` — the fitted m and u

`SCREENING_MATCH_SCORE=1` also copies `match_probability` onto each decision line. The screening review page
and the management page show the probability. Filtering the management page to Review sorts that queue by it.

```
python -m screening_ml.eval_linkage --data data/screening/dev
python -m unittest tests/test_linkage.py
```

The evaluation script needs the dev folder, which is git-ignored. It reports how the probability separates the
practice labels, and what a cost threshold would have cleared. That table is not applied.

## Model and prompts

The rule engine always makes the decision. An **optional** extractor (`screening_ml/llm_extract.py`) reads
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
  output format, extraction edge cases, the model extractor with a fake transport (no network), and the
  Fellegi–Sunter score.
- `python -m unittest tests/test_linkage.py`: the score on its own. The fixtures are synthetic, so this does
  not need `data/screening/dev`.
- `python -m screening_ml.eval_linkage --data data/screening/dev`: the score against the practice labels, when
  that folder is present. It does not change decisions and it does not invent a result when the folder is missing.
- `tests/perturb_screening.py <folder>` rebuilds the folder four ways and scores each copy:
  - KYC dates shifted
  - residence verbs reworded
  - age wording reworded
  - KYC notes rewritten as failed

  All four copies have 0 unsafe and 0 new clearances.

## Experiment: a neural network (not submitted)

`screening_ml/` also contains a small PyTorch network. It is separate from the Fellegi–Sunter score above,
which is part of the submission. The network is a 2-layer MLP with a decision head and a
reason-code head, using 50 features, and runs on CUDA, Apple MPS or CPU. It is **not part of the submission**:
the brief allows only the standard library, and the network does not beat the rules.

- **Why it trains on rule-engine labels**: there are only 50 gold labels, with the smallest class at 10 and
  1 label per feature. So it learns from the rule engine's labels on all 819 hits ("silver" labels), and the
  gold labels are held out for testing.
- **Results**: on a leak-free split (hits sharing an alert, customer or watchlist name are kept on one side),
  it scores 97% on silver validation and 100% on gold. On its own, though, it made 1–2 unsafe clears that
  only the rule guard blocked.
- **How to run it**:
  - `pip install torch numpy pyyaml` (for CUDA, install torch with
    `--index-url https://download.pytorch.org/whl/cu126`)
  - `python -m screening_ml.check_data`, then `python -m screening_ml.train`, then
    `python -m screening_ml.predict`
  - settings are in `screening_ml/config.yaml`

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
- **The match score is unsupervised.** It orders the review queue. It is not a calibrated probability, and it
  does not clear or confirm a hit. On a checkout that has the dev folder, `python -m screening_ml.eval_linkage`
  shows how it ranks the 50 practice labels.
- **Data handling**: the data is for event participants only. `data/screening/dev/`, `artifacts/` and the audit
  log are git-ignored.
