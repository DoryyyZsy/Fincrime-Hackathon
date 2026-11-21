# Same Person or Not? Track 2 system

For every sanctions, PEP, enforcement or adverse-media hit, `system/run` decides whether the hit is the customer
(`SAME_ENTITY`), someone else (`DIFFERENT_ENTITY`), or needs a person (`REVIEW`), and says why, citing the
records that decided it.

```
python3 system/run <data folder> <output file>      # e.g. system/run data/screening/dev output/dev.jsonl
```

Python 3 standard library only. No network, no language model, no API key. It reads only the data folder.

## Architecture

```
                      ┌──────────────── data folder (JSON Lines) ────────────────┐
alerts.jsonl ─────────┤                                                           │
                      ▼                                                           │
          1. CUSTOMER SIDE  (screening_ml/customer.py)                            │
          link? account_ref → accounts (joint holders) │ login → accounts │ email │
          no link → search customers by name parts + DOB, keep records that fit  │
          screened name / DOB / country / city                                    │
                      ▼                                                           │
          one person, one picture (data.py): merge records with the same phone   │
          AND full DOB, or the same document number on approved KYC checks        │
                      ▼                                                           │
          person facts: names, latest APPROVED identity (KYC check or KYC note),  │
          DOB beside a birth cue (verified beats declared), countries (records,   │
          ID document, any country a support note names) — each fact keeps the    │
          record IDs it came from                                                  │
                      │                                                           │
hits.jsonl ───────────┼──► 2. HIT SIDE  (screening_ml/hit.py)                     │
search_results.jsonl ─┘    names + aliases; DOB from `dobs` → age sentences in    │
                           the profile notes → matching web results (same kind of │
                           listing + a country tied to the hit); countries;       │
                           earliest dated report; date of death                   │
                      ▼                                                           │
          3. COMPARE  (screening_ml/compare.py, POLICY.md principles 6-8)         │
          clear only on a contradiction:  type → death → too young → DOB →        │
                                          location → verified legal name          │
          confirm only on agreement:      name + DOB (+ country) │ company: name  │
                                          + country                               │
          otherwise REVIEW; several people fit with different outcomes → REVIEW   │
          AMBIGUOUS_CUSTOMER_RECORD                                               │
                      ▼                                                           │
          4. OUTPUT  one JSON line per hit: decision, reason_code,                │
             customer_record_id, reason (this hit's own facts), evidence (IDs)   ─┘
```

| File | Role |
|---|---|
| `system/run` | entry point; one line per hit; any error on a hit becomes `REVIEW` instead of a crash |
| `screening_ml/data.py` | loads the folder, builds joins and the person-merging union-find |
| `screening_ml/normalize.py` | dates (6+ formats, partial dates stay partial), countries (names, codes, adjectives), names (POLICY.md principle 8) |
| `screening_ml/customer.py` | customer resolution and the person picture |
| `screening_ml/hit.py` | watchlist-side facts from fields, free text and search results |
| `screening_ml/compare.py` | the SOP checks, reason codes, evidence and reason sentences |

## How the customer is found

1. Follow the alert's link: `account_ref` (a joint account gives several holders), `login_handle` or `email`.
2. Without a link, collect customers sharing a name part (or the screened full DOB) and keep those that fit
   what was screened: name agreement (nicknames, typos and variants allowed), DOB equal in every part both give,
   screened country as residence or nationality, screened city in city or address. The same filter picks
   between joint holders.
3. Merge records that are one person: same phone **and** same full DOB, or the same document number on approved
   checks, transitively. A shared phone or address alone does not merge (families share them).
4. Identity: the most recent **approved** KYC check, or an onboarding note that says the check passed, gives
   the legal name, DOB and document country. A note is not evidence when it says the check failed
   (`rejected`, `did not pass`, `not verified`, `resubmit`, `pending`, …) or never says that it passed.
   The DOB is the date next to a birth cue (`DOB`, `born`, `date of birth`), or a date that is the whole
   parenthesis after the name — not "the first full date that is not `created_at`". A date after the
   screening date, or an age outside 0–110, is ignored; two birth dates that disagree become unknown.
   When a note is discarded, the declared record DOB is not used to clear the hit, and a country mismatch
   is not a clearance either: the document country on that note is unknown. Countries come from every
   record's residence and nationality, the ID document, and every country named in a non-KYC note. A
   keyword list (`lived`, `work`, …) is not required, because a reworded sentence (`stayed`, `job`) would
   otherwise drop the country and clear the hit.
5. More than one person fits: decide for each; same outcome → that outcome, otherwise `REVIEW`
   `AMBIGUOUS_CUSTOMER_RECORD`; `customer_record_id` is `AMBIGUOUS` either way. No fit → compare with the screened
   details and report `null`.

## How the listed person is found

- **Names**: name, best matching name, first/middle/last, every alias. Comparison follows principle 8: case,
  accents, brackets and punctuation ignored; initials and placeholders are not parts; `name_variants.json`
  groups; joined forms (Abdulsalam = Abdul Salam); 1 edit for 4-6 letters, 2 for longer; company legal forms
  dropped; agreement = two parts match or every part of the shorter name.
- **Date of birth**: the `dobs` field; else the first paragraph of the profile notes ("aged 30 in November
  1998", "age 30", "30 year old", "30 years of age", "he was 24", "at age 25 when": two possible years
  each); else web results for the name that describe the same kind of listing ("a politically exposed
  person", …) in a country tied to the hit. Most results are namesakes and are rejected by that test.
  A number counts only next to age wording, so a sentence of "28 years" or a fine does not become a birth year.
- **Slash dates on the hit**: `6/20/1955` is 20 June, because the other reading is not a day. `05/06/1980`
  is both 5 June and 6 May. A clearance needs every reading to contradict, and one reading that agrees
  is enough to stop it. Customer records and KYC fields stay day-first: every unambiguous slash date in
  those fields has the day first.
- **Countries**: countries, nationality, citizenship, the last part of each address and the place of birth
  ("International" and "Europe" are not countries).
- **Time**: the earliest "According to …; Month D, YYYY:" report and any date of death.

## How evidence is represented

Every fact carries the record IDs it came from (`Fact(value, sources)`), so each check knows its own evidence:
a DOB from a KYC note cites the `NOTE-…`, a country from an ID document cites the `KYC-…`, a hit DOB found in a
search result cites the `SR-…`. The output cites only the evidence of the check that decided the hit (for
example only the DOB source for `DOB_MISMATCH`), not everything looked at. The reason sentence is generated
from the same facts, so it always names this hit's own dates, countries and names, and says what is missing
when the answer is `REVIEW`.

## Keeping results the same run to run

The system is pure rules: no randomness, no model calls, no network. Candidate lists, country lists and
evidence are sorted or kept in file order, and the output is written in `hits.jsonl` order. Two runs of the
dev set with `PYTHONHASHSEED=0` and `PYTHONHASHSEED=1` were byte-identical.

## Model and prompts

The decision is always the rule engine. An optional extractor in `screening_ml/llm_extract.py` may read
free text first, and only through OpenRouter (`POST https://openrouter.ai/api/v1/chat/completions`, stdlib
`urllib`, no SDK). It runs when `OPENROUTER_API_KEY` is set. `SCREENING_LLM=0` forces the regex path even
if a key is present. No key, a timeout, a bad status, or JSON that fails the schema: that text is unknown
and the regex stands. Nothing is guessed.

Pinned model: `google/gemini-2.5-flash`. `temperature` is 0, `reasoning.max_tokens` is 0 (no thinking
tokens), and `max_tokens` is 2500. Four fixed prompts, one per text kind:

| Prompt | Text | What it must return |
|---|---|---|
| `PROMPT_NOTE` | KYC note | `passed` / `rejected` / `unclear`, a date of birth only when the note states one, countries and cities |
| `PROMPT_NOTE` | support note | the same shape; outcome is ignored and the note is not an identity check |
| `PROMPT_PROFILE` | first paragraph of a hit profile | birth years (aged 45 in March 2019 → 1973 and 1974) and places |
| `PROMPT_SEARCH` | web snippet | whether it is this listed person or a namesake (`high` / `low`), plus birth years and countries |

Every fact carries a quote. The quote has to be a substring of the source (whitespace aside), and a birth
year has to be one the quote can support. A country is added only when the quote contains that country.

How a model fact meets the regex:

- Countries are a union. An extra country can only remove a `LOCATION_MISMATCH`.
- A date of birth is used when the regex and the model agree, or when one of them is unknown. If they
  disagree, the date is unknown. A date the model alone supplies can clear (`DOB_MISMATCH` or
  `TEMPORAL_IMPOSSIBILITY`) only because its quote was checked. Any other new clearance is refused and the
  regex decision is kept.
- A KYC note is evidence only when the regex says it passed and the model does not say `rejected` or
  `unclear`. The model cannot revive a note the regex rejected. A discarded note does not leave the
  declared date of birth, or the remaining countries, free to clear the hit.
- A search snippet contributes a birth date only when the model attributes it to this person with `high`
  confidence, or when the model was not asked and the regex attribution already passed. A namesake call
  drops the snippet and blocks a location clearance from taking its place.

Calls are batched 8 texts at a time and run 8 at a time. Responses are cached for the run under
SHA-256 of `(model, prompt, item text)`. The dev folder is about 2,600 texts and about 320 calls. At the
posted rate of $0.30 / $2.50 per million tokens that is about $0.25 if every text is sent. A healthy run
on ~550 hits should stay well under 30 minutes; a batch that times out (12s, one retry) falls back to the
regex for those texts only.

Within one run the cache, temperature 0, and hit order make the file deterministic given the responses.
Across runs the provider can still differ. A difference can turn a clearance into a review; it cannot
open a clearance the regex did not already have unless the new birth date is quoted from the text. With
no key the output is the regex file, byte-identical across `PYTHONHASHSEED` values. This environment had
no `OPENROUTER_API_KEY`. `SCREENING_LLM_MOCK=1` is an offline stand-in that agrees with the regex, used
by the tests, and on the dev set it reproduces the regex file exactly.

We also built a small neural network on top of these features (`screening_ml/`, PyTorch) as an experiment.
It is not part of this system. `system/run` does not import it. The network's `rule_guard` can only turn a
rule clearance into `REVIEW`, it is seed-unstable, and it needs PyTorch, so the submitted path is the rule
engine (see `screening_ml/README.md`).

## Time and cost

About 4 seconds for the dev set (440 alerts, 819 hits, 72k customers) in this environment, most of it loading
the files. No API calls, so cost is $0.

## Results on the development set

`validate.py --score practice_labels.jsonl`: 819 lines, 0 problems; practice hits 50/50 right decisions,
0 unsafe clearances, decision score 100, reason code 100%, customer record 100%. Dev decisions: 463
`DIFFERENT_ENTITY`, 134 `SAME_ENTITY`, 222 `REVIEW`. One of those clearances, H-3DB30F, is `DOB_MISMATCH`
rather than `LOCATION_MISMATCH`: the profile says he "reached 37 years of age" in November 1981.
Caveat: we looked at the practice labels while building, so this score is optimistic; the unseen set is the real test.

## Three cases

**A clear mismatch: H-D861DA, Michael Earl Tilley (adverse media, US).** The alert's account is joint; the
screened name and DOB pick CUS-804632. Onfido verified a US driving licence with DOB 1979-11-08 (NOTE-8548C1).
The hit has no `dobs`, but its profile notes say he was 46 when questioned in January 2023: born 1976 or 1977.
1979 is more than a year from both, so every birth date the hit allows is contradicted.
```json
{"decision": "DIFFERENT_ENTITY", "reason_code": "DOB_MISMATCH", "customer_record_id": "CUS-804632",
 "reason": "The customer's date of birth is 1979-11-08 (NOTE-8548C1) but the hit's profile notes put its birth date at 1976, 1977.",
 "evidence": ["NOTE-8548C1"]}
```

**A match: H-582410, Elvira Velasco Morillo (PEP, Spain).** Joint account; the screened details pick
CUS-377745. Veriff verified a Spanish passport for ELVIRA VELASCO MORILLO, DOB 1968-03-05 (NOTE-14E25A). The hit
is a Spanish member of parliament described as aged 30 in November 1998 (born 1967 or 1968). Name, DOB and
country agree and nothing contradicts them.
```json
{"decision": "SAME_ENTITY", "reason_code": "NAME_DOB_LOCATION_MATCH", "customer_record_id": "CUS-377745",
 "reason": "The customer's date of birth 1968-03-05 (NOTE-14E25A) agrees with the hit's 1967 or 1968. Both the customer and the hit are linked to Spain. The customer's name agrees with the hit's name Elvira Velasco Morillo.",
 "evidence": ["CUS-377745", "NOTE-14E25A"]}
```

**Left for review: H-FF3096, Abdul Rahman Yasin (Israeli terrorism sanctions list, US).** The email link gives
CUS-459031, an Australian born 1965. It looks like a country mismatch, but a support note (NOTE-10AEAB) says
the customer lived and worked in the United States from 2013 to 2023, so the countries overlap. The hit gives
no date of birth, so nothing contradicts it and nothing confirms it.
```json
{"decision": "REVIEW", "reason_code": "INSUFFICIENT_DATA", "customer_record_id": "CUS-459031",
 "reason": "Both the customer and the hit are linked to United States. The customer's name agrees with the hit's name Abdul Rahman Yasin. Nothing in the customer's records contradicts the hit, and the date of birth cannot be compared with it, so the hit can be neither cleared nor confirmed.",
 "evidence": ["NOTE-10AEAB"]}
```

## Checks

`tests/test_screening_extract.py` is stdlib `unittest` (slash dates, KYC cues, rejection phrases, age paraphrases).
`tests/test_llm_extract.py` checks quotes, disagreement, the clearance guard, the cache, and a rejected
model outcome, with a fake transport and no network.
`tests/perturb_screening.py <data folder>` rebuilds the folder four ways and scores each with `validate.py`:
KYC-note `created_at` shifted by two days, residence verbs paraphrased (`stayed`, `job`, …), age wording
paraphrased, and KYC notes rewritten as "not verified / did not pass / resubmit" with the birth year moved.
On the dev folder the four perturbed runs have 0 unsafe clearances and 0 decisions that flip to
`DIFFERENT_ENTITY`. The rejected-note run reviews more hits (practice score 67.6) because those notes were
real passes and dropping them removes DOB and document-country evidence. The same script then reruns every
copy with `SCREENING_LLM_MOCK=1` and checks that the stand-in opens no new clearance against the regex file.

## Known limits

- Ages are still regexes. A sentence that states an age without putting a number next to "age", "year old"
  or "born" is missed. Missing a DOB removes the rule that blocks a location clearance when name and DOB
  agree, so a badly paraphrased age can still clear a hit the DOB would have saved. The paraphrases in
  `tests/perturb_screening.py` do not.
- A KYC note that gives a date with no birth cue ("national ID dated 1952-01-21") does not yield a DOB.
  Seven development-set notes are in that shape. The declared record DOB is used instead.
- Slash dates on customer records and KYC notes are day-first, because every unambiguous slash date in those
  fields is day-first and the screened date agrees. A hit profile is not written that way; an ambiguous hit
  date keeps both readings. A customer file that switched to month-first would be misread.
- Any country named in a support note is tied to the customer. A note about a transfer destination can add
  a country and suppress a location clearance. That is the safe direction. For a company, the same extra
  country can support a name-and-country match.
- A KYC note we cannot read as a pass is dropped entirely. If that note was the real identity check, the hit
  goes to review instead of being confirmed, and it is not cleared from the declared DOB or from the countries
  that remain. A genuinely failed note plus a true country mismatch is therefore reviewed, not cleared.
