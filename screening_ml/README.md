# screening_ml: hit adjudication with a small neural network

Decides, for every sanctions / PEP / adverse-media hit, whether it is the customer (`SAME_ENTITY`), someone
else (`DIFFERENT_ENTITY`), or needs a person (`REVIEW`). It reads the data room's JSON Lines folder
(`data/screening/dev`, or an unseen folder in the same format) and writes the submission file.

## Why there is a rule engine inside an ML program

`check_data.py` audits the folder before training. On the dev set it finds:

| Check | Value | Verdict |
|---|---|---|
| Hits to decide | 819 | |
| Gold labels (`practice_labels.jsonl`) | 50 (6%) | |
| Smallest decision class | 10 (REVIEW) | too few (want ≥ 100) |
| Labels per feature | 1 (50 features) | too few (want ≥ 10) |
| Reason codes seen | 7 of 11 | 4 never appear |

So a network cannot be trained on the gold labels alone. Instead:

1. **Silver labels.** `compare.py` applies POLICY.md exactly as written (principles 2-8) to every hit, giving
   a decision, reason code, evidence and reason sentence for all 819 hits. It matches all 50 gold labels.
2. **Network.** `HitNet` (MLP, two heads: decision + reason code) learns from silver labels on 50
   fact features (name agreement, DOB agreement/contradiction, country overlap, data coverage, link type…).
3. **Gold labels are held out** and used only to measure the result with the organisers' cost-based score.
4. **Fine-tuning.** When you have more human-checked labels, `finetune.py` adapts the network to them.

```
alerts ─► customer.py ─► person picture (records merged, latest approved KYC, countries)
hits   ─► hit.py      ─► hit facts (names, DOBs from field/notes/search, countries, dated reports)
                 └──► compare.py ─► SOP checks + evidence + reason  ─► silver label
                             └──► features.py ─► 50-d vector ─► HitNet ─► decision + reason code
                                                                  └─► predict.py (cost matrix, safety guards)
```

## Setup

Keep the virtual environment **outside OneDrive** (PyTorch is ~3 GB).

Windows with an NVIDIA GPU:
```powershell
python -m venv $HOME\.venvs\screening-ml
& $HOME\.venvs\screening-ml\Scripts\Activate.ps1
pip install torch --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements-ml.txt
```

macOS on Apple Silicon (uses MPS automatically):
```bash
python3 -m venv ~/.venvs/screening-ml && source ~/.venvs/screening-ml/bin/activate
pip install -r requirements-ml.txt
```

Put the data room's `dev/` folder at `data/screening/dev/` (git-ignored; never commit it).

## Run

```powershell
python -m screening_ml.check_data            # audit the data and the labels
python -m screening_ml.train                 # train on silver, validate on gold -> artifacts/model.pt
python -m screening_ml.predict --data data/screening/dev --out artifacts/dev_nn.jsonl
python data/screening/validate.py --data data/screening/dev --output artifacts/dev_nn.jsonl --score data/screening/dev/practice_labels.jsonl
python -m screening_ml.finetune --labels my_labels.jsonl   # optional
```

Fine-tuning on `practice_labels.jsonl` (the default) uses up the gold labels, so after that the gold score is no
longer an independent check; fine-tune on your own extra labels when you have them.

`--device cuda|mps|cpu` overrides the automatic choice. Feature building takes about 5-10 seconds and is cached in
`artifacts/cache/` (keyed by the data files and the code, so edits invalidate it). Training takes seconds.

## Where to tune

| What | Where |
|---|---|
| Network size, dropout | `config.yaml` → `model` |
| LR, epochs, batch, early stopping, class balancing | `config.yaml` → `train` |
| How hard unsafe clearances are punished in training | `train.cost_weight` (official cost matrix) |
| How confident the model must be to clear | `inference.clear_min_prob` |
| Allow the model to clear without an SOP contradiction | `inference.rule_guard: false` (risky: a wrong clear costs 60) |
| Allow SAME_ENTITY despite a contradiction | `inference.same_guard: false` |
| Leak-free vs per-hit split | `labels.group_split` |
| Feature ablation | `drop_features` |
| New features | append to `FEATURE_NAMES` and `featurize()` in `features.py`, then retrain |
| Better free-text extraction (ages, KYC notes) | `hit.extract_ages`, `customer._kyc_note_identity` (an LLM call can go here) |
| Fine-tuning on new labels | `config.yaml` → `finetune`, then `python -m screening_ml.finetune` |

## Results on the dev set

Seed 42, default config, CPU, leak-free grouped split (below). Network alone, before the guards:

| Set | n | Accuracy | Decision score* | Unsafe clears (FN) |
|---|---|---|---|---|
| Silver train | 407 | 97.8% | 96.6 | 0 |
| Silver validation (no shared alert/customer/watchlist name) | 72 | 97.2% | 47.9 | **1** |
| Gold practice labels (held out) | 50 | 100% | 100.0 | 0 |
| Rule teacher on gold | 50 | 100% | 100.0 | 0 |

\* the organisers' score: share of cost saved versus sending everything to REVIEW. One unsafe clear costs 60,
which is why one mistake in 72 drops validation to 47.9.

**Final output** (`artifacts/dev_nn.jsonl`, network + guards; the submission file `output/dev.jsonl` comes from `system/run`): `validate.py` reports 0 problems, 50/50 practice hits,
reason code 100%, customer record 100%. Against the rule engine on all 819 hits:

| Error type | Meaning here | Count |
|---|---|---|
| False negative (unsafe clear) | cleared a hit the rules keep open | 0 (the guard blocked the network's 1) |
| False positive | kept open a namesake the rules clear | 32 (all sent to REVIEW, none to SAME) |
| Missed SAME_ENTITY | rules confirm, output does not | 0 |

Two training runs give byte-identical output.

### Leakage checks (and what was fixed)

| Check | Finding | Status |
|---|---|---|
| Gold hits in training | none (indices disjoint) | ok |
| Same alert / customer / watchlist name across train, val and gold | first version: 44 alerts shared train/val, 20 train/gold; 31/115 val rows were exact feature copies of train rows, so val was inflated (93.9%) | fixed: `labels.group_split` keeps groups together and drops 290 hits linked to gold from train/val |
| Normalisation stats | computed on train rows only | ok |
| Practice labels used as features | no; only for scoring | ok |
| Features that encode the label | `dob_contra`, `country_disjoint`, `under_min_age`, `type_mismatch`, `deceased_before_screening` are the SOP checks themselves, so the net is distilling the rules | by design; ablation below |
| Tuning on gold | I fixed two rule bugs (accented search keys, accented age pattern) **after** seeing gold misses (48 to 50/50), so the teacher's gold score is optimistic | cannot be undone; the unseen set is the only clean test |
| Fine-tuning on practice labels | uses up the gold labels | documented in the Run section |
| Independent re-audit with wider keys (every candidate record, every alias, source URLs, identifiers) | no alert, customer or candidate record shared between train, val and gold. Shared keys are only generic list URLs (UN/UK sanctions lists, 419 cross-set pairs) and junk alias keys; the 4 URL pairs whose names "agreed" were N26 Bank vs Danske Bank, which agreed only on the word "Bank" | ok, no leak. The name bug behind it is fixed: company parts with digits ("N26", "3M") are now kept (`normalize.name_parts`, tests in `tests/test_screening_names.py`); no dev decision changed |
| Near-duplicates (standardised features) | median nearest-neighbour distance val→train 4.08 and gold→train 4.05, versus 2.31 train→train; the 2 exact copies are different people with the same fact pattern | ok |
| Gold labels in the code path | used only to exclude hits and to print scores; never in the loss, early stopping or prediction | ok |
| Choices made after looking at results | early stopping uses silver val (standard, slightly optimistic); `use_cost_matrix` off, `same_guard` on and `final_fit` off were chosen after seeing dev and gold results | gold is a development check, not a clean test |

Ablation (`drop_features` = the 5 SOP flags + `dob_agree` + `name_agree`): gold falls to 84% (score 77.8) and
validation has 2 unsafe clears. The network does learn from the underlying facts, but far less reliably
than the rules, so keep `rule_guard` on.

Notes: the official cost matrix makes "lowest expected cost" a poor decision rule (a wrong clear costs 60 but a
wrong SAME only 3, so it labels namesakes SAME), so `inference.use_cost_matrix` is off. `train.final_fit` (refit on
all non-gold hits) was tried and under-fits (48 namesakes kept open), so it is off too.

## Limits, honestly

- With silver labels, the network **learns the rules back**. It cannot beat the rule engine on dev until you give
  it gold labels the rules get wrong. Its value is as a scaffold: once you have more labels it can learn
  patterns the hand-written rules miss, and its probabilities give a confidence for each decision.
- The unseen set rewords free text. The regex extractors (`hit.py`, `customer.py`) are the main risk there.
- **Hackathon submission rules**: the brief requires `system/run` using only the Python/Node standard library
  (an LLM only via OpenRouter). PyTorch is not allowed in the submitted system. The SOP engine in this
  package (`data.py`, `normalize.py`, `customer.py`, `hit.py`, `compare.py`) uses only the standard library, so
  it can be wrapped as `system/run`. The network is for experiments and the demo.
