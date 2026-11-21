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
python -m screening_ml.predict --data data/screening/dev --out output/dev.jsonl
python data/screening/validate.py --data data/screening/dev --output output/dev.jsonl --score data/screening/dev/practice_labels.jsonl
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
| New features | append to `FEATURE_NAMES` and `featurize()` in `features.py`, then retrain |
| Better free-text extraction (ages, KYC notes) | `hit.extract_ages`, `customer._kyc_note_identity` (an LLM call can go here) |
| Fine-tuning on new labels | `config.yaml` → `finetune`, then `python -m screening_ml.finetune` |

## Results on the dev set

Seed 42, default config, CPU (about 11 s including feature building):

| Set | n | Accuracy | Decision score* | Unsafe clears |
|---|---|---|---|---|
| Silver train | 654 | 98.6% | 98.0 | 0 |
| Silver validation | 115 | 93.9% | 90.9 | 0 |
| Gold practice labels (held out) | 50 | 100% | 100.0 | 0 |
| Rule teacher on gold | 50 | 100% | 100.0 | 0 |

\* the organisers' score: share of cost saved versus sending everything to REVIEW.

`validate.py` on `output/dev.jsonl`: 819 lines, 0 problems; practice labels 50/50 right, reason code 96%,
customer record 100%. The network agrees with the rule teacher on 803 of 819 hits; all 16 disagreements are
clears the network was not confident about, which it sends to REVIEW (the safe direction). Two training runs
give byte-identical output.

Note: the official cost matrix makes "lowest expected cost" a poor decision rule (a wrong clear costs 60 but a
wrong SAME only 3, so it labels namesakes SAME). `inference.use_cost_matrix` is therefore off by default.

## Limits, honestly

- With silver labels, the network **learns the rules back**. It cannot beat the rule engine on dev until you give
  it gold labels the rules get wrong. Its value is as a scaffold: once you have more labels it can learn
  patterns the hand-written rules miss, and its probabilities give a confidence for each decision.
- The unseen set rewords free text. The regex extractors (`hit.py`, `customer.py`) are the main risk there.
- **Hackathon submission rules**: the brief requires `system/run` using only the Python/Node standard library
  (an LLM only via OpenRouter). PyTorch is not allowed in the submitted system. The SOP engine in this
  package (`data.py`, `normalize.py`, `customer.py`, `hit.py`, `compare.py`) uses only the standard library, so
  it can be wrapped as `system/run`. The network is for experiments and the demo.
