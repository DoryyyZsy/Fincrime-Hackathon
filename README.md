# Right Person Screening

Streamlit hackathon demo for explaining why two records do or do not describe the same person. The packaged snapshot contains a public OpenSanctions sanctions export and 5,190 synthetic customer records.

Authors: Dory Zhu, Jennifer Liu, Yiding Wang

## Run the packaged demo

From the project directory, use Python 3.11:

```shell
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
make app PYTHON=python
```

Open the local URL printed by Streamlit. The Overview links directly to three cases.

| Case | Expected result |
|---|---|
| Min Shi | An identical name scores below the alert threshold because the birth dates conflict. The detail page also cites OFAC’s recorded nationality. |
| Maslennikvo Yevhen Illich | A transliterated alias with a transposition is caught. The detail page shows the alias and matching date of birth. |
| Grey-zone case | Reviewed UN evidence supplies nationality and confirms the listed date of birth. Both fields conflict with the synthetic customer record. |

Use the live form with `Maslennikvo Yevhen Illich` and `1979-12-24` to reproduce the live match. Explainable linkage and all four pages work offline after installation. Analyst actions require a reason and append a UTC-stamped entry to `data/processed/audit_log.jsonl`.

## Rebuild and evaluate

```shell
make test data match alerts eval PYTHON=python
```

The packaged input is the OpenSanctions sanctions export dated 20 November 2026, artifact `20261120194704-jzr`. You can also place another `targets.simple.csv` file at `data/raw/sanctions_targets.csv`. The OpenSanctions `default` collection can include PEPs as well as sanctions. Rebuilding against a different file can change the generated cases and measured results. [Data format](https://www.opensanctions.org/docs/bulk/csv/) · [Snapshot](https://data.opensanctions.org/artifacts/sanctions/20261120194704-jzr/targets.simple.csv)

| Metric | Name-only fuzzy baseline | Right Person Screening |
|---|---:|---:|
| Alerts | 574 | 111 |
| True hits caught | 40 | 40 |
| Missed true hits | 0 | 0 |
| False positives | 534 | 71 |
| Alert reduction | | 80.7% |

An alert is a unique customer–entity pair sent to an analyst queue or marked likely hit. Dismissed pairs are retained for explanation and excluded from the alert count. The baseline uses the shared normalized names, a same-initial block, a length bound and RapidFuzz token-sort ratio ≥85. Initial blocking can omit first-letter typos. Linkage trains random-pair and EM estimates, then replaces the DOB agreement distribution with a stated assumption because same-name nonmatches dominate the training lists. The prior and DOB conflict assumption were tuned on this demo snapshot. The numbers describe the seed-42 synthetic dataset and this baseline. Independent validation and PEP performance measurement remain open.

The simplified CSV mixes nationality with other country affiliations and flattens identifiers. Those values are kept for reference. Only explicit nationality and scoped IDs are scored where present. This snapshot did not exercise nationality or ID comparisons during model training. The exact scoped-ID routing override is covered by tests. Edit `config.yaml` using its JSON-compatible YAML format, then rerun `make alerts eval` to apply route thresholds. Rerun `make match` for model changes.

## Optional Tavily and Grok

Copy `.env.example` to `.env` and set the credentials and Grok model name. To fetch public research on listed entities, run:

```shell
make alerts eval PYTHON=python ALERT_ARGS=--research
```

Tavily receives only public listing properties. The bundled grey-zone example uses reviewed public evidence and works without API credentials. By design, the live form only screens records locally. Grok receives generated demo customer records only when explicitly enabled from the command line:

```shell
make alerts eval PYTHON=python ALERT_ARGS="--research --grok-synthetic"
```

The Grok mode verifies that the customer file came from the synthetic generator. It sends the comparison and public findings as data to a chat completion with no search tools. The system prompt is `src/adjudication_prompt.txt`. Invalid model output and uncertain recommendations route to human review. The packaged benchmark used reviewed public evidence without live API calls.
