# GIST Reasoning Traces

Sample extractions with GPT-OSS and DeepSeek using the installed ClimateXtract
package, followed by descriptive statistics on saved reasoning traces.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Provide the gateway settings from `.env.example` in a local `.env`. Put the
report, existing embedding database and gold standard at the paths configured
in `configs/climatextract_test.toml`. See `data/README.md`.

## Run

```bash
python scripts/sample_pipeline_run.py --check  # local input/cache checks
python scripts/sample_pipeline_run.py          # paid model calls
```

Select reports, model IDs and output limit in the TOML config. Both models use
its retrieval, prompt and extraction settings. Each document is processed
separately; the runner verifies matching prompt sets for paired runs.
Cached embeddings are required; it refuses new embedding calls.

```text
output/
├── gptoss_<effort>_<run-id>/
│   └── <document>_<filename-hash>/
│       ├── reasoning_traces.json
│       ├── config_and_metrics.json
│       └── … pipeline CSV files
└── deepseek_<effort>_<same-run-id>/
    └── <document>_<filename-hash>/
        └── … same file structure
```

Each trace file stores document/model IDs, requested settings, prompt hashes,
answers, reasoning text, raw responses, failures, reported usage/cost and call
duration. Files are updated after every attempted call. Existing runs are
preserved. The output-token limit is not a dollar spending cap. Equal requested
reasoning effort is not an equal internal reasoning budget across model families.
The runner uses ClimateXtract 0.3.4's internal orchestration interface for exact
output paths; package upgrades may need an integration change.

## Analyse

```bash
python scripts/analyse_reasoning_pilot.py
# Or select a subset of saved runs:
python scripts/analyse_reasoning_pilot.py --input output/deepseek_low_RUN_ID
```

Only two derived files are written (replaced on each analysis):

- `analysis/calls.csv`: one row per recorded API attempt.
- `analysis/documents.csv`: one row per model/document/run.

Facts include trace characters/words, missing/empty traces, JSON/schema status,
extracted-record counts, truncation, errors, duration and reported usage/cost.
Missing measurements are blank, not zero. Mean/min/max trace lengths use only
nonempty traces; counts of missing/empty traces remain explicit. Words use
Python `str.split()`, not a tokenizer. Token counts are provider-reported;
completion tokens can include reasoning. Reported zero cost is retained and
may mean unavailable billing. JSON/schema success is not extraction accuracy.
Gold-standard accuracy results remain in the pipeline's evaluation CSV files.
No qualitative labels, hardcoded findings or inferred causal claims are produced.
Different answers are allowed. Older files without document metadata are marked
`unknown`; they are not assigned a document based on page number.

Historical pilot notes/artifacts are retained in `analysis/archive/` and are
not used by these scripts.

## Local verification

```bash
python -m unittest discover -s tests
```
