# GIST Reasoning Traces

This LMU GIST master's seminar project studies how GPT-OSS and DeepSeek extract Scope 1, 2, and 3 greenhouse gas emissions from corporate reports using ClimateXtract.

## Research questions

- How do extraction precision, recall, F1, and cost vary across the tested reasoning settings? Accuracy is measured against reference values.
- What themes appear in the reasoning traces, and how do they change after filtering out sentences that repeat the prompt?

The [seminar presentation](presentation/main.pdf) covers the experimental setup and findings.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Analyze saved runs

Run extraction in the ClimateXtract repository. Copy the saved run directories to `output/` and point `RUN_DIRS` in [analysis/config.py](analysis/config.py) to them.

Run `python analysis/topic_modelling.py` to compare BERTopic results before and after filtering prompt copies. Results go to `analysis/topic_modelling/`. Install `openpyxl` for Excel exports or set `SAVE_EXCEL = False`.

See the [analysis notebooks](analysis/notebooks/) and [archived experiments](experiment/README.md).
