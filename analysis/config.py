"""Configuration for topic modelling on reasoning traces.

Used by ``topic_modelling.py``. Edit values here, then run the script.
"""

from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output"
# All topic-modelling outputs go here (dated subfolder per script run).
OUTPUT_DIR = Path(__file__).resolve().parent / "topic_modelling"
# Shared MiniLM cache (reused across runs with the same sentence set).
EMBEDDINGS_CACHE_DIR = OUTPUT_DIR / "cache"

# ---------------------------------------------------------------------------
# Which model runs to load (label → folder under output/)
# ---------------------------------------------------------------------------
DEEPSEEK = "DeepSeek-V4-Flash-0731"
RUN_DIRS = {
    "gpt-oss low": OUT / "gpt-oss-120b_low" / "1ebb0c8dc26d4681bda9555908104e67",
    "gpt-oss medium": OUT / "gpt-oss-120b_medium" / "8b438992f14c4b6da28ec581572dcc85",
    "gpt-oss high": OUT / "gpt-oss-120b_high" / "10d7970dd8304f8cb06d174a851f8e2a",
    "DeepSeek high": OUT / f"{DEEPSEEK}_high" / "2c50bb4fcde3462f9d6cbe6516b686c9",
    "DeepSeek max": OUT / f"{DEEPSEEK}_max" / "4c3229ed86dc4fe6b1ef4b95c64ca68d",
}
KEEP_RUNS = list(RUN_DIRS)

# ---------------------------------------------------------------------------
# Topic model
# ---------------------------------------------------------------------------
N_TOPICS = 10
EMBED_MODEL = "all-MiniLM-L6-v2"
SEED = 42
# Prompt-copy: cosine similarity of sentence ↔ closest instruction sentence.
THRESHOLD_PROMPT_COPY = 0.85
MIN_SENTENCE_WORDS = 3

# BERTopic / UMAP (clustering space; separate from 2D plot UMAP)
UMAP_N_NEIGHBORS = 15
UMAP_N_COMPONENTS = 20
UMAP_MIN_DIST = 0.0
KMEANS_N_INIT = 10

# ---------------------------------------------------------------------------
# What to run
# ---------------------------------------------------------------------------
# full_trace: fit on all sentences (incl. prompt copies) — like the Excel review
RUN_FULL_TRACE = True
# clean_trace: fit only on sentences that are NOT prompt copies
RUN_CLEAN_TRACE = True

# ---------------------------------------------------------------------------
# What to save
# ---------------------------------------------------------------------------
SAVE_EXCEL = True
# Excel with sentence-level topics + keywords (large).
SAVE_EXCEL_NAME = "sentence_topics.xlsx"
# CSV/parquet tables are always useful for evaluation; Excel is optional above.
SAVE_TABLES = True
SAVE_PROMPT_COPIES = True
SAVE_PROMPT_COPIES_NAME = "prompt_copies.csv"
SAVE_PLOTS = True
SAVE_UMAP = True
# Reuse MiniLM embeddings from EMBEDDINGS_CACHE_DIR across runs.
CACHE_EMBEDDINGS = True
