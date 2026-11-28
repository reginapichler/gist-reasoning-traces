"""Use this script to run topic modelling on the traces.

There are two ways to run the topic modelling:
- full_trace: topic modelling on the full trace (all sentences)
- clean_trace: topic modelling on the trace without prompt copies

Configurations for the run are in ``config.py``:
- n_topics, embedding model, threshold_prompt_copy, seed
- save_excel / save_prompt_copies / paths
- which model runs to load, output folder, BERTopic/UMAP knobs

Run from the repo root::

    .venv/bin/python analysis/topic_modelling.py
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Allow ``import config`` when run as a script from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as cfg

SPLIT = re.compile(r"\n+|(?<=[.!?])\s+(?=[A-Z\"'(\[])")
EXCEL_CELL_LIMIT = 32767
EXCEL_TRUNC = "\n\n[... truncated for Excel cell limit ...]"
ILLEGAL_EXCEL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")


# ---------------------------------------------------------------------------
# Load + segment
# ---------------------------------------------------------------------------

def sentences(text: str, min_words: int = cfg.MIN_SENTENCE_WORDS) -> list[str]:
    out = []
    for part in SPLIT.split(text or ""):
        part = re.sub(r"\s+", " ", part).strip(" -*•\t")
        if len(part.split()) >= min_words:
            out.append(part)
    return out


def load_traces(run_dirs: dict[str, Path] | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (pages, seg). pages = one row per call; seg = one row per sentence."""
    run_dirs = run_dirs or cfg.RUN_DIRS
    rows = []
    for run, run_dir in run_dirs.items():
        trace_dir = Path(run_dir) / "reasoning_traces"
        if not trace_dir.is_dir():
            raise FileNotFoundError(f"missing reasoning_traces for {run}: {trace_dir}")
        for path in sorted(trace_dir.glob("*.jsonl")):
            with path.open(encoding="utf-8") as f:
                for i, line in enumerate(f):
                    c = json.loads(line)
                    reasoning = c.get("reasoning_content")
                    if not reasoning or not str(reasoning).strip():
                        continue
                    rows.append(
                        {
                            "run": run,
                            "report": Path(str(c["pdf"]).replace("\\", "/")).name,
                            "call_index": i,
                            "reasoning": reasoning,
                            "answer": c.get("content"),
                            "prompt": c.get("prompt"),
                        }
                    )

    pages = pd.DataFrame(rows)
    pages["run"] = pd.Categorical(pages["run"], list(run_dirs), ordered=True)
    pages["trace_id"] = np.arange(len(pages))

    seg = pd.DataFrame(
        [
            (row.trace_id, row.run, row.report, i, text)
            for row in pages.itertuples()
            for i, text in enumerate(sentences(row.reasoning))
        ],
        columns=["trace_id", "run", "report", "sentence_pos", "text"],
    )
    seg["run"] = pd.Categorical(seg["run"], list(run_dirs), ordered=True)
    return pages, seg


def _parse_modell_effort(run: str) -> tuple[str, str]:
    """Map run label → (Modell, reasoning_effort) like the old review Excel."""
    s = str(run)
    if s.startswith("gpt-oss"):
        return "gpt-oss-120b", s.replace("gpt-oss", "").strip()
    if s.startswith("DeepSeek"):
        return cfg.DEEPSEEK, s.replace("DeepSeek", "").strip()
    parts = s.rsplit(" ", 1)
    if len(parts) == 2:
        return parts[0], parts[1]
    return s, ""


ERR = {
    "true_positive_value": "tp",
    "false_positive_value": "fp",
    "false_negative_value": "fn",
    "true_negative_value": "tn",
}


def _pdf_name(path_or_name: str) -> str:
    return Path(str(path_or_name).replace("\\", "/")).name


def _norm_page(v) -> str | None:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return None
    s = str(v).strip()
    return s if s and s.lower() != "nan" else None


def _load_raw_page_texts(run_dir: Path) -> pd.DataFrame:
    """Non-empty page texts from pipeline ``raw_results.csv`` (prompt → PDF page)."""
    raw = run_dir / "raw_results.csv"
    if not raw.is_file():
        return pd.DataFrame(columns=["report", "page_label", "page_text"])
    df = pd.read_csv(
        raw,
        dtype=str,
        usecols=["report_name_short", "page_number_to_llm", "page_texts_to_llm"],
    ).rename(
        columns={
            "report_name_short": "report",
            "page_number_to_llm": "page_label",
            "page_texts_to_llm": "page_text",
        }
    )
    df["report"] = df["report"].map(_pdf_name)
    df["page_label"] = df["page_label"].map(_norm_page)
    df = df[["report", "page_label", "page_text"]].dropna(subset=["page_text"])
    df = df[df["page_text"].astype(str).str.strip().ne("")]
    return df.drop_duplicates(["report", "page_label"]).reset_index(drop=True)


def _assign_pdf_pages(calls: pd.DataFrame, page_texts: pd.DataFrame) -> pd.Series:
    """Match each call's prompt to the longest page text it contains → pdf page label."""
    texts = {
        r: [(p, t.strip()) for p, t in zip(g.page_label, g.page_text) if isinstance(t, str) and t.strip()]
        for r, g in page_texts.groupby("report")
    }

    def match(row):
        prompt = row.prompt or ""
        hits = [(len(t), p) for p, t in texts.get(row.report, []) if t in prompt]
        return max(hits)[1] if hits else None

    return calls.apply(match, axis=1)


def _load_eval(run: str, run_dir: Path) -> pd.DataFrame:
    path = run_dir / "eval_results_vs_benchmark.csv"
    if not path.is_file():
        print(f"WARNING: no eval_results_vs_benchmark.csv for {run} ({run_dir})")
        return pd.DataFrame()
    e = pd.read_csv(
        path,
        usecols=[
            "ReportName",
            "page_man",
            "page_number_used_by_llm",
            "error_value",
        ],
        dtype={"page_man": str, "page_number_used_by_llm": str},
    )
    e["report"] = e["ReportName"].map(_pdf_name)
    e["err"] = e["error_value"].map(ERR)
    e["page_used"] = e["page_number_used_by_llm"].map(_norm_page)
    e["page_gold"] = e["page_man"].map(_norm_page)
    e["run"] = run
    return e


def build_page_outcomes(pages: pd.DataFrame, run_dirs: dict[str, Path] | None = None) -> pd.DataFrame:
    """Build page-level gold/correctness from ``output/...`` run folders (no derived/)."""
    run_dirs = run_dirs or cfg.RUN_DIRS
    calls = pages[["run", "report", "call_index", "prompt", "answer"]].copy()
    calls["run"] = calls["run"].astype(str)
    calls["report"] = calls["report"].map(_pdf_name)
    calls["pdf_page"] = None

    evals = []
    for run, run_dir in run_dirs.items():
        run_dir = Path(run_dir)
        mask = calls["run"] == run
        page_texts = _load_raw_page_texts(run_dir)
        if len(page_texts):
            calls.loc[mask, "pdf_page"] = _assign_pdf_pages(calls.loc[mask], page_texts)
        else:
            print(f"WARNING: no non-empty page texts in raw_results for {run}")
        ev = _load_eval(run, run_dir)
        if len(ev):
            evals.append(ev)

    if not evals:
        print("WARNING: no eval files found — Excel will lack gold columns")
        return pd.DataFrame()

    ev = pd.concat(evals, ignore_index=True)
    pred = (
        ev[ev.err.isin(["tp", "fp"])]
        .groupby(["run", "report", "page_used", "err"], dropna=False)
        .size()
        .unstack("err", fill_value=0)
        .rename(columns={"tp": "n_tp_page", "fp": "n_fp_page"})
    )
    gold = (
        ev[ev.err.isin(["tp", "fn"])]
        .groupby(["run", "report", "page_gold", "err"], dropna=False)
        .size()
        .unstack("err", fill_value=0)
    )
    gold["n_gold_page"] = gold.sum(axis=1)
    if "fn" in gold.columns:
        gold = gold.rename(columns={"fn": "n_fn_page"})
    else:
        gold["n_fn_page"] = 0
    gold = gold[["n_gold_page", "n_fn_page"]]
    pred.index = pred.index.set_names(["run", "report", "pdf_page"])
    gold.index = gold.index.set_names(["run", "report", "pdf_page"])

    pc = calls.dropna(subset=["pdf_page"]).copy()
    pc = pc.join(pred, on=["run", "report", "pdf_page"]).join(gold, on=["run", "report", "pdf_page"])
    for c in ["n_tp_page", "n_fp_page", "n_gold_page", "n_fn_page"]:
        pc[c] = pc[c].fillna(0).astype(int)
    pc["page_outcome"] = np.select(
        [
            (pc.n_gold_page == 0) & (pc.n_fp_page == 0),
            (pc.n_gold_page > 0) & (pc.n_fn_page == 0) & (pc.n_fp_page == 0),
        ],
        ["correct, no gold values", "correct, gold values"],
        default="error",
    )
    pc["gold_standard_met"] = pc["page_outcome"].str.startswith("correct")
    pc["gold_values_correct"] = pc["page_outcome"] == "correct, gold values"
    pc["gold_on_page"] = pc["n_gold_page"] > 0
    pc["answer_status"] = np.where(
        pc["answer"].fillna("").astype(str).str.contains('"KPI_Entries": []', regex=False),
        "empty",
        "nonempty",
    )

    report = (
        pc.groupby(["run", "report"], as_index=False)
        .agg(
            report_gold_standard_met=("gold_standard_met", "all"),
            report_tp=("n_tp_page", "sum"),
            report_fp=("n_fp_page", "sum"),
            report_fn=("n_fn_page", "sum"),
            report_tn=("page_outcome", lambda s: int((s == "correct, no gold values").sum())),
        )
    )
    pc = pc.merge(report, on=["run", "report"], how="left")

    # Left-join so traces without a matched pdf_page stay (outcome = NaN)
    out = calls[["run", "report", "call_index", "pdf_page", "answer"]].merge(
        pc.drop(columns=["prompt", "answer", "pdf_page"], errors="ignore"),
        on=["run", "report", "call_index"],
        how="left",
    )
    print(
        f"pdf pages matched: {out['pdf_page'].notna().mean():.1%} of traces  |  "
        f"outcomes: {out['page_outcome'].notna().mean():.1%}"
    )
    return out


def enrich_pages_with_outcomes(pages: pd.DataFrame, outcomes: pd.DataFrame) -> pd.DataFrame:
    """Attach gold/answer columns onto page-level traces."""
    out = pages.copy()
    modell_effort = out["run"].astype(str).map(_parse_modell_effort)
    out["Modell"] = [m for m, _ in modell_effort]
    out["reasoning_effort"] = [e for _, e in modell_effort]

    if outcomes.empty:
        return out

    cols = [c for c in outcomes.columns if c not in {"run", "report", "call_index"}]
    merged = out.merge(
        outcomes[["run", "report", "call_index", *cols]],
        on=["run", "report", "call_index"],
        how="left",
        suffixes=("", "_out"),
    )
    if "answer_out" in merged.columns:
        merged["answer"] = merged["answer_out"].combine_first(merged["answer"])
        merged = merged.drop(columns=["answer_out"])
    if "pdf_page_out" in merged.columns:
        merged["pdf_page"] = merged["pdf_page_out"].combine_first(merged.get("pdf_page"))
        merged = merged.drop(columns=["pdf_page_out"])
    n = merged["page_outcome"].notna().sum() if "page_outcome" in merged.columns else 0
    print(
        f"page outcomes joined: {n:,}/{len(merged):,} traces "
        f"({n / max(len(merged), 1):.1%} coverage)"
    )
    return merged


def attach_page_meta(seg: pd.DataFrame, pages: pd.DataFrame) -> pd.DataFrame:
    """Copy page-level meta (answer, gold, …) onto every sentence row."""
    meta_cols = [
        c
        for c in [
            "trace_id",
            "Modell",
            "reasoning_effort",
            "call_index",
            "pdf_page",
            "page_outcome",
            "gold_standard_met",
            "gold_values_correct",
            "gold_on_page",
            "n_gold_page",
            "n_tp_page",
            "n_fp_page",
            "n_fn_page",
            "report_gold_standard_met",
            "report_tp",
            "report_fp",
            "report_fn",
            "report_tn",
            "answer",
            "answer_status",
        ]
        if c in pages.columns
    ]
    out = seg.drop(columns=[c for c in meta_cols if c != "trace_id" and c in seg.columns], errors="ignore")
    return out.merge(pages[meta_cols], on="trace_id", how="left")


def add_topic_majority(seg: pd.DataFrame) -> pd.DataFrame:
    """Majority topic among labelled sentences in the same trace/page."""
    out = seg.copy()
    labelled = out.dropna(subset=["topic"])
    if labelled.empty:
        out["topic_majority_in_page"] = np.nan
        return out
    maj = (
        labelled.groupby("trace_id")["topic"]
        .agg(lambda s: int(s.value_counts().index[0]))
        .rename("topic_majority_in_page")
    )
    out = out.drop(columns=["topic_majority_in_page"], errors="ignore")
    return out.merge(maj, on="trace_id", how="left")


def review_export_frame(seg: pd.DataFrame) -> pd.DataFrame:
    """Column order / names aligned with the old bertopic10 review Excel."""
    df = seg.copy()
    rename = {
        "report": "file",
        "call_index": "page",
        "text": "sentence",
        "topic": "topic_by_bert",
    }
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})
    preferred = [
        "Modell",
        "reasoning_effort",
        "run",
        "file",
        "page",
        "pdf_page",
        "sentence_pos",
        "sentence",
        "topic_by_bert",
        "topic_keywords",
        "topic_majority_in_page",
        "page_outcome",
        "gold_standard_met",
        "gold_values_correct",
        "gold_on_page",
        "n_gold_page",
        "n_tp_page",
        "n_fp_page",
        "n_fn_page",
        "report_gold_standard_met",
        "report_tp",
        "report_fp",
        "report_fn",
        "report_tn",
        "prompt_sim",
        "prompt_copy",
        "instruction_text",
        "answer",
        "answer_status",
        "trace_id",
    ]
    cols = [c for c in preferred if c in df.columns] + [
        c for c in df.columns if c not in preferred
    ]
    return df[cols]


def print_load_diagnostics(pages: pd.DataFrame, seg: pd.DataFrame) -> None:
    """Short per-model counts after loading / segmenting."""
    print("\n--- load ---")
    print(f"traces: {len(pages):,}  |  sentences: {len(seg):,}  |  unique: {seg.text.nunique():,}")
    per = (
        pd.DataFrame(
            {
                "traces": pages.groupby("run", observed=True).size(),
                "sentences": seg.groupby("run", observed=True).size(),
                "unique": seg.groupby("run", observed=True)["text"].nunique(),
            }
        )
        .assign(
            sent_per_trace=lambda d: (d["sentences"] / d["traces"]).round(1),
        )
    )
    print(per.to_string())
    print(
        f"median sentences/trace (overall): "
        f"{seg.groupby('trace_id').size().median():.0f}"
    )


def print_prompt_diagnostics(seg: pd.DataFrame, per_run: pd.DataFrame) -> None:
    print("\n--- prompt copies ---")
    n_copy = int(seg["prompt_copy"].sum())
    print(
        f"flagged: {n_copy:,} / {len(seg):,} sentences "
        f"({seg['prompt_copy'].mean():.1%})  |  threshold={cfg.THRESHOLD_PROMPT_COPY}"
    )
    # Compact per-model view
    view = per_run.copy()
    view["prompt_copy_share_mean"] = (view["prompt_copy_share_mean"] * 100).round(1).astype(str) + "%"
    view["prompt_sim_mean"] = view["prompt_sim_mean"].round(3)
    print(view.to_string(index=False))


# ---------------------------------------------------------------------------
# Embeddings + prompt-copy scores
# ---------------------------------------------------------------------------

def _try_load_embedding_cache(
    uniq: list[str], emb_path: Path, txt_path: Path
) -> np.ndarray | None:
    if not (emb_path.is_file() and txt_path.is_file()):
        return None
    cached = pd.read_parquet(txt_path)["text"].tolist()
    if cached != uniq:
        return None
    E = np.load(emb_path)
    print(f"loaded cached embeddings {E.shape} ← {emb_path}")
    return E


def embed_unique_sentences(
    texts: list[str],
    *,
    cache_dir: Path | None = None,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Embed unique sentences; cache by content hash under ``cache_dir``."""
    from sentence_transformers import SentenceTransformer

    uniq = list(dict.fromkeys(texts))  # preserve order, drop dups
    key = hashlib.sha1("\n".join(uniq).encode()).hexdigest()[:12]
    emb_name, txt_name = f"embeddings_{key}.npy", f"sentences_{key}.parquet"

    if cfg.CACHE_EMBEDDINGS and cache_dir is not None:
        # 1) Shared cache
        E = _try_load_embedding_cache(uniq, cache_dir / emb_name, cache_dir / txt_name)
        if E is not None:
            return pd.DataFrame({"text": uniq}), E

        # 2) Older run folders under OUTPUT_DIR (e.g. current job still writing there)
        for emb_path in sorted(cfg.OUTPUT_DIR.glob(f"*/{emb_name}")):
            txt_path = emb_path.with_name(txt_name)
            E = _try_load_embedding_cache(uniq, emb_path, txt_path)
            if E is not None:
                # Promote into shared cache for next runs
                cache_dir.mkdir(parents=True, exist_ok=True)
                dest_e, dest_t = cache_dir / emb_name, cache_dir / txt_name
                if not dest_e.exists():
                    dest_e.write_bytes(emb_path.read_bytes())
                    dest_t.write_bytes(txt_path.read_bytes())
                    print(f"copied embeddings into shared cache → {dest_e.name}")
                return pd.DataFrame({"text": uniq}), E

    embedder = SentenceTransformer(cfg.EMBED_MODEL)
    E = embedder.encode(
        uniq, batch_size=256, normalize_embeddings=True, show_progress_bar=True
    )
    if cache_dir is not None and cfg.CACHE_EMBEDDINGS:
        cache_dir.mkdir(parents=True, exist_ok=True)
        emb_path = cache_dir / emb_name
        txt_path = cache_dir / txt_name
        np.save(emb_path, E)
        pd.DataFrame({"text": uniq}).to_parquet(txt_path, index=False)
        print(f"saved embeddings {E.shape} → {emb_path}")
    return pd.DataFrame({"text": uniq}), np.asarray(E)


def score_prompt_copies(
    pages: pd.DataFrame,
    sent: pd.DataFrame,
    E: np.ndarray,
) -> pd.DataFrame:
    """Add prompt_sim / prompt_copy / instruction_text on unique-sentence frame."""
    from sentence_transformers import SentenceTransformer

    sample_prompt = next(
        (p for p in pages["prompt"] if isinstance(p, str) and "Here is the excerpt" in p),
        None,
    )
    if sample_prompt is None:
        sent = sent.copy()
        sent["prompt_sim"] = np.nan
        sent["prompt_copy"] = False
        sent["instruction_text"] = None
        return sent

    instructions = sentences(sample_prompt.split("Here is the excerpt")[0])
    embedder = SentenceTransformer(cfg.EMBED_MODEL)
    I = embedder.encode(instructions, normalize_embeddings=True, show_progress_bar=False)
    sims = E @ I.T
    best = sims.argmax(axis=1)
    sent = sent.copy()
    sent["prompt_sim"] = sims.max(axis=1)
    sent["prompt_copy"] = sent["prompt_sim"] > cfg.THRESHOLD_PROMPT_COPY
    sent["instruction_text"] = [instructions[j] for j in best]
    return sent


def attach_prompt_flags(seg: pd.DataFrame, sent: pd.DataFrame) -> pd.DataFrame:
    cols = ["text", "prompt_sim", "prompt_copy", "instruction_text"]
    return seg.drop(columns=[c for c in cols[1:] if c in seg.columns], errors="ignore").merge(
        sent[cols], on="text", how="left"
    )


# ---------------------------------------------------------------------------
# Prompt-copy analysis
# ---------------------------------------------------------------------------

def analyse_prompt_copies(seg: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-trace and per-run prompt-copy shares."""
    per_trace = (
        seg.groupby(["run", "report", "trace_id"], observed=True)
        .agg(
            n_sentences=("text", "size"),
            prompt_copy_share=("prompt_copy", "mean"),
            prompt_sim_mean=("prompt_sim", "mean"),
        )
        .reset_index()
    )
    per_run = (
        per_trace.groupby("run", observed=True)
        .agg(
            n_traces=("trace_id", "size"),
            prompt_copy_share_mean=("prompt_copy_share", "mean"),
            prompt_copy_share_median=("prompt_copy_share", "median"),
            prompt_sim_mean=("prompt_sim_mean", "mean"),
        )
        .reset_index()
    )
    return per_trace, per_run


# ---------------------------------------------------------------------------
# BERTopic
# ---------------------------------------------------------------------------

def fit_bertopic(
    docs: list[str],
    embeddings: np.ndarray,
    *,
    embedder=None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit BERTopic with fixed k-means. Returns (sent_topics, topic_info)."""
    from bertopic import BERTopic
    from bertopic.representation import KeyBERTInspired
    from sentence_transformers import SentenceTransformer
    from sklearn.cluster import KMeans
    from sklearn.feature_extraction.text import CountVectorizer
    from umap import UMAP

    if embedder is None:
        embedder = SentenceTransformer(cfg.EMBED_MODEL)

    topic_model = BERTopic(
        embedding_model=embedder,
        umap_model=UMAP(
            n_neighbors=cfg.UMAP_N_NEIGHBORS,
            n_components=cfg.UMAP_N_COMPONENTS,
            min_dist=cfg.UMAP_MIN_DIST,
            metric="cosine",
            random_state=cfg.SEED,
        ),
        hdbscan_model=KMeans(
            n_clusters=cfg.N_TOPICS, random_state=cfg.SEED, n_init=cfg.KMEANS_N_INIT
        ),
        vectorizer_model=CountVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            min_df=5,
            token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z\-]+\b",
        ),
        representation_model=KeyBERTInspired(),
        calculate_probabilities=False,
    )
    topics, _ = topic_model.fit_transform(docs, embeddings)
    sent_topics = pd.DataFrame({"text": docs, "topic": topics})
    info = topic_model.get_topic_info()
    topic_info = pd.DataFrame(
        {
            "topic": info.Topic,
            "n": info.Count,
            "keywords": info.Representation.map(lambda w: ", ".join(w[:8])),
        }
    )
    return sent_topics, topic_info


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_prompt_copy_share(per_trace: pd.DataFrame, out_path: Path) -> None:
    order = [r for r in cfg.KEEP_RUNS if r in set(per_trace["run"].astype(str))]
    data = [per_trace.loc[per_trace["run"] == r, "prompt_copy_share"] for r in order]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.boxplot(data, tick_labels=order, showfliers=False)
    ax.set_ylabel("share of prompt-copy sentences")
    ax.set_title("Prompt-copy share per trace")
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_topic_mix(seg: pd.DataFrame, out_path: Path, title: str) -> None:
    labelled = seg.dropna(subset=["topic"]).copy()
    labelled["topic"] = labelled["topic"].astype(int)
    share = (
        labelled.groupby(["run", "topic"], observed=True)
        .size()
        .groupby("run", observed=True)
        .transform(lambda s: s / s.sum())
        .reset_index(name="share")
        .pivot(index="topic", columns="run", values="share")
        .reindex(columns=cfg.KEEP_RUNS)
        .fillna(0)
    )
    ax = share.T.plot(kind="bar", stacked=True, figsize=(9, 4), colormap="tab10")
    ax.set_ylabel("share of sentences")
    ax.set_title(title)
    ax.legend(title="topic", bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    fig = ax.get_figure()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_umap(
    embeddings: np.ndarray,
    topics: np.ndarray,
    out_path: Path,
    title: str,
) -> np.ndarray:
    from umap import UMAP

    xy = UMAP(
        n_neighbors=15,
        n_components=2,
        min_dist=0.1,
        metric="cosine",
        random_state=cfg.SEED,
    ).fit_transform(embeddings)

    rng = np.random.default_rng(cfg.SEED)
    idx = rng.choice(len(xy), size=min(40_000, len(xy)), replace=False)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    sc = axes[0].scatter(
        xy[idx, 0],
        xy[idx, 1],
        c=topics[idx],
        cmap="tab10",
        s=3,
        alpha=0.35,
        linewidths=0,
    )
    axes[0].set_title(title)
    axes[0].set_xticks([])
    axes[0].set_yticks([])
    fig.colorbar(sc, ax=axes[0], ticks=sorted(set(map(int, topics))))

    axes[1].scatter(
        xy[idx, 0], xy[idx, 1], c=topics[idx], cmap="tab10", s=2, alpha=0.2, linewidths=0
    )
    centres = (
        pd.DataFrame({"topic": topics, "x": xy[:, 0], "y": xy[:, 1]})
        .groupby("topic")[["x", "y"]]
        .mean()
    )
    for t, row in centres.iterrows():
        axes[1].text(
            row.x,
            row.y,
            str(int(t)),
            fontsize=10,
            fontweight="bold",
            ha="center",
            va="center",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.85, lw=0),
        )
    axes[1].set_title("UMAP with topic IDs")
    axes[1].set_xticks([])
    axes[1].set_yticks([])
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return xy


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

def _clean_excel_cell(v):
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return v
    if not isinstance(v, str):
        return v
    s = ILLEGAL_EXCEL.sub("", v)
    if len(s) > EXCEL_CELL_LIMIT:
        s = s[: EXCEL_CELL_LIMIT - len(EXCEL_TRUNC)] + EXCEL_TRUNC
    return s


def save_excel(
    sentence_topics: pd.DataFrame,
    topic_info: pd.DataFrame,
    path: Path,
) -> None:
    df = sentence_topics.copy()
    for c in df.columns:
        if df[c].dtype == object:
            df[c] = df[c].map(_clean_excel_cell)
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        topic_info.to_excel(writer, sheet_name="topics", index=False)
        if "run" in df.columns:
            for run, g in df.groupby("run", sort=True, observed=True):
                g.to_excel(writer, sheet_name=str(run)[:31], index=False)
        else:
            df.to_excel(writer, sheet_name="sentences", index=False)


def save_run_config(run_dir: Path) -> None:
    """Snapshot the config module values used for this run."""
    keys = [
        "N_TOPICS",
        "EMBED_MODEL",
        "SEED",
        "THRESHOLD_PROMPT_COPY",
        "MIN_SENTENCE_WORDS",
        "UMAP_N_NEIGHBORS",
        "UMAP_N_COMPONENTS",
        "UMAP_MIN_DIST",
        "KMEANS_N_INIT",
        "RUN_FULL_TRACE",
        "RUN_CLEAN_TRACE",
        "SAVE_EXCEL",
        "SAVE_PROMPT_COPIES",
        "SAVE_PLOTS",
        "SAVE_UMAP",
        "CACHE_EMBEDDINGS",
    ]
    snap = {k: getattr(cfg, k) for k in keys}
    snap["OUTPUT_DIR"] = str(cfg.OUTPUT_DIR)
    snap["EMBEDDINGS_CACHE_DIR"] = str(cfg.EMBEDDINGS_CACHE_DIR)
    snap["KEEP_RUNS"] = list(cfg.KEEP_RUNS)
    snap["RUN_DIRS"] = {k: str(v) for k, v in cfg.RUN_DIRS.items()}
    (run_dir / "config_snapshot.json").write_text(
        json.dumps(snap, indent=2) + "\n", encoding="utf-8"
    )


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def _save_sentence_outputs(
    seg_out: pd.DataFrame,
    topic_info: pd.DataFrame,
    sub_dir: Path,
) -> pd.DataFrame:
    """Write parquet/csv/xlsx with review-style columns (topics + gold + answer)."""
    seg_out = add_topic_majority(seg_out)
    export_df = review_export_frame(seg_out)

    if cfg.SAVE_TABLES:
        export_df.to_parquet(sub_dir / "sentence_topics.parquet", index=False)
        export_df.to_csv(sub_dir / "sentence_topics.csv", index=False)
        print("saved tables → sentence_topics.parquet/csv")

    if cfg.SAVE_EXCEL:
        save_excel(export_df, topic_info, sub_dir / cfg.SAVE_EXCEL_NAME)
        print(f"saved Excel → {cfg.SAVE_EXCEL_NAME} ({len(export_df):,} rows, {export_df.shape[1]} cols)")

    return export_df


def run_topic_fit(
    mode: str,
    seg: pd.DataFrame,
    sent: pd.DataFrame,
    E: np.ndarray,
    run_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray | None]:
    """mode: 'full' | 'clean'. Returns (seg_labelled, topic_info, umap_xy)."""
    assert mode in {"full", "clean"}
    sub_dir = run_dir / f"{mode}_trace"
    sub_dir.mkdir(parents=True, exist_ok=True)

    if mode == "clean":
        mask = ~sent["prompt_copy"].to_numpy()
        docs = sent.loc[mask, "text"].tolist()
        E_fit = E[mask]
        print(f"\n--- topic model ({mode}) ---")
        print(f"fitting BERTopic on {len(docs):,} unique non-copy sentences …")
    else:
        docs = sent["text"].tolist()
        E_fit = E
        print(f"\n--- topic model ({mode}) ---")
        print(f"fitting BERTopic on {len(docs):,} unique sentences (incl. prompt copies) …")

    sent_topics, topic_info = fit_bertopic(docs, E_fit)
    topic_info.to_csv(sub_dir / "topic_info.csv", index=False)

    # Map topics back onto all sentence occurrences (clean: copies stay NaN).
    seg_out = seg.drop(columns=[c for c in ["topic", "topic_keywords", "topic_majority_in_page"] if c in seg.columns], errors="ignore")
    seg_out = seg_out.merge(sent_topics[["text", "topic"]], on="text", how="left")
    kw = topic_info.set_index("topic")["keywords"].to_dict()
    seg_out["topic_keywords"] = seg_out["topic"].map(kw)

    labelled = int(seg_out["topic"].notna().sum())
    print(
        f"topics={topic_info['topic'].nunique()}  |  "
        f"labelled occurrences={labelled:,}/{len(seg_out):,}  |  "
        f"out → {sub_dir.name}/"
    )
    # Size per topic (unique sentences in the fit)
    print(topic_info[["topic", "n", "keywords"]].to_string(index=False))

    # Topic mix sketch per model (share of labelled sentences)
    mix = (
        seg_out.dropna(subset=["topic"])
        .groupby(["run", "topic"], observed=True)
        .size()
        .groupby("run", observed=True)
        .transform(lambda s: s / s.sum())
        .reset_index(name="share")
        .pivot(index="run", columns="topic", values="share")
        .fillna(0)
        .reindex(cfg.KEEP_RUNS)
    )
    print("topic share by model (rows sum ≈ 1):")
    print(mix.round(2).to_string())

    _save_sentence_outputs(seg_out, topic_info, sub_dir)

    if cfg.SAVE_PLOTS:
        plot_topic_mix(
            seg_out,
            sub_dir / "topic_mix_by_run.png",
            title=f"Topic mix by run ({mode} trace, k={cfg.N_TOPICS})",
        )
        print("saved plot → topic_mix_by_run.png")

    umap_xy = None
    if cfg.SAVE_UMAP:
        umap_xy = plot_umap(
            E_fit,
            sent_topics["topic"].to_numpy(),
            sub_dir / "umap.png",
            title=f"UMAP — BERTopic ({mode} trace, k={cfg.N_TOPICS})",
        )
        np.save(sub_dir / "umap2d.npy", umap_xy)
        print("saved UMAP → umap.png / umap2d.npy")

    return seg_out, topic_info, umap_xy


def main() -> None:
    # One folder per script run, e.g. analysis/topic_modelling/2026-11-28_n5/
    date = datetime.now().strftime("%Y-%m-%d")
    run_dir = cfg.OUTPUT_DIR / f"{date}_n{cfg.N_TOPICS}"
    if run_dir.exists():
        # Same day + same n_topics → add time so runs do not overwrite each other.
        run_dir = cfg.OUTPUT_DIR / f"{date}_{datetime.now().strftime('%H%M%S')}_n{cfg.N_TOPICS}"
    run_dir.mkdir(parents=True, exist_ok=True)
    save_run_config(run_dir)
    print(f"run folder → {run_dir}")
    print(
        f"n_topics={cfg.N_TOPICS}  threshold={cfg.THRESHOLD_PROMPT_COPY}  "
        f"full={cfg.RUN_FULL_TRACE}  clean={cfg.RUN_CLEAN_TRACE}  excel={cfg.SAVE_EXCEL}"
    )

    # Load traces and split into sentences
    pages, seg = load_traces()
    print_load_diagnostics(pages, seg)

    # Gold / correctness / answer — from output/ run folders (eval + raw_results)
    print("\n--- page outcomes (from output/ runs) ---")
    outcomes = build_page_outcomes(pages)
    pages = enrich_pages_with_outcomes(pages, outcomes)
    seg = attach_page_meta(seg, pages)
    if "page_outcome" in seg.columns:
        by_run = (
            pages.assign(has=pages["page_outcome"].notna())
            .groupby("run", observed=True)["has"]
            .mean()
            .map(lambda x: f"{100 * x:.0f}%")
        )
        print("outcome coverage by model:", by_run.to_dict())

    # Embeddings (shared cache across runs) + prompt-copy flags
    print("\n--- embeddings ---")
    cache_dir = cfg.EMBEDDINGS_CACHE_DIR if cfg.CACHE_EMBEDDINGS else None
    print(f"cache dir: {cache_dir if cache_dir else '(disabled)'}")
    sent, E = embed_unique_sentences(seg["text"].tolist(), cache_dir=cache_dir)
    print(f"embedding matrix: {E.shape[0]:,} × {E.shape[1]}")
    sent = score_prompt_copies(pages, sent, E)
    seg = attach_prompt_flags(seg, sent)

    # Analyse prompt copies
    per_trace, per_run = analyse_prompt_copies(seg)
    per_trace.to_csv(run_dir / "prompt_copy_per_trace.csv", index=False)
    per_run.to_csv(run_dir / "prompt_copy_per_run.csv", index=False)
    print_prompt_diagnostics(seg, per_run)

    if cfg.SAVE_PROMPT_COPIES:
        copies = seg.loc[seg["prompt_copy"]].copy()
        copies.to_csv(run_dir / cfg.SAVE_PROMPT_COPIES_NAME, index=False)
        print(f"saved prompt copies → {cfg.SAVE_PROMPT_COPIES_NAME} ({len(copies):,} rows)")

    if cfg.SAVE_PLOTS:
        plot_prompt_copy_share(per_trace, run_dir / "prompt_copy_share_boxplot.png")
        print("saved plot → prompt_copy_share_boxplot.png")

    # Topic modelling: full and/or clean
    if cfg.RUN_FULL_TRACE:
        run_topic_fit("full", seg, sent, E, run_dir)

    if cfg.RUN_CLEAN_TRACE:
        run_topic_fit("clean", seg, sent, E, run_dir)

    print(f"\ndone. → {run_dir}")


if __name__ == "__main__":
    main()
