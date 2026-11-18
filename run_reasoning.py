"""Run climatextract with a chosen model and reasoning effort.

Uses the original ``extract_and_evaluate`` entry point (same MLflow logging
as ``sample_run_litellm.py``). Adds on top:
- live per-PDF reasoning traces (survive an aborted run)
- ``--resume`` / ``--redo-failed`` for aborted or partially failed runs
- a short ``run_summary.json`` with reasoning-trace totals

HTTP status codes are logged by the pipeline itself into ``http_errors.log``.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import re
import shutil
import statistics
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import mlflow
from dotenv import load_dotenv

from climatextract import _runtime_config
from climatextract.adapters.litellm_gateway import (
    LiteLLMGatewayEmbeddingHandler,
    LiteLLMGatewayLlmHandler,
)
from climatextract.console import contextvars_filename
from climatextract.data_lake_manager import DataLakeManager
from climatextract.evaluator import evaluate as evaluate_results
from climatextract.main import _load_config, extract_and_evaluate
from climatextract.pipeline import ValueRetrieverPipeline, save_results
from climatextract.semantic_search import EmbeddingsRepository

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_PDF_DIR = PROJECT_ROOT / "data" / "pdfs" / "sample_160"
EXPECTED_CORPUS_SIZE = 139
DEFAULT_EMBEDDINGS_DB = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "embeddings"
    / "text-embedding-ada-002_from_2025_12_23.duckdb"
)
GOLD_STANDARD = (
    PROJECT_ROOT / "data" / "evaluation_dataset" / "gold_standard.csv"
)

MODEL_CHOICES = ("gpt-6-luna", "gpt-oss-120b", "DeepSeek-V4-Flash-0731")
REASONING_CHOICES = ("none", "low", "medium", "high", "max")
# "none" is only sent to the API for models listed in EXPLICIT_NONE_MODELS;
# for the others it means "omit the reasoning_effort parameter".
ALLOWED_REASONING = {
    "gpt-6-luna": ("none", "low", "medium", "high"),
    "gpt-oss-120b": ("low", "medium", "high"),
    "DeepSeek-V4-Flash-0731": ("none", "high", "max"),
}
EXPLICIT_NONE_MODELS = ("DeepSeek-V4-Flash-0731",)

DEFAULT_PARALLEL_LLM_CALLS = 20
EMBEDDING_PARALLEL_CALLS = 30
REASONING_TRACES_DIRNAME = "reasoning_traces"
PDF_RESULTS_DIRNAME = "pdf_results"
HTTP_ERRORS_FILENAME = "http_errors.log"


# ---------------------------------------------------------------------------
# Reasoning traces
# ---------------------------------------------------------------------------

def _pdf_basename(path: str) -> str:
    """Return the file name of a POSIX or Windows path.

    Args:
        path: Full path or file name. Older runs stored Windows paths.

    Returns:
        The trailing file name component.
    """
    return re.split(r"[\\/]", path)[-1]


def reasoning_trace_path(directory: Path, pdf: str) -> Path:
    """Build the JSONL path for one PDF's reasoning trace.

    Args:
        directory: Folder that holds per-PDF ``.jsonl`` files.
        pdf: PDF path or file name.

    Returns:
        ``<directory>/<pdf-stem>.jsonl``.
    """
    stem = re.sub(r"\.pdf$", "", _pdf_basename(pdf), flags=re.IGNORECASE)
    return directory / f"{stem}.jsonl"


def reasoning_record(
    pdf: str,
    model: str,
    reasoning_effort: str | None,
    messages,
    response: dict | None,
    start_ns: int | None,
    end_ns: int | None,
    cost_usd: float | None,
    error: str | None = None,
) -> dict:
    """Build one LLM-call record for a per-PDF reasoning trace.

    Args:
        pdf: PDF path or file name the call belongs to.
        model: Gateway model name.
        reasoning_effort: Reasoning level sent to the model, or ``None``.
        messages: Chat messages passed to the LLM (last user message is stored).
        response: LiteLLM response as a dict, or ``None`` on failure.
        start_ns: Call start time in nanoseconds since the Unix epoch.
        end_ns: Call end time in nanoseconds since the Unix epoch.
        cost_usd: Reported USD cost for the call, or ``None``.
        error: Error string if the call failed after all retries.

    Returns:
        A JSON-serializable dict with timing, tokens, cost, prompt, answer
        and optional ``reasoning_content``.
    """
    response = response or {}
    choice = (response.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    usage = response.get("usage") or {}
    completion_details = usage.get("completion_tokens_details") or {}
    reasoning_content = message.get("reasoning_content")
    prompt = None
    if isinstance(messages, list) and messages:
        prompt = messages[-1].get("content")
    return {
        "pdf": _pdf_basename(pdf),
        "model": model,
        "reasoning_effort": reasoning_effort,
        "started_at": (
            datetime.fromtimestamp(start_ns / 1e9, tz=timezone.utc).isoformat()
            if start_ns is not None else None
        ),
        "duration_seconds": (
            round((end_ns - start_ns) / 1e9, 6)
            if start_ns is not None and end_ns is not None else None
        ),
        "finish_reason": choice.get("finish_reason"),
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "reasoning_tokens": completion_details.get("reasoning_tokens"),
        "total_tokens": usage.get("total_tokens"),
        "cost_usd": cost_usd,
        "reasoning_chars": len(reasoning_content) if reasoning_content else 0,
        "reasoning_content": reasoning_content,
        "content": message.get("content"),
        "prompt": prompt,
        "error": error,
    }


def append_reasoning_record(directory: Path, record: dict) -> None:
    """Append one reasoning-trace record to the PDF's JSONL file.

    Args:
        directory: Folder that holds per-PDF ``.jsonl`` files.
        record: Output of ``reasoning_record``.
    """
    directory.mkdir(parents=True, exist_ok=True)
    with reasoning_trace_path(directory, record["pdf"]).open(
        "a", encoding="utf-8", newline="\n"
    ) as file:
        file.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")


def reasoning_trace_totals(directory: Path) -> dict:
    """Aggregate timing, token and cost totals over all reasoning files.

    Args:
        directory: Folder with per-PDF ``.jsonl`` reasoning traces.

    Returns:
        Dict with call counts, duration stats, token sums and total USD cost.
        Failed calls (non-empty ``error``) are counted but excluded from sums.
    """
    records = [
        json.loads(line)
        for path in sorted(directory.glob("*.jsonl"))
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    ok = [record for record in records if not record.get("error")]
    durations = sorted(
        record["duration_seconds"] for record in ok
        if record.get("duration_seconds") is not None
    )

    def total(key: str) -> int | float:
        return sum(record.get(key) or 0 for record in ok)

    return {
        "pdf_count": len(list(directory.glob("*.jsonl"))),
        "llm_calls": len(ok),
        "failed_llm_calls": len(records) - len(ok),
        "llm_call_seconds_total": round(sum(durations), 3),
        "llm_call_seconds_mean": (
            round(sum(durations) / len(durations), 3) if durations else None
        ),
        "llm_call_seconds_median": (
            round(statistics.median(durations), 3) if durations else None
        ),
        "prompt_tokens": total("prompt_tokens"),
        "completion_tokens": total("completion_tokens"),
        "reasoning_tokens": total("reasoning_tokens"),
        "reasoning_chars": total("reasoning_chars"),
        "cost_usd": round(total("cost_usd"), 6),
    }


def write_json(path: Path, payload) -> None:
    """Write ``payload`` as pretty-printed JSON.

    Args:
        path: Destination file.
        payload: JSON-serializable object.
    """
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False, default=str)


# ---------------------------------------------------------------------------
# Live recording + resume checkpoints
# ---------------------------------------------------------------------------

def active_run_dir(output_dir: Path) -> Path:
    """Return the results folder of the active MLflow run.

    Args:
        output_dir: Base output directory, e.g. ``output/<model>_<reasoning>``.

    Returns:
        ``output_dir/<run_id>``, or ``output_dir/no_mlflow_run`` if no run is active.
    """
    active_run = mlflow.active_run()
    return output_dir / (active_run.info.run_id if active_run else "no_mlflow_run")


class ReasoningRecordingLlmHandler(LiteLLMGatewayLlmHandler):
    """LiteLLM gateway handler that records every LLM call to a per-PDF JSONL file.

    Uses the pipeline's ``contextvars_filename`` so each call is attributed to
    the PDF currently being processed. Traces are written as soon as the call
    returns, so they survive an aborted run.
    """

    def __init__(self, output_dir: Path, reasoning_effort: str, **extra_params):
        """Create the handler.

        Args:
            output_dir: Base output directory for this model/reasoning level.
            reasoning_effort: Reasoning level stored in each trace record.
            **extra_params: Forwarded to ``LiteLLMGatewayLlmHandler``.
        """
        super().__init__(**extra_params)
        self.output_dir = output_dir
        self.reasoning_effort = reasoning_effort
        self.traces_dir: Path | None = None

    def _resolve_traces_dir(self) -> Path:
        """Return the run's ``reasoning_traces`` folder, creating the path lazily."""
        if self.traces_dir is None:
            self.traces_dir = active_run_dir(self.output_dir) / REASONING_TRACES_DIRNAME
        return self.traces_dir

    async def aget_completion_and_cost(self, messages: list[dict]):
        """Call the gateway and append a reasoning-trace record.

        Args:
            messages: Chat messages for the LLM.

        Returns:
            ``(response, cost_usd)`` from the gateway handler.

        Raises:
            Exception: Re-raises gateway errors after writing an error record.
        """
        pdf = contextvars_filename.get() or "unknown_pdf"
        start_ns = time.time_ns()
        try:
            response, cost = await super().aget_completion_and_cost(messages)
        except Exception as error:
            append_reasoning_record(
                self._resolve_traces_dir(),
                reasoning_record(
                    pdf, self.model_name, self.reasoning_effort,
                    messages, None, start_ns, time.time_ns(), None,
                    error=str(error),
                ),
            )
            raise
        append_reasoning_record(
            self._resolve_traces_dir(),
            reasoning_record(
                pdf, self.model_name, self.reasoning_effort,
                messages, response.model_dump(), start_ns, time.time_ns(),
                float(cost or 0.0),
            ),
        )
        return response, cost


def pdf_result_path(directory: Path, pdf: str) -> Path:
    """Build the pickle path for one PDF's stored result tables.

    Args:
        directory: Folder that holds per-PDF ``.pkl`` checkpoints.
        pdf: PDF path or file name.

    Returns:
        ``<directory>/<pdf-stem>.pkl``.
    """
    stem = re.sub(r"\.pdf$", "", _pdf_basename(pdf), flags=re.IGNORECASE)
    return directory / f"{stem}.pkl"


def instrument_pipeline(output_dir: Path) -> None:
    """Patch the pipeline so each finished PDF is stored as a resume checkpoint.

    Wraps ``ValueRetrieverPipeline.retrieve_values_per_doc`` once. The pipeline
    already sets ``contextvars_filename`` for HTTP-error logging and for the
    reasoning-trace handler.

    Args:
        output_dir: Base output directory for this model/reasoning level.
    """
    original = ValueRetrieverPipeline.retrieve_values_per_doc
    if getattr(original, "_instrumented", False):
        return

    async def retrieve_values_per_doc(self, filename: str):
        result = await original(self, filename)
        table_results = result[0]
        if isinstance(table_results, dict) and table_results:
            directory = active_run_dir(output_dir) / PDF_RESULTS_DIRNAME
            directory.mkdir(parents=True, exist_ok=True)
            with pdf_result_path(directory, filename).open("wb") as file:
                pickle.dump(table_results, file)
        return result

    retrieve_values_per_doc._instrumented = True
    ValueRetrieverPipeline.retrieve_values_per_doc = retrieve_values_per_doc


# ---------------------------------------------------------------------------
# Resume / redo-failed
# ---------------------------------------------------------------------------

def find_previous_results(resume_dirs: list[Path]) -> dict[str, Path]:
    """Map PDF stem to stored result tables from earlier run parts.

    Args:
        resume_dirs: Run folders that contain ``pdf_results/*.pkl``.

    Returns:
        ``{pdf_stem: path_to_pkl}``. Later folders overwrite earlier ones.
    """
    previous: dict[str, Path] = {}
    for run_dir in resume_dirs:
        for path in sorted((run_dir / PDF_RESULTS_DIRNAME).glob("*.pkl")):
            previous[path.stem] = path
    return previous


def failed_pdf_stems(previous: dict[str, Path]) -> set[str]:
    """Find PDFs with at least one LLM call that failed after all retries.

    Args:
        previous: Map from ``find_previous_results``.

    Returns:
        Set of PDF stems whose reasoning trace contains a non-empty ``error``.
    """
    failed = set()
    for stem, source in previous.items():
        trace = source.parent.parent / REASONING_TRACES_DIRNAME / f"{stem}.jsonl"
        if not trace.is_file():
            continue
        with trace.open(encoding="utf-8") as file:
            if any(
                line.strip() and json.loads(line).get("error")
                for line in file
            ):
                failed.add(stem)
    return failed


def merge_previous_parts(
    results_dir: Path,
    previous: dict[str, Path],
    pdf_paths: list[Path],
) -> None:
    """Copy earlier results into this run and rebuild the final result files.

    Copies ``pdf_results``, ``reasoning_traces`` and relevant ``http_errors.log``
    lines, then rewrites the CSV result files via the pipeline's ``save_results``.

    Args:
        results_dir: Current run folder that receives the merged files.
        previous: Map from ``find_previous_results`` for PDFs not reprocessed.
        pdf_paths: Full list of PDFs that the final results should cover.
    """
    results_store = results_dir / PDF_RESULTS_DIRNAME
    traces_store = results_dir / REASONING_TRACES_DIRNAME
    results_store.mkdir(parents=True, exist_ok=True)

    for stem, source in previous.items():
        target = results_store / source.name
        if not target.exists():
            shutil.copy2(source, target)
        previous_trace = (
            source.parent.parent / REASONING_TRACES_DIRNAME / f"{stem}.jsonl"
        )
        if previous_trace.is_file():
            traces_store.mkdir(parents=True, exist_ok=True)
            shutil.copy2(previous_trace, traces_store / previous_trace.name)

    # Merge http_errors.log: keep lines for PDFs taken over from earlier parts,
    # then append anything this part already wrote.
    previous_run_dirs = dict.fromkeys(
        source.parent.parent.resolve() for source in previous.values()
    )
    previous_run_dirs.pop(results_dir.resolve(), None)
    lines: list[str] = []
    seen: set[str] = set()
    for run_dir in previous_run_dirs:
        log_path = run_dir / HTTP_ERRORS_FILENAME
        if not log_path.is_file():
            continue
        for line in log_path.read_text(encoding="utf-8").splitlines(True):
            match = re.search(r"\[([^\]]+)\]", line)
            if match and Path(match.group(1)).stem in previous and line not in seen:
                lines.append(line)
                seen.add(line)
    current_log = results_dir / HTTP_ERRORS_FILENAME
    if current_log.is_file():
        for line in current_log.read_text(encoding="utf-8").splitlines(True):
            if line not in seen:
                lines.append(line)
                seen.add(line)
    if lines:
        current_log.write_text("".join(lines), encoding="utf-8", newline="\n")

    all_results = []
    for pdf_path in sorted(pdf_paths, key=lambda path: path.name):
        stored = pdf_result_path(results_store, pdf_path.name)
        if stored.is_file():
            with stored.open("rb") as file:
                all_results.append(pickle.load(file))
    save_results(
        raw_results=all_results,
        path_to_results=str(results_dir),
        first_write=True,
        results_type="final",
    )


# ---------------------------------------------------------------------------
# Config for one run
# ---------------------------------------------------------------------------

def build_config(
    pdf_paths: list[Path],
    model_name: str,
    reasoning_effort: str,
    embeddings_db: Path,
    parallel_llm_calls: int,
) -> str:
    """Write a temporary TOML config for one run.

    Leaves ``climatextract.toml`` unchanged. The caller must delete the file.

    Args:
        pdf_paths: PDFs to process in this part of the run.
        model_name: Gateway model name.
        reasoning_effort: Reasoning level (``none`` / ``low`` / … / ``max``).
        embeddings_db: Path to the shared embeddings DuckDB file.
        parallel_llm_calls: ``max_parallel_llm_prompts_running``.

    Returns:
        Path to the temporary TOML file.
    """
    pdf_list = ", ".join(json.dumps(str(path)) for path in pdf_paths)
    if reasoning_effort == "none" and model_name not in EXPLICIT_NONE_MODELS:
        reasoning_value = ""
    else:
        reasoning_value = reasoning_effort
    run_label = f"{model_name}_{reasoning_effort}"
    content = f"""\
[input]
filename_list = [{pdf_list}]

[models]
llm_model = {json.dumps(model_name)}
emb_model = "text-embedding-ada-002"
max_parallel_llm_prompts_running = {parallel_llm_calls}
max_parallel_embedding_calls = {EMBEDDING_PARALLEL_CALLS}
temperature = 0.0
return_logprobs = false
reasoning_effort = {json.dumps(reasoning_value)}

[extraction]
year_min = 2013
year_max = 2024
input_mode = "text"
embed_only = false
prompt_type = "structured_json"
context_window = 0
similarity_top_k = 7
embeddings_repository = {json.dumps(str(embeddings_db))}

[output]
output_dir = {json.dumps(f"output/{run_label}")}

[evaluation]
gold_standard = {json.dumps(str(GOLD_STANDARD))}

[mlflow]
experiment_name = {json.dumps(run_label)}

[datalake]
blob_path_pdfs = "pdfs/sample_160"
blob_path_embeddings = "embeddings"
"""
    config = tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", suffix=".toml", delete=False
    )
    try:
        config.write(content)
        return config.name
    finally:
        config.close()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    """Run extract_and_evaluate with reasoning-trace recording and optional resume.

    Returns:
        Process exit code (``0`` on success or user cancel, non-zero on error
        via ``parser.error`` / uncaught exceptions).
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pdf",
        type=Path,
        nargs="+",
        default=None,
        help=(
            "PDF file(s) to process. Default: all PDFs in "
            f"{DEFAULT_PDF_DIR}."
        ),
    )
    parser.add_argument(
        "--model",
        choices=MODEL_CHOICES,
        default="gpt-oss-120b",
        help="Gateway model (default: gpt-oss-120b).",
    )
    parser.add_argument(
        "--reasoning",
        choices=REASONING_CHOICES,
        default="low",
        help=(
            "Reasoning effort (default: low). Allowed values depend on the "
            "model: gpt-oss-120b low/medium/high, DeepSeek-V4-Flash-0731 "
            "none/high/max, gpt-6-luna none/low/medium/high."
        ),
    )
    parser.add_argument(
        "--parallel",
        type=int,
        default=DEFAULT_PARALLEL_LLM_CALLS,
        help=(
            "Maximum concurrent LLM calls "
            f"(default: {DEFAULT_PARALLEL_LLM_CALLS})."
        ),
    )
    parser.add_argument(
        "--embeddings-db",
        type=Path,
        default=DEFAULT_EMBEDDINGS_DB,
        help="Local shared embeddings database path.",
    )
    parser.add_argument(
        "--expected-pdfs",
        type=int,
        default=EXPECTED_CORPUS_SIZE,
        help="Expected PDF count for the default corpus (default: 139).",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        nargs="+",
        default=None,
        metavar="RUN_DIR",
        help=(
            "Folder(s) of aborted runs with the same model and reasoning "
            "level. Only PDFs without stored results are processed. Pass "
            "every earlier part if a resume was aborted too."
        ),
    )
    parser.add_argument(
        "--redo-failed",
        action="store_true",
        help=(
            "With --resume: reprocess every PDF that has a failed LLM call "
            "in its reasoning trace; all other PDFs are taken over unchanged."
        ),
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Skip the interactive confirmation prompt.",
    )
    args = parser.parse_args()

    if args.parallel < 1:
        parser.error("--parallel must be at least 1.")
    if args.redo_failed and not args.resume:
        parser.error("--redo-failed requires --resume.")
    allowed_reasoning = ALLOWED_REASONING[args.model]
    if args.reasoning not in allowed_reasoning:
        parser.error(
            f"--reasoning {args.reasoning} is not supported by {args.model}; "
            f"allowed: {', '.join(allowed_reasoning)}."
        )

    # ---- PDFs --------------------------------------------------------------
    if args.pdf is None:
        pdf_paths = sorted(DEFAULT_PDF_DIR.glob("*.pdf"))
        if not pdf_paths:
            print(
                f"No PDFs in {DEFAULT_PDF_DIR}; downloading from Azure Blob "
                "Storage (storage download, not a model call)."
            )
            load_dotenv(PROJECT_ROOT / ".env")
            DataLakeManager(
                blob_path_pdfs="pdfs/sample_160",
                blob_path_embeddings="embeddings",
            ).download_directory_from_blob(str(DEFAULT_PDF_DIR))
            pdf_paths = sorted(DEFAULT_PDF_DIR.glob("*.pdf"))
        if len(pdf_paths) != args.expected_pdfs:
            parser.error(
                f"Expected {args.expected_pdfs} PDFs in {DEFAULT_PDF_DIR}, "
                f"found {len(pdf_paths)}."
            )
    else:
        pdf_paths = args.pdf
    pdf_paths = [path.resolve() for path in pdf_paths]
    missing = [path for path in pdf_paths if not path.is_file()]
    if missing:
        parser.error("PDF not found: " + ", ".join(str(path) for path in missing))

    # ---- Resume ------------------------------------------------------------
    run_label = f"{args.model}_{args.reasoning}"
    all_pdf_paths = pdf_paths
    previous_results: dict[str, Path] = {}
    resume_dirs = [path.resolve() for path in args.resume or []]
    for run_dir in resume_dirs:
        if not run_dir.is_dir():
            parser.error(f"--resume folder not found: {run_dir}")
        if run_dir.parent.name != run_label:
            parser.error(
                f"--resume folder {run_dir} belongs to {run_dir.parent.name}, "
                f"not to {run_label}."
            )
        if not any((run_dir / PDF_RESULTS_DIRNAME).glob("*.pkl")):
            parser.error(
                f"{run_dir} has no {PDF_RESULTS_DIRNAME}/; "
                "runs started before resume support cannot be resumed."
            )
    if resume_dirs:
        previous_results = {
            stem: path
            for stem, path in find_previous_results(resume_dirs).items()
            if stem in {pdf.stem for pdf in all_pdf_paths}
        }
        if args.redo_failed:
            failed = failed_pdf_stems(previous_results)
            if not failed:
                parser.error("No PDF in the given run(s) has a failed LLM call.")
            print(
                f"Redoing {len(failed)} PDFs with failed LLM calls: "
                + ", ".join(sorted(failed))
            )
            previous_results = {
                stem: path for stem, path in previous_results.items()
                if stem not in failed
            }
        pdf_paths = [
            path for path in all_pdf_paths if path.stem not in previous_results
        ]
        if not pdf_paths:
            parser.error("All requested PDFs are already done in the given run(s).")
        print(
            f"Resuming: {len(previous_results)} PDFs done in earlier part(s), "
            f"{len(pdf_paths)} remain. Wall-clock time covers only this part."
        )

    # ---- Environment -------------------------------------------------------
    load_dotenv(PROJECT_ROOT / ".env")
    if not os.getenv("LITELLM_GATEWAY_API_BASE"):
        parser.error("Set LITELLM_GATEWAY_API_BASE in .env or the environment.")
    if not os.getenv("LITELLM_GATEWAY_API_KEY"):
        parser.error("Set LITELLM_GATEWAY_API_KEY in .env or the environment.")

    tracking_dir = PROJECT_ROOT / "mlruns"
    tracking_dir.mkdir(parents=True, exist_ok=True)
    # Keep the same local SQLite store as the earlier DeepSeek high/max runs.
    # This does not change extraction; only where MLflow writes its DB.
    # Without this, the pipeline would use MLFLOW_TRACKING_URI from .env
    # or fall back to ./mlruns (file store).
    os.environ["MLFLOW_TRACKING_URI"] = "sqlite:///mlruns/mlflow.db"
    os.environ["MLFLOW_ENABLE_ARTIFACTS_PROGRESS_BAR"] = "false"
    # Keep traces buffered for long batch runs.
    os.environ["MLFLOW_TRACE_TIMEOUT_SECONDS"] = "0"
    os.environ["MLFLOW_TRACE_BUFFER_TTL_SECONDS"] = str(7 * 24 * 60 * 60)
    os.environ["MLFLOW_TRACE_BUFFER_MAX_SIZE"] = str(max(1000, len(pdf_paths) + 1))
    os.chdir(PROJECT_ROOT)

    embeddings_db = args.embeddings_db.resolve()
    config_path = build_config(
        pdf_paths=pdf_paths,
        model_name=args.model,
        reasoning_effort=args.reasoning,
        embeddings_db=embeddings_db,
        parallel_llm_calls=args.parallel,
    )
    try:
        _, params, _, _, _ = _load_config(config_path)
        _runtime_config.set_current(params)

        embeddings = EmbeddingsRepository(database_name=str(embeddings_db))
        if not embeddings.database_exists():
            print(
                f"Embeddings database not found at {embeddings_db}; "
                "offering download from Azure Blob Storage."
            )
            DataLakeManager(
                blob_path_pdfs="pdfs/sample_160",
                blob_path_embeddings="embeddings",
            ).handle_embedding_database(embeddings)
            embeddings = EmbeddingsRepository(database_name=str(embeddings_db))

        if embeddings.database_exists():
            missing_embeddings = [
                path.name for path in pdf_paths
                if not embeddings.pdf_exists(path.name)
            ]
            if missing_embeddings:
                print(
                    f"{len(missing_embeddings)} PDFs have no cached embeddings "
                    "and will be embedded during the run."
                )
        else:
            print(f"No embeddings database; a new one will be created at {embeddings_db}.")

        print(
            f"\nModel: {args.model}\n"
            f"Reasoning: {args.reasoning}\n"
            f"PDFs this part: {len(pdf_paths)}\n"
            f"Parallel LLM calls: {args.parallel}"
        )
        if not args.yes:
            answer = input("\nStart the run and make the LLM calls? [y/N] ").strip().lower()
            if answer != "y":
                print("Batch not started; no model calls were made.")
                return 0

        run_output_dir = PROJECT_ROOT / "output" / run_label
        instrument_pipeline(run_output_dir)
        llm = ReasoningRecordingLlmHandler(
            output_dir=run_output_dir,
            reasoning_effort=args.reasoning,
        )
        embedder = LiteLLMGatewayEmbeddingHandler()

        print(f"Starting extract_and_evaluate for {len(pdf_paths)} PDF(s)...")
        started = time.perf_counter()
        result_path = extract_and_evaluate(
            pdf_input=[str(path) for path in pdf_paths],
            gold_standard_path=str(GOLD_STANDARD),
            config_path=config_path,
            enable_mlflow=True,
            verbose=True,
            llm=llm,
            embedder=embedder,
        )
        elapsed_seconds = time.perf_counter() - started
        if result_path is None:
            raise RuntimeError("Extraction failed; no results directory was returned.")

        results_dir = Path(result_path)
        # extract_and_evaluate writes config_and_metrics.json (not config.json).
        config_path_out = results_dir / "config_and_metrics.json"
        if not config_path_out.is_file():
            config_path_out = results_dir / "config.json"
        with config_path_out.open(encoding="utf-8") as file:
            run_config = json.load(file)
        run_id = run_config["run_info"]["run_id"]

        cost_metric_keys = {
            "embedding_tokens",
            "llm_prompt_tokens",
            "llm_completion_tokens",
            "llm_reasoning_tokens",
            "total_llm_token_count",
            "total_llm_costs_in_dollar",
        }

        if previous_results:
            # extract_and_evaluate evaluated only this part; rebuild results
            # and metrics for the full corpus after merging earlier parts.
            print(
                f"Merging results of {len(previous_results)} PDFs from earlier "
                "part(s) into this run..."
            )
            merge_previous_parts(results_dir, previous_results, all_pdf_paths)
            print(f"Re-evaluating full corpus against {GOLD_STANDARD}...")
            evaluation_metrics = evaluate_results(
                path_to_results=str(results_dir),
                gold_standard=str(GOLD_STANDARD),
            )
            evaluation_metrics = {
                key: value.item() if hasattr(value, "item") else value
                for key, value in evaluation_metrics.items()
            }
            run_config["metrics"].update(evaluation_metrics)
            write_json(results_dir / "config_and_metrics.json", run_config)
            with mlflow.start_run(run_id=run_id):
                mlflow.log_metrics(
                    {
                        key: value for key, value in evaluation_metrics.items()
                        if isinstance(value, (int, float))
                    }
                )
                mlflow.log_artifacts(str(results_dir))
        else:
            # Metrics written by extract_and_evaluate into config_and_metrics.json.
            evaluation_metrics = {
                key: value for key, value in run_config.get("metrics", {}).items()
                if key not in cost_metric_keys
            }

        write_json(results_dir / "evaluation_metrics.json", evaluation_metrics)

        trace_totals = reasoning_trace_totals(results_dir / REASONING_TRACES_DIRNAME)
        write_json(results_dir / "reasoning_trace_totals.json", trace_totals)

        summary = {
            "pdf": [str(path) for path in all_pdf_paths],
            "model": args.model,
            "reasoning_effort": args.reasoning,
            "parallel_llm_calls": args.parallel,
            "wall_clock_seconds": round(elapsed_seconds, 3),
            "mlflow_run_id": run_id,
            "results_directory": str(results_dir.resolve()),
            "reported_costs": run_config["metrics"],
            "reasoning_trace_totals": trace_totals,
            "resumed_from": [str(path) for path in resume_dirs],
            "pdfs_from_earlier_parts": len(previous_results),
            "pdfs_this_part": len(pdf_paths),
            "scope_notes": (
                "wall_clock_seconds and reported_costs cover only this part; "
                "results, evaluation and reasoning_traces cover all PDFs."
                if previous_results else None
            ),
            "evaluation_metrics": evaluation_metrics,
        }
        write_json(results_dir / "run_summary.json", summary)
        with mlflow.start_run(run_id=run_id):
            mlflow.log_metric("wall_clock_seconds", elapsed_seconds)
            mlflow.log_artifact(str(results_dir / "run_summary.json"))
            mlflow.log_artifact(str(results_dir / "reasoning_trace_totals.json"))

        print(
            "Evaluation metrics: "
            f"precision={evaluation_metrics.get('precision_value', 0):.4f}, "
            f"recall={evaluation_metrics.get('recall_value', 0):.4f}, "
            f"F1={evaluation_metrics.get('f1_value', 0):.4f}"
        )
        print(f"Wall-clock time: {elapsed_seconds:.1f} s")
        print(f"Results and run summary: {results_dir.resolve()}")
        print(
            "Per-PDF reasoning traces: "
            f"{(results_dir / REASONING_TRACES_DIRNAME).resolve()}"
        )
        print(f"HTTP errors: {(results_dir / HTTP_ERRORS_FILENAME).resolve()}")
        print(f"MLflow run ID: {run_id}")
        return 0
    finally:
        Path(config_path).unlink(missing_ok=True)


if __name__ == "__main__":
    raise SystemExit(main())
