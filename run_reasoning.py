"""Run a confirmed batch extraction and store per-PDF MLflow traces locally."""

import argparse
import contextvars
import csv
import json
import os
from pathlib import Path
import pickle
import re
import shutil
import sqlite3
import statistics
import tempfile
import time
from collections import Counter
from datetime import datetime, timezone

from dotenv import load_dotenv
import litellm
import mlflow

from climatextract import _runtime_config
from climatextract.adapters.litellm_gateway import (
    NUM_RETRIES,
    LiteLLMGatewayEmbeddingHandler,
    LiteLLMGatewayLlmHandler,
    describe_error,
)
from climatextract.data_lake_manager import DataLakeManager
from climatextract.evaluator import evaluate as evaluate_results
from climatextract.main import _load_config, extract
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
REFERENCE_RUN_ID = "3473a1ff9d8d468a9efb945966026085"
MODEL_CHOICES = (
    "gpt-6-luna",
    "gpt-oss-120b",
    "DeepSeek-V4-Flash-0731",
)
REASONING_CHOICES = ("none", "low", "medium", "high", "max")
# Values the provider accepts per model. "none" is only sent explicitly where
# it is listed here; for the other models it means "omit reasoning_effort".
ALLOWED_REASONING = {
    "gpt-6-luna": ("none", "low", "medium", "high"),
    "gpt-oss-120b": ("low", "medium", "high"),
    "DeepSeek-V4-Flash-0731": ("none", "high", "max"),
}
EXPLICIT_NONE_MODELS = ("DeepSeek-V4-Flash-0731",)
DEFAULT_PARALLEL_LLM_CALLS = 20
EMBEDDING_PARALLEL_CALLS = 30
TRACE_BUFFER_TTL_SECONDS = 7 * 24 * 60 * 60
TRACE_BUFFER_MAX_SIZE = 1000
REASONING_TRACES_DIRNAME = "reasoning_traces"
PDF_RESULTS_DIRNAME = "pdf_results"
HTTP_ERRORS_FILENAME = "http_errors.log"

_current_pdf: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "current_pdf", default=None
)


def _pdf_basename(path: str) -> str:
    """File name of a POSIX or Windows path (older runs were made on Windows)."""
    return re.split(r"[\\/]", path)[-1]


def reasoning_trace_path(directory: Path, pdf: str) -> Path:
    stem = re.sub(r"\.pdf$", "", _pdf_basename(pdf), flags=re.IGNORECASE)
    return directory / f"{stem}.jsonl"


def reasoning_record(pdf: str, model: str, reasoning_effort: str | None,
                     messages, response: dict | None, start_ns: int | None,
                     end_ns: int | None, cost_usd: float | None,
                     error: str | None = None) -> dict:
    """One LLM call in the per-PDF reasoning trace format.

    Live recording and the export from MLflow traces both go through this
    function, so files from old and new runs have identical fields.
    """
    response = response or {}
    choices = response.get("choices") or [{}]
    message = choices[0].get("message") or {}
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
        "finish_reason": choices[0].get("finish_reason"),
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
    directory.mkdir(parents=True, exist_ok=True)
    with reasoning_trace_path(directory, record["pdf"]).open(
        "a", encoding="utf-8", newline="\n"
    ) as file:
        file.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")


def active_run_dir(output_dir: Path) -> Path:
    """Results folder of the MLflow run that ``extract`` has started."""
    active_run = mlflow.active_run()
    return output_dir / (active_run.info.run_id if active_run else "no_mlflow_run")


class ReasoningRecordingLlmHandler(LiteLLMGatewayLlmHandler):
    """Gateway handler that appends every LLM call to a per-PDF JSONL file
    as soon as the call returns, so traces survive an aborted run."""

    def __init__(self, output_dir: Path, reasoning_effort: str, **extra_params):
        super().__init__(**extra_params)
        self.output_dir = output_dir
        self.reasoning_effort = reasoning_effort
        self.traces_dir: Path | None = None

    def _resolve_traces_dir(self) -> Path:
        if self.traces_dir is None:
            self.traces_dir = active_run_dir(self.output_dir) / REASONING_TRACES_DIRNAME
        return self.traces_dir

    def on_failed_attempt(self, attempt: int, error: Exception,
                          will_retry: bool) -> None:
        """Log to the console and append to the run's ``http_errors.log`` in
        the format ``read_pipeline_warnings`` groups by PDF."""
        super().on_failed_attempt(attempt, error, will_retry)
        pdf = _pdf_basename(_current_pdf.get() or "unknown_pdf")
        level = "WARNING" if will_retry else "ERROR"
        outcome = "retrying" if will_retry else "giving up"
        log_path = active_run_dir(self.output_dir) / HTTP_ERRORS_FILENAME
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8", newline="\n") as file:
            file.write(
                f"{datetime.now(timezone.utc).isoformat()} {level} [{pdf}] "
                f"{self.model_name}: attempt {attempt}/{NUM_RETRIES + 1} "
                f"({outcome}) {describe_error(error)}\n"
            )

    async def aget_completion_and_cost(self, messages: list[dict]):
        pdf = _current_pdf.get() or "unknown_pdf"
        start_ns = time.time_ns()
        try:
            response, cost = await super().aget_completion_and_cost(messages)
        except Exception as error:
            append_reasoning_record(
                self._resolve_traces_dir(),
                reasoning_record(pdf, self.model_name, self.reasoning_effort,
                                 messages, None, start_ns, time.time_ns(), None,
                                 error=str(error)),
            )
            raise
        end_ns = time.time_ns()
        append_reasoning_record(
            self._resolve_traces_dir(),
            reasoning_record(pdf, self.model_name, self.reasoning_effort,
                             messages, response.model_dump(), start_ns, end_ns,
                             float(cost or 0.0)),
        )
        return response, cost


def pdf_result_path(directory: Path, pdf: str) -> Path:
    stem = re.sub(r"\.pdf$", "", _pdf_basename(pdf), flags=re.IGNORECASE)
    return directory / f"{stem}.pkl"


def instrument_pipeline(output_dir: Path) -> None:
    """Track the current PDF for the LLM handler and store each finished
    PDF's result tables, so an aborted run can be resumed.

    Each PDF runs in its own asyncio task and the per-page LLM calls are
    child tasks, so a context variable set here is seen by exactly the calls
    belonging to this PDF.
    """
    original = ValueRetrieverPipeline.retrieve_values_per_doc
    if getattr(original, "_instrumented", False):
        return

    async def retrieve_values_per_doc(self, filename: str):
        _current_pdf.set(filename)
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


def find_previous_results(resume_dirs: list[Path]) -> dict[str, Path]:
    """Map PDF stem to the stored result tables of earlier run parts.

    Later folders win if a PDF appears in more than one of them.
    """
    previous: dict[str, Path] = {}
    for run_dir in resume_dirs:
        for path in sorted((run_dir / PDF_RESULTS_DIRNAME).glob("*.pkl")):
            previous[path.stem] = path
    return previous


def failed_pdf_stems(previous: dict[str, Path]) -> set[str]:
    """PDFs whose stored result contains at least one LLM call that failed
    after all retries, judged by the reasoning trace of the same run part."""
    failed = set()
    for stem, source in previous.items():
        trace = source.parent.parent / REASONING_TRACES_DIRNAME / f"{stem}.jsonl"
        if not trace.is_file():
            continue
        with trace.open(encoding="utf-8") as file:
            if any(line.strip() and json.loads(line).get("error")
                   for line in file):
                failed.add(stem)
    return failed


def _logged_pdf_stem(line: str) -> str | None:
    match = re.search(r"\b(?:WARNING|ERROR|CRITICAL)\s+\[([^\]]+)\]", line)
    return Path(match.group(1)).stem if match else None


def merge_previous_parts(results_dir: Path, previous: dict[str, Path],
                         pdf_paths: list[Path]) -> None:
    """Copy result tables and reasoning traces of earlier parts into this run
    and rewrite the final result files for all PDFs.

    The final files are produced by the pipeline's own ``save_results``, so
    they are built exactly like those of an uninterrupted run.
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

    previous_run_dirs = dict.fromkeys(
        source.parent.parent.resolve() for source in previous.values()
    )
    previous_run_dirs.pop(results_dir.resolve(), None)
    # Earlier logs may already be merged copies and may mention PDFs redone in
    # this part, so keep each line once and only for PDFs taken over from them.
    merged_lines: dict[str, None] = {}
    for run_dir in previous_run_dirs:
        log_path = run_dir / HTTP_ERRORS_FILENAME
        if log_path.is_file():
            for line in log_path.read_text(encoding="utf-8").splitlines(True):
                if _logged_pdf_stem(line) in previous:
                    merged_lines[line] = None
    current_log = results_dir / HTTP_ERRORS_FILENAME
    if current_log.is_file():
        merged_lines.update(
            dict.fromkeys(current_log.read_text(encoding="utf-8").splitlines(True))
        )
    merged_errors = "".join(merged_lines)
    if merged_errors:
        (results_dir / HTTP_ERRORS_FILENAME).write_text(
            merged_errors, encoding="utf-8", newline="\n"
        )

    all_results = []
    for pdf_path in sorted(pdf_paths, key=lambda path: path.name):
        stored = pdf_result_path(results_store, pdf_path.name)
        if stored.is_file():
            with stored.open("rb") as file:
                all_results.append(pickle.load(file))
    save_results(raw_results=all_results, path_to_results=str(results_dir),
                 first_write=True, results_type="final")


def reasoning_trace_totals(directory: Path) -> dict:
    """Per-call timing, token and cost totals over all per-PDF reasoning files."""
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


def export_reasoning_traces(traces: list[dict], directory: Path,
                            model: str, reasoning_effort: str | None) -> int:
    """Write per-PDF reasoning files from MLflow traces (e.g. ``traces.json``).

    Returns the number of LLM calls written. Existing files are replaced.
    """
    records_by_pdf: dict[str, list[tuple[int, dict]]] = {}
    for trace in traces:
        spans = trace["spans"]
        spans_by_id = {span["span_id"]: span for span in spans}
        for span in spans:
            if span.get("type") != "LLM":
                continue
            pdf = None
            parent = spans_by_id.get(span.get("parent_span_id"))
            while parent is not None:
                if parent.get("name") == "retrieve_values_per_doc":
                    pdf = _span_input_filename(parent)
                    break
                parent = spans_by_id.get(parent.get("parent_span_id"))
            if pdf is None:
                continue
            attributes = _span_attributes(span)
            inputs = _decode_json(attributes.get("mlflow.spanInputs")) or {}
            outputs = _decode_json(attributes.get("mlflow.spanOutputs"))
            record = reasoning_record(
                pdf,
                str(_decode_json(attributes.get("model")) or model),
                reasoning_effort,
                inputs.get("messages") if isinstance(inputs, dict) else None,
                outputs if isinstance(outputs, dict) else None,
                span.get("start_time_unix_nano"),
                span.get("end_time_unix_nano"),
                _span_number_attribute(span, "response_cost"),
                error=None if span.get("status") == "OK" else str(span.get("status")),
            )
            records_by_pdf.setdefault(record["pdf"], []).append(
                (span.get("start_time_unix_nano") or 0, record)
            )

    directory.mkdir(parents=True, exist_ok=True)
    for pdf, records in records_by_pdf.items():
        with reasoning_trace_path(directory, pdf).open(
            "w", encoding="utf-8", newline="\n"
        ) as file:
            for _, record in sorted(records, key=lambda item: item[0]):
                file.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
    return sum(len(records) for records in records_by_pdf.values())


def _decode_json(value):
    """Decode JSON-backed MLflow span content while preserving plain strings."""
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def load_run_traces(database_path: Path, run_id: str) -> list[dict]:
    """Read the full MLflow trace records associated with one run."""
    if not database_path.is_file():
        raise FileNotFoundError(f"MLflow database not found: {database_path}")

    with sqlite3.connect(database_path) as connection:
        table_names = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }
        required_tables = {
            "trace_info",
            "trace_request_metadata",
            "trace_tags",
            "spans",
        }
        missing_tables = required_tables - table_names
        if missing_tables:
            raise RuntimeError(
                "MLflow database does not contain trace tables: "
                + ", ".join(sorted(missing_tables))
            )

        request_ids = [
            row[0]
            for row in connection.execute(
                """
                SELECT request_id
                FROM trace_request_metadata
                WHERE key = ? AND value = ?
                """,
                ("mlflow.sourceRun", run_id),
            )
        ]
        traces = []
        for request_id in request_ids:
            info_cursor = connection.execute(
                "SELECT * FROM trace_info WHERE request_id = ?",
                (request_id,),
            )
            info_columns = [column[0] for column in info_cursor.description]
            info_row = info_cursor.fetchone()
            if info_row is None:
                continue
            info = dict(zip(info_columns, info_row))

            tags = dict(
                connection.execute(
                    "SELECT key, value FROM trace_tags WHERE request_id = ?",
                    (request_id,),
                ).fetchall()
            )
            metadata = dict(
                connection.execute(
                    """
                    SELECT key, value
                    FROM trace_request_metadata
                    WHERE request_id = ?
                    """,
                    (request_id,),
                ).fetchall()
            )

            spans_cursor = connection.execute(
                """
                SELECT *
                FROM spans
                WHERE trace_id = ?
                ORDER BY start_time_unix_nano, span_id
                """,
                (request_id,),
            )
            span_columns = [column[0] for column in spans_cursor.description]
            spans = []
            for span_row in spans_cursor.fetchall():
                span = dict(zip(span_columns, span_row))
                span["content"] = _decode_json(span["content"])
                spans.append(span)

            traces.append(
                {
                    "trace_id": request_id,
                    "info": info,
                    "tags": tags,
                    "metadata": metadata,
                    "spans": spans,
                }
            )

    traces.sort(key=lambda trace: trace["info"].get("timestamp_ms") or 0)
    if not traces:
        raise RuntimeError(
            f"No MLflow traces were recorded for run {run_id}."
        )
    return traces


def configure_mlflow_trace_buffer(document_count: int) -> None:
    """Keep long-running batch traces buffered until MLflow closes them."""
    os.environ["MLFLOW_TRACE_TIMEOUT_SECONDS"] = "0"
    os.environ["MLFLOW_TRACE_BUFFER_TTL_SECONDS"] = str(
        TRACE_BUFFER_TTL_SECONDS
    )
    os.environ["MLFLOW_TRACE_BUFFER_MAX_SIZE"] = str(
        max(TRACE_BUFFER_MAX_SIZE, document_count + 1)
    )


def estimate_llm_cost(
    model_name: str,
    document_count: int,
    reference_trace: dict,
) -> dict:
    """Scale observed per-PDF token usage using LiteLLM's Azure AI price map."""
    model_key = f"azure_ai/{model_name}"
    model_info = litellm.model_cost.get(model_key)
    if model_info is None:
        raise RuntimeError(
            f"No Azure AI pricing entry is available for {model_key}; "
            "refusing to show an unsupported cost estimate."
        )

    input_rate = model_info.get("input_cost_per_token")
    output_rate = model_info.get("output_cost_per_token")
    if not isinstance(input_rate, (int, float)) or not isinstance(
        output_rate, (int, float)
    ):
        raise RuntimeError(
            f"Pricing for {model_key} is incomplete; refusing to estimate."
        )

    llm_spans = [
        span for span in reference_trace["spans"] if span.get("type") == "LLM"
    ]
    input_tokens = 0
    output_tokens = 0
    for span in llm_spans:
        usage = _span_usage(span)
        if not usage or "input_tokens" not in usage or "output_tokens" not in usage:
            raise RuntimeError(
                "Reference trace is missing token usage; cannot estimate costs."
            )
        input_tokens += int(usage["input_tokens"])
        output_tokens += int(usage["output_tokens"])
    if not llm_spans:
        raise RuntimeError("Reference trace has no LLM spans.")

    per_pdf_cost = input_tokens * input_rate + output_tokens * output_rate
    estimate = {
        "model": model_name,
        "pricing_model_key": model_key,
        "input_usd_per_token": input_rate,
        "output_usd_per_token": output_rate,
        "reference_run_id": REFERENCE_RUN_ID,
        "reference_pdf_count": 1,
        "reference_llm_call_count": len(llm_spans),
        "reference_input_tokens": input_tokens,
        "reference_output_tokens": output_tokens,
        "estimated_input_tokens": input_tokens * document_count,
        "estimated_output_tokens": output_tokens * document_count,
        "estimated_llm_calls": len(llm_spans) * document_count,
        "estimated_cost_usd": round(per_pdf_cost * document_count, 6),
        "document_count": document_count,
        "basis": (
            "One completed GPT-OSS test PDF scaled linearly by PDF count; "
            "selected model uses LiteLLM azure_ai catalog rates."
        ),
        "limitations": (
            "This is an estimate, not a cap. PDF lengths, outputs, reasoning "
            "effort, gateway pricing, and retries can change actual charges. "
            "Embedding charges for PDFs missing from the embedding cache are "
            "excluded."
        ),
    }
    return estimate


def load_reference_trace(database_path: Path) -> dict:
    try:
        traces = load_run_traces(database_path, REFERENCE_RUN_ID)
    except (FileNotFoundError, RuntimeError):
        trace_file = (
            PROJECT_ROOT
            / "output"
            / "gpt_oss_test"
            / REFERENCE_RUN_ID
            / "traces.json"
        )
        if not trace_file.is_file():
            raise
        with trace_file.open(encoding="utf-8") as file:
            traces = json.load(file)
        if not traces:
            raise RuntimeError(f"No reference traces found in {trace_file}.")
    return max(
        traces,
        key=lambda trace: trace["info"].get("timestamp_ms") or 0,
    )


def confirm_batch(estimate: dict | None, reasoning_effort: str,
                  model_name: str, document_count: int,
                  parallel_llm_calls: int) -> bool:
    if estimate is None:
        print("\nLLM batch (no cost estimate available)")
        print(f"  PDFs: {document_count}")
        print(f"  Model: {model_name}")
        print(f"  Reasoning effort: {reasoning_effort}")
        print(f"  Parallel LLM calls: {parallel_llm_calls}")
    else:
        print("\nEstimated LLM batch")
        print(f"  PDFs: {estimate['document_count']}")
        print(f"  Model: {estimate['model']}")
        print(f"  Reasoning effort: {reasoning_effort}")
        print(f"  Parallel LLM calls: {parallel_llm_calls}")
        print(f"  Estimated LLM calls: {estimate['estimated_llm_calls']}")
        print(
            "  Estimated tokens: "
            f"{estimate['estimated_input_tokens']:,} input + "
            f"{estimate['estimated_output_tokens']:,} output"
        )
        print(f"  Estimated LLM cost: ${estimate['estimated_cost_usd']:.6f}")
        print(f"  Pricing source: {estimate['pricing_model_key']} (LiteLLM catalog)")
        print(f"  Estimate basis: {estimate['basis']}")
        print(f"  Important: {estimate['limitations']}")
    try:
        answer = input(
            "\nStart all PDFs and make the LLM calls? [y/N] "
        ).strip().lower()
    except EOFError:
        return False
    return answer == "y"


def write_json(path: Path, payload) -> None:
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False, default=str)


def read_result_rows(csv_path: Path) -> list[dict[str, str]]:
    csv.field_size_limit(2**31 - 1)
    with csv_path.open(encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def _normalized_path(value: str | Path) -> str:
    return os.path.normcase(os.path.normpath(str(value)))


def _span_input_filename(span: dict) -> str | None:
    content = span.get("content") or {}
    attributes = content.get("attributes") or {}
    inputs = _decode_json(attributes.get("mlflow.spanInputs"))
    if isinstance(inputs, dict):
        filename = inputs.get("filename")
        if isinstance(filename, str):
            return filename
    return None


def validate_trace_coverage(traces: list[dict], pdf_paths: list[Path]) -> None:
    """Fail explicitly unless every requested PDF has exactly one document span."""
    expected = Counter(_normalized_path(path) for path in pdf_paths)
    observed = Counter(
        _normalized_path(filename)
        for trace in traces
        for span in trace["spans"]
        if span.get("name") == "retrieve_values_per_doc"
        and (filename := _span_input_filename(span))
    )
    missing = expected - observed
    unexpected = observed - expected
    if not missing and not unexpected:
        return

    details = []
    if missing:
        details.append(
            "missing: "
            + ", ".join(
                f"{Path(path).name} ({count})"
                for path, count in sorted(missing.items())
            )
        )
    if unexpected:
        details.append(
            "unexpected: "
            + ", ".join(
                f"{Path(path).name} ({count})"
                for path, count in sorted(unexpected.items())
            )
        )
    raise RuntimeError(
        "MLflow trace coverage does not match the requested PDFs; "
        + "; ".join(details)
        + ". No complete trace export was written."
    )


def _span_attributes(span: dict) -> dict:
    content = span.get("content") or {}
    return content.get("attributes") or {}


def _span_number_attribute(span: dict, key: str) -> float | None:
    value = _decode_json(_span_attributes(span).get(key))
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _span_usage(span: dict) -> dict:
    usage = _decode_json(_span_attributes(span).get("mlflow.chat.tokenUsage"))
    return usage if isinstance(usage, dict) else {}


def _trace_spans_for_pdf(trace: dict, pdf_path: Path) -> list[dict]:
    spans = trace["spans"]
    document_spans = [
        span
        for span in spans
        if span.get("name") == "retrieve_values_per_doc"
        and _span_input_filename(span)
        and _normalized_path(_span_input_filename(span)) == _normalized_path(pdf_path)
    ]
    if not document_spans:
        raise RuntimeError(
            f"Trace {trace['trace_id']} has no document span for {pdf_path}."
        )

    spans_by_parent: dict[str | None, list[dict]] = {}
    for span in spans:
        spans_by_parent.setdefault(span.get("parent_span_id"), []).append(span)

    included_ids = {span["span_id"] for span in document_spans}
    pending = [span["span_id"] for span in document_spans]
    while pending:
        parent_id = pending.pop()
        for child in spans_by_parent.get(parent_id, []):
            child_id = child["span_id"]
            if child_id not in included_ids:
                included_ids.add(child_id)
                pending.append(child_id)

    return [span for span in spans if span.get("span_id") in included_ids]


def _pdf_trace_metrics(spans: list[dict]) -> dict:
    document_spans = [
        span for span in spans if span.get("name") == "retrieve_values_per_doc"
    ]
    llm_spans = [span for span in spans if span.get("type") == "LLM"]
    duration_seconds = sum(
        max(0, span["end_time_unix_nano"] - span["start_time_unix_nano"]) / 1e9
        for span in document_spans
        if span.get("start_time_unix_nano") is not None
        and span.get("end_time_unix_nano") is not None
    )
    costs = [
        cost for span in llm_spans
        if (cost := _span_number_attribute(span, "response_cost")) is not None
    ]
    usages = [_span_usage(span) for span in llm_spans]

    return {
        "pipeline_span_seconds": round(duration_seconds, 6),
        "llm_call_count": len(llm_spans),
        "llm_prompt_tokens": sum(int(usage.get("input_tokens", 0)) for usage in usages),
        "llm_completion_tokens": sum(
            int(usage.get("output_tokens", 0)) for usage in usages
        ),
        "llm_total_tokens": sum(int(usage.get("total_tokens", 0)) for usage in usages),
        "llm_reported_cost_usd": round(sum(costs), 10) if costs else None,
        "llm_cost_complete": len(costs) == len(llm_spans),
        "cost_scope": "LLM calls only; embedding costs are not attributed per PDF",
    }


def write_pdf_traces_jsonl(path: Path, pdf_paths: list[Path], summary: dict,
                           result_rows: list[dict], traces: list[dict]) -> None:
    """Write one JSON Lines record per PDF with its own results and trace metrics."""
    run_metadata = {
        "model": summary["model"],
        "reasoning_effort": summary["reasoning_effort"],
        "run_wall_clock_seconds": summary["wall_clock_seconds"],
        "mlflow_run_id": summary["mlflow_run_id"],
        "run_reported_costs": summary["reported_costs"],
    }

    records = []
    for pdf_path in pdf_paths:
        pdf_key = _normalized_path(pdf_path)
        pdf_rows = [
            row for row in result_rows
            if row.get("report_name")
            and _normalized_path(row["report_name"]) == pdf_key
        ]
        pdf_traces = [
            {
                "trace_id": trace["trace_id"],
                "info": trace["info"],
                "tags": trace["tags"],
                "metadata": trace["metadata"],
                "spans": _trace_spans_for_pdf(trace, pdf_path),
            }
            for trace in traces
            if any(
                span.get("name") == "retrieve_values_per_doc"
                and _span_input_filename(span)
                and _normalized_path(_span_input_filename(span)) == pdf_key
                for span in trace["spans"]
            )
        ]
        if not pdf_traces:
            raise RuntimeError(
                f"No traces were found for PDF {pdf_path}; JSONL was not written."
            )
        records.append(
            {
                "pdf": str(pdf_path),
                "run": run_metadata,
                "pdf_metrics": _pdf_trace_metrics(
                    [span for trace in pdf_traces for span in trace["spans"]]
                ),
                "result_count": len(pdf_rows),
                "results": pdf_rows,
                "traces": pdf_traces,
            }
        )

    with path.open("w", encoding="utf-8", newline="\n") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")


def read_pipeline_warnings(path: Path) -> dict[str, list[str]]:
    """Group pipeline warning/error messages by their logged PDF filename."""
    warnings_by_pdf: dict[str, list[str]] = {}
    if not path.is_file():
        return warnings_by_pdf

    pattern = re.compile(
        r"\b(?:WARNING|ERROR|CRITICAL)\s+\[([^\]]+)\]\s+[^:]+:\s*(.*)"
    )
    with path.open(encoding="utf-8", errors="replace") as file:
        for line in file:
            match = pattern.search(line)
            if match:
                warnings_by_pdf.setdefault(match.group(1), []).append(
                    match.group(2).strip()
                )
    return warnings_by_pdf


def write_run_report_csv(
    path: Path,
    summary: dict,
    pdf_paths: list[Path],
    result_rows: list[dict],
    traces: list[dict],
    warnings_by_pdf: dict[str, list[str]],
) -> None:
    """Write a compact per-PDF report; full rows and traces live in other exports."""
    result_counts: dict[str, int] = {}
    for row in result_rows:
        report_name = row.get("report_name")
        if report_name:
            key = _normalized_path(report_name)
            result_counts[key] = result_counts.get(key, 0) + 1

    fields = [
        "pdf",
        "model",
        "reasoning_effort",
        "pdf_pipeline_span_seconds",
        "run_wall_clock_seconds",
        "llm_call_count",
        "llm_prompt_tokens",
        "llm_completion_tokens",
        "llm_total_tokens",
        "llm_reported_cost_usd",
        "llm_cost_complete",
        "result_count",
        "warning_count",
        "warnings_json",
        "trace_ids_json",
        "run_llm_cost_usd",
        "mlflow_run_id",
    ]

    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for pdf_path in pdf_paths:
            pdf_key = _normalized_path(pdf_path)
            pdf_spans = [
                span
                for trace in traces
                for span in _trace_spans_for_pdf(trace, pdf_path)
            ]
            pdf_metrics = _pdf_trace_metrics(pdf_spans)
            warning_messages = warnings_by_pdf.get(pdf_path.name, [])
            pdf_trace_ids = [
                trace["trace_id"]
                for trace in traces
                if any(
                    span.get("name") == "retrieve_values_per_doc"
                    and _span_input_filename(span)
                    and _normalized_path(_span_input_filename(span)) == pdf_key
                    for span in trace["spans"]
                )
            ]
            writer.writerow(
                {
                    "pdf": str(pdf_path),
                    "model": summary["model"],
                    "reasoning_effort": summary["reasoning_effort"],
                    "pdf_pipeline_span_seconds": pdf_metrics[
                        "pipeline_span_seconds"
                    ],
                    "run_wall_clock_seconds": summary["wall_clock_seconds"],
                    "llm_call_count": pdf_metrics["llm_call_count"],
                    "llm_prompt_tokens": pdf_metrics["llm_prompt_tokens"],
                    "llm_completion_tokens": pdf_metrics[
                        "llm_completion_tokens"
                    ],
                    "llm_total_tokens": pdf_metrics["llm_total_tokens"],
                    "llm_reported_cost_usd": pdf_metrics[
                        "llm_reported_cost_usd"
                    ],
                    "llm_cost_complete": pdf_metrics["llm_cost_complete"],
                    "result_count": result_counts.get(pdf_key, 0),
                    "warning_count": len(warning_messages),
                    "warnings_json": json.dumps(
                        warning_messages, ensure_ascii=False
                    ),
                    "trace_ids_json": json.dumps(
                        pdf_trace_ids,
                        ensure_ascii=False,
                    ),
                    "run_llm_cost_usd": summary["reported_costs"].get(
                        "total_llm_costs_in_dollar"
                    ),
                    "mlflow_run_id": summary["mlflow_run_id"],
                }
            )


def build_config(
    pdf_paths: list[Path],
    model_name: str,
    reasoning_effort: str,
    embeddings_db: Path,
    parallel_llm_calls: int,
) -> str:
    """Create the batch config without changing climatextract.toml."""
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pdf",
        type=Path,
        nargs="+",
        default=None,
        help=(
            "PDF file(s) to process. By default, process all PDFs in "
            f"{DEFAULT_PDF_DIR}. For multiple PDFs, pass each path separately."
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
            "Maximum number of concurrent LLM calls "
            f"(default: {DEFAULT_PARALLEL_LLM_CALLS}). Keep it equal across "
            "runs you want to compare by wall-clock time."
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
        help="Expected PDF count when using the default corpus (default: 139).",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        nargs="+",
        default=None,
        metavar="RUN_DIR",
        help=(
            "Folder(s) of aborted runs with the same model and reasoning "
            "level, e.g. output/<model>_<level>/<run-id>. Only PDFs without "
            "stored results are processed; the final results and evaluation "
            "cover all PDFs. Pass every earlier part if a resume was aborted too."
        ),
    )
    parser.add_argument(
        "--redo-failed",
        action="store_true",
        help=(
            "With --resume: process again every PDF whose stored result has an "
            "LLM call that failed after all retries; all other PDFs are taken "
            "over unchanged."
        ),
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

    if args.pdf is None:
        pdf_paths = sorted(DEFAULT_PDF_DIR.glob("*.pdf"))
        if not pdf_paths:
            print(
                f"No PDFs found in {DEFAULT_PDF_DIR}; trying to download them "
                "from Azure Blob Storage. This is a storage download, not a model call."
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
    missing_pdfs = [path for path in pdf_paths if not path.is_file()]
    if missing_pdfs:
        parser.error("PDF not found: " + ", ".join(str(path) for path in missing_pdfs))

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
                f"{run_dir} has no stored per-PDF results ({PDF_RESULTS_DIRNAME}/); "
                "runs started before resume support cannot be resumed."
            )
    if resume_dirs:
        previous_results = {
            stem: path
            for stem, path in find_previous_results(resume_dirs).items()
            if stem in {pdf_path.stem for pdf_path in all_pdf_paths}
        }
        if args.redo_failed:
            failed = failed_pdf_stems(previous_results)
            if not failed:
                parser.error("No PDF in the given run(s) has a failed LLM call.")
            print(f"Redoing {len(failed)} PDFs with failed LLM calls: "
                  + ", ".join(sorted(failed)))
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
            f"Resuming: {len(previous_results)} PDFs are done in earlier part(s), "
            f"{len(pdf_paths)} remain. The wall-clock time covers only this part."
        )

    load_dotenv(PROJECT_ROOT / ".env")
    if not os.getenv("LITELLM_GATEWAY_API_BASE"):
        parser.error("Set LITELLM_GATEWAY_API_BASE in .env or the environment.")
    if not os.getenv("LITELLM_GATEWAY_API_KEY"):
        parser.error("Set LITELLM_GATEWAY_API_KEY in .env or the environment.")

    # Keep traces and artifacts on this machine, regardless of .env settings.
    tracking_dir = PROJECT_ROOT / "mlruns"
    tracking_dir.mkdir(parents=True, exist_ok=True)
    os.environ["MLFLOW_TRACKING_URI"] = "sqlite:///mlruns/mlflow.db"
    os.environ["MLFLOW_ENABLE_ARTIFACTS_PROGRESS_BAR"] = "false"
    configure_mlflow_trace_buffer(len(pdf_paths))

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
        embeddings = EmbeddingsRepository(
            database_name=str(embeddings_db)
        )
        if not embeddings.database_exists():
            print(
                f"Local embeddings database not found: {embeddings_db}\n"
                "The script can offer to download the matching database from "
                "Azure Blob Storage. This is a storage download, not a model call."
            )
            manager = DataLakeManager(
                blob_path_pdfs="pdfs/sample_160",
                blob_path_embeddings="embeddings",
            )
            manager.handle_embedding_database(embeddings)
            embeddings = EmbeddingsRepository(database_name=str(embeddings_db))

        if embeddings.database_exists():
            missing_embeddings = [
                path.name for path in pdf_paths
                if not embeddings.pdf_exists(path.name)
            ]
            search_query_missing = not embeddings.search_query_exists(
                params.semantic_search_params.search_query
            )
        else:
            print(
                "No embeddings database available; a new one will be created "
                f"locally at {embeddings_db}."
            )
            missing_embeddings = [path.name for path in pdf_paths]
            search_query_missing = True

        if missing_embeddings or search_query_missing:
            if missing_embeddings:
                print(
                    f"{len(missing_embeddings)} PDFs have no cached embeddings; "
                    "they will be embedded during the run."
                )
                for name in missing_embeddings[:20]:
                    print(f"  - {name}")
                if len(missing_embeddings) > 20:
                    print(f"  ... and {len(missing_embeddings) - 20} more")
            if search_query_missing:
                print("The search query will be embedded during the run.")
            print(
                "Embedding costs are not part of the estimate but are tracked "
                "in the run metrics. Embedding time counts towards the "
                "wall-clock time."
            )
        else:
            print(
                f"Embedding cache covers all {len(pdf_paths)} PDFs and the search query; "
                "the extraction should make no embedding API calls."
            )

        database_path = tracking_dir / "mlflow.db"
        try:
            reference_trace = load_reference_trace(database_path)
        except (FileNotFoundError, RuntimeError):
            reference_trace = None
            print(
                f"Reference run {REFERENCE_RUN_ID} not found locally; "
                "skipping the cost estimate. Actual costs are still tracked."
            )
        if reference_trace is None:
            estimate = None
        else:
            estimate = estimate_llm_cost(
                model_name=args.model,
                document_count=len(pdf_paths),
                reference_trace=reference_trace,
            )
            estimate["reasoning_effort"] = args.reasoning
        if not confirm_batch(
            estimate, args.reasoning, args.model, len(pdf_paths), args.parallel
        ):
            print("Batch not started; no model calls were made.")
            return 0

        # Construct gateway handlers only after explicit cost confirmation.
        run_output_dir = PROJECT_ROOT / "output" / run_label
        instrument_pipeline(run_output_dir)
        llm = ReasoningRecordingLlmHandler(
            output_dir=run_output_dir,
            reasoning_effort=args.reasoning,
        )
        embedder = LiteLLMGatewayEmbeddingHandler()
        print(
            "MLflow trace buffer TTL: "
            f"{int(os.environ['MLFLOW_TRACE_BUFFER_TTL_SECONDS'])} seconds."
        )
        print(f"Starting extraction of {len(pdf_paths)} PDF(s)...")
        started = time.perf_counter()
        result_path = extract(
            pdf_input=[str(path) for path in pdf_paths],
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
        run_config_path = results_dir / "config.json"
        with run_config_path.open(encoding="utf-8") as file:
            run_config = json.load(file)
        run_id = run_config["run_info"]["run_id"]

        mlflow.flush_trace_async_logging()
        traces = load_run_traces(database_path, run_id)
        validate_trace_coverage(traces, pdf_paths)
        trace_directory = results_dir / "traces"
        trace_directory.mkdir(parents=True, exist_ok=True)
        for trace in traces:
            write_json(
                trace_directory / f"{trace['trace_id']}.json",
                trace,
            )
        traces_path = results_dir / "traces.json"
        write_json(traces_path, traces)

        if previous_results:
            print(
                f"Merging results of {len(previous_results)} PDFs from earlier "
                "part(s) into this run..."
            )
            merge_previous_parts(results_dir, previous_results, all_pdf_paths)

        results_csv_path = results_dir / "raw_results.csv"
        result_rows = read_result_rows(results_csv_path)

        gold_standard_path = (
            PROJECT_ROOT / "data" / "evaluation_dataset" / "gold_standard.csv"
        )
        print(f"Evaluating extraction against {gold_standard_path}...")
        evaluation_metrics = evaluate_results(
            path_to_results=str(results_dir),
            gold_standard=str(gold_standard_path),
        )
        evaluation_metrics = {
            key: value.item() if hasattr(value, "item") else value
            for key, value in evaluation_metrics.items()
        }
        evaluation_metrics_path = results_dir / "evaluation_metrics.json"
        write_json(evaluation_metrics_path, evaluation_metrics)

        trace_totals = reasoning_trace_totals(results_dir / REASONING_TRACES_DIRNAME)
        write_json(results_dir / "reasoning_trace_totals.json", trace_totals)

        summary_path = results_dir / "run_summary.json"
        summary = {
            "pdf": [str(path) for path in all_pdf_paths],
            "model": args.model,
            "reasoning_effort": args.reasoning,
            "parallel_llm_calls": args.parallel,
            "wall_clock_seconds": round(elapsed_seconds, 3),
            "mlflow_tracking_uri": os.environ["MLFLOW_TRACKING_URI"],
            "mlflow_run_id": run_id,
            "mlflow_trace_buffer_ttl_seconds": TRACE_BUFFER_TTL_SECONDS,
            "results_directory": str(results_dir.resolve()),
            "reported_costs": run_config["metrics"],
            "cost_estimate": estimate,
            "trace_ids": [trace["trace_id"] for trace in traces],
            "trace_count": len(traces),
            "result_count": len(result_rows),
            "results_json": str((results_dir / "results.json").resolve()),
            "traces_json": str(traces_path.resolve()),
            "traces_jsonl": str(
                (results_dir / "traces_by_pdf.jsonl").resolve()
            ),
            "trace_directory": str(trace_directory.resolve()),
            "reasoning_traces_directory": str(
                (results_dir / REASONING_TRACES_DIRNAME).resolve()
            ),
            "reasoning_trace_totals": trace_totals,
            "resumed_from": [str(path) for path in resume_dirs],
            "pdfs_from_earlier_parts": len(previous_results),
            "pdfs_this_part": len(pdf_paths),
            "scope_notes": (
                "wall_clock_seconds, reported_costs, traces and the trace-based "
                "exports (traces_by_pdf.jsonl, run_report.csv) cover only this "
                "part; results, evaluation and reasoning_traces cover all PDFs."
                if previous_results else None
            ),
            "evaluation_metrics": evaluation_metrics,
            "evaluation_metrics_json": str(
                evaluation_metrics_path.resolve()
            ),
        }
        results_json_path = results_dir / "results.json"
        write_json(
            results_json_path,
            {
                **summary,
                "results": result_rows,
                "traces": traces,
            },
        )
        write_json(summary_path, summary)

        traces_jsonl_path = results_dir / "traces_by_pdf.jsonl"
        write_pdf_traces_jsonl(
            traces_jsonl_path, pdf_paths, summary, result_rows, traces
        )
        report_csv_path = results_dir / "run_report.csv"
        warnings_by_pdf = read_pipeline_warnings(
            results_dir / "http_errors.log"
        )
        write_run_report_csv(
            report_csv_path,
            summary,
            pdf_paths,
            result_rows,
            traces,
            warnings_by_pdf,
        )

        with mlflow.start_run(run_id=run_id):
            mlflow.log_metric("wall_clock_seconds", elapsed_seconds)
            mlflow.log_artifact(str(summary_path))
            mlflow.log_artifact(str(results_json_path))
            mlflow.log_artifact(str(traces_path))
            mlflow.log_artifact(str(traces_jsonl_path))
            mlflow.log_artifact(str(report_csv_path))
            mlflow.log_artifact(str(evaluation_metrics_path))
            mlflow.log_artifact(
                str(results_dir / "eval_results_vs_benchmark.csv")
            )
            mlflow.log_artifact(
                str(results_dir / "eval_results_metrics_by_ReportName.csv")
            )
            for metric_name, metric_value in evaluation_metrics.items():
                if (
                    isinstance(metric_value, (int, float))
                    and re.fullmatch(r"[A-Za-z0-9_.-]+", metric_name)
                ):
                    mlflow.log_metric(metric_name, metric_value)
            for trace_path in trace_directory.glob("*.json"):
                mlflow.log_artifact(str(trace_path), artifact_path="traces")

        print(
            "Evaluation metrics: "
            f"precision={evaluation_metrics.get('precision_value', 0):.4f}, "
            f"recall={evaluation_metrics.get('recall_value', 0):.4f}, "
            f"F1={evaluation_metrics.get('f1_value', 0):.4f}"
        )
        print(f"Wall-clock time: {elapsed_seconds:.1f} s")
        print(f"Results, traces, and run summary: {results_dir.resolve()}")
        print(f"Per-PDF JSONL: {traces_jsonl_path.resolve()}")
        print(
            "Per-PDF reasoning traces: "
            f"{(results_dir / REASONING_TRACES_DIRNAME).resolve()}"
        )
        print(f"Combined CSV report: {report_csv_path.resolve()}")
        print(f"Local MLflow store: {tracking_dir.resolve()}")
        print(f"MLflow run ID: {run_id}")
        return 0
    finally:
        Path(config_path).unlink(missing_ok=True)


if __name__ == "__main__":
    raise SystemExit(main())
