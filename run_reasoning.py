"""Run a confirmed batch extraction and store per-PDF MLflow traces locally."""

import argparse
import csv
import json
import os
from pathlib import Path
import re
import sqlite3
import sys
import tempfile
import time

from dotenv import load_dotenv


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
REASONING_CHOICES = ("none", "low", "medium", "high")


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


def estimate_llm_cost(
    model_name: str,
    document_count: int,
    reference_trace: dict,
) -> dict:
    """Scale observed per-PDF token usage using LiteLLM's Azure AI price map."""
    import litellm

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
            "Embedding charges are excluded and therefore the batch requires "
            "a complete local embedding cache."
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


def confirm_batch(estimate: dict, reasoning_effort: str) -> bool:
    print("\nEstimated LLM batch")
    print(f"  PDFs: {estimate['document_count']}")
    print(f"  Model: {estimate['model']}")
    print(f"  Reasoning effort: {reasoning_effort}")
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
            "\nStart all PDFs and make the LLM calls? Type JA to confirm: "
        ).strip()
    except EOFError:
        return False
    return answer == "JA"


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
) -> str:
    """Create the batch config without changing climatextract.toml."""
    pdf_list = ", ".join(json.dumps(str(path)) for path in pdf_paths)
    reasoning_value = "" if reasoning_effort == "none" else reasoning_effort
    content = f"""\
[input]
filename_list = [{pdf_list}]

[models]
llm_model = {json.dumps(model_name)}
emb_model = "text-embedding-ada-002"
max_parallel_llm_prompts_running = 1
max_parallel_embedding_calls = 1
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
output_dir = "output/gpt_oss_batch"

[mlflow]
experiment_name = "gpt-oss-batch"

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
        help="Reasoning effort (default: low; use none to omit it).",
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
    args = parser.parse_args()

    if args.pdf is None:
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

    os.chdir(PROJECT_ROOT)
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

    from climatextract import _runtime_config
    from climatextract.main import _load_config, extract
    from climatextract.semantic_search import EmbeddingsRepository
    from climatextract.adapters.litellm_gateway import (
        LiteLLMGatewayEmbeddingHandler,
        LiteLLMGatewayLlmHandler,
    )
    from climatextract.data_lake_manager import DataLakeManager

    embeddings_db = args.embeddings_db.resolve()
    config_path = build_config(
        pdf_paths=pdf_paths,
        model_name=args.model,
        reasoning_effort=args.reasoning,
        embeddings_db=embeddings_db,
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
            if not manager.handle_embedding_database(embeddings):
                print("Embedding database download was not completed.")
                return 2
            embeddings = EmbeddingsRepository(database_name=str(embeddings_db))
            if not embeddings.database_exists():
                print(
                    "Embedding database is still unavailable. No model calls "
                    "were made; check data lake access and the exact database name."
                )
                return 2

        missing_embeddings = [
            path.name for path in pdf_paths if not embeddings.pdf_exists(path.name)
        ]
        if missing_embeddings:
            print(
                f"{len(missing_embeddings)} PDFs do not have cached embeddings. "
                "To prevent unestimated embedding charges, the batch was not started."
            )
            for name in missing_embeddings[:20]:
                print(f"  - {name}")
            if len(missing_embeddings) > 20:
                print(f"  ... and {len(missing_embeddings) - 20} more")
            return 2
        if not embeddings.search_query_exists(params.semantic_search_params.search_query):
            print(
                "The search query is missing from the embedding cache. "
                "To prevent an unestimated embedding call, the batch was not started."
            )
            return 2
        print(
            f"Embedding cache covers all {len(pdf_paths)} PDFs and the search query; "
            "the extraction should make no embedding API calls."
        )

        database_path = tracking_dir / "mlflow.db"
        reference_trace = load_reference_trace(database_path)
        estimate = estimate_llm_cost(
            model_name=args.model,
            document_count=len(pdf_paths),
            reference_trace=reference_trace,
        )
        estimate["reasoning_effort"] = args.reasoning
        if not confirm_batch(estimate, args.reasoning):
            print("Batch not started; no model calls were made.")
            return 0

        # Construct gateway handlers only after explicit cost confirmation.
        llm = LiteLLMGatewayLlmHandler()
        embedder = LiteLLMGatewayEmbeddingHandler()
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

        traces = load_run_traces(database_path, run_id)
        trace_directory = results_dir / "traces"
        trace_directory.mkdir(parents=True, exist_ok=True)
        for trace in traces:
            write_json(
                trace_directory / f"{trace['trace_id']}.json",
                trace,
            )
        traces_path = results_dir / "traces.json"
        write_json(traces_path, traces)

        results_csv_path = results_dir / "raw_results.csv"
        result_rows = read_result_rows(results_csv_path)
        from climatextract.evaluator import evaluate as evaluate_results

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

        summary_path = results_dir / "run_summary.json"
        summary = {
            "pdf": [str(path) for path in pdf_paths],
            "model": args.model,
            "reasoning_effort": args.reasoning,
            "wall_clock_seconds": round(elapsed_seconds, 3),
            "mlflow_tracking_uri": os.environ["MLFLOW_TRACKING_URI"],
            "mlflow_run_id": run_id,
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

        import mlflow

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
        print(f"Combined CSV report: {report_csv_path.resolve()}")
        print(f"Local MLflow store: {tracking_dir.resolve()}")
        print(f"MLflow run ID: {run_id}")
        return 0
    finally:
        Path(config_path).unlink(missing_ok=True)


if __name__ == "__main__":
    raise SystemExit(main())
