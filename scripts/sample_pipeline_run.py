"""Run GPT-OSS and DeepSeek using the installed ClimateXtract package.

From gist-reasoning-traces with its virtual environment:
    python scripts/sample_pipeline_run.py --check  # no model calls
    python scripts/sample_pipeline_run.py          # paid model calls

Inputs and credentials are local: configs/climatextract_test.toml, data/, .env.
Outputs: output/{gptoss,deepseek}_<effort>_<run-id>/<document>/ with CSVs and traces.
Equal requested effort does not imply equal internal reasoning across models.
"""

import argparse
from collections import Counter
import importlib.metadata
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import tempfile
import time
import tomllib
from uuid import uuid4

PROJECT_ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--check", action="store_true", help="Verify local inputs/cache without API calls")
parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/climatextract_test.toml")
ARGS = parser.parse_args()
ARGS.config = ARGS.config.resolve()
# Use the installed package; do not add any source checkout to sys.path.
os.chdir(PROJECT_ROOT)

import duckdb

if "--check" in sys.argv:
    os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"

from climatextract import _runtime_config
from climatextract.adapters.litellm_gateway import (
    LiteLLMGatewayEmbeddingHandler,
    LiteLLMGatewayLlmHandler,
)
from climatextract.main import _load_config, _resolve_pdf_input, _extract_and_evaluate_with_metadata

os.chdir(PROJECT_ROOT)

BASE_CONFIG = ARGS.config
with BASE_CONFIG.open("rb") as stream:
    SAMPLE = tomllib.load(stream)["sample"]
MODELS = SAMPLE["model_ids"]
MAX_OUTPUT_TOKENS = SAMPLE["max_output_tokens"]
if len(MODELS) != 2 or len(set(MODELS)) != 2:
    raise ValueError("Configure exactly two distinct sample.model_ids.")


class CachedEmbeddingHandler(LiteLLMGatewayEmbeddingHandler):
    """Refuse paid embedding calls if the cache is unexpectedly incomplete."""

    def get_embedding_and_cost(self, texts):
        raise RuntimeError("Embedding cache miss; no new embeddings allowed in this test.")

    async def aget_embedding_and_cost(self, texts):
        raise RuntimeError("Embedding cache miss; no new embeddings allowed in this test.")


class TracingLlmHandler(LiteLLMGatewayLlmHandler):
    """Record full responses and persist each completed/failed call immediately."""

    def __init__(self, trace_path, metadata, **extra_params):
        super().__init__(**extra_params)
        self.trace_path = trace_path
        self.metadata = metadata
        self.traces = []
        self.save()

    def save(self):
        self.trace_path.write_text(json.dumps(
            {**self.metadata, "calls": self.traces}, indent=2, ensure_ascii=False
        ) + "\n", encoding="utf-8")

    async def aget_completion_and_cost(self, messages):
        start = time.perf_counter()
        call = {"prompt": messages[-1]["content"],
                "prompt_sha256": hashlib.sha256(messages[-1]["content"].encode()).hexdigest()}
        try:
            response, cost = await super().aget_completion_and_cost(messages)
            message = response.choices[0].message
            usage = response.usage
            details = getattr(usage, "completion_tokens_details", None)
            call.update({
                "answer": message.content,
                "reasoning": getattr(message, "reasoning_content", None),
                "prompt_tokens": getattr(usage, "prompt_tokens", None),
                "completion_tokens": getattr(usage, "completion_tokens", None),
                "reasoning_tokens": getattr(details, "reasoning_tokens", None),
                "finish_reason": response.choices[0].finish_reason,
                "cost": cost,  # Gateway-reported; zero may mean unavailable.
                "raw_response": response.model_dump(mode="json"),
            })
            return response, cost
        except Exception as exc:
            call.update({"error_type": type(exc).__name__})
            raise
        finally:
            call["duration_seconds"] = time.perf_counter() - start
            self.traces.append(call)
            self.save()


def check_cache():
    """Read-only preflight: all reports and the exact search query must be cached."""
    if not BASE_CONFIG.is_file():
        raise FileNotFoundError(BASE_CONFIG)
    config, params, output_dir, *_ = _load_config(str(BASE_CONFIG))
    if params.pipeline_params.embed_only:
        raise ValueError("This test requires embed_only = false.")
    reports = _resolve_pdf_input(config.filename_list)
    import csv
    if not config.gold_standard or not Path(config.gold_standard).is_file():
        raise ValueError("Gold-standard file is missing.")
    with open(config.gold_standard, newline="", encoding="utf-8") as stream:
        gold_reports = {row["report_name"] for row in csv.DictReader(stream)}
    if any(Path(report).name not in gold_reports for report in reports):
        raise ValueError("Every report must have a matching report_name in the gold standard.")
    if not reports or any(not Path(report).is_file() for report in reports):
        raise ValueError("Every configured report must exist locally.")
    if len({Path(report).name for report in reports}) != len(reports):
        raise ValueError("Report filenames must be unique: the cache identifies reports by filename.")
    database = Path(params.semantic_search_params.embeddings_repository or "")
    if not database.is_file():
        raise ValueError("Configure an existing embeddings_repository in climatextract_test.toml.")
    with duckdb.connect(str(database), read_only=True) as connection:
        for report in reports:
            rows = connection.execute(
                "SELECT embed_created_by FROM pages WHERE short_file_name = ?",
                [Path(report).name],
            ).fetchall()
            cached = connection.execute(
                "SELECT number_of_pages FROM pdf_files WHERE short_file_name = ?", [Path(report).name]
            ).fetchone()
            if not rows or not cached or len(rows) != cached[0]:
                raise ValueError(f"No complete cache entry for {report}")
            if any(params.semantic_search_params.emb_model not in (row[0] or "") for row in rows):
                raise ValueError(f"Cached embedding model differs for {report}")
        query = connection.execute(
            "SELECT embed_created_by FROM search_query_embeddings WHERE search_query = ?",
            [params.semantic_search_params.search_query],
        ).fetchone()
        if not query or params.semantic_search_params.emb_model not in (query[0] or ""):
            raise ValueError("Search query embedding is missing or uses a different model.")
    print(f"Cache verified: {len(reports)} report(s), {database}")
    print(f"Models: {', '.join(MODELS)}; requested effort: {params.llm_params.reasoning_effort}")
    print(f"Pages/report: top {params.semantic_search_params.similarity_top_k}; "
          f"output limit/call: {MAX_OUTPUT_TOKENS}; client retries: 0")
    return reports, database, output_dir


def run(model, report, database, output_dir, run_id):
    """Run the original pipeline; save traces even if extraction/evaluation fails."""
    prefix = "deepseek" if model.lower().startswith("deepseek") else "gptoss"
    _, base_params, *_ = _load_config(str(BASE_CONFIG))
    effort = base_params.llm_params.reasoning_effort or "default"
    effort_label = re.sub(r"[^A-Za-z0-9_-]+", "_", effort)
    # Hash avoids collisions between sanitized filenames; original name stays in metadata.
    document_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", Path(report).stem)
    document_id += "_" + hashlib.sha256(Path(report).name.encode()).hexdigest()[:8]
    run_root = Path(output_dir).resolve() / f"{prefix}_{effort_label}_{run_id}" / document_id
    run_root.mkdir(parents=True)
    trace_path = run_root / "reasoning_traces.json"
    text = BASE_CONFIG.read_text()
    text, count = re.subn(r'^llm_model\s*=\s*".*"\s*$', f'llm_model = "{model}"', text, flags=re.M)
    if count != 1:
        raise ValueError("Expected exactly one llm_model setting.")
    # Use a private temporary config, removed automatically; keep main config untouched.
    text, count = re.subn(r'^output_dir\s*=\s*".*"\s*$',
                          f'output_dir = {json.dumps(str(run_root))}', text, flags=re.M)
    if count != 1:
        raise ValueError("Expected exactly one output_dir setting.")
    with tempfile.TemporaryDirectory(prefix="climatextract-test-") as tmp:
        config_path = Path(tmp) / "config.toml"
        config_path.write_text(text)
        _, params, *_ = _load_config(str(config_path))
        _runtime_config.set_current(params)
        settings = {"reasoning_effort": params.llm_params.reasoning_effort,
                    "logprobs": False, "max_tokens": MAX_OUTPUT_TOKENS,
                    "num_retries": 0, "timeout": 90,
                    "extra_body": {"drop_params": True}}
        if model.lower().startswith("deepseek"):
            settings["extra_body"]["thinking"] = {"type": "enabled"}
        metadata = {"model": model, "document": Path(report).name, "reports": [report], "run_id": run_id,
                    "embeddings_repository": str(database), "requested_settings": settings,
                    "base_config": BASE_CONFIG.read_text(), "project_root": str(PROJECT_ROOT),
                    "climatextract_version": importlib.metadata.version("climatextract"),
                    "status": "running"}
        llm = TracingLlmHandler(trace_path, metadata, **settings)
        try:
            # Internal orchestration accepts an exact output folder; the public
            # entry point would add another UUID subfolder. No pipeline edits.
            result = _extract_and_evaluate_with_metadata(
                pdf_input=[report], config_path=str(config_path),
                path_to_results=str(run_root), llm=llm,
                embedder=CachedEmbeddingHandler(),
            )
            metadata.update({"results_dir": str(run_root),
                             "status": ("finished_with_errors" if any("error_type" in c for c in llm.traces) else "finished")
                                       if result and result["has_results"] else "no results"})
        except Exception as exc:
            metadata.update({"status": "failed", "error_type": type(exc).__name__})
            raise
        finally:
            llm.save()
    print(f"{model}: {trace_path}")
    print(f"  Calls: {len(llm.traces)}; visible traces: "
          f"{sum(bool(c.get('reasoning')) for c in llm.traces)}; errors: "
          f"{sum('error_type' in c for c in llm.traces)}; truncated: "
          f"{sum(c.get('finish_reason') == 'length' for c in llm.traces)}; "
          f"reported cost: ${sum(c.get('cost') or 0 for c in llm.traces):.6f}")
    return Counter(c["prompt_sha256"] for c in llm.traces)


if __name__ == "__main__":
    reports, database, output_dir = check_cache()
    if not ARGS.check:
        run_id = uuid4().hex
        issues = []
        for report in reports:
            prompts = []
            for model in MODELS:
                try:
                    prompts.append(run(model, report, database, output_dir, run_id))
                except Exception as exc:
                    issues.append(f"{model} / {Path(report).name}: {type(exc).__name__}")
            if len(prompts) != 2 or not prompts[0] or prompts[0] != prompts[1]:
                issues.append(f"{Path(report).name}: paired prompt sets not verified")
        if issues:
            raise SystemExit("\n".join(issues))
        print("Verified identical prompts for each paired document.")
