"""Measure saved traces. No model calls or qualitative annotations.

python scripts/analyse_reasoning_pilot.py [--input output] [--output analysis]
Writes calls.csv (one row per API attempt) and documents.csv (one row per
model/document/run). Word counts use str.split(); lengths are not token counts.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]


def answer_records(answer):
    """Count schema entries; invalid/absent output remains explicitly marked."""
    if not isinstance(answer, str) or not answer.strip():
        return "missing", None
    text = answer.strip()
    if text.startswith("```") and text.endswith("```"):
        text = "\n".join(text.splitlines()[1:-1])
    try:
        data = json.loads(text)
    except (ValueError, TypeError):
        return "invalid_json", None
    entries = data.get("KPI_Entries") if isinstance(data, dict) else None
    if not isinstance(entries, list):
        return "unexpected_schema", None
    return "valid_json", len(entries)


def analyse(input_dir):
    files = sorted(input_dir.rglob("reasoning_traces.json"))
    if not files:
        raise ValueError(f"No reasoning_traces.json files under {input_dir}")
    calls, documents = [], []
    for path in files:
        saved = json.loads(path.read_text(encoding="utf-8"))
        attempts = saved.get("calls", [])
        rows = []
        for index, call in enumerate(attempts, 1):
            trace = call.get("reasoning")
            state = "missing" if trace is None else "empty" if trace == "" else "present"
            if trace is not None and not isinstance(trace, str):
                raise ValueError(f"Non-text reasoning field: {path}, call {index}")
            status, count = answer_records(call.get("answer"))
            prompt = call.get("prompt", "")
            row = {
                "trace_file": str(path.resolve()), "model": saved.get("model", "unknown"),
                "document": saved.get("document", "unknown"),
                "run_id": saved.get("run_id", path.parent.name),
                "call_index": index,
                "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                "trace_state": state,
                "trace_characters": len(trace) if trace is not None else None,
                "trace_words": len(trace.split()) if trace is not None else None,
                "answer_characters": len(call.get("answer") or ""),
                "answer_status": status, "extracted_records": count,
                "finish_reason": call.get("finish_reason"),
                "error_type": call.get("error_type"),
                "duration_seconds": call.get("duration_seconds"),
                "prompt_tokens_reported": call.get("prompt_tokens"),
                "completion_tokens_reported": call.get("completion_tokens"),
                "reasoning_tokens_reported": call.get("reasoning_tokens"),
                "cost_usd_reported": call.get("cost"),
            }
            rows.append(row)
        calls.extend(rows)
        lengths = [r["trace_words"] for r in rows if r["trace_state"] == "present"]
        costs = [r["cost_usd_reported"] for r in rows if r["cost_usd_reported"] is not None]
        durations = [r["duration_seconds"] for r in rows if r["duration_seconds"] is not None]
        documents.append({
            "trace_file": str(path.resolve()), "model": saved.get("model", "unknown"),
            "document": saved.get("document", "unknown"),
            "run_id": saved.get("run_id", path.parent.name), "run_status": saved.get("status", "unknown"),
            "calls": len(rows), "traces_present": len(lengths),
            "traces_missing": sum(r["trace_state"] == "missing" for r in rows),
            "traces_empty": sum(r["trace_state"] == "empty" for r in rows),
            "failed_calls": sum(bool(r["error_type"]) for r in rows),
            "truncated_calls": sum(r["finish_reason"] == "length" for r in rows),
            "valid_json_answers": sum(r["answer_status"] == "valid_json" for r in rows),
            "invalid_json_answers": sum(r["answer_status"] == "invalid_json" for r in rows),
            "unexpected_schema_answers": sum(r["answer_status"] == "unexpected_schema" for r in rows),
            "missing_answers": sum(r["answer_status"] == "missing" for r in rows),
            "extracted_records": sum(r["extracted_records"] or 0 for r in rows),
            "mean_trace_words_present": mean(lengths) if lengths else None,
            "min_trace_words_present": min(lengths) if lengths else None,
            "max_trace_words_present": max(lengths) if lengths else None,
            "mean_duration_seconds": mean(durations) if durations else None,
            "duration_observations": len(durations),
            "cost_usd_reported_sum": sum(costs) if costs else None,
            "cost_observations": len(costs),
        })
    return calls, documents


def write_csv(path, rows):
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "output")
    parser.add_argument("--output", type=Path, default=ROOT / "analysis")
    args = parser.parse_args()
    calls, documents = analyse(args.input)
    args.output.mkdir(parents=True, exist_ok=True)
    write_csv(args.output / "calls.csv", calls)
    write_csv(args.output / "documents.csv", documents)
    print(f"{len(documents)} saved runs, {len(calls)} calls: {args.output / 'calls.csv'}, {args.output / 'documents.csv'}")


if __name__ == "__main__":
    main()
