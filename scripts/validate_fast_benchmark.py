#!/usr/bin/env python3
"""Validate fast-v1 benchmark artifacts and estimate legacy request volume."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
import threading
from typing import Any, Dict, Mapping, Sequence

from ralphton_icml.learning import load_learning_state
from ralphton_icml.orchestrator import PaperInput, ReviewerOrchestrator
from ralphton_icml.schema import ReviewOutput
from ralphton_icml.track2 import load_track2_bundle


FORBIDDEN_LIVE_KEYS = {
    "decision",
    "final_score",
    "initial_score",
    "metareview",
    "rating",
    "rebuttal_dialogues",
    "review_content",
    "reviews",
}


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_name = ""
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=str(path.parent),
            prefix=path.name + ".",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary_name = stream.name
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, str(path))
        temporary_name = ""
    finally:
        if temporary_name:
            try:
                Path(temporary_name).unlink()
            except FileNotFoundError:
                pass


def read_json(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError("{} must contain a JSON object".format(path))
    return value


class LegacyMeasureBackend:
    """Replay fast outputs through legacy-v1 solely to serialize its requests."""

    def __init__(self, fast_results: Mapping[str, Mapping[str, Any]]) -> None:
        self.fast_results = fast_results
        self.requests = []
        self.lock = threading.Lock()

    def complete(self, request: Any) -> str:
        paper = request.payload.get("paper", {})
        paper_id = str(paper.get("paper_id", ""))
        result = self.fast_results[paper_id]
        encoded_size = len(canonical_bytes(request.as_dict()))
        with self.lock:
            self.requests.append((paper_id, request.stage, encoded_size))

        if request.stage == "extraction":
            task_id = request.payload["extraction_task"]["task_id"]
            evidence = {
                item["task_id"]: item for item in result["context"]["evidence"]
            }[task_id]
            answer = "\n".join(
                "[content] {}".format(line)
                if line.strip().casefold() in {"answer", "sources"}
                else line
                for line in evidence["answer"].splitlines()
            )
            sources = "\n".join(
                "- [source] {}".format(item)
                if item.strip().casefold() in {"answer", "sources"}
                else "- {}".format(item)
                for item in evidence["sources"]
            )
            return "ANSWER\n{}\nSOURCES\n{}".format(answer, sources)
        if request.stage in {"domain_review", "criterion_review", "synthesis"}:
            return json.dumps(result["critiques"], ensure_ascii=False, sort_keys=True)
        if request.stage == "author_rebuttal":
            author = result.get("author_response")
            if isinstance(author, Mapping) and isinstance(author.get("response"), str):
                return author["response"]
            return "No additional frozen evidence is available."
        if request.stage == "final_review":
            return result["rendered_review"]
        raise AssertionError("unexpected legacy stage {}".format(request.stage))


def validate_result(
    result_path: Path,
    markdown_path: Path,
    completion_path: Path,
) -> Mapping[str, Any]:
    result_bytes = result_path.read_bytes()
    markdown_bytes = markdown_path.read_bytes()
    result = json.loads(result_bytes.decode("utf-8"))
    completion = read_json(completion_path)
    ReviewOutput.from_markdown(markdown_bytes.decode("utf-8"))
    if result["rendered_review"] != markdown_bytes.decode("utf-8"):
        raise ValueError("rendered_review differs from Markdown: {}".format(result_path))
    if completion["result_json_sha256"] != sha256_bytes(result_bytes):
        raise ValueError("completion JSON hash mismatch: {}".format(result_path))
    if completion["review_markdown_sha256"] != sha256_bytes(markdown_bytes):
        raise ValueError("completion Markdown hash mismatch: {}".format(markdown_path))

    known = {item["evidence_id"] for item in result["context"]["evidence"]}
    known.update(item["evidence_id"] for item in result["provided_evidence"])
    linked = 0
    major_fatal = 0
    for critique in result["critiques"].values():
        for group in ("strengths", "weaknesses", "questions"):
            for finding in critique[group]:
                evidence_ids = set(finding["evidence_ids"])
                if not evidence_ids or not evidence_ids.issubset(known):
                    raise ValueError("finding has invalid evidence IDs")
                linked += 1
                if finding["severity"] in {"major", "fatal"}:
                    major_fatal += 1
        for contradiction in critique["unresolved_contradictions"]:
            evidence_ids = set(contradiction["evidence_ids"])
            if not evidence_ids or not evidence_ids.issubset(known):
                raise ValueError("contradiction has invalid evidence IDs")
            linked += 1
    if not set(result["chair_selected_evidence_ids"]).issubset(known):
        raise ValueError("chair selected an unknown evidence ID")
    return {
        "result": result,
        "linked_findings_and_contradictions": linked,
        "major_fatal_findings": major_fatal,
    }


def legacy_request_estimate(
    papers: Sequence[Mapping[str, Any]],
    fast_results: Mapping[str, Mapping[str, Any]],
    state_path: Path,
) -> Mapping[str, Any]:
    backend = LegacyMeasureBackend(fast_results)
    state = load_learning_state(state_path)
    for value in papers:
        paper = PaperInput(
            paper_id=value["paper_id"],
            title=value["title"],
            text=value["text"],
            document_id=value["document_id"],
            source_uri=value["source_uri"],
        )
        ReviewerOrchestrator(
            backend,
            max_workers=8,
            learning_state=state,
        ).review(paper, run_author_loop=True)
    per_paper: Dict[str, Dict[str, int]] = {}
    for paper_id, _stage, size in backend.requests:
        value = per_paper.setdefault(paper_id, {"calls": 0, "request_bytes": 0})
        value["calls"] += 1
        value["request_bytes"] += size
    return {
        "method": (
            "Counterfactual legacy-v1 request serialization using the same public papers, "
            "state, fast-v1 extracted evidence, critiques, and final Markdown; no model calls."
        ),
        "calls": len(backend.requests),
        "request_bytes": sum(item[2] for item in backend.requests),
        "per_paper": per_paper,
    }


def render_human_review(acceptance: Mapping[str, Any]) -> str:
    checks = acceptance["acceptance"]
    rows = [
        ("Valid reviews", "{}/{}".format(checks["valid_reviews"], checks["papers_total"])),
        ("Calls", str(acceptance["fast_v1"]["calls"])),
        ("Retries / failures / timeouts", "{} / {} / {}".format(
            acceptance["fast_v1"]["retries"],
            acceptance["fast_v1"]["backend_failures"],
            acceptance["fast_v1"]["backend_timeouts"],
        )),
        ("Wall time", "{:.1f}s".format(acceptance["fast_v1"]["wall_time_seconds"])),
        ("Request bytes", str(acceptance["fast_v1"]["request_bytes"])),
        ("Legacy request estimate", str(acceptance["legacy_estimate"]["request_bytes"])),
        ("Estimated byte reduction", "{:.2%}".format(checks["legacy_request_byte_reduction"])),
        ("Representative max request / paper text", "{:.2f}x".format(
            checks["representative_max_request_to_text_ratio"]
        )),
    ]
    lines = [
        "# Fast-v1 Benchmark Human Review",
        "",
        "| Check | Result |",
        "|---|---:|",
    ]
    lines.extend("| {} | {} |".format(name, value) for name, value in rows)
    lines.extend(
        [
            "",
            "## Acceptance",
            "",
            "- 10/10 strict ReviewOutput and completion hashes: {}".format(
                "PASS" if checks["all_reviews_valid"] else "FAIL"
            ),
            "- Evidence ID linkage for all findings/contradictions: {}".format(
                "PASS" if checks["all_evidence_ids_valid"] else "FAIL"
            ),
            "- Cold-cache 30-minute deadline and <=60 calls: {}".format(
                "PASS" if checks["deadline_and_call_budget"] else "FAIL"
            ),
            "- Legacy request byte reduction >=85%: {}".format(
                "PASS" if checks["legacy_byte_reduction_at_least_85_percent"] else "FAIL"
            ),
            "- Historical dev MAE/Brier non-regression tolerance 0.02: {}".format(
                "PASS" if checks["historical_non_regression"] else "FAIL"
            ),
            "",
            "## Scope",
            "",
            "The ten inputs contain public paper fields only. This run validates pipeline "
            "contracts, latency, and evidence linkage; it does not establish review quality "
            "or generalization. Public texts may be revised versions rather than initial submissions.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark-dir", type=Path, required=True)
    parser.add_argument("--input-root", type=Path, default=Path("data/benchmark_fast_v1"))
    parser.add_argument("--state", type=Path, default=Path("artifacts/real_seed_v2/best_state.json"))
    parser.add_argument("--seed-summary", type=Path, default=Path("artifacts/real_seed_v2/summary.json"))
    parser.add_argument("--representative-root", type=Path, required=True)
    args = parser.parse_args()

    benchmark_dir = args.benchmark_dir.resolve(strict=True)
    summary = read_json(benchmark_dir / "summary.json")
    manifest = read_json(args.input_root / "papers.manifest.json")
    papers = []
    for relative in manifest["papers"]:
        value = read_json(args.input_root / relative)
        if set(value).intersection(FORBIDDEN_LIVE_KEYS):
            raise ValueError("live paper input contains forbidden supervision")
        papers.append(value)

    fast_results: Dict[str, Mapping[str, Any]] = {}
    evidence_links = 0
    major_fatal = 0
    for paper_record in summary["papers"]:
        result_path = Path(paper_record["result_json"])
        markdown_path = Path(paper_record["review_markdown"])
        completion_path = result_path.with_name(result_path.stem + ".complete.json")
        validation = validate_result(result_path, markdown_path, completion_path)
        result = validation["result"]
        fast_results[result["paper_id"]] = result
        evidence_links += validation["linked_findings_and_contradictions"]
        major_fatal += validation["major_fatal_findings"]

    legacy = legacy_request_estimate(papers, fast_results, args.state)
    reduction = 1.0 - float(summary["request_bytes"]) / legacy["request_bytes"]
    representative_bundle = load_track2_bundle(args.representative_root)
    representative_starts = [
        json.loads(line)
        for line in (args.representative_root / "backend-progress.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip() and json.loads(line).get("event") == "start"
    ]
    representative_ratio = max(item["request_bytes"] for item in representative_starts) / len(
        representative_bundle.paper_text.encode("utf-8")
    )
    seed = read_json(args.seed_summary)
    tolerance = 0.02
    non_regression = (
        seed["best_dev"]["mae"] <= seed["baseline_dev"]["mae"] + tolerance
        and seed["best_dev"]["brier"] <= seed["baseline_dev"]["brier"] + tolerance
    )
    acceptance = {
        "schema_version": 1,
        "fast_v1": dict(summary),
        "legacy_estimate": legacy,
        "evidence_audit": {
            "linked_findings_and_contradictions": evidence_links,
            "major_fatal_findings": major_fatal,
        },
        "acceptance": {
            "papers_total": len(papers),
            "valid_reviews": len(fast_results),
            "all_reviews_valid": len(fast_results) == len(papers) == 10,
            "all_evidence_ids_valid": True,
            "deadline_and_call_budget": (
                summary["deadline_met"]
                and summary["calls"] <= 60
                and summary["backend_failures"] == 0
                and summary["backend_timeouts"] == 0
            ),
            "legacy_request_byte_reduction": reduction,
            "legacy_byte_reduction_at_least_85_percent": reduction >= 0.85,
            "representative_max_request_to_text_ratio": representative_ratio,
            "representative_request_ratio_at_most_3": representative_ratio <= 3.0,
            "historical_non_regression": non_regression,
            "historical_tolerance": tolerance,
        },
    }
    atomic_write(
        benchmark_dir / "acceptance.json",
        json.dumps(acceptance, ensure_ascii=False, indent=2, sort_keys=True).encode("utf-8") + b"\n",
    )
    atomic_write(
        benchmark_dir / "HUMAN_REVIEW.md",
        render_human_review(acceptance).encode("utf-8"),
    )
    print(json.dumps(acceptance["acceptance"], sort_keys=True))


if __name__ == "__main__":
    main()
