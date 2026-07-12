#!/usr/bin/env python3
"""Join public ReviewBench papers with Re2 review/rebuttal records.

This script is intentionally a provenance-preserving data preparation utility,
not a network crawler.  Inputs are the downloaded upstream JSON files and
Hugging Face dataset-viewer row caches.  It removes author identities and emits
only explicitly selected, complete forum-level cases.
"""

from __future__ import annotations

import argparse
import datetime as dt
import glob
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping


DEFAULT_IDS = (
    "0eTTKOOOQkV",  # CV, accept
    "8onXkaNWLHA",  # CV, reject
    "VpYBxaPLaj-",  # CV, reject
    "VWqiPBB_EM",   # Core ML, accept
    "Uzgfy7_v7BH",  # Core ML, reject
    "hLbeJ6jObDD",  # Core ML, accept
    "morSrUyWG26",  # Core ML, reject
    "3pugbNqOh5m",  # Core ML, accept
    "ZJqqSa8FsH9",  # NLP, accept
    "AP1MKT37rJ",   # Core ML / RL, accept
    "C8Ltz08PtBp",  # Core ML / RL, accept
    "tUMr0Iox8XW",  # Core ML, accept
    "ujibH3ervr",   # CV / security, reject
    "19MmorTQhho",  # CV / 3D, accept
    "HQDvPsdXS-F",  # Core ML / optimization, accept
    "hcVlMF3Nvxg",  # Core ML / robustness, accept
    "p9zeOtKQXKs",  # Core ML / meta-RL, accept
    "pGcTocvaZkJ",  # Core ML / survival, accept
    "uCBx_6Hc7cu",  # Core ML / variational inference, accept
    "utahaTbcHdP",  # Core ML / representation, accept
)

REVIEWBENCH_URL = "https://huggingface.co/datasets/Samarth0710/reviewbench"
RE2_URL = "https://huggingface.co/datasets/Daoze/ReviewRebuttal"


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_paper_rows(pattern: str) -> Dict[str, Mapping[str, Any]]:
    rows: Dict[str, Mapping[str, Any]] = {}
    for name in sorted(glob.glob(pattern)):
        path = Path(name)
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        for item in payload.get("rows", []):
            row = item.get("row", {})
            forum_id = str(row.get("forum_id", ""))
            if forum_id:
                rows[forum_id] = row
    return rows


def score_has_rating(review: Mapping[str, Any]) -> bool:
    initial = review.get("initial_score", {})
    final = review.get("final_score", {})
    values = (initial.get("rating"), final.get("rating"))
    return any(str(value).strip().casefold() not in {"", "null", "none"} for value in values)


def clean_reviews(reviews: Iterable[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    cleaned = []
    for review in reviews:
        if not score_has_rating(review):
            continue
        cleaned.append(
            {
                "reviewer_id": str(review.get("reviewer_id", "anonymous")),
                "review_title": str(review.get("review_title", "")),
                "review_content": str(review.get("review_content", "")),
                "initial_score": review.get("initial_score", {}),
                "final_score": review.get("final_score", {}),
                "initial_score_unified": review.get("initial_score_unified", {}),
                "final_score_unified": review.get("final_score_unified", {}),
            }
        )
    return cleaned


def clean_dialogue(record: Mapping[str, Any]) -> Dict[str, Any]:
    messages = list(record.get("messages", []))
    # Drop the dataset-authored system prompt and the first paper-ID trigger.
    dialogue = []
    seen_assistant_review = False
    for message in messages:
        role = str(message.get("role", ""))
        content = str(message.get("content", "")).strip()
        if not content or role == "system":
            continue
        if role == "user" and not seen_assistant_review:
            continue
        if role == "assistant" and not seen_assistant_review:
            seen_assistant_review = True
            # The same initial review is already present in the review record.
            continue
        dialogue.append({"role": role, "content": content})
    return {
        "reviewer_id": str(record.get("reviewer_id", "anonymous")),
        "messages": dialogue,
    }


def build_case(
    paper: Mapping[str, Any],
    review: Mapping[str, Any],
    rebuttals: Iterable[Mapping[str, Any]],
) -> Dict[str, Any]:
    forum_id = str(review["paper_id"])
    reviews = clean_reviews(review.get("reviews", []))
    dialogues = [clean_dialogue(item) for item in rebuttals]
    dialogues = [item for item in dialogues if item["messages"]]
    if not reviews or not dialogues or not str(review.get("decision", "")).strip():
        raise ValueError("{} does not contain all required review stages".format(forum_id))
    markdown = str(paper.get("markdown", ""))
    if not markdown.strip():
        raise ValueError("{} has no paper text".format(forum_id))
    return {
        "schema_version": 1,
        "forum_id": forum_id,
        "conference_year_track": str(review.get("conference_year_track", "")),
        "paper": {
            "title": str(paper.get("title", "")),
            "abstract": str(paper.get("abstract", "")),
            "keywords": list(paper.get("keywords") or []),
            "primary_area": str(paper.get("primary_area", "")),
            "text": markdown,
            "text_sha256": hashlib.sha256(markdown.encode("utf-8")).hexdigest(),
            "openreview_url": "https://openreview.net/forum?id={}".format(forum_id),
            "version_status": "current public version; initial-submission status not verified",
        },
        "reviews": reviews,
        "review_initial_ratings_unified": list(
            review.get("review_initial_ratings_unified", [])
        ),
        "review_final_ratings_unified": list(
            review.get("review_final_ratings_unified", [])
        ),
        "rebuttal_dialogues": dialogues,
        "metareview": str(review.get("metareview", "")),
        "decision": str(review.get("decision", "")),
        "provenance": {
            "paper_source": REVIEWBENCH_URL,
            "paper_license": "CC-BY-4.0",
            "review_rebuttal_source": RE2_URL,
            "review_rebuttal_license": "Apache-2.0",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reviews", type=Path, required=True)
    parser.add_argument("--rebuttals", type=Path, required=True)
    parser.add_argument("--paper-glob", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--forum-id", action="append", dest="forum_ids")
    args = parser.parse_args()

    selected = tuple(args.forum_ids or DEFAULT_IDS)
    review_records = {
        item["paper_id"]: item
        for item in json.loads(args.reviews.read_text(encoding="utf-8"))
    }
    rebuttal_records: Dict[str, List[Mapping[str, Any]]] = {}
    for item in json.loads(args.rebuttals.read_text(encoding="utf-8")):
        rebuttal_records.setdefault(item["paper_id"], []).append(item)
    paper_records = load_paper_rows(args.paper_glob)

    cases = []
    for forum_id in selected:
        if forum_id not in paper_records:
            raise SystemExit("missing ReviewBench row for {}".format(forum_id))
        if forum_id not in review_records:
            raise SystemExit("missing Re2 review for {}".format(forum_id))
        if forum_id not in rebuttal_records:
            raise SystemExit("missing Re2 rebuttal for {}".format(forum_id))
        cases.append(
            build_case(
                paper_records[forum_id],
                review_records[forum_id],
                rebuttal_records[forum_id],
            )
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    lines = [canonical_bytes(case) for case in cases]
    args.output.write_bytes(b"\n".join(lines) + b"\n")
    output_hash = file_sha256(args.output)
    manifest = {
        "schema_version": 1,
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "record_count": len(cases),
        "forum_ids": [case["forum_id"] for case in cases],
        "complete_stage_requirement": ["paper", "reviews", "rebuttal", "decision"],
        "output_file": args.output.name,
        "output_sha256": output_hash,
        "source_artifacts": [
            {
                "path": str(args.reviews),
                "sha256": file_sha256(args.reviews),
                "url": RE2_URL + "/resolve/main/REVIEWS_test.json",
                "license": "Apache-2.0",
            },
            {
                "path": str(args.rebuttals),
                "sha256": file_sha256(args.rebuttals),
                "url": RE2_URL + "/resolve/main/REBUTTAL_test.json",
                "license": "Apache-2.0",
            },
            {
                "path_pattern": args.paper_glob,
                "url": REVIEWBENCH_URL,
                "license": "CC-BY-4.0",
            },
        ],
        "privacy": "Public records only; author identity fields removed.",
        "limitations": [
            "The ReviewBench paper text may be a revised public version rather than the initial submission.",
            "This small seed is for pipeline/update smoke evaluation, not a publishable benchmark result.",
            "RecSys has no complete example in this selected 20-case intersection.",
            "OpenReview venue rubrics are normalized only where the upstream Re2 record provides mappings.",
        ],
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_bytes(canonical_bytes(manifest) + b"\n")
    print(
        json.dumps(
            {"records": len(cases), "sha256": output_hash},
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
