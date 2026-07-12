#!/usr/bin/env python3
"""Build the fixed public-paper-only input set for the fast-v1 benchmark."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Dict, Iterable, Mapping


EXPECTED_FORUM_IDS = (
    "0eTTKOOOQkV",
    "8onXkaNWLHA",
    "VpYBxaPLaj-",
    "VWqiPBB_EM",
    "Uzgfy7_v7BH",
    "hLbeJ6jObDD",
    "morSrUyWG26",
    "3pugbNqOh5m",
    "ZJqqSa8FsH9",
    "AP1MKT37rJ",
)

_PAPER_KEYS = {
    "schema_version",
    "paper_id",
    "title",
    "text",
    "document_id",
    "source_uri",
    "version_status",
    "license",
}


def _canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write(path: Path, payload: bytes) -> None:
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


def _load_cases(path: Path) -> Dict[str, Mapping[str, Any]]:
    cases: Dict[str, Mapping[str, Any]] = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, Mapping):
            raise ValueError("line {} is not an object".format(line_number))
        forum_id = value.get("forum_id")
        if not isinstance(forum_id, str) or not forum_id:
            raise ValueError("line {} has no forum_id".format(line_number))
        if forum_id in cases:
            raise ValueError("duplicate forum_id: {}".format(forum_id))
        cases[forum_id] = value
    return cases


def _paper_record(case: Mapping[str, Any]) -> Dict[str, Any]:
    forum_id = str(case["forum_id"])
    source = case.get("paper")
    if not isinstance(source, Mapping):
        raise ValueError("{} has no paper object".format(forum_id))
    title = source.get("title")
    text = source.get("text")
    if not isinstance(title, str) or not title.strip():
        raise ValueError("{} has no paper title".format(forum_id))
    if not isinstance(text, str) or not text.strip():
        raise ValueError("{} has no paper text".format(forum_id))
    record = {
        "schema_version": 1,
        "paper_id": forum_id,
        "title": title.strip(),
        "text": text,
        "document_id": "openreview-{}".format(forum_id),
        "source_uri": source.get("openreview_url")
        or "https://openreview.net/forum?id={}".format(forum_id),
        "version_status": source.get("version_status")
        or "current public version; initial-submission status not verified",
        "license": "CC-BY-4.0",
    }
    if set(record) != _PAPER_KEYS:
        raise AssertionError("paper-only record key drift")
    return record


def build(source: Path, output_dir: Path, forum_ids: Iterable[str]) -> Mapping[str, Any]:
    selected = tuple(forum_ids)
    if len(selected) != 10 or len(selected) != len(set(selected)):
        raise ValueError("benchmark requires exactly 10 distinct forum IDs")
    cases = _load_cases(source)
    papers_dir = output_dir / "papers"
    manifest_paths = []
    file_records = []
    for index, forum_id in enumerate(selected, 1):
        if forum_id not in cases:
            raise ValueError("source corpus is missing {}".format(forum_id))
        record = _paper_record(cases[forum_id])
        relative = Path("papers") / "paper-{:02d}.json".format(index)
        payload = _canonical_bytes(record)
        _atomic_write(output_dir / relative, payload)
        manifest_paths.append(relative.as_posix())
        file_records.append(
            {
                "path": relative.as_posix(),
                "paper_id": forum_id,
                "sha256": _sha256_bytes(payload),
                "text_sha256": _sha256_bytes(record["text"].encode("utf-8")),
                "text_characters": len(record["text"]),
            }
        )
    batch_manifest = {"papers": manifest_paths}
    batch_manifest_payload = _canonical_bytes(batch_manifest)
    provenance = {
        "schema_version": 1,
        "benchmark": "track2-fast-v1-public-paper-only-10",
        "selection_rule": "first 10 fixed IDs from scripts/build_real_seed.py DEFAULT_IDS",
        "batch_manifest": "papers.manifest.json",
        "batch_manifest_sha256": _sha256_bytes(batch_manifest_payload),
        "paper_files": file_records,
        "source": {
            "path": source.as_posix(),
            "sha256": _sha256_file(source),
            "paper_source": "https://huggingface.co/datasets/Samarth0710/reviewbench",
            "license": "CC-BY-4.0",
        },
        "information_boundary": (
            "Only public paper fields are materialized. Human reviews, rebuttals, "
            "metareviews, ratings, and decisions are excluded from every live input file."
        ),
        "limitations": [
            "The public paper text may be a revised version rather than the initial submission.",
            "This fixed set measures pipeline validity and latency, not review quality or generalization.",
        ],
    }
    _atomic_write(output_dir / "papers.manifest.json", batch_manifest_payload)
    _atomic_write(output_dir / "input_provenance.json", _canonical_bytes(provenance))
    return {"manifest": batch_manifest, "provenance": provenance}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=Path("data/real/seed_cases.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/benchmark_fast_v1"))
    args = parser.parse_args()
    result = build(args.source, args.output_dir, EXPECTED_FORUM_IDS)
    print(
        json.dumps(
            {
                "manifest": str(args.output_dir / "papers.manifest.json"),
                "papers": len(result["manifest"]["papers"]),
                "source_sha256": result["provenance"]["source"]["sha256"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
