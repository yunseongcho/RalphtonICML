"""Adapters for the small, public, provenance-tracked seed corpus.

The paper-only signals below are intentionally simple structural heuristics.
They make the update loop executable without an LLM and are not claimed to be a
competitive reviewer.  Human reviews, rebuttals, and decisions are used only as
supervision after forum-level splitting.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
import re
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

from .learning import LearningExample, seed_case_to_learning_example
from .openreview import DatasetSplit, deterministic_forum_split


class SeedDataError(ValueError):
    """Raised when a real-seed record is incomplete or malformed."""


def load_seed_cases(path: Path) -> Tuple[Mapping[str, Any], ...]:
    cases = []
    seen = set()
    with Path(path).open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                case = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SeedDataError("invalid JSON at line {}".format(line_number)) from exc
            if not isinstance(case, Mapping):
                raise SeedDataError("line {} must be a JSON object".format(line_number))
            forum_id = str(case.get("forum_id", "")).strip()
            if not forum_id or forum_id in seen:
                raise SeedDataError("missing or duplicate forum_id at line {}".format(line_number))
            seen.add(forum_id)
            paper = case.get("paper")
            if not isinstance(paper, Mapping) or not str(paper.get("text", "")).strip():
                raise SeedDataError("{} has no paper text".format(forum_id))
            if "authors" in paper:
                raise SeedDataError("{} contains forbidden author identity fields".format(forum_id))
            if not case.get("reviews"):
                raise SeedDataError("{} has no reviews".format(forum_id))
            if not case.get("rebuttal_dialogues"):
                raise SeedDataError("{} has no rebuttal".format(forum_id))
            if not str(case.get("decision", "")).strip():
                raise SeedDataError("{} has no decision".format(forum_id))
            cases.append(case)
    if not cases:
        raise SeedDataError("seed corpus is empty")
    return tuple(cases)


def _bounded_density(text: str, patterns: Sequence[str], base: float = 1.8) -> float:
    matches = sum(len(re.findall(pattern, text, flags=re.IGNORECASE)) for pattern in patterns)
    words = max(1, len(re.findall(r"\b\w+\b", text)))
    normalized = matches / max(1.0, words / 1500.0)
    return min(4.0, max(1.0, base + 0.55 * math.log1p(normalized)))


def paper_only_signals(case: Mapping[str, Any]) -> Tuple[Tuple[str, float], ...]:
    """Derive weak structural inputs without touching review-stage labels."""

    paper = case["paper"]
    text = "{}\n{}\n{}".format(
        paper.get("title", ""), paper.get("abstract", ""), paper.get("text", "")
    )
    soundness = _bounded_density(
        text,
        (
            r"\btheorem\b", r"\bproof\b", r"\blemma\b", r"\bexperiment\w*\b",
            r"\bablation\b", r"\bbaseline\w*\b", r"\bdataset\w*\b", r"\bconfidence interval\b",
        ),
    )
    presentation = _bounded_density(
        text,
        (
            r"(?m)^#{1,4}\s+", r"\bfigure\s+\d+", r"\btable\s+\d+",
            r"\bequation\s+\d+", r"\blimitation\w*\b", r"\bappendix\b",
        ),
        base=2.0,
    )
    significance = _bounded_density(
        text,
        (
            r"\bcontribution\w*\b", r"\bstate[- ]of[- ]the[- ]art\b",
            r"\bbenchmark\w*\b", r"\bimprov\w*\b", r"\bimpact\w*\b",
            r"\bpractical\b", r"\bapplication\w*\b",
        ),
    )
    originality = _bounded_density(
        text,
        (
            r"\bwe propose\b", r"\bwe introduce\b", r"\bnovel\b",
            r"\bnew\b", r"\bfirst\b", r"\boriginal\w*\b",
        ),
    )
    evidence_axes = sum(value >= 2.25 for value in (soundness, presentation, significance, originality))
    confidence = min(4.0, 2.0 + evidence_axes * 0.4)
    return tuple(
        sorted(
            (
                ("soundness", soundness),
                ("presentation", presentation),
                ("significance", significance),
                ("originality", originality),
                ("confidence", confidence),
            )
        )
    )


def seed_case_to_example(case: Mapping[str, Any]) -> LearningExample:
    """Use the canonical final-score/rebuttal adapter with paper-only inputs."""

    return seed_case_to_learning_example(case, paper_only_signals(case))


def split_seed_examples(
    cases: Iterable[Mapping[str, Any]],
    seed: str = "ralphton-icml-real-seed-v1",
) -> Tuple[Tuple[LearningExample, ...], Tuple[LearningExample, ...], Tuple[LearningExample, ...], DatasetSplit]:
    cases_tuple = tuple(cases)
    examples = {str(case["forum_id"]): seed_case_to_example(case) for case in cases_tuple}
    split = deterministic_forum_split(examples, seed=seed)
    return (
        tuple(examples[forum_id] for forum_id in split.train),
        tuple(examples[forum_id] for forum_id in split.dev),
        tuple(examples[forum_id] for forum_id in split.test),
        split,
    )


__all__ = [
    "SeedDataError",
    "load_seed_cases",
    "paper_only_signals",
    "seed_case_to_example",
    "split_seed_examples",
]
