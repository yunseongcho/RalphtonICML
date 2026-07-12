"""Versioned reviewer/author memory learning for the ICML review team.

This module does **not** fine-tune LLM weights.  It implements versioned rubric
weights, score calibration, and bounded retrieval-memory updates, which are the
practical default for a 16 GB Apple M2 machine.  A remote or CUDA-backed model
fine-tuner can later be placed behind the prediction boundary without changing
the train/dev/test isolation and convergence logic implemented here.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


CRITERIA_FIELDS = ("soundness", "presentation", "significance", "originality")
SCORE_FIELDS = CRITERIA_FIELDS + ("overall_recommendation", "confidence")
REQUIRED_REVIEW_FIELDS = SCORE_FIELDS + ("comment",)
GENERIC_REVIEW_COMMENT = (
    "Improve the AI review agent by requiring evidence-linked checks of claims, "
    "limitations, and reproducibility; improve the paper with specific, "
    "constructive suggestions grounded in this submission."
)
DEFAULT_RETRIEVAL_RELEVANCE = 0.05
_MEMORY_GUIDANCE_AXES = (
    (
        "technical assumptions, derivations, and proof completeness",
        (r"\bproof\w*\b", r"\btheorem\w*\b", r"\blemma\w*\b", r"\bderiv\w*\b", r"\bassum\w*\b"),
    ),
    (
        "experimental controls, baselines, ablations, and statistical support",
        (r"\bexperiment\w*\b", r"\bbaseline\w*\b", r"\bablation\w*\b", r"\bstatistic\w*\b", r"\bcomparison\w*\b"),
    ),
    (
        "generalization, robustness, failure cases, and limitations",
        (r"\bgenerali[sz]\w*\b", r"\brobust\w*\b", r"\bfailure\w*\b", r"\blimitation\w*\b", r"\bout.of.distribution\b"),
    ),
    (
        "novelty claims and positioning against prior work",
        (r"\bnovel\w*\b", r"\boriginal\w*\b", r"\bprior work\b", r"\brelated work\b", r"\bincremental\b"),
    ),
    (
        "reproducibility, data, hyperparameters, compute, and released resources",
        (r"\breproduc\w*\b", r"\bdataset\w*\b", r"\bhyperparameter\w*\b", r"\bcompute\b", r"\bcode\b"),
    ),
    (
        "clarity, notation, organization, and implementation detail",
        (r"\bclar\w*\b", r"\bnotation\b", r"\bwriting\b", r"\bpseudocode\b", r"\bimplementation\b"),
    ),
    (
        "ethics, privacy, bias, consent, and potential misuse",
        (r"\bethic\w*\b", r"\bprivacy\b", r"\bbias\w*\b", r"\bconsent\b", r"\bmisuse\b"),
    ),
)
_RETRIEVAL_STOPWORDS = frozenset({
    "about", "after", "also", "and", "approach", "are", "based", "data", "for", "from", "have",
    "into", "learning", "method", "model", "models", "network", "neural",
    "paper", "propose", "proposed", "results", "show", "study", "their",
    "studies", "the", "these", "this", "training", "using", "was", "were", "with", "without",
})
SCORE_RANGES = {
    "soundness": (1.0, 4.0),
    "presentation": (1.0, 4.0),
    "significance": (1.0, 4.0),
    "originality": (1.0, 4.0),
    "overall_recommendation": (1.0, 6.0),
    "confidence": (1.0, 5.0),
    "accept_probability": (0.0, 1.0),
}


class DataLeakageError(ValueError):
    """Raised when forum IDs cross immutable split boundaries."""


def _finite_number(value: Any) -> Optional[float]:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _clip(value: float, lower: float, upper: float) -> float:
    return min(upper, max(lower, value))


def _pairs_to_dict(pairs: Sequence[Tuple[str, float]]) -> Dict[str, float]:
    return {str(key): float(value) for key, value in pairs}


def _validated_pairs(pairs: Sequence[Tuple[str, float]], label: str) -> Tuple[Tuple[str, float], ...]:
    result = []
    seen = set()
    for key, value in pairs:
        key = str(key).strip()
        number = _finite_number(value)
        if not key:
            raise ValueError("%s contains an empty key" % label)
        if key in seen:
            raise ValueError("%s contains duplicate key %s" % (label, key))
        if number is None:
            raise ValueError("%s[%s] must be finite" % (label, key))
        seen.add(key)
        result.append((key, number))
    return tuple(sorted(result))


def _canonical_hash(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def compute_prompt_manifest_digest(manifest: Mapping[str, Any]) -> str:
    """Hash the immutable prompt, schema, team, and model configuration manifest."""

    if not isinstance(manifest, Mapping) or not manifest:
        raise ValueError("prompt manifest must be a non-empty mapping")
    return _canonical_hash(dict(manifest))


DEFAULT_PROMPT_MANIFEST_DIGEST = compute_prompt_manifest_digest({
    "status": "unconfigured",
    "contract": "prompt-schema-team-manifest-v1",
})


def _tokens(text: str) -> set:
    return set(re.findall(r"[A-Za-z0-9가-힣_]+", text.lower()))


def _retrieval_tokens(text: str) -> set:
    return {
        token
        for token in _tokens(text)
        if len(token) >= 3 and token not in _RETRIEVAL_STOPWORDS
    }


def _jaccard_distance(left: str, right: str) -> float:
    left_tokens, right_tokens = _tokens(left), _tokens(right)
    union = left_tokens.union(right_tokens)
    if not union:
        return 0.0
    return 1.0 - len(left_tokens.intersection(right_tokens)) / float(len(union))


@dataclass(frozen=True)
class MemoryItem:
    role: str
    forum_id: str
    text: str
    cue: str = ""

    def __post_init__(self) -> None:
        if self.role not in ("reviewer", "author"):
            raise ValueError("memory role must be reviewer or author")
        if not self.forum_id.strip():
            raise ValueError("memory forum_id cannot be empty")
        if not self.text.strip():
            raise ValueError("memory text cannot be empty")

    @property
    def identity(self) -> str:
        return _canonical_hash((self.role, self.forum_id, self.text, self.cue))


def _default_rubric() -> Tuple[Tuple[str, float], ...]:
    return tuple((field_name, 1.0 / len(CRITERIA_FIELDS)) for field_name in CRITERIA_FIELDS)


@dataclass(frozen=True)
class LearningState:
    """Immutable version of rubric, calibration, and both agents' memories."""

    version: int = 0
    rubric_weights: Tuple[Tuple[str, float], ...] = field(default_factory=_default_rubric)
    calibration_scale: float = 1.0
    calibration_bias: float = 0.0
    prompt_manifest_digest: str = DEFAULT_PROMPT_MANIFEST_DIGEST
    reviewer_memory: Tuple[MemoryItem, ...] = ()
    author_memory: Tuple[MemoryItem, ...] = ()
    parent_digest: str = ""

    def __post_init__(self) -> None:
        if type(self.version) is not int or self.version < 0:
            raise ValueError("state version cannot be negative")
        weights = _validated_pairs(self.rubric_weights, "rubric_weights")
        if {key for key, _ in weights} != set(CRITERIA_FIELDS):
            raise ValueError("rubric_weights must contain exactly the four reviewer criteria")
        if any(value < 0 for _, value in weights) or not math.isclose(
            sum(value for _, value in weights), 1.0, rel_tol=0.0, abs_tol=1e-8
        ):
            raise ValueError("rubric_weights must be non-negative and sum to one")
        if _finite_number(self.calibration_scale) is None or self.calibration_scale <= 0:
            raise ValueError("calibration_scale must be finite and positive")
        if _finite_number(self.calibration_bias) is None:
            raise ValueError("calibration_bias must be finite")
        if not isinstance(self.prompt_manifest_digest, str) or not re.fullmatch(
            r"[0-9a-f]{64}", self.prompt_manifest_digest
        ):
            raise ValueError("prompt_manifest_digest must be a lowercase SHA-256 digest")
        if any(item.role != "reviewer" for item in self.reviewer_memory):
            raise ValueError("reviewer_memory contains a non-reviewer item")
        if any(item.role != "author" for item in self.author_memory):
            raise ValueError("author_memory contains a non-author item")
        memory_ids = [item.identity for item in self.reviewer_memory + self.author_memory]
        if len(memory_ids) != len(set(memory_ids)):
            raise ValueError("state memory contains duplicate items")
        object.__setattr__(self, "rubric_weights", weights)

    @property
    def digest(self) -> str:
        return _canonical_hash({
            "version": self.version,
            "rubric_weights": self.rubric_weights,
            "calibration_scale": self.calibration_scale,
            "calibration_bias": self.calibration_bias,
            "prompt_manifest_digest": self.prompt_manifest_digest,
            "reviewer_memory": [
                (item.role, item.forum_id, item.text, item.cue) for item in self.reviewer_memory
            ],
            "author_memory": [
                (item.role, item.forum_id, item.text, item.cue) for item in self.author_memory
            ],
            "parent_digest": self.parent_digest,
        })

    def as_dict(self) -> Dict[str, Any]:
        """Return a JSON-safe state object with an integrity digest."""

        return {
            "schema_version": 1,
            "version": self.version,
            "rubric_weights": [list(item) for item in self.rubric_weights],
            "calibration_scale": self.calibration_scale,
            "calibration_bias": self.calibration_bias,
            "prompt_manifest_digest": self.prompt_manifest_digest,
            "reviewer_memory": [
                {"role": item.role, "forum_id": item.forum_id, "text": item.text, "cue": item.cue}
                for item in self.reviewer_memory
            ],
            "author_memory": [
                {"role": item.role, "forum_id": item.forum_id, "text": item.text, "cue": item.cue}
                for item in self.author_memory
            ],
            "parent_digest": self.parent_digest,
            "digest": self.digest,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "LearningState":
        """Restore and integrity-check :meth:`as_dict` output."""

        if not isinstance(payload, Mapping):
            raise ValueError("learning state payload must be an object")
        if payload.get("schema_version", 1) != 1:
            raise ValueError("unsupported learning state schema_version")

        def memories(name: str) -> Tuple[MemoryItem, ...]:
            values = payload.get(name, [])
            if not isinstance(values, list):
                raise ValueError("%s must be a list" % name)
            try:
                return tuple(MemoryItem(
                    role=str(item["role"]),
                    forum_id=str(item["forum_id"]),
                    text=str(item["text"]),
                    cue=str(item.get("cue", "")),
                ) for item in values)
            except (KeyError, TypeError) as exc:
                raise ValueError("invalid %s entry: %s" % (name, exc))

        state = cls(
            version=payload.get("version", 0),
            rubric_weights=tuple(tuple(item) for item in payload.get("rubric_weights", _default_rubric())),
            calibration_scale=payload.get("calibration_scale", 1.0),
            calibration_bias=payload.get("calibration_bias", 0.0),
            prompt_manifest_digest=payload.get(
                "prompt_manifest_digest", DEFAULT_PROMPT_MANIFEST_DIGEST
            ),
            reviewer_memory=memories("reviewer_memory"),
            author_memory=memories("author_memory"),
            parent_digest=str(payload.get("parent_digest", "")),
        )
        expected_digest = payload.get("digest")
        if expected_digest is not None and expected_digest != state.digest:
            raise ValueError("learning state digest does not match payload")
        return state


def serialize_learning_state(state: LearningState) -> str:
    if not isinstance(state, LearningState):
        raise TypeError("state must be a LearningState")
    return json.dumps(
        state.as_dict(), ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")
    )


def deserialize_learning_state(payload: str) -> LearningState:
    if not isinstance(payload, str):
        raise TypeError("payload must be a JSON string")
    try:
        decoded = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise ValueError("invalid learning state JSON: %s" % exc)
    return LearningState.from_dict(decoded)


def dump_learning_state(state: LearningState, path: Any) -> None:
    """Write a state atomically without depending on third-party serialization."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary_name = ""
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=str(target.parent),
            prefix=target.name + ".",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary_name = stream.name
            stream.write(serialize_learning_state(state) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, str(target))
    finally:
        if temporary_name:
            try:
                Path(temporary_name).unlink()
            except FileNotFoundError:
                pass


def load_learning_state(path: Any) -> LearningState:
    return deserialize_learning_state(Path(path).read_text(encoding="utf-8"))


@dataclass(frozen=True)
class LearningExample:
    """One submission with paper-only signals and post-review supervision.

    ``paper_signals`` must be derived without reading reviews, rebuttals, or the
    decision.  ``target_scores`` and memory lessons are supervision and may only
    be consumed by the updater for the train split.
    """

    forum_id: str
    paper_signals: Tuple[Tuple[str, float], ...] = ()
    target_scores: Tuple[Tuple[str, float], ...] = ()
    accepted: Optional[bool] = None
    reviewer_lessons: Tuple[str, ...] = ()
    author_lessons: Tuple[str, ...] = ()
    retrieval_text: str = ""

    def __post_init__(self) -> None:
        if not self.forum_id.strip():
            raise ValueError("forum_id cannot be empty")
        signals = _validated_pairs(self.paper_signals, "paper_signals")
        targets = _validated_pairs(self.target_scores, "target_scores")
        for key, value in targets:
            if key not in SCORE_FIELDS:
                raise ValueError("unsupported target score field %s" % key)
            lower, upper = SCORE_RANGES[key]
            if value < lower or value > upper:
                raise ValueError("target score %s is outside [%s, %s]" % (key, lower, upper))
        if self.accepted is not None and not isinstance(self.accepted, bool):
            raise ValueError("accepted must be bool or None")
        reviewer = tuple(text.strip() for text in self.reviewer_lessons if text.strip())
        author = tuple(text.strip() for text in self.author_lessons if text.strip())
        object.__setattr__(self, "paper_signals", signals)
        object.__setattr__(self, "target_scores", targets)
        object.__setattr__(self, "reviewer_lessons", reviewer)
        object.__setattr__(self, "author_lessons", author)

    @property
    def fingerprint(self) -> str:
        return _canonical_hash({
            "forum_id": self.forum_id,
            "paper_signals": self.paper_signals,
            "target_scores": self.target_scores,
            "accepted": self.accepted,
            "reviewer_lessons": self.reviewer_lessons,
            "author_lessons": self.author_lessons,
            "retrieval_text": self.retrieval_text,
        })

    @classmethod
    def from_forum_record(
        cls,
        record: Any,
        paper_signals: Sequence[Tuple[str, float]],
    ) -> "LearningExample":
        """Build supervision from a normalized ``ForumRecord``.

        Paper signals are required explicitly so human review scores cannot be
        accidentally fed back as reviewer inputs.
        """

        target_pairs = []
        reviews = tuple(getattr(record, "reviews", ()))
        for field_name in SCORE_FIELDS:
            values = [
                float(getattr(review, field_name))
                for review in reviews
                if getattr(review, field_name, None) is not None
            ]
            if values:
                target_pairs.append((field_name, sum(values) / len(values)))
        reviewer_lessons = []
        for review in reviews:
            for attribute in ("weaknesses", "questions", "comment"):
                text = str(getattr(review, attribute, "")).strip()
                if text:
                    reviewer_lessons.append(text)
        rebuttals = tuple(getattr(record, "rebuttals", ()))
        author_lessons = tuple(str(getattr(item, "text", "")).strip() for item in rebuttals)
        paper = getattr(record, "paper", None)
        retrieval_text = ""
        if paper is not None:
            retrieval_text = "%s\n%s" % (getattr(paper, "title", ""), getattr(paper, "abstract", ""))
        decision = getattr(record, "decision", None)
        accepted = getattr(decision, "accepted", None) if decision is not None else None
        return cls(
            forum_id=str(getattr(record, "forum_id")),
            paper_signals=tuple(paper_signals),
            target_scores=tuple(target_pairs),
            accepted=accepted,
            reviewer_lessons=tuple(reviewer_lessons),
            author_lessons=tuple(text for text in author_lessons if text),
            retrieval_text=retrieval_text,
        )


@dataclass(frozen=True)
class PredictionInput:
    """Paper-only view passed across the predictor information boundary."""

    forum_id: str
    paper_signals: Tuple[Tuple[str, float], ...]
    retrieval_text: str = ""

    def __post_init__(self) -> None:
        if not self.forum_id.strip():
            raise ValueError("forum_id cannot be empty")
        object.__setattr__(
            self,
            "paper_signals",
            _validated_pairs(self.paper_signals, "paper_signals"),
        )

    @classmethod
    def from_example(cls, example: LearningExample) -> "PredictionInput":
        return cls(example.forum_id, example.paper_signals, example.retrieval_text)


DEFAULT_SEED_SIGNALS = tuple((name, 2.5) for name in CRITERIA_FIELDS)


def _seed_number(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, Mapping) and "value" in value:
        value = value["value"]
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return _finite_number(value)
    text = str(value).strip()
    if not text or text.casefold() in ("null", "none", "n/a", "nan"):
        return None
    match = re.search(r"[-+]?\d+(?:\.\d+)?", text)
    return float(match.group(0)) if match else None


def _seed_local_rating(value: Any, venue: str) -> Optional[float]:
    """Map common ICLR/OpenReview recommendations to the local 1--6 scale."""

    text = str(value or "").casefold().replace("_", " ")
    labels = (
        ("strong reject", 1.0),
        ("weak reject", 3.0),
        ("strong accept", 6.0),
        ("weak accept", 4.0),
    )
    for label, score in labels:
        if label in text:
            return score
    number = _seed_number(value)
    if number is None:
        if "reject" in text:
            return 2.0
        if "accept" in text:
            return 5.0
        return None
    # ICLR historically uses a 1--10 recommendation scale.  Preserve local
    # 1--6 values for other venues unless a value above six proves otherwise.
    if "iclr" in venue.casefold() or number > 6.0:
        if number >= 9.0:
            return 6.0
        if number >= 7.0:
            return 5.0
        if number >= 5.5:
            return 4.0
        if number >= 4.0:
            return 3.0
        if number >= 2.0:
            return 2.0
        return 1.0
    return _clip(number, 1.0, 6.0)


def _seed_accepted(decision: Any) -> Optional[bool]:
    text = re.sub(r"[^a-z]+", " ", str(decision or "").casefold()).strip()
    if not text:
        return None
    if any(marker in text for marker in ("reject", "withdraw", "not accept")):
        return False
    if "accept" in text:
        return True
    return None


def _seed_aspect_scores(value: Any) -> Dict[str, float]:
    text = str(value or "")
    collected: Dict[str, List[float]] = {}
    patterns = {
        "soundness": ("correctness", "soundness", "quality"),
        "presentation": ("presentation", "clarity"),
        "significance": ("significance", "contribution"),
        "originality": ("originality",),
    }
    for target, aliases in patterns.items():
        for alias in aliases:
            match = re.search(r"(?im)^\s*%s\s*:\s*(\d+(?:\.\d+)?)" % re.escape(alias), text)
            if match:
                collected.setdefault(target, []).append(
                    _clip(float(match.group(1)), 1.0, 4.0)
                )
                break
    novelty = []
    for alias in ("technical_novelty_and_significance", "empirical_novelty_and_significance"):
        match = re.search(r"(?im)^\s*%s\s*:\s*(\d+(?:\.\d+)?)" % re.escape(alias), text)
        if match:
            novelty.append(_clip(float(match.group(1)), 1.0, 4.0))
    if novelty:
        collected.setdefault("significance", []).extend(novelty)
        collected.setdefault("originality", []).extend(novelty)
    return {
        name: sum(values) / len(values) for name, values in collected.items()
    }


def seed_case_to_learning_example(
    case: Mapping[str, Any],
    paper_signals: Optional[Sequence[Tuple[str, float]]] = None,
) -> LearningExample:
    """Convert one provenance-preserving real seed case to trainable supervision.

    The default paper signals are neutral placeholders, never human ratings.  A
    real reviewer run should pass its paper-only criterion outputs explicitly.
    """

    if not isinstance(case, Mapping):
        raise ValueError("seed case must be a JSON object")
    forum_id = str(case.get("forum_id", "")).strip()
    if not forum_id:
        raise ValueError("seed case has no forum_id")
    venue = str(case.get("conference_year_track", ""))
    reviews = case.get("reviews", [])
    if not isinstance(reviews, list):
        raise ValueError("seed case reviews must be a list")
    overall_values: List[float] = []
    confidence_values: List[float] = []
    aspects: Dict[str, List[float]] = {}
    reviewer_lessons: List[str] = []
    for review in reviews:
        if not isinstance(review, Mapping):
            continue
        final_score = review.get("final_score", {})
        initial_score = review.get("initial_score", {})
        if not isinstance(final_score, Mapping):
            final_score = {}
        if not isinstance(initial_score, Mapping):
            initial_score = {}
        rating_value = final_score.get("rating")
        if _seed_number(rating_value) is None:
            rating_value = initial_score.get("rating")
        rating = _seed_local_rating(rating_value, venue)
        if rating is not None:
            overall_values.append(rating)
        confidence_value = final_score.get("confidence")
        if _seed_number(confidence_value) is None:
            confidence_value = initial_score.get("confidence")
        confidence = _seed_number(confidence_value)
        if confidence is not None:
            confidence_values.append(_clip(confidence, 1.0, 5.0))
        aspect_value = final_score.get("aspect_score") or initial_score.get("aspect_score")
        for name, value in _seed_aspect_scores(aspect_value).items():
            aspects.setdefault(name, []).append(value)
        review_text = str(review.get("review_content", "")).strip()
        if review_text:
            reviewer_lessons.append(review_text)
    metareview = str(case.get("metareview", "")).strip()
    if metareview:
        reviewer_lessons.append(metareview)

    targets = []
    for name, values in aspects.items():
        targets.append((name, sum(values) / len(values)))
    if overall_values:
        targets.append(("overall_recommendation", sum(overall_values) / len(overall_values)))
    if confidence_values:
        targets.append(("confidence", sum(confidence_values) / len(confidence_values)))

    author_lessons = []
    dialogues = case.get("rebuttal_dialogues", [])
    if isinstance(dialogues, list):
        for dialogue in dialogues:
            if not isinstance(dialogue, Mapping):
                continue
            messages = dialogue.get("messages", [])
            if not isinstance(messages, list):
                continue
            for message in messages:
                if not isinstance(message, Mapping) or str(message.get("role", "")) != "user":
                    continue
                text = str(message.get("content", "")).strip()
                if text:
                    author_lessons.append(text)

    paper = case.get("paper", {})
    if not isinstance(paper, Mapping):
        paper = {}
    retrieval_text = "\n".join(
        str(value).strip()
        for value in (
            paper.get("title", ""),
            paper.get("abstract", ""),
            paper.get("primary_area", ""),
            " ".join(str(item) for item in (paper.get("keywords", []) or [])),
        )
        if str(value).strip()
    )
    return LearningExample(
        forum_id=forum_id,
        paper_signals=tuple(paper_signals or DEFAULT_SEED_SIGNALS),
        target_scores=tuple(targets),
        accepted=_seed_accepted(case.get("decision")),
        reviewer_lessons=tuple(reviewer_lessons),
        author_lessons=tuple(author_lessons),
        retrieval_text=retrieval_text,
    )


def load_seed_examples(
    path: Any,
    signal_builder: Optional[Callable[[Mapping[str, Any]], Sequence[Tuple[str, float]]]] = None,
) -> Tuple[LearningExample, ...]:
    """Load either a JSON array or JSONL seed corpus into immutable examples."""

    source = Path(path)
    text = source.read_text(encoding="utf-8")
    stripped = text.lstrip()
    try:
        if stripped.startswith("["):
            records = json.loads(text)
        else:
            records = [json.loads(line) for line in text.splitlines() if line.strip()]
    except json.JSONDecodeError as exc:
        raise ValueError("invalid seed JSON in %s: %s" % (source, exc))
    if not isinstance(records, list):
        raise ValueError("seed JSON root must be an array or JSONL records")
    examples = []
    for record in records:
        signals = signal_builder(record) if signal_builder is not None else None
        examples.append(seed_case_to_learning_example(record, signals))
    if len({example.forum_id for example in examples}) != len(examples):
        raise ValueError("seed corpus contains duplicate forum IDs")
    return tuple(examples)


@dataclass(frozen=True)
class EvaluationResult:
    n_examples: int
    field_coverage: float
    complete_coverage: float
    mae: Optional[float]
    normalized_mae: Optional[float]
    per_field_mae: Tuple[Tuple[str, float], ...]
    brier: Optional[float]
    utility: float


@dataclass(frozen=True)
class LearningConfig:
    min_iterations: int = 2
    max_iterations: int = 12
    patience: int = 3
    epsilon_quality: float = 1e-4
    epsilon_behavior: float = 1e-3
    epsilon_state: float = 1e-3
    non_regression_tolerance: float = 1e-6
    learning_rate: float = 0.25
    decision_calibration_weight: float = 0.25
    memory_limit: int = 256

    def __post_init__(self) -> None:
        if self.min_iterations < 1:
            raise ValueError("min_iterations must be at least one")
        if self.max_iterations < self.min_iterations:
            raise ValueError("max_iterations must be >= min_iterations")
        if self.patience < 1:
            raise ValueError("patience must be at least one")
        for name in ("epsilon_quality", "epsilon_behavior", "epsilon_state", "non_regression_tolerance"):
            value = getattr(self, name)
            if _finite_number(value) is None or value < 0:
                raise ValueError("%s must be finite and non-negative" % name)
        if _finite_number(self.learning_rate) is None or self.learning_rate < 0 or self.learning_rate > 1:
            raise ValueError("learning_rate must be in [0, 1]")
        if (
            _finite_number(self.decision_calibration_weight) is None
            or self.decision_calibration_weight < 0
            or self.decision_calibration_weight > 1
        ):
            raise ValueError("decision_calibration_weight must be in [0, 1]")
        if self.memory_limit < 0:
            raise ValueError("memory_limit cannot be negative")


@dataclass(frozen=True)
class IterationRecord:
    iteration: int
    candidate_version: int
    accepted: bool
    utility: float
    quality_improvement: float
    behavioral_delta: float
    state_delta: float
    plateau_count: int
    evaluation: EvaluationResult


@dataclass(frozen=True)
class LearningRun:
    state: LearningState
    best_iteration: int
    stop_reason: str
    history: Tuple[IterationRecord, ...]
    dev_evaluation: EvaluationResult
    test_evaluation: EvaluationResult
    test_fingerprint: str


def _base_overall(state: LearningState, example: PredictionInput) -> float:
    signals = _pairs_to_dict(example.paper_signals)
    weights = _pairs_to_dict(state.rubric_weights)
    criterion_average = sum(
        weights[field_name] * _clip(signals.get(field_name, 2.5), 1.0, 4.0)
        for field_name in CRITERIA_FIELDS
    )
    return 1.0 + (criterion_average - 1.0) * (5.0 / 3.0)


def _accept_probability(overall: float) -> float:
    logit = _clip((overall - 3.5) * 1.2, -30.0, 30.0)
    return 1.0 / (1.0 + math.exp(-logit))


def retrieve_memory(
    state: LearningState,
    role: str,
    query_text: str,
    limit: int = 3,
    minimum_relevance: float = DEFAULT_RETRIEVAL_RELEVANCE,
    exclude_forum_ids: Iterable[str] = (),
) -> Tuple[MemoryItem, ...]:
    """Return deterministic lexical retrieval from one role's isolated memory."""

    if role not in ("reviewer", "author"):
        raise ValueError("role must be reviewer or author")
    if limit < 0:
        raise ValueError("limit cannot be negative")
    if _finite_number(minimum_relevance) is None or not 0.0 <= minimum_relevance <= 1.0:
        raise ValueError("minimum_relevance must be in [0, 1]")
    if isinstance(exclude_forum_ids, str):
        excluded = {exclude_forum_ids}
    else:
        excluded = {str(value) for value in exclude_forum_ids}
    return tuple(
        item for item, _relevance in _rank_memory(state, role, query_text)
        if _relevance >= minimum_relevance and item.forum_id not in excluded
    )[:limit]


def _rank_memory(
    state: LearningState,
    role: str,
    query_text: str,
) -> Tuple[Tuple[MemoryItem, float], ...]:
    """Rank memory while keeping source and score internal to the predictor."""

    if role not in ("reviewer", "author"):
        raise ValueError("role must be reviewer or author")
    memory = state.reviewer_memory if role == "reviewer" else state.author_memory
    query = _retrieval_tokens(query_text)

    def relevance(item: MemoryItem) -> float:
        # Reviews contain broad ML vocabulary that creates false matches.  The
        # retrieval key is only the source paper's title/abstract cue.
        memory_tokens = _retrieval_tokens(item.cue)
        intersection = query.intersection(memory_tokens)
        if len(intersection) < 2:
            return 0.0
        union = query.union(memory_tokens)
        return len(intersection) / float(len(union)) if union else 0.0

    scored = [(item, relevance(item)) for item in memory]
    return tuple(sorted(scored, key=lambda pair: (-pair[1], pair[0].identity)))


def memory_guidance(text: str, role: str) -> str:
    """Generalize raw train memory into paper-agnostic role guidance.

    Raw human reviews and rebuttals remain internal retrieval material.  They are
    never copied into a new paper's reviewer Comment or live agent payload.
    """

    if role not in ("reviewer", "author"):
        raise ValueError("role must be reviewer or author")
    matched = []
    for label, patterns in _MEMORY_GUIDANCE_AXES:
        if any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in patterns):
            matched.append(label)
    if not matched:
        matched = ["claims, evidence, limitations, and reproducibility"]
    focus = "; ".join(matched[:4])
    if role == "reviewer":
        return (
            "Improve the AI review agent by independently checking {} against "
            "evidence in the current paper; improve the paper with concrete, "
            "constructive revisions. Do not transfer facts or judgments from "
            "training cases."
        ).format(focus)
    return (
        "For the current paper, address concerns about {} point by point using only "
        "existing current-paper evidence; concede unsupported claims and do not copy "
        "facts from training rebuttals."
    ).format(focus)


def _retrieve_comment(
    state: LearningState,
    example: PredictionInput,
) -> Tuple[str, Mapping[str, Any]]:
    ranked = _rank_memory(state, "reviewer", example.retrieval_text)
    ranked = tuple(pair for pair in ranked if pair[0].forum_id != example.forum_id)
    if not ranked or ranked[0][1] < DEFAULT_RETRIEVAL_RELEVANCE:
        best_relevance = ranked[0][1] if ranked else 0.0
        return GENERIC_REVIEW_COMMENT, {
            "matched": False,
            "source_forum": None,
            "memory_id": None,
            "relevance": best_relevance,
        }
    item, relevance = ranked[0]
    return memory_guidance(item.text, "reviewer"), {
        "matched": True,
        "source_forum": item.forum_id,
        "memory_id": item.identity,
        "relevance": relevance,
    }


def author_memory_context(
    state: LearningState,
    prediction_input: PredictionInput,
    limit: int = 3,
) -> Tuple[str, ...]:
    """Expose author-only lessons without mixing them into reviewer memory."""

    return tuple(
        dict.fromkeys(
            memory_guidance(item.text, "author")
            for item in retrieve_memory(
                state,
                "author",
                prediction_input.retrieval_text,
                limit,
                exclude_forum_ids=(prediction_input.forum_id,),
            )
        )
    )


def predict_review(state: LearningState, example: PredictionInput) -> Mapping[str, Any]:
    """Apply rubric/calibration state to paper-only signals."""

    signals = _pairs_to_dict(example.paper_signals)
    output: Dict[str, Any] = {}
    for field_name in CRITERIA_FIELDS:
        value = _clip(signals.get(field_name, 2.5), 1.0, 4.0)
        output[field_name] = int(math.floor(value + 0.5))
    continuous_overall = state.calibration_scale * _base_overall(state, example) + state.calibration_bias
    continuous_overall = _clip(continuous_overall, 1.0, 6.0)
    output["overall_recommendation"] = int(math.floor(continuous_overall + 0.5))
    confidence = _clip(signals.get("confidence", 3.0), 1.0, 5.0)
    output["confidence"] = int(math.floor(confidence + 0.5))
    comment, retrieval_metadata = _retrieve_comment(state, example)
    output["comment"] = comment
    output["_retrieval"] = retrieval_metadata
    output["accept_probability"] = _accept_probability(continuous_overall)
    return output


def predict_many(
    state: LearningState,
    examples: Sequence[LearningExample],
    predictor: Callable[[LearningState, PredictionInput], Mapping[str, Any]] = predict_review,
) -> Dict[str, Mapping[str, Any]]:
    predictions = {}
    for example in examples:
        if example.forum_id in predictions:
            raise ValueError("duplicate example forum_id: %s" % example.forum_id)
        prediction = predictor(state, PredictionInput.from_example(example))
        if not isinstance(prediction, Mapping):
            raise TypeError("predictor must return a mapping")
        predictions[example.forum_id] = dict(prediction)
    return predictions


def _valid_output_field(field_name: str, value: Any) -> bool:
    if field_name == "comment":
        return isinstance(value, str) and bool(value.strip())
    if field_name in SCORE_FIELDS and type(value) is not int:
        return False
    number = _finite_number(value)
    if number is None or field_name not in SCORE_RANGES:
        return False
    lower, upper = SCORE_RANGES[field_name]
    return lower <= number <= upper


def evaluate_predictions(
    predictions: Mapping[str, Mapping[str, Any]],
    examples: Sequence[LearningExample],
) -> EvaluationResult:
    """Evaluate form coverage, rating MAE, and accept-probability Brier score."""

    if not examples:
        return EvaluationResult(0, 0.0, 0.0, None, None, (), None, 0.0)
    field_hits = 0
    complete_hits = 0
    absolute_errors: List[float] = []
    normalized_errors: List[float] = []
    errors_by_field: Dict[str, List[float]] = {}
    brier_terms: List[float] = []
    for example in examples:
        prediction = predictions.get(example.forum_id, {})
        if not isinstance(prediction, Mapping):
            prediction = {}
        valid = [_valid_output_field(name, prediction.get(name)) for name in REQUIRED_REVIEW_FIELDS]
        field_hits += sum(valid)
        complete_hits += int(all(valid))
        targets = _pairs_to_dict(example.target_scores)
        for field_name, target in targets.items():
            if field_name not in SCORE_RANGES or not _valid_output_field(field_name, prediction.get(field_name)):
                continue
            predicted = float(prediction[field_name])
            error = abs(predicted - target)
            width = SCORE_RANGES[field_name][1] - SCORE_RANGES[field_name][0]
            absolute_errors.append(error)
            normalized_errors.append(error / width if width else 0.0)
            errors_by_field.setdefault(field_name, []).append(error)
        probability = _finite_number(prediction.get("accept_probability"))
        if example.accepted is not None and probability is not None and 0.0 <= probability <= 1.0:
            brier_terms.append((probability - float(example.accepted)) ** 2)
    n_examples = len(examples)
    coverage = field_hits / float(n_examples * len(REQUIRED_REVIEW_FIELDS))
    complete = complete_hits / float(n_examples)
    mae = sum(absolute_errors) / len(absolute_errors) if absolute_errors else None
    normalized_mae = sum(normalized_errors) / len(normalized_errors) if normalized_errors else None
    brier = sum(brier_terms) / len(brier_terms) if brier_terms else None
    components = [coverage, complete]
    if normalized_mae is not None:
        components.append(1.0 - _clip(normalized_mae, 0.0, 1.0))
    if brier is not None:
        components.append(1.0 - _clip(brier, 0.0, 1.0))
    utility = sum(components) / len(components)
    per_field = tuple(sorted(
        (field_name, sum(values) / len(values)) for field_name, values in errors_by_field.items()
    ))
    return EvaluationResult(
        n_examples=n_examples,
        field_coverage=coverage,
        complete_coverage=complete,
        mae=mae,
        normalized_mae=normalized_mae,
        per_field_mae=per_field,
        brier=brier,
        utility=utility,
    )


def behavioral_delta(
    previous: Mapping[str, Mapping[str, Any]],
    current: Mapping[str, Mapping[str, Any]],
) -> float:
    """Measure output change on a fixed probe set in the unit interval."""

    forum_ids = sorted(set(previous).union(current))
    if not forum_ids:
        return 0.0
    forum_deltas = []
    for forum_id in forum_ids:
        if forum_id not in previous or forum_id not in current:
            forum_deltas.append(1.0)
            continue
        left, right = previous[forum_id], current[forum_id]
        values = []
        for field_name in SCORE_FIELDS + ("accept_probability",):
            left_value, right_value = _finite_number(left.get(field_name)), _finite_number(right.get(field_name))
            if left_value is None and right_value is None:
                continue
            if left_value is None or right_value is None:
                values.append(1.0)
                continue
            lower, upper = SCORE_RANGES[field_name]
            values.append(min(1.0, abs(left_value - right_value) / (upper - lower)))
        values.append(_jaccard_distance(str(left.get("comment", "")), str(right.get("comment", ""))))
        forum_deltas.append(sum(values) / len(values))
    return sum(forum_deltas) / len(forum_deltas)


def state_delta(previous: LearningState, current: LearningState) -> float:
    """Measure rubric, calibration, prompt manifest, and memory change in [0, 1]."""

    left_weights, right_weights = _pairs_to_dict(previous.rubric_weights), _pairs_to_dict(current.rubric_weights)
    weight_delta = sum(abs(left_weights[key] - right_weights[key]) for key in CRITERIA_FIELDS) / 2.0
    scale_delta = min(1.0, abs(previous.calibration_scale - current.calibration_scale) / 2.0)
    bias_delta = min(1.0, abs(previous.calibration_bias - current.calibration_bias) / 5.0)
    left_memory = {item.identity for item in previous.reviewer_memory + previous.author_memory}
    right_memory = {item.identity for item in current.reviewer_memory + current.author_memory}
    union = left_memory.union(right_memory)
    memory_delta = 1.0 - len(left_memory.intersection(right_memory)) / float(len(union)) if union else 0.0
    manifest_delta = float(previous.prompt_manifest_digest != current.prompt_manifest_digest)
    return _clip(
        (weight_delta + scale_delta + bias_delta + manifest_delta + memory_delta) / 5.0,
        0.0,
        1.0,
    )


def is_non_regression(
    candidate: EvaluationResult,
    reference: EvaluationResult,
    tolerance: float = 0.0,
) -> bool:
    """Require every available quality axis to stay within its regression budget."""

    if candidate.field_coverage + tolerance < reference.field_coverage:
        return False
    if candidate.complete_coverage + tolerance < reference.complete_coverage:
        return False
    for name in ("normalized_mae", "brier"):
        candidate_value, reference_value = getattr(candidate, name), getattr(reference, name)
        if reference_value is not None and candidate_value is None:
            return False
        if reference_value is not None and candidate_value is not None and candidate_value > reference_value + tolerance:
            return False
    return True


def _target_overall(example: LearningExample) -> Optional[float]:
    targets = _pairs_to_dict(example.target_scores)
    if "overall_recommendation" in targets:
        return targets["overall_recommendation"]
    criteria = [targets[name] for name in CRITERIA_FIELDS if name in targets]
    if not criteria:
        return None
    mean = sum(criteria) / len(criteria)
    return 1.0 + (mean - 1.0) * (5.0 / 3.0)


def _merge_memories(
    existing: Sequence[MemoryItem],
    new_items: Sequence[MemoryItem],
    limit: int,
) -> Tuple[MemoryItem, ...]:
    """Select an idempotent, forum-stratified bounded memory set."""

    if limit < 0:
        raise ValueError("memory limit cannot be negative")
    if limit == 0:
        return ()

    merged: Dict[str, MemoryItem] = {
        item.identity: item for item in tuple(existing) + tuple(new_items)
    }
    grouped: Dict[str, List[MemoryItem]] = {}
    for item in merged.values():
        grouped.setdefault(item.forum_id, []).append(item)
    for items in grouped.values():
        items.sort(key=lambda item: item.identity)

    # Hash ordering avoids favoring lexically early forum IDs. Round-robin depth
    # guarantees one item per forum whenever the capacity permits it.
    forum_ids = sorted(
        grouped,
        key=lambda forum_id: (_canonical_hash(("memory-forum", forum_id)), forum_id),
    )
    selected: List[MemoryItem] = []
    depth = 0
    while len(selected) < limit:
        added = False
        for forum_id in forum_ids:
            items = grouped[forum_id]
            if depth >= len(items):
                continue
            selected.append(items[depth])
            added = True
            if len(selected) == limit:
                break
        if not added:
            break
        depth += 1
    return tuple(selected)


def propose_update(
    state: LearningState,
    train_examples: Sequence[LearningExample],
    config: LearningConfig,
) -> LearningState:
    """Create one train-only rubric, calibration, and dual-memory update."""

    if len({example.forum_id for example in train_examples}) != len(train_examples):
        raise ValueError("training examples contain duplicate forum IDs")
    rows = []
    for example in train_examples:
        target = _target_overall(example)
        if target is not None:
            prediction_input = PredictionInput.from_example(example)
            rows.append((example, _base_overall(state, prediction_input), target))

    new_scale = state.calibration_scale
    new_bias = state.calibration_bias
    if rows and config.learning_rate > 0:
        bases = [row[1] for row in rows]
        targets = [row[2] for row in rows]
        mean_base, mean_target = sum(bases) / len(bases), sum(targets) / len(targets)
        variance = sum((value - mean_base) ** 2 for value in bases)
        if variance > 1e-12:
            slope = sum((base - mean_base) * (target - mean_target) for base, target in zip(bases, targets)) / variance
            slope = _clip(slope, 0.5, 1.5)
            new_scale = (1.0 - config.learning_rate) * state.calibration_scale + config.learning_rate * slope
        desired_bias = mean_target - new_scale * mean_base
        new_bias = (1.0 - config.learning_rate) * state.calibration_bias + config.learning_rate * desired_bias
        new_bias = _clip(new_bias, -5.0, 5.0)

    # Final decisions provide a separate train-only calibration signal.  This
    # small residual step cannot bypass the dev MAE/Brier non-regression gate.
    decision_residuals = []
    if config.learning_rate > 0 and config.decision_calibration_weight > 0:
        for example in train_examples:
            if example.accepted is None:
                continue
            prediction_input = PredictionInput.from_example(example)
            continuous = _clip(
                new_scale * _base_overall(state, prediction_input) + new_bias,
                1.0,
                6.0,
            )
            probability = _accept_probability(continuous)
            decision_residuals.append(float(example.accepted) - probability)
    if decision_residuals:
        new_bias += (
            config.learning_rate
            * config.decision_calibration_weight
            * sum(decision_residuals)
            / len(decision_residuals)
        )
        new_bias = _clip(new_bias, -5.0, 5.0)

    old_weights = _pairs_to_dict(state.rubric_weights)
    evidence_weights: Dict[str, float] = {}
    for field_name in CRITERIA_FIELDS:
        errors = []
        for example, _, target in rows:
            signal = _pairs_to_dict(example.paper_signals).get(field_name)
            if signal is not None:
                mapped = 1.0 + (_clip(signal, 1.0, 4.0) - 1.0) * (5.0 / 3.0)
                errors.append(abs(mapped - target))
        if errors:
            evidence_weights[field_name] = 1.0 / (1e-6 + sum(errors) / len(errors))
    if evidence_weights and config.learning_rate > 0:
        total = sum(evidence_weights.values())
        desired = {
            name: evidence_weights.get(name, 0.0) / total for name in CRITERIA_FIELDS
        }
        blended = {
            name: (1.0 - config.learning_rate) * old_weights[name] + config.learning_rate * desired[name]
            for name in CRITERIA_FIELDS
        }
        normalization = sum(blended.values())
        weights = tuple((name, blended[name] / normalization) for name in CRITERIA_FIELDS)
    else:
        weights = state.rubric_weights

    reviewer_items, author_items = [], []
    for example in train_examples:
        cue = example.retrieval_text[:1000]
        reviewer_items.extend(
            MemoryItem("reviewer", example.forum_id, text[:4000], cue)
            for text in example.reviewer_lessons
        )
        author_items.extend(
            MemoryItem("author", example.forum_id, text[:4000], cue)
            for text in example.author_lessons
        )
    return LearningState(
        version=state.version + 1,
        rubric_weights=weights,
        calibration_scale=new_scale,
        calibration_bias=new_bias,
        prompt_manifest_digest=state.prompt_manifest_digest,
        reviewer_memory=_merge_memories(state.reviewer_memory, reviewer_items, config.memory_limit),
        author_memory=_merge_memories(state.author_memory, author_items, config.memory_limit),
        parent_digest=state.digest,
    )


def _dataset_fingerprint(examples: Sequence[LearningExample]) -> str:
    return _canonical_hash(tuple((example.forum_id, example.fingerprint) for example in examples))


def _validate_splits(
    train: Sequence[LearningExample],
    dev: Sequence[LearningExample],
    test: Sequence[LearningExample],
    state: LearningState,
) -> None:
    split_ids = []
    for label, examples in (("train", train), ("dev", dev), ("test", test)):
        ids = [example.forum_id for example in examples]
        if len(ids) != len(set(ids)):
            raise DataLeakageError("%s split contains duplicate forum IDs" % label)
        split_ids.append(set(ids))
    if split_ids[0].intersection(split_ids[1]) or split_ids[0].intersection(split_ids[2]) or split_ids[1].intersection(split_ids[2]):
        raise DataLeakageError("forum IDs overlap across train/dev/test")
    forbidden = split_ids[1].union(split_ids[2])
    memory_forums = {item.forum_id for item in state.reviewer_memory + state.author_memory}
    leaked = forbidden.intersection(memory_forums)
    if leaked:
        raise DataLeakageError("initial state memory contains held-out forums: %s" % sorted(leaked))


def run_learning_loop(
    train_examples: Iterable[LearningExample],
    dev_examples: Iterable[LearningExample],
    test_examples: Iterable[LearningExample],
    initial_state: Optional[LearningState] = None,
    config: Optional[LearningConfig] = None,
    predictor: Callable[[LearningState, PredictionInput], Mapping[str, Any]] = predict_review,
) -> LearningRun:
    """Update on train, stop on dev, restore best state, then evaluate test once."""

    train, dev, test = tuple(train_examples), tuple(dev_examples), tuple(test_examples)
    if not train:
        raise ValueError("train_examples cannot be empty")
    if not dev:
        raise ValueError("dev_examples cannot be empty")
    state = initial_state or LearningState()
    resolved_config = config or LearningConfig()
    _validate_splits(train, dev, test, state)
    test_fingerprint = _dataset_fingerprint(test)

    current_predictions = predict_many(state, dev, predictor)
    current_evaluation = evaluate_predictions(current_predictions, dev)
    best_state, best_evaluation, best_iteration = state, current_evaluation, 0
    history: List[IterationRecord] = []
    plateau_count = 0
    stop_reason = "max_iterations"

    for iteration in range(1, resolved_config.max_iterations + 1):
        candidate = propose_update(state, train, resolved_config)
        candidate_predictions = predict_many(candidate, dev, predictor)
        candidate_evaluation = evaluate_predictions(candidate_predictions, dev)
        accepted = is_non_regression(
            candidate_evaluation,
            current_evaluation,
            resolved_config.non_regression_tolerance,
        )
        previous_state = state
        previous_predictions = current_predictions
        previous_evaluation = current_evaluation
        if accepted:
            state = candidate
            current_predictions = candidate_predictions
            current_evaluation = candidate_evaluation
        improvement = current_evaluation.utility - previous_evaluation.utility
        behavior_change = behavioral_delta(previous_predictions, current_predictions)
        state_change = state_delta(previous_state, state)

        if accepted and current_evaluation.utility + 1e-12 >= best_evaluation.utility and is_non_regression(
            current_evaluation, best_evaluation, resolved_config.non_regression_tolerance
        ):
            best_state, best_evaluation, best_iteration = state, current_evaluation, iteration

        plateau = (
            iteration >= resolved_config.min_iterations
            and abs(improvement) <= resolved_config.epsilon_quality
            and behavior_change <= resolved_config.epsilon_behavior
            and state_change <= resolved_config.epsilon_state
        )
        plateau_count = plateau_count + 1 if plateau else 0
        history.append(IterationRecord(
            iteration=iteration,
            candidate_version=candidate.version,
            accepted=accepted,
            utility=current_evaluation.utility,
            quality_improvement=improvement,
            behavioral_delta=behavior_change,
            state_delta=state_change,
            plateau_count=plateau_count,
            evaluation=current_evaluation,
        ))
        if plateau_count >= resolved_config.patience:
            stop_reason = "converged"
            break

    # Best-state restore occurs before the only test prediction/evaluation call.
    state = best_state
    current_evaluation = best_evaluation
    held_out_forums = {example.forum_id for example in dev + test}
    memory_forums = {item.forum_id for item in state.reviewer_memory + state.author_memory}
    if held_out_forums.intersection(memory_forums):
        raise DataLeakageError("updated state contains held-out forum memory")
    if _dataset_fingerprint(test) != test_fingerprint:
        raise DataLeakageError("immutable test examples changed during learning")
    test_predictions = predict_many(state, test, predictor)
    test_evaluation = evaluate_predictions(test_predictions, test)
    if _dataset_fingerprint(test) != test_fingerprint:
        raise DataLeakageError("test examples changed during evaluation")
    return LearningRun(
        state=state,
        best_iteration=best_iteration,
        stop_reason=stop_reason,
        history=tuple(history),
        dev_evaluation=current_evaluation,
        test_evaluation=test_evaluation,
        test_fingerprint=test_fingerprint,
    )
