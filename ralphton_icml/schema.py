"""Typed contracts shared by the reviewer-agent pipeline.

The extraction contract and the final-review contract intentionally use
different types.  Extraction workers emit ``ExtractionOutput`` objects using
the ANSWER/SOURCES format from ``prompts.py``; only the final chair emits a
``ReviewOutput`` matching ``reviewer_instruction.md``.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any, Dict, Iterable, Optional, Tuple


class ReviewValidationError(ValueError):
    """Raised when a final review violates the ICML reviewer form."""


class ReviewParseError(ReviewValidationError):
    """Raised when Markdown does not match the final-review contract."""


_REVIEW_FIELDS = (
    ("soundness", "#### **Soundness**", 1, 4),
    ("presentation", "#### **Presentation**", 1, 4),
    ("significance", "#### **Significance**", 1, 4),
    ("originality", "#### **Originality**", 1, 4),
    ("overall_recommendation", "#### **Overall Recommendation**", 1, 6),
    ("confidence", "#### **Confidence**", 1, 5),
)
_COMMENT_HEADING = "#### Comment"


def _canonical_text(value: Any, field_name: str) -> str:
    if not isinstance(value, str):
        raise ValueError("{} must be a string".format(field_name))
    canonical = value.replace("\r\n", "\n").replace("\r", "\n").strip()
    if not canonical:
        raise ValueError("{} must not be empty".format(field_name))
    if "\x00" in canonical:
        raise ValueError("{} must not contain NUL characters".format(field_name))
    return canonical


def validate_comment(comment: Any) -> str:
    """Validate and canonicalize the required constructive Comment field."""

    try:
        return _canonical_text(comment, "comment")
    except ValueError as exc:
        raise ReviewValidationError(str(exc)) from exc


def _validate_score(name: str, value: Any, minimum: int, maximum: int) -> int:
    # bool is an int subclass, but accepting True as score 1 is never intended.
    if type(value) is not int:
        raise ReviewValidationError("{} must be an integer".format(name))
    if not minimum <= value <= maximum:
        raise ReviewValidationError(
            "{} must be in the inclusive range {}..{}".format(
                name, minimum, maximum
            )
        )
    return value


@dataclass(frozen=True)
class ReviewOutput:
    """The exact seven-field final output required by reviewer_instruction.md."""

    soundness: int
    presentation: int
    significance: int
    originality: int
    overall_recommendation: int
    confidence: int
    comment: str

    def __post_init__(self) -> None:
        for name, _heading, minimum, maximum in _REVIEW_FIELDS:
            _validate_score(name, getattr(self, name), minimum, maximum)
        object.__setattr__(self, "comment", validate_comment(self.comment))

    def to_markdown(self) -> str:
        """Render the canonical reviewer form without extraction headings."""

        return render_review_output(self)

    @classmethod
    def from_markdown(cls, markdown: str) -> "ReviewOutput":
        """Parse canonical reviewer-form Markdown."""

        return parse_review_output(markdown)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "soundness": self.soundness,
            "presentation": self.presentation,
            "significance": self.significance,
            "originality": self.originality,
            "overall_recommendation": self.overall_recommendation,
            "confidence": self.confidence,
            "comment": self.comment,
        }


def render_review_output(review: ReviewOutput) -> str:
    """Render a validated review using the exact rubric heading order."""

    if not isinstance(review, ReviewOutput):
        raise TypeError("review must be a ReviewOutput")
    blocks = []
    for name, heading, _minimum, _maximum in _REVIEW_FIELDS:
        blocks.append("{}\n\n{}".format(heading, getattr(review, name)))
    blocks.append("{}\n\n{}".format(_COMMENT_HEADING, review.comment))
    return "\n\n".join(blocks) + "\n"


def parse_review_output(markdown: str) -> ReviewOutput:
    """Strictly parse Markdown produced by :func:`render_review_output`.

    Headings, field order, blank lines, and integer-only score payloads are
    deliberate.  In particular, an extraction response headed ANSWER/SOURCES
    cannot accidentally be accepted as a final review.
    """

    if not isinstance(markdown, str):
        raise ReviewParseError("review Markdown must be a string")
    normalized = markdown.replace("\r\n", "\n").replace("\r", "\n")
    if normalized.endswith("\n"):
        normalized = normalized[:-1]
    lines = normalized.split("\n")
    position = 0
    values: Dict[str, Any] = {}

    def require_line(expected: str) -> None:
        nonlocal position
        if position >= len(lines) or lines[position] != expected:
            actual = "<end>" if position >= len(lines) else repr(lines[position])
            raise ReviewParseError(
                "expected {!r} at line {}, got {}".format(
                    expected, position + 1, actual
                )
            )
        position += 1

    for name, heading, minimum, maximum in _REVIEW_FIELDS:
        require_line(heading)
        require_line("")
        if position >= len(lines):
            raise ReviewParseError("missing score for {}".format(name))
        score_text = lines[position]
        position += 1
        if not score_text.isascii() or not score_text.isdigit():
            raise ReviewParseError("{} score must be an integer".format(name))
        try:
            values[name] = _validate_score(name, int(score_text), minimum, maximum)
        except ReviewValidationError as exc:
            raise ReviewParseError(str(exc)) from exc
        require_line("")

    require_line(_COMMENT_HEADING)
    require_line("")
    if position >= len(lines):
        raise ReviewParseError("missing Comment")
    try:
        values["comment"] = validate_comment("\n".join(lines[position:]))
    except ReviewValidationError as exc:
        raise ReviewParseError(str(exc)) from exc
    return ReviewOutput(**values)


@dataclass(frozen=True)
class ContextTask:
    """One leaf prompt from the general-paper prompt map."""

    task_id: str
    section: str
    item: str
    prompt: str
    tier: str
    ordinal: int

    def __post_init__(self) -> None:
        for name in ("task_id", "section", "item", "prompt"):
            object.__setattr__(self, name, _canonical_text(getattr(self, name), name))
        if self.tier not in {"A", "B", "C", "D"}:
            raise ValueError("tier must be one of A, B, C, or D")
        if type(self.ordinal) is not int or self.ordinal < 0:
            raise ValueError("ordinal must be a non-negative integer")

    @property
    def name(self) -> str:
        return self.item


@dataclass(frozen=True)
class ExtractionOutput:
    """Parsed ANSWER/SOURCES payload produced by a context extraction task."""

    answer: str
    sources: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "answer", _canonical_text(self.answer, "answer"))
        if isinstance(self.sources, str):
            raise ValueError("sources must be an iterable of source strings")
        canonical_sources = tuple(
            _canonical_text(source, "source") for source in self.sources
        )
        object.__setattr__(self, "sources", canonical_sources)


@dataclass(frozen=True)
class Provenance:
    """Immutable origin metadata for extracted evidence."""

    paper_id: str
    document_id: str
    agent_id: str
    iteration: int = 0
    source_type: str = "paper"
    source_uri: str = ""

    def __post_init__(self) -> None:
        for name in ("paper_id", "document_id", "agent_id", "source_type"):
            object.__setattr__(self, name, _canonical_text(getattr(self, name), name))
        if type(self.iteration) is not int or self.iteration < 0:
            raise ValueError("iteration must be a non-negative integer")
        if not isinstance(self.source_uri, str):
            raise ValueError("source_uri must be a string")
        object.__setattr__(self, "source_uri", self.source_uri.strip())


@dataclass(frozen=True)
class Evidence:
    """An immutable context fact with citations and provenance."""

    task_id: str
    answer: str
    sources: Tuple[str, ...]
    provenance: Provenance
    evidence_id: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "task_id", _canonical_text(self.task_id, "task_id"))
        object.__setattr__(self, "answer", _canonical_text(self.answer, "answer"))
        if not isinstance(self.provenance, Provenance):
            raise ValueError("provenance must be a Provenance")
        if isinstance(self.sources, str):
            raise ValueError("sources must be an iterable of source strings")
        canonical_sources = tuple(
            _canonical_text(source, "source") for source in self.sources
        )
        object.__setattr__(self, "sources", canonical_sources)
        if self.evidence_id:
            object.__setattr__(
                self,
                "evidence_id",
                _canonical_text(self.evidence_id, "evidence_id"),
            )
        else:
            object.__setattr__(self, "evidence_id", self._content_id())

    def _content_id(self) -> str:
        payload = {
            "task_id": self.task_id,
            "answer": self.answer,
            "sources": self.sources,
            "provenance": {
                "paper_id": self.provenance.paper_id,
                "document_id": self.provenance.document_id,
                "agent_id": self.provenance.agent_id,
                "iteration": self.provenance.iteration,
                "source_type": self.provenance.source_type,
                "source_uri": self.provenance.source_uri,
            },
        }
        encoded = json.dumps(
            payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    @classmethod
    def from_extraction(
        cls,
        task: ContextTask,
        extraction: ExtractionOutput,
        provenance: Provenance,
    ) -> "Evidence":
        if not isinstance(task, ContextTask):
            raise TypeError("task must be a ContextTask")
        if not isinstance(extraction, ExtractionOutput):
            raise TypeError("extraction must be an ExtractionOutput")
        return cls(
            task_id=task.task_id,
            answer=extraction.answer,
            sources=extraction.sources,
            provenance=provenance,
        )


@dataclass(frozen=True)
class ContextPacket:
    """Immutable, versioned snapshot shared among reviewer agents."""

    paper_id: str
    revision: int
    evidence: Tuple[Evidence, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "paper_id", _canonical_text(self.paper_id, "paper_id"))
        if type(self.revision) is not int or self.revision < 0:
            raise ValueError("revision must be a non-negative integer")
        if isinstance(self.evidence, Evidence):
            raise ValueError("evidence must be an iterable of Evidence")
        canonical_evidence = tuple(self.evidence)
        identifiers = set()
        for item in canonical_evidence:
            if not isinstance(item, Evidence):
                raise ValueError("every evidence item must be an Evidence")
            if item.provenance.paper_id != self.paper_id:
                raise ValueError("evidence paper_id does not match ContextPacket")
            if item.evidence_id in identifiers:
                raise ValueError("duplicate evidence_id in ContextPacket")
            identifiers.add(item.evidence_id)
        object.__setattr__(self, "evidence", canonical_evidence)

    def for_task(self, task_id: str) -> Tuple[Evidence, ...]:
        return tuple(item for item in self.evidence if item.task_id == task_id)

    def latest_for_task(self, task_id: str) -> Optional[Evidence]:
        matches = self.for_task(task_id)
        return matches[-1] if matches else None

    def __len__(self) -> int:
        return len(self.evidence)


__all__ = [
    "ContextPacket",
    "ContextTask",
    "Evidence",
    "ExtractionOutput",
    "Provenance",
    "ReviewOutput",
    "ReviewParseError",
    "ReviewValidationError",
    "parse_review_output",
    "render_review_output",
    "validate_comment",
]
