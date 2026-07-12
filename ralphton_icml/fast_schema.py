"""Structured contracts for the Track 2 ``fast-v1`` reviewer pipeline."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

from .schema import ContextTask, ReviewOutput


class FastContractError(ValueError):
    """Raised when a fast-v1 stage violates its structured contract."""


SEVERITIES = ("positive", "minor", "major", "fatal")
TECHNICAL_CRITERIA = ("Soundness", "Reproducibility", "Ethics")
CONTRIBUTION_CRITERIA = ("Presentation", "Significance", "Originality")


def _decode_object(value: Any, stage: str) -> Mapping[str, Any]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as exc:
            raise FastContractError("{} response must be JSON: {}".format(stage, exc)) from exc
    if not isinstance(value, Mapping):
        raise FastContractError("{} response must be a JSON object".format(stage))
    return value


def _exact_keys(value: Mapping[str, Any], expected: Iterable[str], stage: str) -> None:
    expected_set = set(expected)
    actual = set(value)
    if actual != expected_set:
        raise FastContractError(
            "{} fields differ: missing={}, unknown={}".format(
                stage,
                sorted(expected_set - actual),
                sorted(actual - expected_set),
            )
        )


def _text(value: Any, name: str, maximum: Optional[int] = None) -> str:
    if not isinstance(value, str):
        raise FastContractError("{} must be a string".format(name))
    canonical = value.replace("\r\n", "\n").replace("\r", "\n").strip()
    if not canonical:
        raise FastContractError("{} must not be empty".format(name))
    if "\x00" in canonical:
        raise FastContractError("{} must not contain NUL".format(name))
    if maximum is not None and len(canonical) > maximum:
        raise FastContractError(
            "{} exceeds {} characters".format(name, maximum)
        )
    return canonical


def _string_tuple(
    value: Any,
    name: str,
    maximum_items: int,
    maximum_characters: int,
    minimum_items: int = 0,
) -> Tuple[str, ...]:
    if not isinstance(value, list):
        raise FastContractError("{} must be a JSON array".format(name))
    if not minimum_items <= len(value) <= maximum_items:
        raise FastContractError(
            "{} must contain {}..{} items".format(
                name, minimum_items, maximum_items
            )
        )
    result = tuple(
        _text(item, "{} item".format(name), maximum_characters) for item in value
    )
    if len(result) != len(set(result)):
        raise FastContractError("{} contains duplicate items".format(name))
    return result


def _id_tuple(
    value: Any,
    name: str,
    maximum_items: int,
    allowed: Optional[Iterable[str]] = None,
    minimum_items: int = 0,
) -> Tuple[str, ...]:
    result = _string_tuple(value, name, maximum_items, 128, minimum_items)
    if allowed is not None:
        unknown = set(result) - set(allowed)
        if unknown:
            raise FastContractError(
                "{} contains unknown IDs: {}".format(name, sorted(unknown))
            )
    return result


def _stable_id(prefix: str, value: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return "{}.{}".format(prefix, hashlib.sha256(encoded).hexdigest()[:24])


@dataclass(frozen=True)
class BatchedExtractionItem:
    task_id: str
    answer: str
    sources: Tuple[str, ...]

    def as_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "answer": self.answer,
            "sources": list(self.sources),
        }


@dataclass(frozen=True)
class BatchedExtractionOutput:
    items: Tuple[BatchedExtractionItem, ...]

    @classmethod
    def from_response(
        cls,
        response: Any,
        expected_tasks: Sequence[ContextTask],
    ) -> "BatchedExtractionOutput":
        payload = _decode_object(response, "batched extraction")
        _exact_keys(payload, ("items",), "batched extraction")
        raw_items = payload["items"]
        if not isinstance(raw_items, list):
            raise FastContractError("batched extraction items must be an array")
        if len(raw_items) != len(expected_tasks):
            raise FastContractError(
                "batched extraction expected {} items, got {}".format(
                    len(expected_tasks), len(raw_items)
                )
            )
        tasks_by_id = {task.task_id: task for task in expected_tasks}
        parsed: Dict[str, BatchedExtractionItem] = {}
        for index, raw in enumerate(raw_items):
            if not isinstance(raw, Mapping):
                raise FastContractError("extraction item {} must be an object".format(index))
            _exact_keys(raw, ("task_id", "answer", "sources"), "extraction item")
            task_id = _text(raw["task_id"], "task_id", 256)
            if task_id not in tasks_by_id:
                raise FastContractError("unknown extraction task_id: {}".format(task_id))
            if task_id in parsed:
                raise FastContractError("duplicate extraction task_id: {}".format(task_id))
            maximum = 1200 if tasks_by_id[task_id].item == "Paper Summary" else 700
            parsed[task_id] = BatchedExtractionItem(
                task_id=task_id,
                answer=_text(raw["answer"], "answer for {}".format(task_id), maximum),
                sources=_string_tuple(raw["sources"], "sources", 3, 500),
            )
        missing = set(tasks_by_id) - set(parsed)
        if missing:
            raise FastContractError("missing extraction task IDs: {}".format(sorted(missing)))
        return cls(tuple(parsed[task.task_id] for task in expected_tasks))

    def as_dict(self) -> Dict[str, Any]:
        return {"items": [item.as_dict() for item in self.items]}


@dataclass(frozen=True)
class ReviewFinding:
    criterion: str
    severity: str
    text: str
    evidence_ids: Tuple[str, ...]
    finding_id: str

    @classmethod
    def from_mapping(
        cls,
        value: Any,
        criteria: Iterable[str],
        known_evidence_ids: Iterable[str],
        name: str,
    ) -> "ReviewFinding":
        if not isinstance(value, Mapping):
            raise FastContractError("{} must be an object".format(name))
        _exact_keys(value, ("criterion", "severity", "text", "evidence_ids"), name)
        criterion = _text(value["criterion"], "{}.criterion".format(name), 64)
        if criterion not in set(criteria):
            raise FastContractError("{} has invalid criterion {}".format(name, criterion))
        severity = _text(value["severity"], "{}.severity".format(name), 16)
        if severity not in SEVERITIES:
            raise FastContractError("{} has invalid severity {}".format(name, severity))
        text = _text(value["text"], "{}.text".format(name), 500)
        evidence_ids = _id_tuple(
            value["evidence_ids"],
            "{}.evidence_ids".format(name),
            3,
            known_evidence_ids,
            minimum_items=1,
        )
        material = {
            "criterion": criterion,
            "severity": severity,
            "text": text,
            "evidence_ids": evidence_ids,
        }
        return cls(
            criterion=criterion,
            severity=severity,
            text=text,
            evidence_ids=evidence_ids,
            finding_id=_stable_id("finding", material),
        )

    def as_dict(self) -> Dict[str, Any]:
        return {
            "finding_id": self.finding_id,
            "criterion": self.criterion,
            "severity": self.severity,
            "text": self.text,
            "evidence_ids": list(self.evidence_ids),
        }


@dataclass(frozen=True)
class UnresolvedContradiction:
    text: str
    evidence_ids: Tuple[str, ...]
    contradiction_id: str

    @classmethod
    def from_mapping(
        cls, value: Any, known_evidence_ids: Iterable[str], name: str
    ) -> "UnresolvedContradiction":
        if not isinstance(value, Mapping):
            raise FastContractError("{} must be an object".format(name))
        _exact_keys(value, ("text", "evidence_ids"), name)
        text = _text(value["text"], "{}.text".format(name), 500)
        evidence_ids = _id_tuple(
            value["evidence_ids"],
            "{}.evidence_ids".format(name),
            3,
            known_evidence_ids,
            minimum_items=1,
        )
        material = {"text": text, "evidence_ids": evidence_ids}
        return cls(text, evidence_ids, _stable_id("contradiction", material))

    def as_dict(self) -> Dict[str, Any]:
        return {
            "contradiction_id": self.contradiction_id,
            "text": self.text,
            "evidence_ids": list(self.evidence_ids),
        }


@dataclass(frozen=True)
class ConsolidatedReviewOutput:
    strengths: Tuple[ReviewFinding, ...]
    weaknesses: Tuple[ReviewFinding, ...]
    questions: Tuple[ReviewFinding, ...]
    memory_candidate_ids_used: Tuple[str, ...]
    unresolved_contradictions: Tuple[UnresolvedContradiction, ...]

    @classmethod
    def from_response(
        cls,
        response: Any,
        criteria: Iterable[str],
        known_evidence_ids: Iterable[str],
        known_memory_ids: Iterable[str],
    ) -> "ConsolidatedReviewOutput":
        payload = _decode_object(response, "consolidated review")
        _exact_keys(
            payload,
            (
                "strengths",
                "weaknesses",
                "questions",
                "memory_candidate_ids_used",
                "unresolved_contradictions",
            ),
            "consolidated review",
        )
        criteria_tuple = tuple(criteria)
        evidence_tuple = tuple(known_evidence_ids)

        def findings(key: str, maximum: int) -> Tuple[ReviewFinding, ...]:
            values = payload[key]
            if not isinstance(values, list) or len(values) > maximum:
                raise FastContractError("{} must contain at most {} findings".format(key, maximum))
            parsed = tuple(
                ReviewFinding.from_mapping(
                    item,
                    criteria_tuple,
                    evidence_tuple,
                    "{}[{}]".format(key, index),
                )
                for index, item in enumerate(values)
            )
            ids = [item.finding_id for item in parsed]
            if len(ids) != len(set(ids)):
                raise FastContractError("{} contains duplicate findings".format(key))
            return parsed

        contradictions_raw = payload["unresolved_contradictions"]
        if not isinstance(contradictions_raw, list) or len(contradictions_raw) > 3:
            raise FastContractError("unresolved_contradictions must contain at most 3 items")
        contradictions = tuple(
            UnresolvedContradiction.from_mapping(
                item,
                evidence_tuple,
                "unresolved_contradictions[{}]".format(index),
            )
            for index, item in enumerate(contradictions_raw)
        )
        return cls(
            strengths=findings("strengths", 3),
            weaknesses=findings("weaknesses", 4),
            questions=findings("questions", 2),
            memory_candidate_ids_used=_id_tuple(
                payload["memory_candidate_ids_used"],
                "memory_candidate_ids_used",
                8,
                known_memory_ids,
            ),
            unresolved_contradictions=contradictions,
        )

    @property
    def finding_ids(self) -> Tuple[str, ...]:
        return tuple(
            finding.finding_id
            for finding in self.strengths + self.weaknesses + self.questions
        )

    def as_dict(self) -> Dict[str, Any]:
        return {
            "strengths": [item.as_dict() for item in self.strengths],
            "weaknesses": [item.as_dict() for item in self.weaknesses],
            "questions": [item.as_dict() for item in self.questions],
            "memory_candidate_ids_used": list(self.memory_candidate_ids_used),
            "unresolved_contradictions": [
                item.as_dict() for item in self.unresolved_contradictions
            ],
        }


@dataclass(frozen=True)
class Track2Narrative:
    summary: str
    strengths: Tuple[str, ...]
    weaknesses: Tuple[str, ...]
    questions_for_authors: Tuple[str, ...]
    contribution: str
    ethics_and_limitations: str
    ai_agent_improvements: Tuple[str, ...]

    def render_comment(self) -> str:
        def bullets(items: Tuple[str, ...]) -> str:
            return "\n".join("- {}".format(item) for item in items) if items else "- None."

        return "\n\n".join(
            (
                "### Summary\n\n{}".format(self.summary),
                "### Strengths\n\n{}".format(bullets(self.strengths)),
                "### Weaknesses\n\n{}".format(bullets(self.weaknesses)),
                "### Questions for the Authors\n\n{}".format(
                    bullets(self.questions_for_authors)
                ),
                "### Contribution\n\n{}".format(self.contribution),
                "### Ethics and Limitations\n\n{}".format(
                    self.ethics_and_limitations
                ),
                "### AI Agent Improvements\n\n{}".format(
                    bullets(self.ai_agent_improvements)
                ),
            )
        )

    def as_dict(self) -> Dict[str, Any]:
        return {
            "summary": self.summary,
            "strengths": list(self.strengths),
            "weaknesses": list(self.weaknesses),
            "questions_for_authors": list(self.questions_for_authors),
            "contribution": self.contribution,
            "ethics_and_limitations": self.ethics_and_limitations,
            "ai_agent_improvements": list(self.ai_agent_improvements),
        }


@dataclass(frozen=True)
class ChairOutput:
    review: ReviewOutput
    narrative: Track2Narrative
    needs_refinement: bool
    refinement_reasons: Tuple[str, ...]

    @classmethod
    def from_response(cls, response: Any) -> "ChairOutput":
        payload = _decode_object(response, "chair")
        expected = (
            "soundness",
            "presentation",
            "significance",
            "originality",
            "overall_recommendation",
            "confidence",
            "summary",
            "strengths",
            "weaknesses",
            "questions_for_authors",
            "contribution",
            "ethics_and_limitations",
            "ai_agent_improvements",
            "needs_refinement",
            "refinement_reasons",
        )
        _exact_keys(payload, expected, "chair")
        narrative = Track2Narrative(
            summary=_text(payload["summary"], "summary", 2400),
            strengths=_string_tuple(payload["strengths"], "strengths", 3, 1200, 1),
            weaknesses=_string_tuple(payload["weaknesses"], "weaknesses", 4, 1200, 1),
            questions_for_authors=_string_tuple(
                payload["questions_for_authors"],
                "questions_for_authors",
                2,
                1000,
            ),
            contribution=_text(payload["contribution"], "contribution", 1600),
            ethics_and_limitations=_text(
                payload["ethics_and_limitations"],
                "ethics_and_limitations",
                1600,
            ),
            ai_agent_improvements=_string_tuple(
                payload["ai_agent_improvements"],
                "ai_agent_improvements",
                3,
                1000,
                1,
            ),
        )
        needs_refinement = payload["needs_refinement"]
        if type(needs_refinement) is not bool:
            raise FastContractError("needs_refinement must be a boolean")
        reasons = _string_tuple(
            payload["refinement_reasons"], "refinement_reasons", 4, 500
        )
        if needs_refinement and not reasons:
            raise FastContractError(
                "refinement_reasons must be non-empty when needs_refinement is true"
            )
        if not needs_refinement and reasons:
            raise FastContractError(
                "refinement_reasons must be empty when needs_refinement is false"
            )
        review = ReviewOutput(
            soundness=payload["soundness"],
            presentation=payload["presentation"],
            significance=payload["significance"],
            originality=payload["originality"],
            overall_recommendation=payload["overall_recommendation"],
            confidence=payload["confidence"],
            comment=narrative.render_comment(),
        )
        return cls(review, narrative, needs_refinement, reasons)

    def as_dict(self) -> Dict[str, Any]:
        value = self.review.as_dict()
        value["narrative"] = self.narrative.as_dict()
        value["needs_refinement"] = self.needs_refinement
        value["refinement_reasons"] = list(self.refinement_reasons)
        return value


@dataclass(frozen=True)
class AuthorRefinementOutput:
    response: str
    addressed_finding_ids: Tuple[str, ...]
    addressed_contradiction_ids: Tuple[str, ...]
    memory_candidate_ids_used: Tuple[str, ...]

    @classmethod
    def from_response(
        cls,
        response: Any,
        known_finding_ids: Iterable[str],
        known_contradiction_ids: Iterable[str],
        known_memory_ids: Iterable[str],
    ) -> "AuthorRefinementOutput":
        payload = _decode_object(response, "author refinement")
        _exact_keys(
            payload,
            (
                "response",
                "addressed_finding_ids",
                "addressed_contradiction_ids",
                "memory_candidate_ids_used",
            ),
            "author refinement",
        )
        return cls(
            response=_text(payload["response"], "author response", 12000),
            addressed_finding_ids=_id_tuple(
                payload["addressed_finding_ids"],
                "addressed_finding_ids",
                18,
                known_finding_ids,
            ),
            addressed_contradiction_ids=_id_tuple(
                payload["addressed_contradiction_ids"],
                "addressed_contradiction_ids",
                6,
                known_contradiction_ids,
            ),
            memory_candidate_ids_used=_id_tuple(
                payload["memory_candidate_ids_used"],
                "author memory_candidate_ids_used",
                8,
                known_memory_ids,
            ),
        )

    def as_dict(self) -> Dict[str, Any]:
        return {
            "response": self.response,
            "addressed_finding_ids": list(self.addressed_finding_ids),
            "addressed_contradiction_ids": list(
                self.addressed_contradiction_ids
            ),
            "memory_candidate_ids_used": list(self.memory_candidate_ids_used),
        }


def _object_schema(properties: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "type": "object",
        "properties": dict(properties),
        "required": list(properties),
        "additionalProperties": False,
    }


def extraction_json_schema(expected_tasks: Sequence[ContextTask]) -> Dict[str, Any]:
    task_ids = [task.task_id for task in expected_tasks]
    item = _object_schema(
        {
            "task_id": {"type": "string", "enum": task_ids},
            "answer": {"type": "string", "minLength": 1, "maxLength": 700},
            "sources": {
                "type": "array",
                "items": {"type": "string", "minLength": 1, "maxLength": 500},
                "maxItems": 3,
            },
        }
    )
    return _object_schema(
        {
            "items": {
                "type": "array",
                "items": item,
                "minItems": len(expected_tasks),
                "maxItems": len(expected_tasks),
            }
        }
    )


def consolidated_review_json_schema(criteria: Sequence[str]) -> Dict[str, Any]:
    finding = _object_schema(
        {
            "criterion": {"type": "string", "enum": list(criteria)},
            "severity": {"type": "string", "enum": list(SEVERITIES)},
            "text": {"type": "string", "minLength": 1, "maxLength": 500},
            "evidence_ids": {
                "type": "array",
                "items": {"type": "string", "minLength": 1, "maxLength": 128},
                "minItems": 1,
                "maxItems": 3,
            },
        }
    )
    contradiction = _object_schema(
        {
            "text": {"type": "string", "minLength": 1, "maxLength": 500},
            "evidence_ids": {
                "type": "array",
                "items": {"type": "string", "minLength": 1, "maxLength": 128},
                "minItems": 1,
                "maxItems": 3,
            },
        }
    )
    return _object_schema(
        {
            "strengths": {"type": "array", "items": finding, "maxItems": 3},
            "weaknesses": {"type": "array", "items": finding, "maxItems": 4},
            "questions": {"type": "array", "items": finding, "maxItems": 2},
            "memory_candidate_ids_used": {
                "type": "array",
                "items": {"type": "string", "minLength": 1, "maxLength": 128},
                "maxItems": 8,
            },
            "unresolved_contradictions": {
                "type": "array",
                "items": contradiction,
                "maxItems": 3,
            },
        }
    )


def chair_json_schema() -> Dict[str, Any]:
    def string_array(maximum: int, item_maximum: int, minimum: int = 0) -> Dict[str, Any]:
        value: Dict[str, Any] = {
            "type": "array",
            "items": {"type": "string", "minLength": 1, "maxLength": item_maximum},
            "maxItems": maximum,
        }
        if minimum:
            value["minItems"] = minimum
        return value

    return _object_schema(
        {
            "soundness": {"type": "integer", "minimum": 1, "maximum": 4},
            "presentation": {"type": "integer", "minimum": 1, "maximum": 4},
            "significance": {"type": "integer", "minimum": 1, "maximum": 4},
            "originality": {"type": "integer", "minimum": 1, "maximum": 4},
            "overall_recommendation": {
                "type": "integer",
                "minimum": 1,
                "maximum": 6,
            },
            "confidence": {"type": "integer", "minimum": 1, "maximum": 5},
            "summary": {"type": "string", "minLength": 1, "maxLength": 2400},
            "strengths": string_array(3, 1200, 1),
            "weaknesses": string_array(4, 1200, 1),
            "questions_for_authors": string_array(2, 1000),
            "contribution": {"type": "string", "minLength": 1, "maxLength": 1600},
            "ethics_and_limitations": {
                "type": "string",
                "minLength": 1,
                "maxLength": 1600,
            },
            "ai_agent_improvements": string_array(3, 1000, 1),
            "needs_refinement": {"type": "boolean"},
            "refinement_reasons": string_array(4, 500),
        }
    )


def author_json_schema() -> Dict[str, Any]:
    return _object_schema(
        {
            "response": {"type": "string", "minLength": 1, "maxLength": 12000},
            "addressed_finding_ids": {
                "type": "array",
                "items": {"type": "string", "minLength": 1, "maxLength": 128},
                "maxItems": 18,
            },
            "addressed_contradiction_ids": {
                "type": "array",
                "items": {"type": "string", "minLength": 1, "maxLength": 128},
                "maxItems": 6,
            },
            "memory_candidate_ids_used": {
                "type": "array",
                "items": {"type": "string", "minLength": 1, "maxLength": 128},
                "maxItems": 8,
            },
        }
    )


__all__ = [
    "AuthorRefinementOutput",
    "BatchedExtractionItem",
    "BatchedExtractionOutput",
    "ChairOutput",
    "ConsolidatedReviewOutput",
    "CONTRIBUTION_CRITERIA",
    "FastContractError",
    "ReviewFinding",
    "SEVERITIES",
    "TECHNICAL_CRITERIA",
    "Track2Narrative",
    "UnresolvedContradiction",
    "author_json_schema",
    "chair_json_schema",
    "consolidated_review_json_schema",
    "extraction_json_schema",
]
