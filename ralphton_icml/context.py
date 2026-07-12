"""Context extraction tasks and append-only shared evidence storage."""

from __future__ import annotations

from collections import OrderedDict
import importlib
import re
from threading import RLock
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

from .schema import (
    ContextPacket,
    ContextTask,
    Evidence,
    ExtractionOutput,
    Provenance,
)


class ExtractionParseError(ValueError):
    """Raised when an extraction result is not an ANSWER/SOURCES document."""


_EXTRACTION_HEADER = re.compile(
    r"^\s*(?:#{1,6}\s*)?(?:\*\*)?(ANSWER|SOURCES)\s*:?(?:\*\*)?\s*$",
    re.IGNORECASE,
)
_BULLET = re.compile(r"^\s*(?:[-*+]\s+|\d+[.)]\s+)(.*)$")


def _default_prompt_map() -> Mapping[str, Mapping[str, Sequence[object]]]:
    module = importlib.import_module("prompts")
    prompt_map = getattr(module, "prompts", None)
    if not isinstance(prompt_map, Mapping):
        raise RuntimeError("prompts.py must expose a mapping named 'prompts'")
    return prompt_map


def load_context_tasks(
    prompt_map: Optional[Mapping[str, Mapping[str, Sequence[object]]]] = None,
) -> Tuple[ContextTask, ...]:
    """Convert the general-paper prompt leaves into immutable tasks.

    The repository's default map is required to contain exactly the 19 general
    paper leaves.  A supplied map is useful for validation and unit tests and
    is not constrained to that count.
    """

    is_default = prompt_map is None
    source = _default_prompt_map() if prompt_map is None else prompt_map
    tasks: List[ContextTask] = []
    seen_ids = set()
    for section, items in source.items():
        if not isinstance(section, str) or not isinstance(items, Mapping):
            raise ValueError("prompt sections must map strings to mappings")
        for item, leaves in items.items():
            if not isinstance(item, str):
                raise ValueError("prompt item names must be strings")
            if isinstance(leaves, (str, bytes)) or len(leaves) != 1:
                raise ValueError(
                    "each prompt item must contain exactly one prompt leaf"
                )
            leaf = leaves[0]
            if not isinstance(leaf, tuple) or len(leaf) != 2:
                raise ValueError("each prompt leaf must be a (prompt, tier) tuple")
            prompt, tier = leaf
            task_id = "{}/{}".format(section, item)
            if task_id in seen_ids:
                raise ValueError("duplicate context task id: {}".format(task_id))
            seen_ids.add(task_id)
            tasks.append(
                ContextTask(
                    task_id=task_id,
                    section=section,
                    item=item,
                    prompt=prompt,
                    tier=tier,
                    ordinal=len(tasks),
                )
            )
    if is_default and len(tasks) != 19:
        raise RuntimeError(
            "the general-paper prompt map must expose 19 leaves, got {}".format(
                len(tasks)
            )
        )
    return tuple(tasks)


def _parse_source_lines(lines: Sequence[str]) -> Tuple[str, ...]:
    while lines and not lines[0].strip():
        lines = lines[1:]
    while lines and not lines[-1].strip():
        lines = lines[:-1]
    if not lines:
        return ()

    has_bullets = any(_BULLET.match(line) for line in lines if line.strip())
    if not has_bullets:
        paragraphs = []
        current: List[str] = []
        for line in lines:
            if line.strip():
                current.append(line.strip())
            elif current:
                paragraphs.append(" ".join(current))
                current = []
        if current:
            paragraphs.append(" ".join(current))
        # A common SOURCES form is one citation per line without bullets.
        if len(paragraphs) == 1 and len([line for line in lines if line.strip()]) > 1:
            return tuple(line.strip() for line in lines if line.strip())
        return tuple(paragraphs)

    sources: List[str] = []
    current = []
    for line in lines:
        match = _BULLET.match(line)
        if match:
            if current:
                sources.append(" ".join(current))
            current = [match.group(1).strip()]
        elif line.strip():
            if not current:
                raise ExtractionParseError(
                    "SOURCES content before the first bullet is ambiguous"
                )
            current.append(line.strip())
        elif current:
            sources.append(" ".join(current))
            current = []
    if current:
        sources.append(" ".join(current))
    return tuple(source for source in sources if source)


def parse_extraction_output(markdown: str) -> ExtractionOutput:
    """Parse the extraction-only ANSWER/SOURCES contract.

    Markdown headings (``## ANSWER``), bold headings (``**ANSWER**``), and bare
    headings are accepted.  Exactly one ANSWER followed by one SOURCES heading
    is required; reviewer-form Markdown is therefore rejected by construction.
    """

    if not isinstance(markdown, str):
        raise ExtractionParseError("extraction output must be a string")
    normalized = markdown.replace("\r\n", "\n").replace("\r", "\n")
    lines = normalized.split("\n")
    headers = []
    for index, line in enumerate(lines):
        match = _EXTRACTION_HEADER.match(line)
        if match:
            headers.append((index, match.group(1).upper()))
    if len(headers) != 2 or [name for _index, name in headers] != [
        "ANSWER",
        "SOURCES",
    ]:
        raise ExtractionParseError(
            "extraction output must contain exactly ANSWER then SOURCES"
        )
    answer_index, _ = headers[0]
    sources_index, _ = headers[1]
    if any(line.strip() for line in lines[:answer_index]):
        raise ExtractionParseError("content before ANSWER is not allowed")
    answer = "\n".join(lines[answer_index + 1 : sources_index]).strip()
    if not answer:
        raise ExtractionParseError("ANSWER must not be empty")
    sources = _parse_source_lines(lines[sources_index + 1 :])
    return ExtractionOutput(answer=answer, sources=sources)


class SharedContextStore:
    """Thread-safe, append-only store returning immutable context snapshots."""

    def __init__(self) -> None:
        self._lock = RLock()
        self._records: Dict[str, "OrderedDict[str, Evidence]"] = {}
        self._revisions: Dict[str, int] = {}

    def merge(self, evidence: Evidence) -> ContextPacket:
        """Append evidence, deduplicating exact immutable evidence IDs."""

        return self.merge_many((evidence,))

    def merge_many(self, evidence: Iterable[Evidence]) -> ContextPacket:
        items = tuple(evidence)
        if not items:
            raise ValueError("at least one Evidence is required")
        if any(not isinstance(item, Evidence) for item in items):
            raise TypeError("all merged items must be Evidence")
        paper_ids = {item.provenance.paper_id for item in items}
        if len(paper_ids) != 1:
            raise ValueError("one merge transaction may target only one paper")
        paper_id = next(iter(paper_ids))

        with self._lock:
            records = self._records.get(paper_id, OrderedDict())
            new_items = []
            pending = {}
            for item in items:
                existing = records.get(item.evidence_id, pending.get(item.evidence_id))
                if existing is not None and existing != item:
                    raise ValueError(
                        "evidence_id collision for {}".format(item.evidence_id)
                    )
                if existing is None:
                    new_items.append(item)
                    pending[item.evidence_id] = item
            if new_items:
                if paper_id not in self._records:
                    self._records[paper_id] = records
                for item in new_items:
                    records[item.evidence_id] = item
                self._revisions[paper_id] = self._revisions.get(paper_id, 0) + 1
            return self._snapshot_unlocked(paper_id)

    def merge_extraction(
        self,
        task: ContextTask,
        output: Union[str, ExtractionOutput],
        provenance: Provenance,
    ) -> ContextPacket:
        """Parse, provenance-wrap, and atomically merge one task result."""

        if not isinstance(task, ContextTask):
            raise TypeError("task must be a ContextTask")
        extraction = (
            parse_extraction_output(output) if isinstance(output, str) else output
        )
        if not isinstance(extraction, ExtractionOutput):
            raise TypeError("output must be Markdown or an ExtractionOutput")
        evidence = Evidence.from_extraction(task, extraction, provenance)
        return self.merge(evidence)

    def snapshot(self, paper_id: str) -> ContextPacket:
        """Return the current immutable packet, including an empty revision 0."""

        if not isinstance(paper_id, str) or not paper_id.strip():
            raise ValueError("paper_id must be a non-empty string")
        canonical_id = paper_id.strip()
        with self._lock:
            return self._snapshot_unlocked(canonical_id)

    def _snapshot_unlocked(self, paper_id: str) -> ContextPacket:
        records = self._records.get(paper_id, OrderedDict())
        return ContextPacket(
            paper_id=paper_id,
            revision=self._revisions.get(paper_id, 0),
            evidence=tuple(records.values()),
        )

    def paper_ids(self) -> Tuple[str, ...]:
        with self._lock:
            return tuple(sorted(self._records))


CONTEXT_TASKS = load_context_tasks()


def get_context_tasks() -> Tuple[ContextTask, ...]:
    """Return the immutable default 19-task sequence."""

    return CONTEXT_TASKS


__all__ = [
    "CONTEXT_TASKS",
    "ExtractionParseError",
    "SharedContextStore",
    "get_context_tasks",
    "load_context_tasks",
    "parse_extraction_output",
]
