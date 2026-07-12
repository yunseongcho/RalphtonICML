"""Validation and loading of the user-owned reviewer instruction contract."""

from __future__ import annotations

from pathlib import Path
import re
import sys
from typing import Optional, Tuple


class InstructionContractError(ValueError):
    """Raised when reviewer_instruction.md no longer matches ReviewOutput."""


_SECTIONS: Tuple[Tuple[str, Tuple[int, ...]], ...] = (
    ("Soundness", (4, 3, 2, 1)),
    ("Presentation", (4, 3, 2, 1)),
    ("Significance", (4, 3, 2, 1)),
    ("Originality", (4, 3, 2, 1)),
    ("Overall Recommendation", (6, 5, 4, 3, 2, 1)),
    ("Confidence", (5, 4, 3, 2, 1)),
)


def default_instruction_path() -> Path:
    repository_candidate = Path(__file__).resolve().parent.parent / "reviewer_instruction.md"
    if repository_candidate.is_file():
        return repository_candidate
    # setuptools ``data_files`` installs the user-owned contract under the
    # environment prefix rather than inside this Python package.
    installed_candidate = Path(sys.prefix) / "reviewer_instruction.md"
    if installed_candidate.is_file():
        return installed_candidate
    return repository_candidate


def validate_reviewer_instruction(text: str) -> str:
    """Validate headings and score ranges consumed by ``ReviewOutput``.

    Descriptive prose remains user-owned and may evolve freely.  This guard
    prevents a changed field name or rating scale from silently diverging from
    the machine-readable renderer/parser.
    """

    if not isinstance(text, str) or not text.strip():
        raise InstructionContractError("reviewer instruction must not be empty")
    normalized = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    if "## Paper Review Criteria" not in normalized:
        raise InstructionContractError("missing Paper Review Criteria section")
    if "https://icml.cc/Conferences/2026/ReviewerInstructions" not in normalized:
        raise InstructionContractError("missing ICML 2026 reviewer instruction source")

    positions = []
    for name, expected in _SECTIONS:
        heading = "#### **{}**".format(name)
        start = normalized.find(heading)
        if start < 0:
            raise InstructionContractError("missing heading {}".format(heading))
        positions.append((start, name, expected, len(heading)))
    comment_heading = "#### Comment"
    comment_start = normalized.find(comment_heading)
    if comment_start < 0:
        raise InstructionContractError("missing Comment heading")
    positions.sort()
    if [name for _start, name, _expected, _length in positions] != [
        name for name, _expected in _SECTIONS
    ]:
        raise InstructionContractError("review form headings are out of order")

    for index, (start, name, expected, heading_length) in enumerate(positions):
        end = positions[index + 1][0] if index + 1 < len(positions) else comment_start
        body = normalized[start + heading_length : end]
        scores = tuple(
            int(match.group(1))
            for match in re.finditer(r"(?m)^\s*-\s*(\d+)\s*:", body)
        )
        if scores != expected:
            raise InstructionContractError(
                "{} scale must be {}, got {}".format(name, expected, scores)
            )
    if comment_start < positions[-1][0]:
        raise InstructionContractError("Comment heading is out of order")
    comment_body = normalized[comment_start + len(comment_heading) :].strip()
    if not comment_body:
        raise InstructionContractError("Comment instruction must not be empty")
    return normalized


def load_reviewer_instruction(path: Optional[Path] = None) -> str:
    resolved = default_instruction_path() if path is None else Path(path)
    try:
        text = resolved.read_text(encoding="utf-8")
    except OSError as exc:
        raise InstructionContractError(
            "cannot read reviewer instruction {}: {}".format(resolved, exc)
        ) from exc
    return validate_reviewer_instruction(text)


__all__ = [
    "InstructionContractError",
    "default_instruction_path",
    "load_reviewer_instruction",
    "validate_reviewer_instruction",
]
