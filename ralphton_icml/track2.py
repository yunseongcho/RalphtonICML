"""Immutable Track 2 review inputs and human-readable agent manifests.

The live reviewer pipeline is Track 2: it evaluates a frozen Track 1 paper and
optional, already-existing evidence.  This module deliberately does not run a
review or create experiments.  It snapshots inputs, records their identities,
and verifies them again immediately before a caller starts model work.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import subprocess
import tempfile
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple, Union


TRACK2_SCHEMA_VERSION = 2
TRACK2_MAX_EVIDENCE_BYTES = 1024 * 1024
TRACK2_REVIEW_INSTRUCTIONS = (
    "Treat this entire reviewer pipeline as Track 2 over immutable Track 1 inputs.",
    "Use only the frozen paper and explicitly supplied evidence; do not modify the paper.",
    "Do not invent, request, or imply new experiments, compute runs, measurements, or results.",
    "When supplied evidence is absent or does not verify a claim, label that claim evidence-insufficient.",
)
TRACK2_OUTPUT_CONTRACT = (
    "Soundness",
    "Presentation",
    "Significance",
    "Originality",
    "Overall Recommendation",
    "Confidence",
    "Comment",
)
TRACK2_COMMENT_SECTIONS = (
    "Summary",
    "Strengths",
    "Weaknesses",
    "Questions for the Authors",
    "Contribution",
    "Ethics and Limitations",
    "AI Agent Improvements",
)

_MANIFEST_BEGIN = "<!-- TRACK2_MANIFEST_BEGIN -->"
_MANIFEST_END = "<!-- TRACK2_MANIFEST_END -->"
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_SAFE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")
_SAFE_VERSION = re.compile(r"[A-Za-z0-9][A-Za-z0-9._+-]{0,63}\Z")
_SAFE_RESULT_FILENAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*\.md\Z")


class Track2InputError(ValueError):
    """Raised when a Track 2 manifest or input contract is invalid."""


class Track2IntegrityError(Track2InputError):
    """Raised when a frozen file no longer matches its recorded snapshot."""


class Track2ExtractionError(Track2InputError):
    """Raised when paper text cannot be extracted deterministically."""


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _text_sha256(value: str) -> str:
    return _sha256_bytes(value.encode("utf-8"))


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _one_line(value: Any, name: str, maximum: int = 200) -> str:
    if not isinstance(value, str):
        raise Track2InputError("{} must be a string".format(name))
    canonical = value.strip()
    if not canonical:
        raise Track2InputError("{} must not be empty".format(name))
    if len(canonical) > maximum:
        raise Track2InputError("{} exceeds {} characters".format(name, maximum))
    if any(character in canonical for character in ("\x00", "\r", "\n")):
        raise Track2InputError("{} must be one line without NUL".format(name))
    return canonical


def _utf8(raw: bytes, label: str) -> str:
    try:
        content = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise Track2InputError("{} must be valid UTF-8: {}".format(label, exc)) from exc
    if "\x00" in content:
        raise Track2InputError("{} must not contain NUL characters".format(label))
    return content


def _safe_relative_path(value: Any, name: str) -> PurePosixPath:
    text = _one_line(value, name, maximum=1000)
    if "\\" in text:
        raise Track2InputError("{} must use POSIX separators".format(name))
    relative = PurePosixPath(text)
    if relative.is_absolute() or not relative.parts:
        raise Track2InputError("{} must be a relative path".format(name))
    if any(part in ("", ".", "..") for part in relative.parts):
        raise Track2InputError("{} contains an unsafe path component".format(name))
    return relative


def _resolve_input(root: Path, value: Union[str, Path], name: str) -> Path:
    supplied = Path(value)
    candidate = supplied if supplied.is_absolute() else root / supplied
    try:
        resolved = candidate.resolve(strict=True)
    except FileNotFoundError as exc:
        raise Track2InputError("{} does not exist: {}".format(name, candidate)) from exc
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise Track2InputError("{} must stay inside {}".format(name, root)) from exc
    if not resolved.is_file():
        raise Track2InputError("{} must be a regular file: {}".format(name, resolved))
    return resolved


def _relative(root: Path, path: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError as exc:
        raise Track2InputError("input path must stay inside Track 2 root") from exc


@dataclass(frozen=True)
class FrozenInputFile:
    """A regular file fixed by resolved path, byte count, and SHA-256."""

    path: Path
    sha256: str
    size_bytes: int

    def __post_init__(self) -> None:
        path = Path(self.path).resolve(strict=True)
        if not path.is_file():
            raise Track2InputError("frozen input must be a regular file: {}".format(path))
        if not isinstance(self.sha256, str) or _SHA256.fullmatch(self.sha256) is None:
            raise Track2InputError("sha256 must be a lowercase SHA-256 digest")
        if type(self.size_bytes) is not int or self.size_bytes < 0:
            raise Track2InputError("size_bytes must be a non-negative integer")
        object.__setattr__(self, "path", path)

    @classmethod
    def snapshot(cls, path: Union[str, Path]) -> "FrozenInputFile":
        resolved = Path(path).resolve(strict=True)
        if not resolved.is_file():
            raise Track2InputError("input must be a regular file: {}".format(resolved))
        raw = resolved.read_bytes()
        return cls(path=resolved, sha256=_sha256_bytes(raw), size_bytes=len(raw))

    def verify(self) -> None:
        try:
            raw = self.path.read_bytes()
        except OSError as exc:
            raise Track2IntegrityError(
                "cannot read frozen input {}: {}".format(self.path, exc)
            ) from exc
        actual_digest = _sha256_bytes(raw)
        if len(raw) != self.size_bytes or actual_digest != self.sha256:
            raise Track2IntegrityError(
                "frozen input changed: {} (expected {} bytes/{}, got {} bytes/{})".format(
                    self.path,
                    self.size_bytes,
                    self.sha256,
                    len(raw),
                    actual_digest,
                )
            )


@dataclass(frozen=True)
class ProvidedEvidence:
    """One UTF-8 evidence file and the exact content supplied to reviewers."""

    evidence_id: str
    file: FrozenInputFile
    content: str
    content_sha256: str = ""

    def __post_init__(self) -> None:
        evidence_id = _one_line(self.evidence_id, "evidence_id", maximum=128)
        if _SAFE_ID.fullmatch(evidence_id) is None:
            raise Track2InputError(
                "evidence_id must match [A-Za-z0-9][A-Za-z0-9._-]{0,127}"
            )
        if not isinstance(self.file, FrozenInputFile):
            raise Track2InputError("evidence file must be a FrozenInputFile")
        if not isinstance(self.content, str):
            raise Track2InputError("evidence content must be UTF-8 text")
        if self.file.size_bytes > TRACK2_MAX_EVIDENCE_BYTES:
            raise Track2InputError(
                "evidence exceeds the {}-byte total input limit".format(
                    TRACK2_MAX_EVIDENCE_BYTES
                )
            )
        if not self.content.strip():
            raise Track2InputError("evidence content must not be empty")
        if "\x00" in self.content:
            raise Track2InputError("evidence content must not contain NUL")
        digest = _text_sha256(self.content)
        if self.content_sha256 and self.content_sha256 != digest:
            raise Track2IntegrityError("evidence content_sha256 does not match content")
        if digest != self.file.sha256 or len(self.content.encode("utf-8")) != self.file.size_bytes:
            raise Track2IntegrityError(
                "evidence UTF-8 snapshot does not match its frozen file identity"
            )
        object.__setattr__(self, "evidence_id", evidence_id)
        object.__setattr__(self, "content_sha256", digest)

    @classmethod
    def snapshot(
        cls, path: Union[str, Path], evidence_id: str
    ) -> "ProvidedEvidence":
        frozen = FrozenInputFile.snapshot(path)
        content = _utf8(frozen.path.read_bytes(), "evidence {}".format(frozen.path))
        if frozen.path.suffix.casefold() == ".json":
            try:
                json.loads(content)
            except json.JSONDecodeError as exc:
                raise Track2InputError(
                    "JSON evidence is invalid: {}: {}".format(frozen.path, exc)
                ) from exc
        return cls(evidence_id=evidence_id, file=frozen, content=content)

    @property
    def path(self) -> Path:
        return self.file.path

    @property
    def sha256(self) -> str:
        return self.file.sha256

    @property
    def size_bytes(self) -> int:
        return self.file.size_bytes

    @property
    def text(self) -> str:
        """Compatibility name used by reviewer payload builders."""

        return self.content

    @property
    def available(self) -> bool:
        """A provided snapshot is always available and integrity-checked."""

        return True

    def as_payload(self) -> Dict[str, Any]:
        """Return the bounded metadata/content shape sent to extraction workers."""

        return {
            "evidence_id": self.evidence_id,
            "path": self.path.name,
            "sha256": self.sha256,
            "content_sha256": self.content_sha256,
            "content": self.content,
            "available": True,
        }

    def verify(self) -> None:
        self.file.verify()
        current = _utf8(self.path.read_bytes(), "evidence {}".format(self.path))
        if current != self.content:
            raise Track2IntegrityError(
                "evidence content changed after snapshot: {}".format(self.path)
            )


@dataclass(frozen=True)
class Track2AgentManifest:
    """Review-agent identity, output location, and non-negotiable contract."""

    agent_name: str
    agent_version: str
    result_path: str = "outputs/review-result.md"
    review_instructions: Tuple[str, ...] = TRACK2_REVIEW_INSTRUCTIONS
    output_contract: Tuple[str, ...] = TRACK2_OUTPUT_CONTRACT
    comment_sections: Tuple[str, ...] = TRACK2_COMMENT_SECTIONS

    def __post_init__(self) -> None:
        name = _one_line(self.agent_name, "agent_name")
        version = _one_line(self.agent_version, "agent_version", maximum=64)
        if _SAFE_VERSION.fullmatch(version) is None:
            raise Track2InputError("agent_version contains unsupported characters")
        result = _safe_relative_path(self.result_path, "result_path")
        if len(result.parts) != 2 or result.parts[0] != "outputs":
            raise Track2InputError("result_path must be outputs/<filename>.md")
        if _SAFE_RESULT_FILENAME.fullmatch(result.name) is None:
            raise Track2InputError("result filename must be a safe Markdown filename")

        instructions = tuple(
            _one_line(value, "review instruction", maximum=1000)
            for value in self.review_instructions
        )
        missing = [value for value in TRACK2_REVIEW_INSTRUCTIONS if value not in instructions]
        if missing:
            raise Track2InputError(
                "review instructions may not weaken the Track 2 input policy"
            )
        contract = tuple(
            _one_line(value, "output contract field", maximum=100)
            for value in self.output_contract
        )
        if contract != TRACK2_OUTPUT_CONTRACT:
            raise Track2InputError("output_contract must match the Track 2 review form")
        comment_sections = tuple(
            _one_line(value, "Comment subsection", maximum=100)
            for value in self.comment_sections
        )
        if comment_sections != TRACK2_COMMENT_SECTIONS:
            raise Track2InputError(
                "comment_sections must match the Track 2 Comment contract"
            )
        object.__setattr__(self, "agent_name", name)
        object.__setattr__(self, "agent_version", version)
        object.__setattr__(self, "result_path", result.as_posix())
        object.__setattr__(self, "review_instructions", instructions)
        object.__setattr__(self, "output_contract", contract)
        object.__setattr__(self, "comment_sections", comment_sections)

    @property
    def result_filename(self) -> str:
        return PurePosixPath(self.result_path).name


@dataclass(frozen=True)
class _PaperSnapshot:
    paper_id: str
    title: str
    text: str
    document_id: str
    source_uri: str
    media_type: str
    extractor_name: str
    extractor_version: str

    @property
    def text_sha256(self) -> str:
        return _text_sha256(self.text)


def _pdf_version(executable: str, timeout: float) -> str:
    try:
        completed = subprocess.run(
            [executable, "-v"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=timeout,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise Track2ExtractionError("cannot run pdftotext -v: {}".format(exc)) from exc
    if completed.returncode != 0:
        raise Track2ExtractionError(
            "pdftotext -v exited with code {}".format(completed.returncode)
        )
    version_text = (completed.stdout + b"\n" + completed.stderr).decode(
        "utf-8", "replace"
    )
    first_line = next((line.strip() for line in version_text.splitlines() if line.strip()), "")
    if not first_line:
        raise Track2ExtractionError("pdftotext did not report a version")
    return first_line[:300]


def _pdf_snapshot(path: Path, executable: str, timeout: float) -> _PaperSnapshot:
    version = _pdf_version(executable, timeout)
    try:
        completed = subprocess.run(
            [executable, "-layout", str(path), "-"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=timeout,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise Track2ExtractionError("cannot extract PDF text: {}".format(exc)) from exc
    if completed.returncode != 0:
        error = completed.stderr.decode("utf-8", "replace").strip()
        raise Track2ExtractionError(
            "pdftotext exited with code {}: {}".format(
                completed.returncode, error[-1000:]
            )
        )
    text = _utf8(completed.stdout, "pdftotext output")
    text = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    if not text:
        raise Track2ExtractionError("pdftotext returned empty paper text")
    title = path.stem
    for raw_line in text.splitlines()[:20]:
        candidate = " ".join(raw_line.split())
        lowered = candidate.casefold()
        if (
            5 <= len(candidate) <= 300
            and any(character.isalpha() for character in candidate)
            and not lowered.startswith(("arxiv:", "published as", "proceedings of"))
        ):
            title = candidate
            break
    paper_id = path.stem
    if paper_id.casefold() in {"paper", "manuscript", "submission"}:
        paper_id = "paper-{}".format(_sha256_bytes(path.read_bytes())[:16])
    return _PaperSnapshot(
        paper_id=paper_id,
        title=title,
        text=text,
        document_id=path.name,
        source_uri="",
        media_type="application/pdf",
        extractor_name="pdftotext -layout",
        extractor_version=version,
    )


def _json_paper_snapshot(path: Path, raw: bytes) -> _PaperSnapshot:
    content = _utf8(raw, "JSON paper {}".format(path))
    try:
        decoded = json.loads(content)
    except json.JSONDecodeError as exc:
        raise Track2ExtractionError("invalid JSON paper {}: {}".format(path, exc)) from exc
    if not isinstance(decoded, Mapping):
        raise Track2ExtractionError("JSON paper must be an object")
    top = decoded
    paper = decoded.get("paper")
    if isinstance(paper, Mapping):
        record = paper
    else:
        record = decoded

    text = record.get("text")
    title = record.get("title")
    if not isinstance(text, str) or not text.strip():
        raise Track2ExtractionError("JSON paper requires non-empty text")
    if not isinstance(title, str) or not title.strip():
        raise Track2ExtractionError("JSON paper requires non-empty title")
    paper_id = (
        record.get("paper_id")
        or record.get("forum_id")
        or top.get("forum_id")
        or path.stem
    )
    document_id = record.get("document_id") or path.name
    source_uri = record.get("source_uri") or record.get("openreview_url") or ""
    return _PaperSnapshot(
        paper_id=_one_line(str(paper_id), "paper_id", maximum=500),
        title=_one_line(title, "paper title", maximum=1000),
        text=text.replace("\r\n", "\n").replace("\r", "\n").strip(),
        document_id=_one_line(str(document_id), "document_id", maximum=500),
        source_uri=str(source_uri).strip(),
        media_type="application/json",
        extractor_name="ralphton-json-paper",
        extractor_version="1",
    )


def _markdown_snapshot(path: Path, raw: bytes) -> _PaperSnapshot:
    content = _utf8(raw, "Markdown paper {}".format(path))
    text = content.replace("\r\n", "\n").replace("\r", "\n").strip()
    if not text:
        raise Track2ExtractionError("Markdown paper must not be empty")
    title = path.stem
    for line in text.splitlines():
        match = re.match(r"^#\s+(.+?)\s*$", line)
        if match:
            title = match.group(1).strip()
            break
    return _PaperSnapshot(
        paper_id=path.stem,
        title=_one_line(title, "paper title", maximum=1000),
        text=text,
        document_id=path.name,
        source_uri="",
        media_type="text/markdown",
        extractor_name="utf-8-markdown",
        extractor_version="1",
    )


def _load_paper_snapshot(
    path: Path, pdftotext_executable: str, timeout: float
) -> _PaperSnapshot:
    suffix = path.suffix.casefold()
    if suffix == ".pdf":
        return _pdf_snapshot(path, pdftotext_executable, timeout)
    raw = path.read_bytes()
    if suffix == ".json":
        return _json_paper_snapshot(path, raw)
    if suffix in (".md", ".markdown"):
        return _markdown_snapshot(path, raw)
    raise Track2ExtractionError(
        "paper must be PDF, Markdown, or compatible JSON: {}".format(path)
    )


@dataclass(frozen=True)
class Track2InputBundle:
    """A portable, deterministic snapshot of all inputs to one Track 2 review."""

    root: Path
    paper: FrozenInputFile
    paper_id: str
    title: str
    paper_text: str
    document_id: str
    source_uri: str
    paper_media_type: str
    paper_text_sha256: str
    paper_extractor: str
    paper_extractor_version: str
    evidence: Tuple[ProvidedEvidence, ...]
    agent_manifest: Track2AgentManifest
    bundle_digest: str = ""
    manifest_source: Optional[FrozenInputFile] = field(default=None, compare=False)
    pdftotext_executable: str = field(default="pdftotext", compare=False, repr=False)

    def __post_init__(self) -> None:
        root = Path(self.root).resolve(strict=True)
        if not root.is_dir():
            raise Track2InputError("Track 2 root must be a directory")
        if not isinstance(self.paper, FrozenInputFile):
            raise Track2InputError("paper must be a FrozenInputFile")
        _relative(root, self.paper.path)
        paper_id = _one_line(self.paper_id, "paper_id", maximum=500)
        title = _one_line(self.title, "paper title", maximum=1000)
        document_id = _one_line(self.document_id, "document_id", maximum=500)
        if not isinstance(self.paper_text, str) or not self.paper_text.strip():
            raise Track2InputError("paper_text must be non-empty text")
        if "\x00" in self.paper_text:
            raise Track2InputError("paper_text must not contain NUL")
        if _text_sha256(self.paper_text) != self.paper_text_sha256:
            raise Track2IntegrityError("paper_text_sha256 does not match paper_text")
        if not isinstance(self.source_uri, str):
            raise Track2InputError("source_uri must be a string")
        for name in (
            "paper_media_type",
            "paper_extractor",
            "paper_extractor_version",
        ):
            _one_line(getattr(self, name), name, maximum=500)
        if not isinstance(self.agent_manifest, Track2AgentManifest):
            raise Track2InputError("agent_manifest must be a Track2AgentManifest")

        evidence = tuple(sorted(self.evidence, key=lambda item: item.evidence_id))
        if any(not isinstance(item, ProvidedEvidence) for item in evidence):
            raise Track2InputError("every evidence item must be ProvidedEvidence")
        evidence_ids = [item.evidence_id for item in evidence]
        evidence_paths = [item.path for item in evidence]
        if len(evidence_ids) != len(set(evidence_ids)):
            raise Track2InputError("evidence IDs must be unique")
        if len(evidence_paths) != len(set(evidence_paths)):
            raise Track2InputError("evidence paths must be unique")
        for item in evidence:
            _relative(root, item.path)
        evidence_bytes = sum(item.size_bytes for item in evidence)
        if evidence_bytes > TRACK2_MAX_EVIDENCE_BYTES:
            raise Track2InputError(
                "provided evidence totals {} bytes; limit is {}".format(
                    evidence_bytes, TRACK2_MAX_EVIDENCE_BYTES
                )
            )
        if self.paper.path in evidence_paths:
            raise Track2InputError("paper cannot also be supplied as evidence")

        output = (root / self.agent_manifest.result_path).resolve(strict=False)
        try:
            output.relative_to(root)
        except ValueError as exc:
            raise Track2InputError("result path escapes the Track 2 root") from exc
        manifest_source = self.manifest_source
        if manifest_source is not None:
            if not isinstance(manifest_source, FrozenInputFile):
                raise Track2InputError("manifest_source must be a FrozenInputFile or None")
            _relative(root, manifest_source.path)

        object.__setattr__(self, "root", root)
        object.__setattr__(self, "paper_id", paper_id)
        object.__setattr__(self, "title", title)
        object.__setattr__(self, "document_id", document_id)
        object.__setattr__(self, "source_uri", self.source_uri.strip())
        object.__setattr__(self, "evidence", evidence)

        calculated = _sha256_bytes(_canonical_json_bytes(self._identity_dict()))
        if self.bundle_digest and self.bundle_digest != calculated:
            raise Track2IntegrityError(
                "bundle digest mismatch: expected {}, calculated {}".format(
                    self.bundle_digest, calculated
                )
            )
        object.__setattr__(self, "bundle_digest", calculated)

    @property
    def result_path(self) -> Path:
        return (self.root / self.agent_manifest.result_path).resolve(strict=False)

    @property
    def has_evidence(self) -> bool:
        return bool(self.evidence)

    @property
    def evidence_files(self) -> Tuple[ProvidedEvidence, ...]:
        """Return evidence snapshots accepted by fast-v1 payload builders."""

        return self.evidence

    @property
    def frozen_evidence_files(self) -> Tuple[FrozenInputFile, ...]:
        """Return only the underlying frozen file identities."""

        return tuple(item.file for item in self.evidence)

    @property
    def paper_input(self) -> Any:
        """Compatibility property for orchestrators that consume ``PaperInput``."""

        return self.to_paper_input()

    def _identity_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": TRACK2_SCHEMA_VERSION,
            "track": "track2",
            "agent": {
                "name": self.agent_manifest.agent_name,
                "version": self.agent_manifest.agent_version,
            },
            "paper": {
                "path": _relative(self.root, self.paper.path),
                "sha256": self.paper.sha256,
                "size_bytes": self.paper.size_bytes,
                "paper_id": self.paper_id,
                "title": self.title,
                "document_id": self.document_id,
                "source_uri": self.source_uri,
                "media_type": self.paper_media_type,
                "text_sha256": self.paper_text_sha256,
                "extractor": {
                    "name": self.paper_extractor,
                    "version": self.paper_extractor_version,
                },
            },
            "evidence": [
                {
                    "evidence_id": item.evidence_id,
                    "path": _relative(self.root, item.path),
                    "sha256": item.sha256,
                    "content_sha256": item.content_sha256,
                    "size_bytes": item.size_bytes,
                }
                for item in self.evidence
            ],
            "result": {"path": self.agent_manifest.result_path},
            "review_instructions": list(self.agent_manifest.review_instructions),
            "output_contract": list(self.agent_manifest.output_contract),
            "comment_sections": list(self.agent_manifest.comment_sections),
        }

    def to_manifest_dict(self) -> Dict[str, Any]:
        payload = self._identity_dict()
        payload["bundle_digest"] = self.bundle_digest
        return payload

    def as_paper_mapping(self) -> Dict[str, str]:
        """Return the existing JSON-paper shape consumed by reviewer orchestration."""

        return {
            "paper_id": self.paper_id,
            "title": self.title,
            "text": self.paper_text,
            "document_id": self.document_id,
            "source_uri": self.source_uri,
        }

    def to_paper_input(self) -> Any:
        """Create ``PaperInput`` lazily without introducing an import cycle."""

        from .orchestrator import PaperInput

        return PaperInput(**self.as_paper_mapping())

    def verify_frozen_inputs(
        self,
        *,
        pdftotext_executable: Optional[str] = None,
        extraction_timeout: float = 120.0,
    ) -> None:
        """Re-read and re-extract every input immediately before review execution."""

        if extraction_timeout <= 0:
            raise Track2InputError("extraction_timeout must be positive")
        self.paper.verify()
        executable = pdftotext_executable or self.pdftotext_executable
        snapshot = _load_paper_snapshot(self.paper.path, executable, extraction_timeout)
        expected = (
            self.paper_id,
            self.title,
            self.paper_text,
            self.document_id,
            self.source_uri,
            self.paper_media_type,
            self.paper_text_sha256,
            self.paper_extractor,
            self.paper_extractor_version,
        )
        actual = (
            snapshot.paper_id,
            snapshot.title,
            snapshot.text,
            snapshot.document_id,
            snapshot.source_uri,
            snapshot.media_type,
            snapshot.text_sha256,
            snapshot.extractor_name,
            snapshot.extractor_version,
        )
        if actual != expected:
            raise Track2IntegrityError(
                "paper extraction no longer matches the frozen Track 2 snapshot"
            )
        for item in self.evidence:
            item.verify()
        if self.manifest_source is not None:
            self.manifest_source.verify()

    def revalidate(self, **kwargs: Any) -> None:
        """Alias for callers that perform an explicit pre-execution gate."""

        self.verify_frozen_inputs(**kwargs)

    def verify_unchanged(self, **kwargs: Any) -> None:
        """Fast-orchestrator compatibility alias for the integrity gate."""

        self.verify_frozen_inputs(**kwargs)


def _default_evidence_id(path: Path) -> str:
    stem = re.sub(r"[^A-Za-z0-9._-]+", "-", path.stem).strip("-._")
    if not stem:
        stem = "item"
    return "evidence.{}".format(stem[:100])


def create_track2_bundle(
    root: Union[str, Path],
    paper_path: Union[str, Path],
    *,
    evidence_paths: Sequence[Union[str, Path]] = (),
    evidence_ids: Optional[Sequence[str]] = None,
    agent_name: str,
    agent_version: str = "1.0.0",
    result_filename: str = "review-result.md",
    pdftotext_executable: str = "pdftotext",
    extraction_timeout: float = 120.0,
) -> Track2InputBundle:
    """Snapshot paper/evidence files under ``root`` into an immutable bundle."""

    root_path = Path(root).resolve(strict=True)
    if not root_path.is_dir():
        raise Track2InputError("Track 2 root must be a directory")
    if extraction_timeout <= 0:
        raise Track2InputError("extraction_timeout must be positive")
    paper_resolved = _resolve_input(root_path, paper_path, "paper_path")
    paper_file = FrozenInputFile.snapshot(paper_resolved)
    snapshot = _load_paper_snapshot(
        paper_resolved, pdftotext_executable, extraction_timeout
    )

    evidence_values = tuple(evidence_paths)
    if evidence_ids is not None and len(evidence_ids) != len(evidence_values):
        raise Track2InputError("evidence_ids must match evidence_paths in length")
    supplied_ids = tuple(evidence_ids) if evidence_ids is not None else ()
    evidence = []
    used_ids = set()
    for index, value in enumerate(evidence_values):
        resolved = _resolve_input(root_path, value, "evidence_paths[{}]".format(index))
        candidate_id = supplied_ids[index] if supplied_ids else _default_evidence_id(resolved)
        if candidate_id in used_ids and not supplied_ids:
            candidate_id = "{}.{}".format(
                candidate_id, FrozenInputFile.snapshot(resolved).sha256[:8]
            )
        if candidate_id in used_ids:
            raise Track2InputError("duplicate evidence_id: {}".format(candidate_id))
        used_ids.add(candidate_id)
        evidence.append(ProvidedEvidence.snapshot(resolved, candidate_id))

    agent = Track2AgentManifest(
        agent_name=agent_name,
        agent_version=agent_version,
        result_path="outputs/{}".format(result_filename),
    )
    return Track2InputBundle(
        root=root_path,
        paper=paper_file,
        paper_id=snapshot.paper_id,
        title=snapshot.title,
        paper_text=snapshot.text,
        document_id=snapshot.document_id,
        source_uri=snapshot.source_uri,
        paper_media_type=snapshot.media_type,
        paper_text_sha256=snapshot.text_sha256,
        paper_extractor=snapshot.extractor_name,
        paper_extractor_version=snapshot.extractor_version,
        evidence=tuple(evidence),
        agent_manifest=agent,
        pdftotext_executable=pdftotext_executable,
    )


prepare_track2_bundle = create_track2_bundle


def default_track2_template_path() -> Path:
    return (
        Path(__file__).resolve().parent.parent
        / "skills"
        / "auto-research"
        / "assets"
        / "track-2-agent-template.md"
    )


def _markdown_code(value: str) -> str:
    return "`{}`".format(value.replace("`", "\\`"))


def render_review_agent(
    bundle: Track2InputBundle,
    template_path: Optional[Union[str, Path]] = None,
) -> str:
    """Render a human-inspectable ``review-agent.md`` with embedded JSON identity."""

    if not isinstance(bundle, Track2InputBundle):
        raise TypeError("bundle must be a Track2InputBundle")
    path = Path(template_path) if template_path is not None else default_track2_template_path()
    try:
        template = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise Track2InputError("cannot read Track 2 agent template {}: {}".format(path, exc)) from exc
    evidence_rows = (
        "\n".join(
            "- {}: {} (`{}`)".format(
                _markdown_code(item.evidence_id),
                _markdown_code(_relative(bundle.root, item.path)),
                item.sha256,
            )
            for item in bundle.evidence
        )
        if bundle.evidence
        else "- None. Unverifiable claims must be marked `evidence-insufficient`."
    )
    replacements = {
        "{{AGENT_NAME}}": bundle.agent_manifest.agent_name,
        "{{AGENT_VERSION}}": bundle.agent_manifest.agent_version,
        "{{PAPER_PATH}}": _relative(bundle.root, bundle.paper.path),
        "{{PAPER_SHA256}}": bundle.paper.sha256,
        "{{PAPER_TEXT_SHA256}}": bundle.paper_text_sha256,
        "{{PAPER_EXTRACTOR}}": "{} ({})".format(
            bundle.paper_extractor, bundle.paper_extractor_version
        ),
        "{{EVIDENCE_ROWS}}": evidence_rows,
        "{{RESULT_PATH}}": bundle.agent_manifest.result_path,
        "{{BUNDLE_DIGEST}}": bundle.bundle_digest,
        "{{REVIEW_INSTRUCTION_ROWS}}": "\n".join(
            "- {}".format(value)
            for value in bundle.agent_manifest.review_instructions
        ),
        "{{OUTPUT_CONTRACT_ROWS}}": "\n".join(
            "{}. {}".format(index, value)
            for index, value in enumerate(bundle.agent_manifest.output_contract, start=1)
        ),
        "{{COMMENT_SECTION_ROWS}}": "\n".join(
            "{}. {}".format(index, value)
            for index, value in enumerate(bundle.agent_manifest.comment_sections, start=1)
        ),
        "{{TRACK2_MANIFEST_JSON}}": json.dumps(
            bundle.to_manifest_dict(),
            ensure_ascii=False,
            allow_nan=False,
            indent=2,
            sort_keys=True,
        ),
    }
    rendered = template
    for marker, value in replacements.items():
        rendered = rendered.replace(marker, value)
    leftovers = sorted(set(re.findall(r"\{\{[A-Z0-9_]+\}\}", rendered)))
    if leftovers:
        raise Track2InputError(
            "unresolved Track 2 template placeholders: {}".format(leftovers)
        )
    return rendered.rstrip() + "\n"


def write_review_agent(
    bundle: Track2InputBundle,
    path: Optional[Union[str, Path]] = None,
    template_path: Optional[Union[str, Path]] = None,
) -> Path:
    """Atomically write the rendered agent file inside the Track 2 root."""

    target = Path(path) if path is not None else bundle.root / "review-agent.md"
    if not target.is_absolute():
        target = bundle.root / target
    target = target.resolve(strict=False)
    try:
        target.relative_to(bundle.root)
    except ValueError as exc:
        raise Track2InputError("review-agent.md path escapes the Track 2 root") from exc
    target.parent.mkdir(parents=True, exist_ok=True)
    rendered = render_review_agent(bundle, template_path=template_path)
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
            stream.write(rendered)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, str(target))
    finally:
        if temporary_name:
            try:
                Path(temporary_name).unlink()
            except FileNotFoundError:
                pass
    return target


def _extract_manifest_payload(markdown: str) -> Mapping[str, Any]:
    if markdown.count(_MANIFEST_BEGIN) != 1 or markdown.count(_MANIFEST_END) != 1:
        raise Track2InputError(
            "review-agent.md must contain exactly one Track 2 manifest block"
        )
    start = markdown.index(_MANIFEST_BEGIN) + len(_MANIFEST_BEGIN)
    end = markdown.index(_MANIFEST_END, start)
    block = markdown[start:end].strip()
    if not block.startswith("```json\n") or not block.endswith("\n```"):
        raise Track2InputError("Track 2 manifest block must be a fenced JSON object")
    encoded = block[len("```json\n") : -len("\n```")]
    try:
        payload = json.loads(encoded)
    except json.JSONDecodeError as exc:
        raise Track2InputError("invalid embedded Track 2 manifest JSON: {}".format(exc)) from exc
    if not isinstance(payload, Mapping):
        raise Track2InputError("embedded Track 2 manifest must be an object")
    return payload


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise Track2InputError("{} must be an object".format(name))
    return value


def _string(value: Any, name: str) -> str:
    if not isinstance(value, str):
        raise Track2InputError("{} must be a string".format(name))
    return value


def load_track2_bundle(
    path: Union[str, Path],
    *,
    pdftotext_executable: str = "pdftotext",
    extraction_timeout: float = 120.0,
) -> Track2InputBundle:
    """Load a batch-ready bundle from ``review-agent.md`` or its directory."""

    manifest_path = Path(path)
    if manifest_path.is_dir():
        manifest_path = manifest_path / "review-agent.md"
    manifest_path = manifest_path.resolve(strict=True)
    manifest_file = FrozenInputFile.snapshot(manifest_path)
    markdown = _utf8(manifest_path.read_bytes(), "review-agent.md")
    payload = _extract_manifest_payload(markdown)
    if payload.get("schema_version") != TRACK2_SCHEMA_VERSION:
        raise Track2InputError("unsupported Track 2 manifest schema_version")
    if payload.get("track") != "track2":
        raise Track2InputError("manifest track must be track2")
    root = manifest_path.parent.resolve(strict=True)

    agent_payload = _mapping(payload.get("agent"), "agent")
    result_payload = _mapping(payload.get("result"), "result")
    instructions = payload.get("review_instructions")
    contract = payload.get("output_contract")
    comment_sections = payload.get("comment_sections")
    if not all(
        isinstance(value, list)
        for value in (instructions, contract, comment_sections)
    ):
        raise Track2InputError(
            "review_instructions, output_contract, and comment_sections must be arrays"
        )
    agent = Track2AgentManifest(
        agent_name=_string(agent_payload.get("name"), "agent.name"),
        agent_version=_string(agent_payload.get("version"), "agent.version"),
        result_path=_string(result_payload.get("path"), "result.path"),
        review_instructions=tuple(instructions),
        output_contract=tuple(contract),
        comment_sections=tuple(comment_sections),
    )

    paper_payload = _mapping(payload.get("paper"), "paper")
    paper_relative = _safe_relative_path(paper_payload.get("path"), "paper.path")
    paper_path = _resolve_input(root, paper_relative.as_posix(), "paper.path")
    paper_file = FrozenInputFile.snapshot(paper_path)
    paper_snapshot = _load_paper_snapshot(
        paper_path, pdftotext_executable, extraction_timeout
    )

    evidence_payload = payload.get("evidence")
    if not isinstance(evidence_payload, list):
        raise Track2InputError("evidence must be an array")
    evidence = []
    for index, raw_entry in enumerate(evidence_payload):
        entry = _mapping(raw_entry, "evidence[{}]".format(index))
        relative = _safe_relative_path(
            entry.get("path"), "evidence[{}].path".format(index)
        )
        evidence_path = _resolve_input(
            root, relative.as_posix(), "evidence[{}].path".format(index)
        )
        evidence.append(
            ProvidedEvidence.snapshot(
                evidence_path,
                _string(entry.get("evidence_id"), "evidence_id"),
            )
        )

    declared_digest = _string(payload.get("bundle_digest"), "bundle_digest")
    bundle = Track2InputBundle(
        root=root,
        paper=paper_file,
        paper_id=paper_snapshot.paper_id,
        title=paper_snapshot.title,
        paper_text=paper_snapshot.text,
        document_id=paper_snapshot.document_id,
        source_uri=paper_snapshot.source_uri,
        paper_media_type=paper_snapshot.media_type,
        paper_text_sha256=paper_snapshot.text_sha256,
        paper_extractor=paper_snapshot.extractor_name,
        paper_extractor_version=paper_snapshot.extractor_version,
        evidence=tuple(evidence),
        agent_manifest=agent,
        bundle_digest=declared_digest,
        manifest_source=manifest_file,
        pdftotext_executable=pdftotext_executable,
    )
    if dict(payload) != bundle.to_manifest_dict():
        raise Track2IntegrityError(
            "embedded manifest metadata does not match the frozen input snapshot"
        )
    return bundle


load_track2_bundle_from_path = load_track2_bundle
load_track2_input_bundle = load_track2_bundle


__all__ = [
    "FrozenInputFile",
    "ProvidedEvidence",
    "TRACK2_COMMENT_SECTIONS",
    "TRACK2_MAX_EVIDENCE_BYTES",
    "TRACK2_OUTPUT_CONTRACT",
    "TRACK2_REVIEW_INSTRUCTIONS",
    "TRACK2_SCHEMA_VERSION",
    "Track2AgentManifest",
    "Track2ExtractionError",
    "Track2InputBundle",
    "Track2InputError",
    "Track2IntegrityError",
    "create_track2_bundle",
    "default_track2_template_path",
    "load_track2_bundle",
    "load_track2_bundle_from_path",
    "load_track2_input_bundle",
    "prepare_track2_bundle",
    "render_review_agent",
    "write_review_agent",
]
