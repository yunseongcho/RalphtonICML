"""Batch scheduling and crash-safe artifacts for Track 2 ``fast-v1`` reviews."""

from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass, replace
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import tempfile
import threading
import time
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple

from .backend import ModelBackend
from .fast import (
    FastReviewerOrchestrator,
    FastReviewerRun,
    fast_run_as_dict,
)
from .instruction import load_reviewer_instruction
from .learning import LearningState
from .schema import ReviewOutput
from .track2 import (
    Track2AgentManifest,
    Track2InputBundle,
    Track2InputError,
    create_track2_bundle,
    default_track2_template_path,
    load_track2_bundle,
    write_review_agent,
)


_FORMAT_VERSION = 1
_SAFE_OUTPUT = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")
_SAFE_FILE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*\Z")
_PIPELINE_FILES = (
    "batch.py",
    "fast.py",
    "fast_schema.py",
    "track2.py",
    "schema.py",
    "team.py",
)


class BatchError(RuntimeError):
    """Base error for batch preflight and scheduling failures."""


class BatchManifestError(BatchError):
    """Raised before model work when a batch manifest is unsafe or malformed."""


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


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


def _atomic_json(path: Path, value: Any) -> None:
    _atomic_write(path, _canonical_json_bytes(value) + b"\n")


def _relative_input(base: Path, value: Any, name: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise BatchManifestError("{} must be a non-empty relative path".format(name))
    text = value.strip()
    if "\\" in text or "\x00" in text:
        raise BatchManifestError("{} contains an unsafe path character".format(name))
    relative = PurePosixPath(text)
    if relative.is_absolute() or any(part in ("", ".", "..") for part in relative.parts):
        raise BatchManifestError("{} must not contain traversal".format(name))
    candidate = base.joinpath(*relative.parts)
    try:
        resolved = candidate.resolve(strict=True)
    except FileNotFoundError as exc:
        raise BatchManifestError("{} does not exist: {}".format(name, candidate)) from exc
    try:
        resolved.relative_to(base)
    except ValueError as exc:
        raise BatchManifestError("{} escapes the manifest directory".format(name)) from exc
    if not resolved.is_file():
        raise BatchManifestError("{} must resolve to a regular file".format(name))
    return resolved


def _paper_id(path: Path, raw: bytes) -> str:
    if path.suffix.casefold() != ".json":
        value = path.stem
    else:
        try:
            payload = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise BatchManifestError("invalid JSON paper {}: {}".format(path, exc)) from exc
        if not isinstance(payload, Mapping):
            raise BatchManifestError("JSON paper must be an object: {}".format(path))
        nested = payload.get("paper")
        record = nested if isinstance(nested, Mapping) else payload
        value = (
            record.get("paper_id")
            or record.get("forum_id")
            or payload.get("forum_id")
            or path.stem
        )
        value = str(value)
    value = value.strip()
    if not value or "\x00" in value or "\n" in value or "\r" in value:
        raise BatchManifestError("paper_id must be non-empty one-line text")
    return value


def _safe_output_key(paper_id: str) -> str:
    if _SAFE_OUTPUT.fullmatch(paper_id) and paper_id not in (".", ".."):
        return paper_id
    stem = re.sub(r"[^A-Za-z0-9._-]+", "-", paper_id).strip("-._")[:80]
    if not stem:
        stem = "paper"
    return "{}-{}".format(stem, _sha256(paper_id.encode("utf-8"))[:12])


def _safe_materialized_name(path: Path, prefix: str = "") -> str:
    name = path.name
    if not _SAFE_FILE.fullmatch(name) or name in (".", ".."):
        suffix = path.suffix if _SAFE_FILE.fullmatch(path.suffix.lstrip(".")) else ""
        stem = re.sub(r"[^A-Za-z0-9._-]+", "-", path.stem).strip("-._") or "input"
        name = "{}{}".format(stem[:80], suffix)
    return prefix + name


def _assert_output_path(root: Path, path: Path, directory: bool = False) -> None:
    try:
        path.resolve(strict=False).relative_to(root)
    except ValueError as exc:
        raise BatchManifestError("output path escapes output-dir: {}".format(path)) from exc
    if path.is_symlink():
        raise BatchManifestError("symlink output path is not allowed: {}".format(path))
    if path.exists() and directory and not path.is_dir():
        raise BatchManifestError("expected output directory: {}".format(path))


@dataclass(frozen=True)
class BatchManifestEntry:
    index: int
    paper_source: Path
    paper_bytes: bytes
    paper_sha256: str
    paper_id: str
    output_key: str
    evidence_sources: Tuple[Path, ...]
    evidence_bytes: Tuple[bytes, ...]
    evidence_ids: Optional[Tuple[str, ...]]
    agent_name: str
    agent_version: str
    result_filename: str


@dataclass(frozen=True)
class PreparedBatchPaper:
    entry: BatchManifestEntry
    root: Path
    bundle: Track2InputBundle
    review_agent_path: Path
    result_json_path: Path
    result_markdown_path: Path
    completion_path: Path
    failure_path: Path


@dataclass(frozen=True)
class PreparedBatch:
    manifest_path: Path
    manifest_sha256: str
    output_dir: Path
    template_path: Path
    template_sha256: str
    papers: Tuple[PreparedBatchPaper, ...]


@dataclass(frozen=True)
class BatchPaperOutcome:
    paper_id: str
    input_index: int
    status: str
    result_json_path: Path
    review_markdown_path: Path
    resumed: bool = False
    error: str = ""
    run: Optional[FastReviewerRun] = None

    @property
    def has_valid_review(self) -> bool:
        return self.resumed or self.run is not None


@dataclass(frozen=True)
class BatchReviewResult:
    outcomes: Tuple[BatchPaperOutcome, ...]
    summary: Mapping[str, Any]

    @property
    def exit_code(self) -> int:
        failing = {"failed", "hard_deadline", "refinement_failed"}
        return 1 if any(item.status in failing for item in self.outcomes) else 0


def _parse_manifest(manifest_path: Path) -> Tuple[str, Tuple[BatchManifestEntry, ...]]:
    manifest_path = manifest_path.resolve(strict=True)
    base = manifest_path.parent.resolve(strict=True)
    raw = manifest_path.read_bytes()
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise BatchManifestError("invalid batch manifest: {}".format(exc)) from exc
    if not isinstance(payload, Mapping) or set(payload) != {"papers"}:
        raise BatchManifestError("batch manifest must contain only a papers array")
    papers = payload["papers"]
    if not isinstance(papers, list) or not papers:
        raise BatchManifestError("papers must be a non-empty array")

    entries = []
    seen_ids = set()
    seen_outputs = set()
    for index, item in enumerate(papers):
        if isinstance(item, str):
            value: Mapping[str, Any] = {"paper": item}
        elif isinstance(item, Mapping):
            value = item
        else:
            raise BatchManifestError("papers[{}] must be a string or object".format(index))
        allowed = {
            "paper",
            "evidence",
            "evidence_ids",
            "agent_name",
            "agent_version",
            "result_filename",
        }
        unknown = set(value) - allowed
        if unknown:
            raise BatchManifestError(
                "papers[{}] has unknown fields: {}".format(index, sorted(unknown))
            )
        paper_path = _relative_input(base, value.get("paper"), "papers[{}].paper".format(index))
        paper_bytes = paper_path.read_bytes()
        paper_id = _paper_id(paper_path, paper_bytes)
        if paper_id in seen_ids:
            raise BatchManifestError("duplicate paper_id: {}".format(paper_id))
        seen_ids.add(paper_id)

        raw_evidence = value.get("evidence", [])
        if not isinstance(raw_evidence, list) or any(not isinstance(item, str) for item in raw_evidence):
            raise BatchManifestError("papers[{}].evidence must be a string array".format(index))
        evidence_paths = tuple(
            _relative_input(
                base,
                item_path,
                "papers[{}].evidence[{}]".format(index, evidence_index),
            )
            for evidence_index, item_path in enumerate(raw_evidence)
        )
        if len(evidence_paths) != len(set(evidence_paths)) or paper_path in evidence_paths:
            raise BatchManifestError("paper/evidence paths must be distinct per entry")
        evidence_bytes = tuple(path.read_bytes() for path in evidence_paths)
        for evidence_path, content in zip(evidence_paths, evidence_bytes):
            try:
                text = content.decode("utf-8")
                if evidence_path.suffix.casefold() == ".json":
                    json.loads(text)
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise BatchManifestError(
                    "invalid evidence {}: {}".format(evidence_path, exc)
                ) from exc

        raw_ids = value.get("evidence_ids")
        if raw_ids is None:
            evidence_ids = None
        else:
            if not isinstance(raw_ids, list) or any(not isinstance(item, str) for item in raw_ids):
                raise BatchManifestError("evidence_ids must be a string array")
            if len(raw_ids) != len(evidence_paths):
                raise BatchManifestError("evidence_ids must match evidence length")
            evidence_ids = tuple(raw_ids)

        agent_name = value.get("agent_name", "Ralphton Track 2 Review Agent")
        agent_version = value.get("agent_version", "fast-v1")
        result_filename = value.get("result_filename", "review-result.md")
        # Reuse Track 2's exact validation before materializing any output.
        try:
            Track2AgentManifest(
                agent_name=agent_name,
                agent_version=agent_version,
                result_path="outputs/{}".format(result_filename),
            )
        except Track2InputError as exc:
            raise BatchManifestError(
                "invalid Track 2 settings for papers[{}]: {}".format(index, exc)
            ) from exc
        output_key = _safe_output_key(paper_id)
        result_identity = (output_key, result_filename)
        if result_identity in seen_outputs:
            raise BatchManifestError("duplicate result path for {}".format(paper_id))
        seen_outputs.add(result_identity)
        entries.append(
            BatchManifestEntry(
                index=index,
                paper_source=paper_path,
                paper_bytes=paper_bytes,
                paper_sha256=_sha256(paper_bytes),
                paper_id=paper_id,
                output_key=output_key,
                evidence_sources=evidence_paths,
                evidence_bytes=evidence_bytes,
                evidence_ids=evidence_ids,
                agent_name=agent_name,
                agent_version=agent_version,
                result_filename=result_filename,
            )
        )
    return _sha256(raw), tuple(entries)


def prepare_batch(
    manifest_path: Path,
    output_dir: Path,
    template_path: Optional[Path] = None,
    *,
    deadline_at: Optional[float] = None,
    monotonic: Callable[[], float] = time.monotonic,
) -> PreparedBatch:
    """Preflight every entry, then atomically materialize immutable Track 2 roots."""

    manifest = Path(manifest_path).resolve(strict=True)
    manifest_sha256, entries = _parse_manifest(manifest)
    output = Path(output_dir).resolve(strict=False)
    output.mkdir(parents=True, exist_ok=True)
    output = output.resolve(strict=True)
    template = (
        default_track2_template_path()
        if template_path is None
        else Path(template_path).resolve(strict=True)
    )
    template_bytes = template.read_bytes()
    load_reviewer_instruction()

    def extraction_timeout(commands_remaining: int) -> float:
        if deadline_at is None:
            return 120.0
        remaining = deadline_at - monotonic()
        if remaining <= 0:
            raise BatchError("hard deadline reached during Track 2 materialization")
        return min(120.0, max(0.01, remaining / commands_remaining))

    # Validate the entire output topology before writing snapshots.
    result_paths = set()
    for entry in entries:
        paper_root = output / entry.output_key
        _assert_output_path(output, paper_root, directory=True)
        for directory in (paper_root / "inputs", paper_root / "evidence", paper_root / "outputs"):
            _assert_output_path(output, directory, directory=True)
        result_path = paper_root / "outputs" / entry.result_filename
        _assert_output_path(output, result_path)
        identity = str(result_path.resolve(strict=False))
        if identity in result_paths:
            raise BatchManifestError("duplicate result path: {}".format(result_path))
        result_paths.add(identity)

    prepared = []
    for entry in entries:
        paper_root = output / entry.output_key
        for directory in (paper_root, paper_root / "inputs", paper_root / "evidence", paper_root / "outputs"):
            directory.mkdir(parents=True, exist_ok=True)
        # Preserve the basename because PDF/Markdown paper_id is defined by its stem.
        paper_target = paper_root / "inputs" / entry.paper_source.name
        _assert_output_path(output, paper_target)
        _atomic_write(paper_target, entry.paper_bytes)
        evidence_targets = []
        for evidence_index, (source, content) in enumerate(
            zip(entry.evidence_sources, entry.evidence_bytes)
        ):
            target = paper_root / "evidence" / _safe_materialized_name(
                source, prefix="{:03d}-".format(evidence_index)
            )
            _assert_output_path(output, target)
            _atomic_write(target, content)
            evidence_targets.append(target)
        bundle = create_track2_bundle(
            paper_root,
            paper_target,
            evidence_paths=tuple(evidence_targets),
            evidence_ids=entry.evidence_ids,
            agent_name=entry.agent_name,
            agent_version=entry.agent_version,
            result_filename=entry.result_filename,
            extraction_timeout=extraction_timeout(4),
        )
        if bundle.paper_id != entry.paper_id:
            raise BatchManifestError("materialized paper identity changed")
        if len(bundle.paper_text) > 240000:
            raise BatchManifestError(
                "paper {} exceeds the 240000-character fast-v1 limit".format(entry.paper_id)
            )
        review_agent = write_review_agent(
            bundle, paper_root / "review-agent.md", template_path=template
        )
        bundle = load_track2_bundle(
            review_agent, extraction_timeout=extraction_timeout(2)
        )
        stem = Path(entry.result_filename).stem
        result_json = paper_root / "outputs" / (stem + ".json")
        completion = paper_root / "outputs" / (stem + ".complete.json")
        failure = paper_root / "outputs" / (stem + ".failure.json")
        prepared.append(
            PreparedBatchPaper(
                entry=entry,
                root=paper_root,
                bundle=bundle,
                review_agent_path=review_agent,
                result_json_path=result_json,
                result_markdown_path=bundle.result_path,
                completion_path=completion,
                failure_path=failure,
            )
        )
    return PreparedBatch(
        manifest_path=manifest,
        manifest_sha256=manifest_sha256,
        output_dir=output,
        template_path=template,
        template_sha256=_sha256(template_bytes),
        papers=tuple(prepared),
    )


class _ProgressLog:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.lock = threading.Lock()

    def write(self, event: str, **details: Any) -> None:
        record = {"time_unix": time.time(), "event": event}
        record.update(details)
        payload = _canonical_json_bytes(record) + b"\n"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.lock:
            descriptor = os.open(
                str(self.path), os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644
            )
            try:
                os.write(descriptor, payload)
            finally:
                os.close(descriptor)


class BatchReviewScheduler:
    """Run all base reviews first, then globally selected Track 2 refinements."""

    def __init__(
        self,
        backend: ModelBackend,
        *,
        learning_state: Optional[LearningState] = None,
        paper_workers: int = 4,
        attempts: int = 2,
        memory_limit: int = 8,
        author_loop: str = "conditional",
        max_refinements: int = 2,
        deadline_seconds: float = 1800.0,
        soft_deadline_seconds: float = 1440.0,
        model: Optional[str] = None,
        cli_version: Optional[str] = None,
        state_digest: Optional[str] = None,
        pipeline_config: Optional[Mapping[str, Any]] = None,
        template_path: Optional[Path] = None,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        if type(paper_workers) is not int or paper_workers < 1:
            raise ValueError("paper_workers must be positive")
        if type(attempts) is not int or attempts < 1:
            raise ValueError("attempts must be positive")
        if type(max_refinements) is not int or max_refinements < 0:
            raise ValueError("max_refinements cannot be negative")
        if author_loop not in {"conditional", "always", "never"}:
            raise ValueError("author_loop must be conditional, always, or never")
        for name, value in (
            ("deadline_seconds", deadline_seconds),
            ("soft_deadline_seconds", soft_deadline_seconds),
        ):
            if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError("{} must be finite and non-negative".format(name))
        if deadline_seconds <= 0 or soft_deadline_seconds > deadline_seconds:
            raise ValueError("deadlines must satisfy 0 <= soft <= hard and hard > 0")
        self.backend = backend
        self.learning_state = learning_state
        self.paper_workers = paper_workers
        self.attempts = attempts
        self.memory_limit = memory_limit
        self.author_loop = author_loop
        self.max_refinements = max_refinements
        self.deadline_seconds = float(deadline_seconds)
        self.soft_deadline_seconds = float(soft_deadline_seconds)
        self.model = model or str(getattr(backend, "model", "<injected-backend>"))
        self.cli_version = cli_version or str(
            getattr(backend, "cli_version", "<injected-backend>")
        )
        if state_digest:
            self.state_digest = state_digest
        elif learning_state is not None:
            self.state_digest = learning_state.digest
        else:
            self.state_digest = str(getattr(backend, "state_digest", "<none>"))
        self.pipeline_config = dict(pipeline_config or {})
        self.template_path = template_path
        self.monotonic = monotonic
        self.reviewer_instruction = load_reviewer_instruction()

    def _pipeline_digest(self) -> str:
        root = Path(__file__).resolve().parent
        files = {}
        for name in _PIPELINE_FILES:
            path = root / name
            files[name] = _sha256(path.read_bytes())
        return _sha256(_canonical_json_bytes(files))

    def _fingerprint(self, prepared: PreparedBatch, paper: PreparedBatchPaper) -> str:
        config = {
            "pipeline": "fast-v1",
            "paper_workers": self.paper_workers,
            "attempts": self.attempts,
            "memory_limit": self.memory_limit,
            "author_loop": self.author_loop,
            "max_refinements": self.max_refinements,
            "deadline_seconds": self.deadline_seconds,
            "soft_deadline_seconds": self.soft_deadline_seconds,
        }
        config.update(self.pipeline_config)
        envelope = {
            "format_version": _FORMAT_VERSION,
            "input_index": paper.entry.index,
            "bundle_digest": paper.bundle.bundle_digest,
            "review_agent_sha256": _sha256(paper.review_agent_path.read_bytes()),
            "template_sha256": prepared.template_sha256,
            "reviewer_instruction_sha256": _sha256(
                self.reviewer_instruction.encode("utf-8")
            ),
            "model": self.model,
            "cli_version": self.cli_version,
            "state_digest": self.state_digest,
            "pipeline_source_digest": self._pipeline_digest(),
            "pipeline_config": config,
        }
        return _sha256(_canonical_json_bytes(envelope))

    def _orchestrator(self, deadline_at: float) -> FastReviewerOrchestrator:
        return FastReviewerOrchestrator(
            self.backend,
            learning_state=self.learning_state,
            reviewer_instruction=self.reviewer_instruction,
            attempts=self.attempts,
            memory_limit=self.memory_limit,
            deadline_at=deadline_at,
            monotonic=self.monotonic,
        )

    def _write_run(
        self,
        paper: PreparedBatchPaper,
        run: FastReviewerRun,
        fingerprint: str,
        phase: str,
        complete: bool,
    ) -> None:
        markdown = run.effective_review.to_markdown()
        ReviewOutput.from_markdown(markdown)
        payload = fast_run_as_dict(run)
        payload["batch"] = {
            "format_version": _FORMAT_VERSION,
            "phase": phase,
            "run_fingerprint": fingerprint,
            "input_index": paper.entry.index,
            "output_key": paper.entry.output_key,
        }
        encoded_json = _canonical_json_bytes(payload) + b"\n"
        encoded_markdown = markdown.encode("utf-8")
        if complete:
            try:
                paper.completion_path.unlink()
            except FileNotFoundError:
                pass
        _atomic_write(paper.result_json_path, encoded_json)
        _atomic_write(paper.result_markdown_path, encoded_markdown)
        if complete:
            marker = {
                "format_version": _FORMAT_VERSION,
                "paper_id": run.paper_id,
                "run_fingerprint": fingerprint,
                "phase": phase,
                "result_json": paper.result_json_path.name,
                "result_json_sha256": _sha256(encoded_json),
                "review_markdown": paper.result_markdown_path.name,
                "review_markdown_sha256": _sha256(encoded_markdown),
            }
            _atomic_json(paper.completion_path, marker)

    def _write_failure(
        self, paper: PreparedBatchPaper, fingerprint: str, error: Exception
    ) -> None:
        try:
            paper.completion_path.unlink()
        except FileNotFoundError:
            pass
        _atomic_json(
            paper.failure_path,
            {
                "format_version": _FORMAT_VERSION,
                "paper_id": paper.entry.paper_id,
                "run_fingerprint": fingerprint,
                "error_type": type(error).__name__,
                "error": str(error)[:4000],
            },
        )

    def _resume(
        self, paper: PreparedBatchPaper, fingerprint: str
    ) -> Optional[BatchPaperOutcome]:
        try:
            marker_bytes = paper.completion_path.read_bytes()
            marker = json.loads(marker_bytes.decode("utf-8"))
            if not isinstance(marker, Mapping):
                return None
            if marker.get("format_version") != _FORMAT_VERSION:
                return None
            if marker.get("run_fingerprint") != fingerprint:
                return None
            json_bytes = paper.result_json_path.read_bytes()
            markdown_bytes = paper.result_markdown_path.read_bytes()
            if marker.get("result_json_sha256") != _sha256(json_bytes):
                return None
            if marker.get("review_markdown_sha256") != _sha256(markdown_bytes):
                return None
            payload = json.loads(json_bytes.decode("utf-8"))
            markdown = markdown_bytes.decode("utf-8")
            ReviewOutput.from_markdown(markdown)
            if payload.get("input_digest") != paper.bundle.bundle_digest:
                return None
            if payload.get("rendered_review") != markdown:
                return None
            batch = payload.get("batch")
            if not isinstance(batch, Mapping) or batch.get("run_fingerprint") != fingerprint:
                return None
            status = str(payload.get("refinement_status") or marker.get("phase"))
            return BatchPaperOutcome(
                paper_id=paper.entry.paper_id,
                input_index=paper.entry.index,
                status=status,
                result_json_path=paper.result_json_path,
                review_markdown_path=paper.result_markdown_path,
                resumed=True,
            )
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError):
            return None

    def _cancel_backend(self) -> None:
        cancel = getattr(self.backend, "cancel_all", None)
        if callable(cancel):
            cancel()

    def _metrics(self) -> Mapping[str, Any]:
        snapshot = getattr(self.backend, "snapshot_metrics", None)
        if callable(snapshot):
            return snapshot()
        requests = getattr(self.backend, "requests", ())
        return {"calls": len(requests), "stage_durations": {}}

    @staticmethod
    def _metric_delta(before: Mapping[str, Any], after: Mapping[str, Any]) -> Dict[str, Any]:
        result = {}
        for name in (
            "calls",
            "cache_hits",
            "failures",
            "timeouts",
            "request_bytes",
            "output_bytes",
        ):
            result[name] = int(after.get(name, 0)) - int(before.get(name, 0))
        before_stages = before.get("stage_durations", {})
        after_stages = after.get("stage_durations", {})
        stages = {}
        if isinstance(after_stages, Mapping):
            for stage, values in after_stages.items():
                old = before_stages.get(stage, []) if isinstance(before_stages, Mapping) else []
                stages[stage] = list(values)[len(old) :]
        result["stage_durations"] = stages
        return result

    @staticmethod
    def _percentile(values: Sequence[float], fraction: float) -> Optional[float]:
        if not values:
            return None
        ordered = sorted(float(value) for value in values)
        if len(ordered) == 1:
            return ordered[0]
        position = fraction * (len(ordered) - 1)
        lower = int(math.floor(position))
        upper = int(math.ceil(position))
        if lower == upper:
            return ordered[lower]
        weight = position - lower
        return ordered[lower] * (1.0 - weight) + ordered[upper] * weight

    def run(
        self,
        manifest_path: Path,
        output_dir: Path,
        *,
        resume: bool = False,
    ) -> BatchReviewResult:
        started = self.monotonic()
        deadline_at = started + self.deadline_seconds
        soft_deadline_at = started + self.soft_deadline_seconds
        prepared = prepare_batch(
            manifest_path,
            output_dir,
            self.template_path,
            deadline_at=deadline_at,
            monotonic=self.monotonic,
        )
        progress = _ProgressLog(prepared.output_dir / "progress.jsonl")
        metrics_before = self._metrics()
        fingerprints = {
            item.entry.index: self._fingerprint(prepared, item)
            for item in prepared.papers
        }
        outcomes: Dict[int, BatchPaperOutcome] = {}
        base_runs: Dict[int, FastReviewerRun] = {}
        pending_papers = []

        for paper in prepared.papers:
            fingerprint = fingerprints[paper.entry.index]
            resumed = self._resume(paper, fingerprint) if resume else None
            if resumed is not None:
                outcomes[paper.entry.index] = resumed
                progress.write("resume_hit", paper_id=paper.entry.paper_id)
                continue
            try:
                paper.completion_path.unlink()
            except FileNotFoundError:
                pass
            pending_papers.append(paper)

        def run_base(paper: PreparedBatchPaper) -> FastReviewerRun:
            progress.write("base_start", paper_id=paper.entry.paper_id)
            return self._orchestrator(deadline_at).run_base(paper.bundle)

        hard_deadline_reached = self.monotonic() >= deadline_at
        if hard_deadline_reached:
            for paper in pending_papers:
                error = BatchError("hard deadline reached during batch preflight")
                self._write_failure(paper, fingerprints[paper.entry.index], error)
                outcomes[paper.entry.index] = BatchPaperOutcome(
                    paper_id=paper.entry.paper_id,
                    input_index=paper.entry.index,
                    status="hard_deadline",
                    result_json_path=paper.result_json_path,
                    review_markdown_path=paper.result_markdown_path,
                    error=str(error),
                )
            pending_papers = []
            self._cancel_backend()
        executor = ThreadPoolExecutor(max_workers=self.paper_workers)
        future_papers: Dict[Future, PreparedBatchPaper] = {
            executor.submit(run_base, paper): paper for paper in pending_papers
        }
        pending = set(future_papers)
        try:
            while pending:
                remaining = deadline_at - self.monotonic()
                if remaining <= 0:
                    hard_deadline_reached = True
                    break
                done, pending = wait(
                    pending,
                    timeout=remaining,
                    return_when=FIRST_COMPLETED,
                )
                if not done:
                    hard_deadline_reached = True
                    break
                for future in done:
                    paper = future_papers[future]
                    fingerprint = fingerprints[paper.entry.index]
                    try:
                        run = future.result()
                        base_runs[paper.entry.index] = run
                        self._write_run(paper, run, fingerprint, "base_pending", False)
                        progress.write("base_success", paper_id=paper.entry.paper_id)
                    except Exception as exc:
                        self._write_failure(paper, fingerprint, exc)
                        outcomes[paper.entry.index] = BatchPaperOutcome(
                            paper_id=paper.entry.paper_id,
                            input_index=paper.entry.index,
                            status="failed",
                            result_json_path=paper.result_json_path,
                            review_markdown_path=paper.result_markdown_path,
                            error=str(exc),
                        )
                        progress.write(
                            "base_failure", paper_id=paper.entry.paper_id, error=str(exc)[:1000]
                        )
            if hard_deadline_reached:
                self._cancel_backend()
                for future in pending:
                    future.cancel()
        finally:
            executor.shutdown(wait=True, cancel_futures=True)

        if hard_deadline_reached:
            for future in pending:
                paper = future_papers[future]
                if paper.entry.index not in outcomes:
                    error = BatchError("hard deadline reached during base review")
                    self._write_failure(paper, fingerprints[paper.entry.index], error)
                    outcomes[paper.entry.index] = BatchPaperOutcome(
                        paper_id=paper.entry.paper_id,
                        input_index=paper.entry.index,
                        status="hard_deadline",
                        result_json_path=paper.result_json_path,
                        review_markdown_path=paper.result_markdown_path,
                        error=str(error),
                    )

        # All base work has ended. Select refinements globally only now.
        candidates = []
        for index, run in base_runs.items():
            orchestrator = self._orchestrator(deadline_at)
            eligible = (
                self.author_loop == "always"
                or (
                    self.author_loop == "conditional"
                    and orchestrator.eligible_for_refinement(run)
                )
            )
            if eligible:
                candidates.append((orchestrator.refinement_priority(run, index), index))
            else:
                status = "disabled" if self.author_loop == "never" else "not_needed"
                terminal = replace(run, refinement_status=status)
                paper = prepared.papers[index]
                self._write_run(
                    paper, terminal, fingerprints[index], status, True
                )
                outcomes[index] = BatchPaperOutcome(
                    paper_id=run.paper_id,
                    input_index=index,
                    status=status,
                    result_json_path=paper.result_json_path,
                    review_markdown_path=paper.result_markdown_path,
                    run=terminal,
                )
        candidates.sort()
        selected = [index for _priority, index in candidates[: self.max_refinements]]
        unselected = [index for _priority, index in candidates[self.max_refinements :]]
        for index in unselected:
            run = replace(
                base_runs[index],
                refinement_status="not_selected",
                refinement_reason="global max_refinements limit",
            )
            paper = prepared.papers[index]
            self._write_run(paper, run, fingerprints[index], "not_selected", True)
            outcomes[index] = BatchPaperOutcome(
                paper_id=run.paper_id,
                input_index=index,
                status="not_selected",
                result_json_path=paper.result_json_path,
                review_markdown_path=paper.result_markdown_path,
                run=run,
            )

        refine_futures: Dict[Future, int] = {}
        refine_executor = ThreadPoolExecutor(max_workers=max(1, min(self.paper_workers, 2)))
        for index in selected:
            if self.monotonic() >= soft_deadline_at:
                run = replace(
                    base_runs[index],
                    refinement_status="soft_deadline",
                    refinement_reason="soft deadline prevented refinement start",
                )
                paper = prepared.papers[index]
                self._write_run(paper, run, fingerprints[index], "soft_deadline", True)
                outcomes[index] = BatchPaperOutcome(
                    paper_id=run.paper_id,
                    input_index=index,
                    status="soft_deadline",
                    result_json_path=paper.result_json_path,
                    review_markdown_path=paper.result_markdown_path,
                    run=run,
                )
                continue
            paper = prepared.papers[index]
            progress.write("refinement_start", paper_id=paper.entry.paper_id)
            refine_futures[
                refine_executor.submit(
                    self._orchestrator(deadline_at).refine,
                    paper.bundle,
                    base_runs[index],
                )
            ] = index
        refine_pending = set(refine_futures)
        try:
            while refine_pending:
                remaining = deadline_at - self.monotonic()
                if remaining <= 0:
                    hard_deadline_reached = True
                    break
                done, refine_pending = wait(
                    refine_pending, timeout=remaining, return_when=FIRST_COMPLETED
                )
                if not done:
                    hard_deadline_reached = True
                    break
                for future in done:
                    index = refine_futures[future]
                    paper = prepared.papers[index]
                    try:
                        run = future.result()
                        self._write_run(
                            paper, run, fingerprints[index], "completed", True
                        )
                        outcomes[index] = BatchPaperOutcome(
                            paper_id=run.paper_id,
                            input_index=index,
                            status="completed",
                            result_json_path=paper.result_json_path,
                            review_markdown_path=paper.result_markdown_path,
                            run=run,
                        )
                        progress.write("refinement_success", paper_id=run.paper_id)
                    except Exception as exc:
                        base = replace(
                            base_runs[index],
                            refinement_status="refinement_failed",
                            refinement_reason=str(exc)[:1000],
                        )
                        self._write_run(
                            paper,
                            base,
                            fingerprints[index],
                            "refinement_failed",
                            True,
                        )
                        outcomes[index] = BatchPaperOutcome(
                            paper_id=base.paper_id,
                            input_index=index,
                            status="refinement_failed",
                            result_json_path=paper.result_json_path,
                            review_markdown_path=paper.result_markdown_path,
                            error=str(exc),
                            run=base,
                        )
                        progress.write(
                            "refinement_failure", paper_id=base.paper_id, error=str(exc)[:1000]
                        )
            if hard_deadline_reached and refine_pending:
                self._cancel_backend()
                for future in refine_pending:
                    future.cancel()
        finally:
            refine_executor.shutdown(wait=True, cancel_futures=True)
        for future in refine_pending:
            index = refine_futures[future]
            if index in outcomes:
                continue
            paper = prepared.papers[index]
            base = replace(
                base_runs[index],
                refinement_status="hard_deadline",
                refinement_reason="hard deadline interrupted refinement",
            )
            self._write_run(
                paper, base, fingerprints[index], "hard_deadline", True
            )
            outcomes[index] = BatchPaperOutcome(
                paper_id=base.paper_id,
                input_index=index,
                status="hard_deadline",
                result_json_path=paper.result_json_path,
                review_markdown_path=paper.result_markdown_path,
                error=base.refinement_reason,
                run=base,
            )

        ended = self.monotonic()
        metrics = self._metric_delta(metrics_before, self._metrics())
        ordered_outcomes = tuple(outcomes[index] for index in sorted(outcomes))
        valid_count = sum(item.has_valid_review for item in ordered_outcomes)
        logical_calls = sum(
            item.run.logical_call_count
            for item in ordered_outcomes
            if item.run is not None and not item.resumed
        )
        latency = {
            stage: {
                "p50": self._percentile(values, 0.50),
                "p95": self._percentile(values, 0.95),
            }
            for stage, values in metrics["stage_durations"].items()
        }
        wall = max(0.0, ended - started)
        summary = {
            "format_version": _FORMAT_VERSION,
            "pipeline": "fast-v1",
            "pipeline_source_digest": self._pipeline_digest(),
            "pipeline_config": dict(self.pipeline_config),
            "model": self.model,
            "codex_cli_version": self.cli_version,
            "state_digest": self.state_digest,
            "manifest_sha256": prepared.manifest_sha256,
            "papers_total": len(prepared.papers),
            "papers_valid": valid_count,
            "papers_failed": len(prepared.papers) - valid_count,
            "papers_resumed": sum(item.resumed for item in ordered_outcomes),
            "logical_calls": logical_calls,
            "expected_base_calls": len(prepared.papers) * 5,
            "calls": metrics["calls"],
            "cache_hits": metrics["cache_hits"],
            "stage_latency_seconds": latency,
            "request_bytes": metrics["request_bytes"],
            "output_bytes": metrics["output_bytes"],
            "retries": max(0, metrics["calls"] - logical_calls),
            "backend_failures": metrics["failures"],
            "backend_timeouts": metrics["timeouts"],
            "wall_time_seconds": wall,
            "papers_per_minute": 0.0 if wall == 0 else valid_count * 60.0 / wall,
            "hard_deadline_seconds": self.deadline_seconds,
            "soft_deadline_seconds": self.soft_deadline_seconds,
            "hard_deadline_reached": hard_deadline_reached,
            "deadline_met": not hard_deadline_reached and ended <= deadline_at,
            "papers": [
                {
                    "paper_id": item.paper_id,
                    "input_index": item.input_index,
                    "status": item.status,
                    "resumed": item.resumed,
                    "error": item.error,
                    "result_json": str(item.result_json_path),
                    "review_markdown": str(item.review_markdown_path),
                }
                for item in ordered_outcomes
            ],
        }
        _atomic_json(prepared.output_dir / "summary.json", summary)
        progress.write("batch_complete", exit_code=(1 if any(
            item.status in {"failed", "hard_deadline", "refinement_failed"}
            for item in ordered_outcomes
        ) else 0))
        return BatchReviewResult(ordered_outcomes, summary)


def run_batch_reviews(
    backend: ModelBackend,
    manifest_path: Path,
    output_dir: Path,
    **kwargs: Any
) -> BatchReviewResult:
    """Convenience wrapper used by the public CLI."""

    resume = bool(kwargs.pop("resume", False))
    return BatchReviewScheduler(backend, **kwargs).run(
        manifest_path, output_dir, resume=resume
    )


__all__ = [
    "BatchError",
    "BatchManifestEntry",
    "BatchManifestError",
    "BatchPaperOutcome",
    "BatchReviewResult",
    "BatchReviewScheduler",
    "PreparedBatch",
    "PreparedBatchPaper",
    "prepare_batch",
    "run_batch_reviews",
]
