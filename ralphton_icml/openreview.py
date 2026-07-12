"""OpenReview API2 ingestion and deterministic dataset preparation.

The module intentionally uses only the Python standard library.  Network access is
limited to public API2 GET requests.  Downloaded forum graphs are stored as
content-addressed, write-once JSON snapshots with a SHA-256 sidecar manifest so
that experiments can cite the exact source bytes they consumed.
"""

from __future__ import annotations

import datetime as _datetime
import hashlib
import json
import math
import os
import re
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


DEFAULT_API2_URL = "https://api2.openreview.net"


class OpenReviewError(RuntimeError):
    """Raised when OpenReview data cannot be fetched or validated."""


class SnapshotIntegrityError(OpenReviewError):
    """Raised when a content-addressed snapshot does not match its manifest."""


def unwrap_content_value(value: Any) -> Any:
    """Unwrap OpenReview API2 ``{"value": ...}`` content recursively.

    API1 stores field values directly, whereas API2 usually wraps them in a
    dictionary that may also contain readers or edit metadata.  Lists and nested
    dictionaries are handled as well so callers receive the same logical value
    for either representation.
    """

    if isinstance(value, Mapping):
        if "value" in value:
            return unwrap_content_value(value["value"])
        return {str(key): unwrap_content_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [unwrap_content_value(item) for item in value]
    return value


def normalize_content(note_or_content: Mapping[str, Any]) -> Dict[str, Any]:
    """Return API1/API2 content as a plain dictionary."""

    content = note_or_content.get("content", note_or_content)
    if not isinstance(content, Mapping):
        return {}
    return {str(key): unwrap_content_value(value) for key, value in content.items()}


def normalize_score(value: Any) -> Optional[float]:
    """Extract a finite numeric score from common OpenReview encodings.

    Examples include ``6: Weak Accept``, ``3 (fair)``, numeric API1 values, and
    textual labels without a number.  Scores are not silently clipped because
    different venues use different scales.
    """

    value = unwrap_content_value(value)
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        score = float(value)
        return score if math.isfinite(score) else None
    text = str(value).strip()
    if not text:
        return None
    match = re.search(r"[-+]?\d+(?:\.\d+)?", text)
    if match:
        try:
            score = float(match.group(0))
        except ValueError:
            return None
        return score if math.isfinite(score) else None
    compact = re.sub(r"[^a-z]+", " ", text.lower()).strip()
    labels = {
        "strong reject": 1.0,
        "reject": 2.0,
        "weak reject": 3.0,
        "weak accept": 4.0,
        "accept": 5.0,
        "strong accept": 6.0,
    }
    return labels.get(compact)


def _canonical_json_bytes(value: Any) -> bytes:
    try:
        text = json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    except (TypeError, ValueError) as exc:
        raise OpenReviewError("OpenReview payload is not canonical JSON: %s" % exc)
    return text.encode("utf-8")


def _utc_now() -> str:
    return _datetime.datetime.now(_datetime.timezone.utc).isoformat()


def _safe_component(value: str) -> str:
    component = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._")
    return component or "forum"


def _write_once(path: Path, payload: bytes) -> None:
    """Create ``path`` exactly once, accepting an identical existing file."""

    try:
        descriptor = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    except FileExistsError:
        if path.read_bytes() != payload:
            raise SnapshotIntegrityError("refusing to overwrite immutable file %s" % path)
        return
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    except Exception:
        try:
            path.unlink()
        except OSError:
            pass
        raise


@dataclass(frozen=True)
class RawSnapshot:
    forum_id: str
    sha256: str
    byte_count: int
    snapshot_path: str
    manifest_path: str
    retrieved_at: str
    source_url: str


class OpenReviewClient:
    """Small public OpenReview API2 client backed by ``urllib``.

    ``opener`` is injectable for deterministic tests.  Authentication is
    deliberately unsupported: this ingestion path is for publicly available
    research artifacts only.
    """

    def __init__(
        self,
        base_url: str = DEFAULT_API2_URL,
        timeout: float = 30.0,
        page_size: int = 1000,
        user_agent: str = "ralphton-icml/1.0 (+public-research-ingestion)",
        opener: Any = None,
    ) -> None:
        if timeout <= 0:
            raise ValueError("timeout must be positive")
        if page_size <= 0:
            raise ValueError("page_size must be positive")
        self.base_url = base_url.rstrip("/")
        self.timeout = float(timeout)
        self.page_size = int(page_size)
        self.user_agent = user_agent
        self._opener = opener or urllib.request.urlopen

    def _get_json(self, path: str, params: Mapping[str, Any]) -> Mapping[str, Any]:
        encoded = urllib.parse.urlencode(
            [(key, item) for key, value in params.items() for item in (
                value if isinstance(value, (list, tuple)) else [value]
            ) if item is not None],
            doseq=True,
        )
        url = "%s/%s" % (self.base_url, path.lstrip("/"))
        if encoded:
            url += "?" + encoded
        request = urllib.request.Request(
            url,
            headers={"Accept": "application/json", "User-Agent": self.user_agent},
            method="GET",
        )
        try:
            response = self._opener(request, timeout=self.timeout)
            with response:
                payload = response.read()
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")[:500]
            raise OpenReviewError("OpenReview HTTP %s for %s: %s" % (exc.code, url, detail))
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise OpenReviewError("OpenReview request failed for %s: %s" % (url, exc))
        try:
            decoded = json.loads(payload.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise OpenReviewError("OpenReview returned invalid JSON for %s: %s" % (url, exc))
        if not isinstance(decoded, Mapping):
            raise OpenReviewError("OpenReview response for %s is not an object" % url)
        return decoded

    def get_notes(self, **filters: Any) -> List[Mapping[str, Any]]:
        """Fetch every page of API2 notes matching ``filters``."""

        offset = int(filters.pop("offset", 0))
        requested_limit = filters.pop("limit", None)
        remaining = int(requested_limit) if requested_limit is not None else None
        if remaining is not None and remaining < 0:
            raise ValueError("limit cannot be negative")
        notes: List[Mapping[str, Any]] = []
        seen_ids = set()
        page_count = 0
        while remaining is None or remaining > 0:
            limit = self.page_size if remaining is None else min(self.page_size, remaining)
            response = self._get_json("notes", dict(filters, offset=offset, limit=limit))
            page = response.get("notes", [])
            if not isinstance(page, list):
                raise OpenReviewError("OpenReview notes response has a non-list 'notes' field")
            for note in page:
                if not isinstance(note, Mapping):
                    continue
                note_id = str(note.get("id", ""))
                identity = note_id or hashlib.sha256(_canonical_json_bytes(note)).hexdigest()
                if identity not in seen_ids:
                    seen_ids.add(identity)
                    notes.append(note)
            received = len(page)
            offset += received
            if remaining is not None:
                remaining -= received
            page_count += 1
            total = response.get("count")
            if received == 0 or received < limit:
                break
            if isinstance(total, int) and offset >= total:
                break
            if page_count >= 10000:
                raise OpenReviewError("pagination exceeded 10,000 pages")
        return notes

    def fetch_forum(self, forum_id: str) -> Mapping[str, Any]:
        """Fetch a forum's root note and all public replies."""

        if not forum_id or not forum_id.strip():
            raise ValueError("forum_id cannot be empty")
        forum_id = forum_id.strip()
        notes = self.get_notes(forum=forum_id)
        if not any(str(note.get("id", "")) == forum_id for note in notes):
            roots = self.get_notes(id=forum_id, limit=1)
            existing = {str(note.get("id", "")) for note in notes}
            notes = roots + [note for note in notes if str(note.get("id", "")) not in existing.intersection(
                {str(root.get("id", "")) for root in roots}
            )]
        return {
            "api": "openreview-api2",
            "forum_id": forum_id,
            "notes": notes,
        }


def write_raw_snapshot(
    raw_forum: Mapping[str, Any],
    output_dir: os.PathLike,
    forum_id: Optional[str] = None,
    source_url: str = DEFAULT_API2_URL,
    retrieved_at: Optional[str] = None,
) -> RawSnapshot:
    """Persist a canonical, content-addressed, immutable forum snapshot."""

    resolved_forum = forum_id or str(raw_forum.get("forum_id", "")).strip()
    if not resolved_forum:
        raise ValueError("forum_id is required for a raw snapshot")
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    raw_bytes = _canonical_json_bytes(raw_forum)
    digest = hashlib.sha256(raw_bytes).hexdigest()
    stem = "%s.%s" % (_safe_component(resolved_forum), digest)
    snapshot_path = directory / (stem + ".json")
    manifest_path = directory / (stem + ".manifest.json")
    _write_once(snapshot_path, raw_bytes)

    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise SnapshotIntegrityError("invalid existing manifest %s: %s" % (manifest_path, exc))
    else:
        manifest = {
            "schema_version": 1,
            "forum_id": resolved_forum,
            "source_url": source_url,
            "retrieved_at": retrieved_at or _utc_now(),
            "sha256": digest,
            "byte_count": len(raw_bytes),
            "snapshot_file": snapshot_path.name,
            "canonicalization": "json-sort-keys-utf8-no-whitespace-v1",
        }
        _write_once(manifest_path, _canonical_json_bytes(manifest))

    expected = {
        "forum_id": resolved_forum,
        "sha256": digest,
        "byte_count": len(raw_bytes),
        "snapshot_file": snapshot_path.name,
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise SnapshotIntegrityError("manifest field %s does not match snapshot" % key)
    return RawSnapshot(
        forum_id=resolved_forum,
        sha256=digest,
        byte_count=len(raw_bytes),
        snapshot_path=str(snapshot_path),
        manifest_path=str(manifest_path),
        retrieved_at=str(manifest.get("retrieved_at", "")),
        source_url=str(manifest.get("source_url", source_url)),
    )


def snapshot_forum(
    client: OpenReviewClient,
    forum_id: str,
    output_dir: os.PathLike,
) -> RawSnapshot:
    raw = client.fetch_forum(forum_id)
    return write_raw_snapshot(raw, output_dir, forum_id, client.base_url)


def load_raw_snapshot(snapshot_path: os.PathLike, manifest_path: Optional[os.PathLike] = None) -> Mapping[str, Any]:
    """Load a raw snapshot and verify its SHA-256 manifest."""

    path = Path(snapshot_path)
    manifest = Path(manifest_path) if manifest_path is not None else path.with_name(
        path.name[:-5] + ".manifest.json" if path.name.endswith(".json") else path.name + ".manifest.json"
    )
    raw_bytes = path.read_bytes()
    try:
        metadata = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SnapshotIntegrityError("cannot read manifest %s: %s" % (manifest, exc))
    digest = hashlib.sha256(raw_bytes).hexdigest()
    if metadata.get("sha256") != digest or metadata.get("byte_count") != len(raw_bytes):
        raise SnapshotIntegrityError("snapshot hash or byte count does not match manifest")
    if metadata.get("snapshot_file") != path.name:
        raise SnapshotIntegrityError("manifest points to a different snapshot file")
    try:
        payload = json.loads(raw_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SnapshotIntegrityError("snapshot is not valid UTF-8 JSON: %s" % exc)
    if not isinstance(payload, Mapping):
        raise SnapshotIntegrityError("snapshot root must be a JSON object")
    return payload


def _canonical_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def _as_strings(value: Any) -> Tuple[str, ...]:
    value = unwrap_content_value(value)
    if value is None:
        return ()
    values = value if isinstance(value, (list, tuple, set)) else [value]
    result = []
    for item in values:
        text = str(item).strip()
        if text:
            result.append(text)
    return tuple(result)


def _note_invitations(note: Mapping[str, Any]) -> Tuple[str, ...]:
    values: List[str] = []
    values.extend(_as_strings(note.get("invitation")))
    values.extend(_as_strings(note.get("invitations")))
    return tuple(dict.fromkeys(values))


def _find_content(content: Mapping[str, Any], aliases: Sequence[str]) -> Any:
    indexed = {_canonical_key(str(key)): value for key, value in content.items()}
    for alias in aliases:
        key = _canonical_key(alias)
        if key in indexed:
            return indexed[key]
    return None


def _as_text(value: Any) -> str:
    value = unwrap_content_value(value)
    if value is None:
        return ""
    if isinstance(value, (list, tuple)):
        return "\n".join(str(item).strip() for item in value if str(item).strip())
    if isinstance(value, Mapping):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value).strip()


def classify_note(note: Mapping[str, Any], forum_id: Optional[str] = None) -> str:
    """Classify a forum note using invitation, signature, and content heuristics."""

    content = normalize_content(note)
    invitations = _note_invitations(note)
    signatures = _as_strings(note.get("signatures"))
    keys = " ".join(_canonical_key(str(key)) for key in content)
    labels = " ".join(_canonical_key(value) for value in invitations + signatures)
    combined = labels + " " + keys
    invitation_roles = tuple(
        _canonical_key(value.rsplit("/-/", 1)[-1]) for value in invitations
    )

    if "decision" in combined or _find_content(content, ("decision", "final_decision")) is not None:
        return "decision"
    author_marked = "author" in labels
    if (
        "rebuttal" in combined
        or "author_response" in combined
        or _find_content(content, ("rebuttal", "author_response")) is not None
        or ("response" in combined and author_marked)
        or (author_marked and any(
            role in ("official_comment", "author_comment", "comment")
            for role in invitation_roles
        ))
    ):
        return "rebuttal"
    review_keys = (
        "summary",
        "strengths",
        "weaknesses",
        "rating",
        "recommendation",
        "soundness",
        "originality",
        "metareview",
    )
    reviewer_marked = "reviewer" in labels or "official_review" in labels or "meta_review" in labels
    review_invitation = any("review" in role for role in invitation_roles)
    has_review_content = any(
        _find_content(content, (key,)) is not None for key in review_keys
    )
    if (review_invitation and has_review_content) or (reviewer_marked and has_review_content):
        return "review"
    note_id = str(note.get("id", ""))
    has_paper_fields = _find_content(content, ("title",)) is not None and _find_content(
        content, ("abstract", "paper_abstract")
    ) is not None
    paper_invitation = any(
        role in ("submission", "blind_submission", "paper_submission")
        for role in invitation_roles
    )
    if (forum_id and note_id == forum_id) or paper_invitation or has_paper_fields:
        return "paper"
    return "other"


@dataclass(frozen=True)
class PaperRecord:
    note_id: str
    forum_id: str
    title: str
    abstract: str
    authors: Tuple[str, ...]
    keywords: Tuple[str, ...]
    pdf: str
    supplemental_material: str
    invitations: Tuple[str, ...]


@dataclass(frozen=True)
class ReviewRecord:
    note_id: str
    forum_id: str
    invitations: Tuple[str, ...]
    signatures: Tuple[str, ...]
    summary: str
    strengths: str
    weaknesses: str
    questions: str
    limitations: str
    comment: str
    soundness: Optional[float]
    presentation: Optional[float]
    significance: Optional[float]
    originality: Optional[float]
    overall_recommendation: Optional[float]
    confidence: Optional[float]

    @property
    def text(self) -> str:
        parts = (
            self.summary,
            self.strengths,
            self.weaknesses,
            self.questions,
            self.limitations,
            self.comment,
        )
        return "\n\n".join(part for part in parts if part)


@dataclass(frozen=True)
class RebuttalRecord:
    note_id: str
    forum_id: str
    invitations: Tuple[str, ...]
    signatures: Tuple[str, ...]
    text: str


@dataclass(frozen=True)
class DecisionRecord:
    note_id: str
    forum_id: str
    invitations: Tuple[str, ...]
    signatures: Tuple[str, ...]
    decision: str
    comment: str

    @property
    def accepted(self) -> Optional[bool]:
        compact = _canonical_key(self.decision)
        if any(token in compact for token in ("reject", "desk_reject", "withdraw", "not_accept")):
            return False
        if "accept" in compact:
            return True
        return None


@dataclass(frozen=True)
class Completeness:
    has_paper: bool
    review_count: int
    scored_review_count: int
    rebuttal_count: int
    has_decision: bool
    fraction: float
    missing: Tuple[str, ...]


@dataclass(frozen=True)
class ForumRecord:
    forum_id: str
    paper: Optional[PaperRecord]
    reviews: Tuple[ReviewRecord, ...]
    rebuttals: Tuple[RebuttalRecord, ...]
    decision: Optional[DecisionRecord]
    completeness: Completeness
    unclassified_note_ids: Tuple[str, ...]


def _paper_from_note(note: Mapping[str, Any], forum_id: str) -> PaperRecord:
    content = normalize_content(note)
    return PaperRecord(
        note_id=str(note.get("id", "")),
        forum_id=forum_id,
        title=_as_text(_find_content(content, ("title",))),
        abstract=_as_text(_find_content(content, ("abstract", "paper_abstract"))),
        authors=_as_strings(_find_content(content, ("authors", "author_names"))),
        keywords=_as_strings(_find_content(content, ("keywords", "subject_areas"))),
        pdf=_as_text(_find_content(content, ("pdf", "paper"))),
        supplemental_material=_as_text(_find_content(content, ("supplementary_material", "supplemental_material"))),
        invitations=_note_invitations(note),
    )


def _review_from_note(note: Mapping[str, Any], forum_id: str) -> ReviewRecord:
    content = normalize_content(note)
    return ReviewRecord(
        note_id=str(note.get("id", "")),
        forum_id=forum_id,
        invitations=_note_invitations(note),
        signatures=_as_strings(note.get("signatures")),
        summary=_as_text(_find_content(content, ("summary", "summary_of_the_paper", "paper_summary"))),
        strengths=_as_text(_find_content(content, ("strengths", "strength"))),
        weaknesses=_as_text(_find_content(content, ("weaknesses", "weakness"))),
        questions=_as_text(_find_content(content, ("questions", "questions_for_authors"))),
        limitations=_as_text(_find_content(content, ("limitations", "limitations_and_societal_impact"))),
        comment=_as_text(_find_content(content, ("comment", "comments", "review", "main_review"))),
        soundness=normalize_score(_find_content(content, ("soundness", "technical_quality"))),
        presentation=normalize_score(_find_content(content, ("presentation", "clarity"))),
        significance=normalize_score(_find_content(content, ("significance", "impact"))),
        originality=normalize_score(_find_content(content, ("originality", "novelty"))),
        overall_recommendation=normalize_score(_find_content(content, (
            "overall_recommendation", "overall_rating", "recommendation", "rating"
        ))),
        confidence=normalize_score(_find_content(content, ("confidence", "reviewer_confidence"))),
    )


def _rebuttal_from_note(note: Mapping[str, Any], forum_id: str) -> RebuttalRecord:
    content = normalize_content(note)
    text = _as_text(_find_content(content, (
        "rebuttal", "author_response", "response", "comment", "comments"
    )))
    if not text:
        text = "\n\n".join(_as_text(value) for value in content.values() if _as_text(value))
    return RebuttalRecord(
        note_id=str(note.get("id", "")),
        forum_id=forum_id,
        invitations=_note_invitations(note),
        signatures=_as_strings(note.get("signatures")),
        text=text,
    )


def _decision_from_note(note: Mapping[str, Any], forum_id: str) -> DecisionRecord:
    content = normalize_content(note)
    return DecisionRecord(
        note_id=str(note.get("id", "")),
        forum_id=forum_id,
        invitations=_note_invitations(note),
        signatures=_as_strings(note.get("signatures")),
        decision=_as_text(_find_content(content, ("decision", "final_decision", "recommendation"))),
        comment=_as_text(_find_content(content, ("comment", "comments", "metareview"))),
    )


def _note_timestamp(note: Mapping[str, Any]) -> int:
    value = note.get("mdate", note.get("tcdate", note.get("cdate", 0)))
    try:
        return int(value or 0)
    except (TypeError, ValueError, OverflowError):
        return 0


def normalize_forum(raw_forum: Mapping[str, Any], forum_id: Optional[str] = None) -> ForumRecord:
    """Normalize one raw OpenReview forum note graph into immutable records."""

    raw_notes = raw_forum.get("notes", [])
    if not isinstance(raw_notes, list):
        raise OpenReviewError("raw forum 'notes' must be a list")
    notes = [note for note in raw_notes if isinstance(note, Mapping)]
    resolved_forum = (forum_id or str(raw_forum.get("forum_id", ""))).strip()
    if not resolved_forum:
        for note in notes:
            candidate = str(note.get("forum", "") or note.get("id", "")).strip()
            if candidate:
                resolved_forum = candidate
                break
    if not resolved_forum:
        raise ValueError("cannot infer forum_id from an empty note graph")

    paper_notes: List[Mapping[str, Any]] = []
    review_notes: List[Mapping[str, Any]] = []
    rebuttal_notes: List[Mapping[str, Any]] = []
    decision_notes: List[Mapping[str, Any]] = []
    unknown: List[str] = []
    for note in notes:
        kind = classify_note(note, resolved_forum)
        if kind == "paper":
            paper_notes.append(note)
        elif kind == "review":
            review_notes.append(note)
        elif kind == "rebuttal":
            rebuttal_notes.append(note)
        elif kind == "decision":
            decision_notes.append(note)
        else:
            unknown.append(str(note.get("id", "")))

    paper_note = None
    if paper_notes:
        paper_note = sorted(
            paper_notes,
            key=lambda note: (str(note.get("id", "")) != resolved_forum, str(note.get("id", ""))),
        )[0]
    paper = _paper_from_note(paper_note, resolved_forum) if paper_note is not None else None
    reviews = tuple(_review_from_note(note, resolved_forum) for note in sorted(
        review_notes, key=lambda item: str(item.get("id", ""))
    ))
    rebuttals = tuple(_rebuttal_from_note(note, resolved_forum) for note in sorted(
        rebuttal_notes, key=lambda item: str(item.get("id", ""))
    ))
    decisions = tuple(_decision_from_note(note, resolved_forum) for note in sorted(
        decision_notes,
        key=lambda item: (_note_timestamp(item), str(item.get("id", ""))),
    ))
    decision = decisions[-1] if decisions else None
    scored = sum(review.overall_recommendation is not None for review in reviews)
    checks = {
        "paper": paper is not None,
        "reviews": bool(reviews),
        "review_scores": scored > 0,
        "rebuttal": bool(rebuttals),
        "decision": decision is not None and bool(decision.decision),
    }
    missing = tuple(key for key, present in checks.items() if not present)
    completeness = Completeness(
        has_paper=paper is not None,
        review_count=len(reviews),
        scored_review_count=scored,
        rebuttal_count=len(rebuttals),
        has_decision=checks["decision"],
        fraction=sum(checks.values()) / float(len(checks)),
        missing=missing,
    )
    return ForumRecord(
        forum_id=resolved_forum,
        paper=paper,
        reviews=reviews,
        rebuttals=rebuttals,
        decision=decision,
        completeness=completeness,
        unclassified_note_ids=tuple(sorted(value for value in unknown if value)),
    )


@dataclass(frozen=True)
class DatasetSplit:
    train: Tuple[str, ...]
    dev: Tuple[str, ...]
    test: Tuple[str, ...]
    seed: str

    def __post_init__(self) -> None:
        train, dev, test = set(self.train), set(self.dev), set(self.test)
        if train.intersection(dev) or train.intersection(test) or dev.intersection(test):
            raise ValueError("forum-level splits overlap")


@dataclass(frozen=True)
class ForumRecordSplit:
    train: Tuple[ForumRecord, ...]
    dev: Tuple[ForumRecord, ...]
    test: Tuple[ForumRecord, ...]
    seed: str


def _apportion(total: int, ratios: Sequence[float]) -> List[int]:
    exact = [total * ratio for ratio in ratios]
    counts = [int(math.floor(value)) for value in exact]
    remainder = total - sum(counts)
    order = sorted(range(len(ratios)), key=lambda index: (-(exact[index] - counts[index]), index))
    for index in order[:remainder]:
        counts[index] += 1
    positive = [index for index, ratio in enumerate(ratios) if ratio > 0]
    if total >= len(positive):
        for empty in [index for index in positive if counts[index] == 0]:
            donors = [index for index in positive if counts[index] > 1]
            if not donors:
                break
            donor = max(donors, key=lambda index: (counts[index], ratios[index], -index))
            counts[donor] -= 1
            counts[empty] += 1
    return counts


def deterministic_forum_split(
    forum_ids: Iterable[str],
    seed: str = "ralphton-icml-v1",
    train_ratio: float = 0.8,
    dev_ratio: float = 0.1,
    test_ratio: float = 0.1,
) -> DatasetSplit:
    """Assign unique forum IDs to deterministic, non-overlapping splits."""

    ratios = (float(train_ratio), float(dev_ratio), float(test_ratio))
    if any(not math.isfinite(value) or value < 0 for value in ratios):
        raise ValueError("split ratios must be finite and non-negative")
    if not math.isclose(sum(ratios), 1.0, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError("split ratios must sum to 1")
    identifiers = sorted({str(value).strip() for value in forum_ids if str(value).strip()})
    ordered = sorted(
        identifiers,
        key=lambda forum: (hashlib.sha256((str(seed) + "\0" + forum).encode("utf-8")).digest(), forum),
    )
    train_count, dev_count, test_count = _apportion(len(ordered), ratios)
    train_end = train_count
    dev_end = train_end + dev_count
    return DatasetSplit(
        train=tuple(ordered[:train_end]),
        dev=tuple(ordered[train_end:dev_end]),
        test=tuple(ordered[dev_end:dev_end + test_count]),
        seed=str(seed),
    )


def split_forum_records(
    records: Iterable[ForumRecord],
    seed: str = "ralphton-icml-v1",
    train_ratio: float = 0.8,
    dev_ratio: float = 0.1,
    test_ratio: float = 0.1,
) -> ForumRecordSplit:
    """Split records by forum ID, rejecting duplicate forum objects."""

    by_id: Dict[str, ForumRecord] = {}
    for record in records:
        if record.forum_id in by_id:
            raise ValueError("duplicate forum record: %s" % record.forum_id)
        by_id[record.forum_id] = record
    split = deterministic_forum_split(by_id, seed, train_ratio, dev_ratio, test_ratio)
    return ForumRecordSplit(
        train=tuple(by_id[value] for value in split.train),
        dev=tuple(by_id[value] for value in split.dev),
        test=tuple(by_id[value] for value in split.test),
        seed=split.seed,
    )
