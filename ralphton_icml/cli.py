"""Command-line interface for team inspection, review runs, and learning."""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
from pathlib import Path
import shlex
import sys
import tempfile
import time
from typing import Any, Mapping, Optional, Sequence

from .backend import SubprocessBackend
from .batch import BatchReviewScheduler
from .codex_backend import CodexExecBackend
from .experiment import build_prompt_manifest, run_real_seed_experiment
from .fast import FastReviewerOrchestrator, fast_pipeline_digest, fast_run_as_dict
from .instruction import load_reviewer_instruction
from .learning import load_learning_state
from .openreview import OpenReviewClient, normalize_forum, snapshot_forum
from .orchestrator import PaperInput, ReviewerOrchestrator
from .schema import ReviewOutput
from .team import DEFAULT_REVIEWER_TEAM
from .track2 import (
    create_track2_bundle,
    load_track2_bundle,
    write_review_agent,
)


def _json_default(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        return dataclasses.asdict(value)
    if hasattr(value, "value"):
        return value.value
    raise TypeError("cannot serialize {}".format(type(value).__name__))


def _print_json(value: Any) -> None:
    print(json.dumps(value, default=_json_default, ensure_ascii=False, indent=2, sort_keys=True))


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary_name = ""
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=str(target.parent),
            prefix=target.name + ".",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary_name = stream.name
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, str(target))
        temporary_name = ""
    finally:
        if temporary_name:
            try:
                Path(temporary_name).unlink()
            except FileNotFoundError:
                pass


def _atomic_write_json(path: Path, value: Any) -> None:
    payload = (
        json.dumps(
            value,
            default=_json_default,
            ensure_ascii=False,
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")
    _atomic_write_bytes(path, payload)


def _atomic_write_text(path: Path, value: str) -> None:
    _atomic_write_bytes(path, value.encode("utf-8"))


def _team_command(_args: argparse.Namespace) -> int:
    _print_json(
        {
            "domain_experts": DEFAULT_REVIEWER_TEAM.domain_experts,
            "criterion_experts": DEFAULT_REVIEWER_TEAM.criterion_experts,
            "author": DEFAULT_REVIEWER_TEAM.author,
            "synthesizer": DEFAULT_REVIEWER_TEAM.synthesizer,
            "chair": DEFAULT_REVIEWER_TEAM.chair,
        }
    )
    return 0


def _validate_instruction_command(_args: argparse.Namespace) -> int:
    text = load_reviewer_instruction()
    _print_json({"valid": True, "characters": len(text)})
    return 0


def _validate_review_command(args: argparse.Namespace) -> int:
    review = ReviewOutput.from_markdown(Path(args.path).read_text(encoding="utf-8"))
    _print_json({"valid": True, "review": review.as_dict()})
    return 0


def _snapshot_command(args: argparse.Namespace) -> int:
    snapshot = snapshot_forum(
        OpenReviewClient(timeout=args.timeout), args.forum_id, args.output_dir
    )
    _print_json(snapshot)
    return 0


def _normalize_command(args: argparse.Namespace) -> int:
    raw = json.loads(Path(args.input).read_text(encoding="utf-8"))
    record = normalize_forum(raw, forum_id=args.forum_id)
    payload = dataclasses.asdict(record)
    if args.output:
        Path(args.output).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    else:
        _print_json(payload)
    return 0


def _load_compatible_learning_state(path: str, root: Path):
    """Load an integrity-protected state bound to the current prompt manifest."""

    state_path = Path(path)
    payload = json.loads(state_path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping) or not isinstance(payload.get("digest"), str):
        raise ValueError("learning state must contain an integrity digest")
    state = load_learning_state(state_path)
    current_digest = build_prompt_manifest(Path(root))["digest"]
    if state.prompt_manifest_digest != current_digest:
        raise ValueError(
            "learning state prompt manifest mismatch: state={}, current={}; "
            "rerun run-seed for this source tree or select a compatible state".format(
                state.prompt_manifest_digest, current_digest
            )
        )
    return state


def _legacy_paper(path: Path) -> PaperInput:
    paper_data = json.loads(Path(path).read_text(encoding="utf-8"))
    if "paper" in paper_data and isinstance(paper_data["paper"], Mapping):
        paper_data = dict(paper_data["paper"], forum_id=paper_data.get("forum_id", ""))
    return PaperInput(
        paper_id=str(paper_data.get("paper_id") or paper_data.get("forum_id") or "paper"),
        title=str(paper_data["title"]),
        text=str(paper_data["text"]),
        document_id=str(paper_data.get("document_id", "paper")),
        source_uri=str(paper_data.get("source_uri") or paper_data.get("openreview_url") or ""),
    )


def _author_loop_value(args: argparse.Namespace) -> str:
    value = args.author_loop
    if args.no_author_loop:
        if value not in (None, "conditional"):
            raise ValueError("--no-author-loop conflicts with --author-loop")
        print(
            "warning: --no-author-loop is deprecated; use --author-loop never",
            file=sys.stderr,
        )
        value = "never"
    return value or "conditional"


def _model_backend(
    args: argparse.Namespace,
    *,
    pipeline_digest: str,
    state_digest: str,
    deadline_at: Optional[float],
):
    if args.backend == "codex":
        if args.backend_command:
            raise ValueError("--backend codex and --backend-command are mutually exclusive")
        if not args.model:
            raise ValueError("--model is required for the Codex backend")
        return CodexExecBackend(
            args.model,
            concurrency=args.codex_concurrency,
            hard_deadline=deadline_at,
            cache_dir=(None if not args.cache_dir else Path(args.cache_dir)),
            progress_path=(None if not args.progress else Path(args.progress)),
            pipeline_digest=pipeline_digest,
            state_digest=state_digest,
        )
    if not args.backend_command:
        raise ValueError("--backend-command is required for the subprocess backend")
    if args.model:
        raise ValueError("--model is only valid with --backend codex")
    return SubprocessBackend(shlex.split(args.backend_command), timeout=args.timeout)


def _legacy_review_command(args: argparse.Namespace, learning_state: Any) -> int:
    if args.backend == "codex":
        raise ValueError("legacy-v1 requires the subprocess compatibility backend")
    paper = _legacy_paper(Path(args.paper))
    backend = _model_backend(
        args,
        pipeline_digest="legacy-v1",
        state_digest="" if learning_state is None else learning_state.digest,
        deadline_at=None,
    )
    author_loop = _author_loop_value(args)
    run = ReviewerOrchestrator(
        backend,
        max_workers=args.max_workers,
        learning_state=learning_state,
    ).review(paper, run_author_loop=author_loop != "never")
    output = {
        "paper_id": run.paper_id,
        "context_revision": run.context.revision,
        "context_evidence_count": len(run.context),
        "domain_critiques": dict(run.domain_critiques),
        "criterion_critiques": dict(run.criterion_critiques),
        "synthesis": run.synthesis,
        "initial_review": run.initial_review.as_dict(),
        "author_rebuttal": run.author_rebuttal,
        "final_review": None if run.final_review is None else run.final_review.as_dict(),
        "rendered_review": run.effective_review.to_markdown(),
    }
    _atomic_write_json(Path(args.output), output)
    print(run.effective_review.to_markdown(), end="")
    return 0


def _fast_review_command(args: argparse.Namespace, learning_state: Any) -> int:
    if args.deadline_seconds <= 0:
        raise ValueError("--deadline-seconds must be positive")
    deadline_at = time.monotonic() + args.deadline_seconds

    def extraction_timeout(commands_remaining: int) -> float:
        remaining = deadline_at - time.monotonic()
        if remaining <= 0:
            raise ValueError("Track 2 deadline reached during input materialization")
        return min(120.0, max(0.01, remaining / commands_remaining))

    paper_path = Path(args.paper).resolve(strict=True)
    track2_root = (
        Path(args.track2_root).resolve(strict=False)
        if args.track2_root
        else (paper_path.parent.parent if paper_path.parent.name == "inputs" else paper_path.parent)
    )
    track2_root.mkdir(parents=True, exist_ok=True)
    (track2_root / "outputs").mkdir(parents=True, exist_ok=True)
    evidence_paths = tuple(Path(value).resolve(strict=True) for value in args.evidence)
    bundle = create_track2_bundle(
        track2_root,
        paper_path,
        evidence_paths=evidence_paths,
        agent_name=args.agent_name,
        agent_version=args.agent_version,
        result_filename=args.result_filename,
        pdftotext_executable=args.pdftotext,
        extraction_timeout=extraction_timeout(4),
    )
    agent_path = write_review_agent(bundle, path=args.review_agent)
    bundle = load_track2_bundle(
        agent_path,
        pdftotext_executable=args.pdftotext,
        extraction_timeout=extraction_timeout(2),
    )
    author_loop = _author_loop_value(args)
    config = {
        "author_loop": author_loop,
        "attempts": args.attempts,
        "memory_limit": args.memory_limit,
        "pipeline": "fast-v1",
    }
    pipeline_digest = fast_pipeline_digest(config)
    backend = _model_backend(
        args,
        pipeline_digest=pipeline_digest,
        state_digest="" if learning_state is None else learning_state.digest,
        deadline_at=deadline_at,
    )
    run = FastReviewerOrchestrator(
        backend,
        learning_state=learning_state,
        attempts=args.attempts,
        memory_limit=args.memory_limit,
        deadline_at=deadline_at,
    ).review(bundle, author_loop=author_loop)
    output = fast_run_as_dict(run)
    output["track2_manifest"] = bundle.to_manifest_dict()
    output["pipeline_digest"] = pipeline_digest
    metrics = getattr(backend, "snapshot_metrics", None)
    if callable(metrics):
        output["backend_metrics"] = metrics()
    _atomic_write_json(Path(args.output), output)
    _atomic_write_text(bundle.result_path, run.effective_review.to_markdown())
    print(run.effective_review.to_markdown(), end="")
    return 0


def _review_command(args: argparse.Namespace) -> int:
    learning_state = (
        _load_compatible_learning_state(args.state, Path(args.root))
        if args.state
        else None
    )
    if args.pipeline == "legacy-v1":
        return _legacy_review_command(args, learning_state)
    return _fast_review_command(args, learning_state)


def _review_batch_command(args: argparse.Namespace) -> int:
    if args.pipeline != "fast-v1":
        raise ValueError("review-batch supports only fast-v1")
    if args.backend != "codex":
        raise ValueError("review-batch requires the first-class Codex backend")
    output_dir = Path(args.output_dir).resolve(strict=False)
    output_dir.mkdir(parents=True, exist_ok=True)
    learning_state = (
        _load_compatible_learning_state(args.state, Path(args.root))
        if args.state
        else None
    )
    author_loop = _author_loop_value(args)
    config = {
        "pipeline": "fast-v1",
        "author_loop": author_loop,
        "attempts": args.attempts,
        "memory_limit": args.memory_limit,
        "paper_workers": args.paper_workers,
        "codex_concurrency": args.codex_concurrency,
        "max_refinements": args.max_refinements,
        "deadline_seconds": args.deadline_seconds,
        "soft_deadline_seconds": args.soft_deadline_seconds,
    }
    pipeline_digest = fast_pipeline_digest(config)
    if not args.cache_dir:
        args.cache_dir = str(output_dir / "backend_cache")
    if not args.progress:
        args.progress = str(output_dir / "backend-progress.jsonl")
    backend = _model_backend(
        args,
        pipeline_digest=pipeline_digest,
        state_digest="" if learning_state is None else learning_state.digest,
        deadline_at=None,
    )
    result = BatchReviewScheduler(
        backend,
        learning_state=learning_state,
        paper_workers=args.paper_workers,
        attempts=args.attempts,
        memory_limit=args.memory_limit,
        author_loop=author_loop,
        max_refinements=args.max_refinements,
        deadline_seconds=args.deadline_seconds,
        soft_deadline_seconds=args.soft_deadline_seconds,
        model=args.model,
        cli_version=getattr(backend, "cli_version", None),
        state_digest=(None if learning_state is None else learning_state.digest),
        pipeline_config=dict(config, pipeline_digest=pipeline_digest),
        template_path=(None if not args.template else Path(args.template)),
    ).run(Path(args.manifest), output_dir, resume=args.resume)
    _print_json(result.summary)
    return result.exit_code


def _run_seed_command(args: argparse.Namespace) -> int:
    result = run_real_seed_experiment(
        root=Path(args.root),
        seed_path=Path(args.seed),
        seed_manifest_path=Path(args.seed_manifest),
        output_dir=Path(args.output_dir),
        config_path=Path(args.config),
        split_seed=args.split_seed,
    )
    _print_json(result)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ralphton-icml")
    subparsers = parser.add_subparsers(dest="command", required=True)

    team = subparsers.add_parser("team", help="print the complete agent registry")
    team.set_defaults(func=_team_command)

    instruction = subparsers.add_parser(
        "validate-instruction", help="validate reviewer_instruction.md against ReviewOutput"
    )
    instruction.set_defaults(func=_validate_instruction_command)

    validate = subparsers.add_parser("validate-review", help="validate rendered review Markdown")
    validate.add_argument("path")
    validate.set_defaults(func=_validate_review_command)

    snapshot = subparsers.add_parser("snapshot", help="snapshot one public OpenReview forum")
    snapshot.add_argument("forum_id")
    snapshot.add_argument("output_dir")
    snapshot.add_argument("--timeout", type=float, default=30.0)
    snapshot.set_defaults(func=_snapshot_command)

    normalize = subparsers.add_parser("normalize", help="normalize a raw OpenReview forum graph")
    normalize.add_argument("input")
    normalize.add_argument("--forum-id")
    normalize.add_argument("--output")
    normalize.set_defaults(func=_normalize_command)

    review = subparsers.add_parser("review", help="run the Track 2 model-backed reviewer team")
    review.add_argument("paper", help="frozen Track 1 PDF, Markdown, or compatible paper JSON")
    review.add_argument(
        "--pipeline",
        choices=("fast-v1", "legacy-v1"),
        default="fast-v1",
    )
    review.add_argument(
        "--backend",
        choices=("subprocess", "codex"),
        default="codex",
    )
    review.add_argument("--backend-command", help="quoted JSON-over-stdin model command")
    review.add_argument("--model", help="explicit Codex model; required with --backend codex")
    review.add_argument("--output", required=True)
    review.add_argument("--timeout", type=float, default=300.0)
    review.add_argument("--max-workers", type=int, default=4)
    review.add_argument("--codex-concurrency", type=int, default=4)
    review.add_argument("--deadline-seconds", type=float, default=1800.0)
    review.add_argument("--attempts", type=int, default=2)
    review.add_argument("--memory-limit", type=int, default=8)
    review.add_argument("--cache-dir")
    review.add_argument("--progress")
    review.add_argument("--track2-root")
    review.add_argument("--evidence", action="append", default=[])
    review.add_argument("--agent-name", default="Ralphton Track 2 Review Agent")
    review.add_argument("--agent-version", default="fast-v1")
    review.add_argument("--result-filename", default="review-result.md")
    review.add_argument("--review-agent")
    review.add_argument("--pdftotext", default="pdftotext")
    review.add_argument("--state", help="best_state.json from a completed update run")
    review.add_argument(
        "--root",
        default=".",
        help="source root used to verify the state's prompt manifest (default: .)",
    )
    review.add_argument(
        "--author-loop",
        choices=("always", "conditional", "never"),
        default="conditional",
    )
    review.add_argument("--no-author-loop", action="store_true")
    review.set_defaults(func=_review_command)

    batch = subparsers.add_parser(
        "review-batch", help="run a bounded Track 2 fast-v1 batch"
    )
    batch.add_argument("manifest", help="paper-only batch manifest JSON")
    batch.add_argument("--output-dir", required=True)
    batch.add_argument("--pipeline", choices=("fast-v1",), default="fast-v1")
    batch.add_argument("--backend", choices=("codex",), default="codex")
    batch.add_argument("--model", required=True, help="explicit hosted Codex model")
    batch.add_argument("--backend-command")
    batch.add_argument("--timeout", type=float, default=300.0)
    batch.add_argument("--paper-workers", type=int, default=4)
    batch.add_argument("--codex-concurrency", type=int, default=4)
    batch.add_argument("--attempts", type=int, default=2)
    batch.add_argument("--memory-limit", type=int, default=8)
    batch.add_argument("--max-refinements", type=int, default=2)
    batch.add_argument("--deadline-seconds", type=float, default=1800.0)
    batch.add_argument("--soft-deadline-seconds", type=float, default=1440.0)
    batch.add_argument("--cache-dir")
    batch.add_argument("--progress")
    batch.add_argument("--template")
    batch.add_argument("--state", help="compatible best_state.json")
    batch.add_argument("--root", default=".")
    batch.add_argument(
        "--author-loop",
        choices=("always", "conditional", "never"),
        default="conditional",
    )
    batch.add_argument("--no-author-loop", action="store_true")
    batch.add_argument("--resume", action="store_true")
    batch.set_defaults(func=_review_batch_command)

    seed = subparsers.add_parser("run-seed", help="run the public 20-case learning smoke")
    seed.add_argument("--root", default=".")
    seed.add_argument("--seed", default="data/real/seed_cases.jsonl")
    seed.add_argument("--seed-manifest", default="data/real/seed_manifest.json")
    seed.add_argument("--output-dir", default="artifacts/real_seed_v2")
    seed.add_argument("--config", default="configs/learning.json")
    seed.add_argument("--split-seed", default="ralphton-icml-real-seed-v1")
    seed.set_defaults(func=_run_seed_command)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except Exception as exc:
        parser.exit(1, "error: {}\n".format(exc))
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
