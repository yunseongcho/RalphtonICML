"""Command-line interface for team inspection, review runs, and learning."""

from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path
import shlex
import sys
from typing import Any, Mapping, Optional, Sequence

from .backend import SubprocessBackend
from .experiment import build_prompt_manifest, run_real_seed_experiment
from .instruction import load_reviewer_instruction
from .learning import load_learning_state
from .openreview import OpenReviewClient, normalize_forum, snapshot_forum
from .orchestrator import PaperInput, ReviewerOrchestrator
from .schema import ReviewOutput
from .team import DEFAULT_REVIEWER_TEAM


def _json_default(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        return dataclasses.asdict(value)
    if hasattr(value, "value"):
        return value.value
    raise TypeError("cannot serialize {}".format(type(value).__name__))


def _print_json(value: Any) -> None:
    print(json.dumps(value, default=_json_default, ensure_ascii=False, indent=2, sort_keys=True))


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


def _review_command(args: argparse.Namespace) -> int:
    paper_data = json.loads(Path(args.paper).read_text(encoding="utf-8"))
    if "paper" in paper_data and isinstance(paper_data["paper"], Mapping):
        paper_data = dict(paper_data["paper"], forum_id=paper_data.get("forum_id", ""))
    paper = PaperInput(
        paper_id=str(paper_data.get("paper_id") or paper_data.get("forum_id") or "paper"),
        title=str(paper_data["title"]),
        text=str(paper_data["text"]),
        document_id=str(paper_data.get("document_id", "paper")),
        source_uri=str(paper_data.get("source_uri") or paper_data.get("openreview_url") or ""),
    )
    command = shlex.split(args.backend_command)
    learning_state = (
        _load_compatible_learning_state(args.state, Path(args.root))
        if args.state
        else None
    )
    run = ReviewerOrchestrator(
        SubprocessBackend(command, timeout=args.timeout),
        max_workers=args.max_workers,
        learning_state=learning_state,
    ).review(paper, run_author_loop=not args.no_author_loop)
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
    Path(args.output).write_text(
        json.dumps(output, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(run.effective_review.to_markdown(), end="")
    return 0


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

    review = subparsers.add_parser("review", help="run the full model-backed reviewer team")
    review.add_argument("paper", help="JSON containing paper_id/title/text")
    review.add_argument("--backend-command", required=True, help="quoted JSON-over-stdin model command")
    review.add_argument("--output", required=True)
    review.add_argument("--timeout", type=float, default=300.0)
    review.add_argument("--max-workers", type=int, default=4)
    review.add_argument("--state", help="best_state.json from a completed update run")
    review.add_argument(
        "--root",
        default=".",
        help="source root used to verify the state's prompt manifest (default: .)",
    )
    review.add_argument("--no-author-loop", action="store_true")
    review.set_defaults(func=_review_command)

    seed = subparsers.add_parser("run-seed", help="run the public 20-case learning smoke")
    seed.add_argument("--root", default=".")
    seed.add_argument("--seed", default="data/real/seed_cases.jsonl")
    seed.add_argument("--seed-manifest", default="data/real/seed_manifest.json")
    seed.add_argument("--output-dir", default="artifacts/real_seed_v1")
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
