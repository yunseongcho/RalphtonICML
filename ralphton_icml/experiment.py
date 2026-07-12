"""Reproducible real-seed learning experiment and artifact manifests."""

from __future__ import annotations

import dataclasses
import datetime as dt
import hashlib
import json
from pathlib import Path
import platform
import sys
from typing import Any, Dict, Mapping, Optional, Sequence

from .learning import (
    LearningConfig,
    LearningState,
    compute_prompt_manifest_digest,
    dump_learning_state,
    evaluate_predictions,
    predict_many,
    run_learning_loop,
)
from .schema import ReviewOutput
from .seed import load_seed_cases, split_seed_examples


PROMPT_MANIFEST_FILES = (
    "prompts.py",
    "review_prompts.py",
    "reviewer_instruction.md",
    "ralphton_icml/schema.py",
    "ralphton_icml/context.py",
    "ralphton_icml/team.py",
    "ralphton_icml/orchestrator.py",
    "ralphton_icml/learning.py",
    "ralphton_icml/seed.py",
    "ralphton_icml/experiment.py",
    "configs/learning.json",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(_json_bytes(value))
    temporary.replace(path)


def _write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def build_prompt_manifest(root: Path) -> Dict[str, Any]:
    root = Path(root).resolve()
    files = {}
    for relative in PROMPT_MANIFEST_FILES:
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError("prompt manifest input missing: {}".format(path))
        files[relative] = sha256_file(path)
    payload: Dict[str, Any] = {
        "schema_version": 1,
        "files": files,
        "review_output_contract": "reviewer_instruction.md seven-field subset",
        "paper_signal_adapter": "paper-only-structural-heuristics-v1",
        "model_backend": "none (deterministic learning smoke)",
    }
    payload["digest"] = compute_prompt_manifest_digest(payload)
    return payload


def _evaluation_dict(value: Any) -> Dict[str, Any]:
    return dataclasses.asdict(value)


def load_learning_config(path: Path) -> LearningConfig:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("learning config must be a JSON object")
    try:
        return LearningConfig(**payload)
    except TypeError as exc:
        raise ValueError("invalid learning config fields: {}".format(exc)) from exc


def run_real_seed_experiment(
    root: Path,
    seed_path: Path,
    seed_manifest_path: Path,
    output_dir: Path,
    config: Optional[LearningConfig] = None,
    config_path: Optional[Path] = None,
    split_seed: str = "ralphton-icml-real-seed-v1",
) -> Mapping[str, Any]:
    """Run train updates, dev convergence, best restore, and one test eval."""

    root = Path(root).resolve()
    seed_path = Path(seed_path).resolve()
    seed_manifest_path = Path(seed_manifest_path).resolve()
    output_dir = Path(output_dir).resolve()
    if config is not None and config_path is not None:
        raise ValueError("provide config or config_path, not both")
    registered_config_path: Optional[Path] = None
    if config is None:
        requested_config = Path(config_path or Path("configs") / "learning.json")
        registered_config_path = (
            requested_config
            if requested_config.is_absolute()
            else root / requested_config
        ).resolve()
        resolved_config = load_learning_config(registered_config_path)
    else:
        resolved_config = config
    data_manifest = json.loads(seed_manifest_path.read_text(encoding="utf-8"))
    seed_hash = sha256_file(seed_path)
    if data_manifest.get("output_sha256") != seed_hash:
        raise ValueError("seed corpus hash does not match seed_manifest.json")

    cases = load_seed_cases(seed_path)
    train, dev, test, split = split_seed_examples(cases, seed=split_seed)
    prompt_manifest = build_prompt_manifest(root)
    initial_state = LearningState(prompt_manifest_digest=prompt_manifest["digest"])
    baseline_dev = evaluate_predictions(predict_many(initial_state, dev), dev)
    run = run_learning_loop(
        train,
        dev,
        test,
        initial_state=initial_state,
        config=resolved_config,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    split_payload = {
        "schema_version": 1,
        "seed": split.seed,
        "train": list(split.train),
        "dev": list(split.dev),
        "test": list(split.test),
        "grouping": "OpenReview forum_id",
        "ratios": {"train": 0.8, "dev": 0.1, "test": 0.1},
    }
    history_payload = [
        {
            **dataclasses.asdict(item),
            "evaluation": _evaluation_dict(item.evaluation),
        }
        for item in run.history
    ]
    summary: Dict[str, Any] = {
        "schema_version": 1,
        "claim_scope": "deterministic pipeline/update smoke; not reviewer-quality evidence",
        "record_count": len(cases),
        "split_counts": {"train": len(train), "dev": len(dev), "test": len(test)},
        "baseline_state_digest": initial_state.digest,
        "best_state_digest": run.state.digest,
        "best_state_version": run.state.version,
        "best_iteration": run.best_iteration,
        "iterations": len(run.history),
        "stop_reason": run.stop_reason,
        "converged": run.stop_reason == "converged",
        "baseline_dev": _evaluation_dict(baseline_dev),
        "best_dev": _evaluation_dict(run.dev_evaluation),
        "final_test": _evaluation_dict(run.test_evaluation),
        "dev_utility_change": run.dev_evaluation.utility - baseline_dev.utility,
        "test_evaluation_count": 1,
        "test_fingerprint": run.test_fingerprint,
        "last_iteration": history_payload[-1] if history_payload else None,
        "reviewer_memory_items": len(run.state.reviewer_memory),
        "author_memory_items": len(run.state.author_memory),
        "memory_forum_ids": sorted(
            {
                item.forum_id
                for item in run.state.reviewer_memory + run.state.author_memory
            }
        ),
        "known_limitations": data_manifest.get("limitations", []),
    }
    config_payload = dataclasses.asdict(resolved_config)
    run_id_payload = {
        "seed_sha256": seed_hash,
        "prompt_manifest_digest": prompt_manifest["digest"],
        "split_seed": split_seed,
        "config": config_payload,
    }
    run_id = hashlib.sha256(
        json.dumps(run_id_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:20]

    _write_json(output_dir / "prompt_manifest.json", prompt_manifest)
    _write_json(output_dir / "split.json", split_payload)
    _write_json(output_dir / "config.json", config_payload)
    _write_json(output_dir / "history.json", history_payload)
    _write_json(output_dir / "summary.json", summary)
    dump_learning_state(run.state, output_dir / "best_state.json")

    dev_preview_prediction = predict_many(run.state, dev)[dev[0].forum_id]
    dev_preview = ReviewOutput(
        soundness=dev_preview_prediction["soundness"],
        presentation=dev_preview_prediction["presentation"],
        significance=dev_preview_prediction["significance"],
        originality=dev_preview_prediction["originality"],
        overall_recommendation=dev_preview_prediction["overall_recommendation"],
        confidence=dev_preview_prediction["confidence"],
        comment=dev_preview_prediction["comment"],
    )
    _write_text(
        output_dir / "dev_review_preview.md",
        dev_preview.to_markdown(),
    )
    human_review = """# Human Review Checkpoint

## Run Identity

- Run ID: `{run_id}`
- Public complete cases: {records}
- Forum-level split: {train} train / {dev} dev / {test} test
- Seed corpus SHA-256: `{seed_hash}`
- Prompt/schema/team manifest: `{prompt_digest}`

## Convergence

- Stop reason: **{stop_reason}**
- Iterations executed: {iterations}
- Best iteration/state version: {best_iteration} / {best_version}
- Last plateau count: {plateau_count}
- Last quality/behavior/state delta: {quality_delta:.6f} / {behavior_delta:.6f} / {state_delta:.6f}
- Reviewer/author memory items: {reviewer_memory} / {author_memory}

The final artifact restores the best state rather than the last candidate. Open
`history.json` to inspect every accepted/rejected update and the three-step
plateau that triggered stopping.

## Metrics

| Metric | Baseline dev | Best dev | Final sealed test |
|---|---:|---:|---:|
| Schema field coverage | {base_coverage:.4f} | {dev_coverage:.4f} | {test_coverage:.4f} |
| Complete-form coverage | {base_complete:.4f} | {dev_complete:.4f} | {test_complete:.4f} |
| MAE | {base_mae:.4f} | {dev_mae:.4f} | {test_mae:.4f} |
| Brier | {base_brier:.4f} | {dev_brier:.4f} | {test_brier:.4f} |
| Utility | {base_utility:.4f} | {dev_utility:.4f} | {test_utility:.4f} |

Dev utility change: **{utility_change:+.6f}**. Test evaluation count: **1**.

## What To Inspect

1. `split.json`: no forum ID occurs in more than one split.
2. `best_state.json`: every retrieval-memory forum is in the train split.
3. `history.json`: non-regression decisions and convergence deltas are explicit.
4. `dev_review_preview.md`: renderer output exactly matches `reviewer_instruction.md`.
5. `prompt_manifest.json`: prompt, schema, team, and orchestration hashes are fixed.

The preview is a deterministic surrogate for contract inspection, not an LLM
review-quality result. Its source forum is `{preview_forum}`.

## Interpretation Limits

- This is a deterministic pipeline/update smoke, not evidence that the reviewer is scientifically good.
- Dev and test contain only two forums each; confidence intervals and domain claims are invalid.
- The public paper text may be a revised version rather than the initial submission.
- RecSys has no complete case in this seed.
- No hosted or local LLM was used in this run; the paper-only input is a structural heuristic.
""".format(
        records=len(cases),
        run_id=run_id,
        train=len(train),
        dev=len(dev),
        test=len(test),
        seed_hash=seed_hash,
        prompt_digest=prompt_manifest["digest"],
        stop_reason=run.stop_reason,
        iterations=len(run.history),
        best_iteration=run.best_iteration,
        best_version=run.state.version,
        plateau_count=run.history[-1].plateau_count,
        quality_delta=run.history[-1].quality_improvement,
        behavior_delta=run.history[-1].behavioral_delta,
        state_delta=run.history[-1].state_delta,
        reviewer_memory=len(run.state.reviewer_memory),
        author_memory=len(run.state.author_memory),
        base_coverage=baseline_dev.field_coverage,
        dev_coverage=run.dev_evaluation.field_coverage,
        test_coverage=run.test_evaluation.field_coverage,
        base_complete=baseline_dev.complete_coverage,
        dev_complete=run.dev_evaluation.complete_coverage,
        test_complete=run.test_evaluation.complete_coverage,
        base_mae=baseline_dev.mae or 0.0,
        dev_mae=run.dev_evaluation.mae or 0.0,
        test_mae=run.test_evaluation.mae or 0.0,
        base_brier=baseline_dev.brier or 0.0,
        dev_brier=run.dev_evaluation.brier or 0.0,
        test_brier=run.test_evaluation.brier or 0.0,
        base_utility=baseline_dev.utility,
        dev_utility=run.dev_evaluation.utility,
        test_utility=run.test_evaluation.utility,
        utility_change=run.dev_evaluation.utility - baseline_dev.utility,
        preview_forum=dev[0].forum_id,
    )
    _write_text(output_dir / "HUMAN_REVIEW.md", human_review)

    artifact_names = (
        "prompt_manifest.json",
        "split.json",
        "config.json",
        "history.json",
        "summary.json",
        "best_state.json",
        "dev_review_preview.md",
        "HUMAN_REVIEW.md",
    )
    artifact_hashes = {
        name: sha256_file(output_dir / name) for name in artifact_names
    }
    if registered_config_path is None:
        config_source = "inline"
    else:
        try:
            config_source = str(registered_config_path.relative_to(root))
        except ValueError:
            config_source = str(registered_config_path)
    manifest = {
        "schema_version": 1,
        "run_id": run_id,
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "seed_corpus": str(seed_path.relative_to(root)),
        "seed_corpus_sha256": seed_hash,
        "seed_manifest": str(seed_manifest_path.relative_to(root)),
        "seed_manifest_sha256": sha256_file(seed_manifest_path),
        "registered_config": config_source,
        "registered_config_sha256": (
            None if registered_config_path is None else sha256_file(registered_config_path)
        ),
        "prompt_manifest_digest": prompt_manifest["digest"],
        "artifacts": artifact_hashes,
        "environment": {
            "python": sys.version.split()[0],
            "implementation": platform.python_implementation(),
            "os": platform.platform(),
            "machine": platform.machine(),
            "accelerator": "Apple M2 Pro / MPS available; not used by deterministic smoke",
        },
        "test_evaluation_count": 1,
    }
    _write_json(output_dir / "run_manifest.json", manifest)
    return {"manifest": manifest, "summary": summary}


__all__ = [
    "PROMPT_MANIFEST_FILES",
    "build_prompt_manifest",
    "load_learning_config",
    "run_real_seed_experiment",
    "sha256_file",
]
