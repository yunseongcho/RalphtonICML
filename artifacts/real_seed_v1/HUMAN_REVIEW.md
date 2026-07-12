# Human Review Checkpoint

## Run Identity

- Run ID: `77771e943444d32dc716`
- Public complete cases: 20
- Forum-level split: 16 train / 2 dev / 2 test
- Seed corpus SHA-256: `02586f1cfef1921d6660dc17692ead8a3762f2421c32b8c94bfb60289c8a92ab`
- Prompt/schema/team manifest: `c0562f49640cbe6b33e59a6bcb2e673a2bd7f04cb6915a95f9805a7cf18bcbc2`

## Convergence

- Stop reason: **converged**
- Iterations executed: 5
- Best iteration/state version: 1 / 1
- Last plateau count: 3
- Last quality/behavior/state delta: 0.000000 / 0.000000 / 0.000000
- Reviewer/author memory items: 64 / 53

The final artifact restores the best state rather than the last candidate. Open
`history.json` to inspect every accepted/rejected update and the three-step
plateau that triggered stopping.

## Metrics

| Metric | Baseline dev | Best dev | Final sealed test |
|---|---:|---:|---:|
| Schema field coverage | 1.0000 | 1.0000 | 1.0000 |
| Complete-form coverage | 1.0000 | 1.0000 | 1.0000 |
| MAE | 0.7083 | 0.7083 | 0.3750 |
| Brier | 0.2863 | 0.2796 | 0.1229 |
| Utility | 0.8762 | 0.8778 | 0.9426 |

Dev utility change: **+0.001656**. Test evaluation count: **1**.

## What To Inspect

1. `split.json`: no forum ID occurs in more than one split.
2. `best_state.json`: every retrieval-memory forum is in the train split.
3. `history.json`: non-regression decisions and convergence deltas are explicit.
4. `dev_review_preview.md`: renderer output exactly matches `reviewer_instruction.md`.
5. `prompt_manifest.json`: prompt, schema, team, and orchestration hashes are fixed.

The preview is a deterministic surrogate for contract inspection, not an LLM
review-quality result. Its source forum is `VWqiPBB_EM`.

## Interpretation Limits

- This is a deterministic pipeline/update smoke, not evidence that the reviewer is scientifically good.
- Dev and test contain only two forums each; confidence intervals and domain claims are invalid.
- The public paper text may be a revised version rather than the initial submission.
- RecSys has no complete case in this seed.
- No hosted or local LLM was used in this run; the paper-only input is a structural heuristic.
