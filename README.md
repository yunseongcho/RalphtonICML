# Ralphton ICML Reviewer Team

이 저장소는 논문 evidence extraction, 분야/평가별 reviewer subagent,
author rebuttal, chair aggregation, public review-history 기반 update와 convergence
gate를 하나의 재현 가능한 pipeline으로 구성합니다. 최종 Markdown은
`reviewer_instruction.md`의 7개 field와 score range를 strict하게 따릅니다.

## Team

- Domain: CV, Core ML, NLP, RecSys, General fallback
- Criterion: Soundness, Presentation, Significance, Originality
- Advisory: Reproducibility, Ethics
- Coordination: Author, Review Synthesizer, Reviewer Chair

`prompts.py`의 일반 논문 19개 항목은 `ANSWER/SOURCES` evidence stage에서만
사용합니다. 최종 review stage와 type을 분리해 두 contract가 섞이지 않습니다.
Agent 간 공유 정보는 raw chat이 아니라 provenance가 붙은 append-only
`SharedContextStore` snapshot입니다.

## Installation

Runtime dependency는 Python 3.9+ standard library뿐입니다. Source checkout에서는
설치 없이 아래 Quick Start를 실행할 수 있습니다. Console entry point가 필요하면
build dependency인 `setuptools`와 `wheel`이 있는 환경에서 설치합니다.

```bash
python3 -m pip install .
ralphton-icml team
ralphton-icml validate-instruction
```

`run-seed`, `make report`, 그리고 learned `--state`의 manifest 검증은 repository의
versioned data/config/source가 필요하므로 checkout root에서 실행해야 합니다.
설치된 console script로 learned state를 사용할 때도 `--root /path/to/checkout`을
지정하십시오.

## Quick Start

다음 명령은 repository root 기준입니다.

```bash
make test
python3 -m ralphton_icml team
python3 -m ralphton_icml validate-instruction
python3 -m ralphton_icml run-seed
make report
```

`run-seed`는 20개 public OpenReview forum의 paper, human reviews, rebuttal,
final decision을 join한 `data/real/seed_cases.jsonl`을 forum 단위로 16/2/2
split합니다. Train에서만 rubric/calibration 및 reviewer/author retrieval memory를
update하고, frozen dev set의 quality/behavior/state delta가 epsilon 이하인 상태가
patience만큼 반복되면 중단합니다. Best state restore 후 test는 한 번 평가합니다.
결과는 `artifacts/real_seed_v1/`에 hash와 함께 저장됩니다.

## Model Backend

실제 agent 실행은 provider-independent JSON-over-stdin command를 사용합니다.
Command는 한 JSON request를 stdin으로 받고 response text 또는
`{"text": "..."}`를 stdout으로 반환해야 합니다.

```bash
python3 -m ralphton_icml review paper.json \
  --backend-command "python3 my_model_adapter.py" \
  --state artifacts/real_seed_v1/best_state.json \
  --root . \
  --output artifacts/review.json
```

`--state`가 없으면 initial reviewer는 현재 paper evidence만, author는
paper evidence+initial review만, post-rebuttal reviewer는 여기에 rebuttal을 더해
받습니다. `--state`를 사용하면 reviewer stage에는 train-only reviewer retrieval
memory와 calibration이, author stage에는 별도의 train-only author memory가 추가됩니다.
현재 paper의 human review와 final decision은 어느 live agent에도 전달되지 않고
offline train updater만 볼 수 있습니다.

State 파일은 자체 integrity digest가 있어야 하며, `--root`에서 계산한 현재
prompt/schema/team manifest와 정확히 일치해야 합니다. 불일치하면 review 실행 전에
중단되므로, source나 prompt를 변경한 뒤에는 `run-seed`로 state를 다시 생성해야 합니다.

### Hosted Codex Adapter

`scripts/codex_backend_adapter.py`는 `codex exec`를 ephemeral read-only worker로
호출합니다. `codex login status`가 ChatGPT login을 표시하면 별도 API key는 필요하지
않습니다. 공개 논문에만 사용하고 confidential submission은 전송하지 마십시오.

```bash
CODEX_BACKEND_LOG=artifacts/review/progress.jsonl \
CODEX_BACKEND_CACHE=artifacts/review/backend_cache \
CODEX_BACKEND_ATTEMPTS=2 CODEX_BACKEND_TIMEOUT=900 \
python3 -B -m ralphton_icml review paper.json \
  --backend-command "python3 scripts/codex_backend_adapter.py" \
  --state artifacts/real_seed_v1/best_state.json \
  --root . --output artifacts/review/review_run.json \
  --timeout 1800 --max-workers 2
```

Cache key는 전체 request JSON, selected model, adapter format version의 SHA-256이며
response만 local JSON으로 저장합니다. Final chair 응답은 정확한 field label/order,
bare integer score/range, non-empty Comment를 검증한 뒤 canonical Markdown heading과
blank-line layout으로 정규화됩니다.

## Scope And Policy

Seed corpus는 pipeline smoke용이며 benchmark 결과가 아닙니다. ReviewBench의
paper text가 initial submission임을 보장하지 않고 RecSys 완전 사례도 없습니다.
ICML 2026 confidential submission에는 이 system을 사용하면 안 됩니다. 실제
review에서는 reviewer에게 배정된 ICML LLM policy와 confidentiality rule이 이
repository보다 우선합니다. 상세 설계, evaluation protocol, limitation은
`report/report.pdf`와 `report/report.tex`에 정리되어 있습니다.
