# Ralphton ICML Reviewer Team

이 저장소의 review pipeline 전체가 Track 2입니다. 최종 Track 1 논문과 기존
evidence를 SHA-256으로 고정한 뒤, read-only reviewer가 extraction, consolidated
review, chair aggregation을 수행합니다. 논문을 수정하거나 새 실험을 만들지 않으며,
검증할 수 없는 주장은 `evidence-insufficient`로 표시합니다. 최종 Markdown은
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
결과는 `artifacts/real_seed_v2/`에 hash와 함께 저장됩니다.

## Track 2 Review

입력은 다음 구조로 준비합니다. `review`가 raw file hash, PDF text extractor version,
extracted-text hash와 출력 계약을 `review-agent.md`에 기록하고 실행 전후 다시
검증합니다.

```text
track2/
├── inputs/paper.pdf
├── evidence/results.json       # optional, 기존 evidence만 허용
└── outputs/
```

Hosted 실행은 first-class `CodexExecBackend`를 사용합니다. 기본 base path는
extraction 2회, reviewer 2회, chair 1회의 총 5회이며, conditional internal author
stress-test가 필요한 경우에만 2회가 추가됩니다.

```bash
python3 -B -m ralphton_icml review track2/inputs/paper.pdf \
  --backend codex --model gpt-5.6-sol \
  --track2-root track2 \
  --evidence track2/evidence/results.json \
  --state artifacts/real_seed_v2/best_state.json --root . \
  --cache-dir track2/backend_cache --progress track2/progress.jsonl \
  --output track2/outputs/review-run.json
```

필수 human-readable 산출물은 `track2/review-agent.md`와
`track2/outputs/review-result.md`입니다. JSON 결과에는 19개 evidence registry,
reviewer finding ID, chair-selected evidence ID, call/latency metadata가 함께 남습니다.

State는 integrity digest와 현재 prompt/schema/team manifest가 정확히 일치해야 합니다.
Source나 prompt를 변경한 뒤에는 `run-seed`로 state를 다시 생성해야 합니다. Live
request에는 현재 paper와 명시적 evidence, train-only generalized memory만 들어가며,
현재 paper의 human review, rebuttal, metareview, rating, decision은 전달되지 않습니다.

### Batch

Batch manifest는 공격 표면을 줄이기 위해 `papers` 외의 key를 허용하지 않습니다.
상대 경로는 manifest 기준으로 해석합니다.

```json
{"papers":["papers/paper-01.json","papers/paper-02.json"]}
```

```bash
python3 -B -m ralphton_icml review-batch papers.manifest.json \
  --backend codex --model gpt-5.6-sol \
  --state artifacts/real_seed_v2/best_state.json --root . \
  --output-dir artifacts/batch_review \
  --paper-workers 4 --codex-concurrency 4 \
  --author-loop conditional --max-refinements 2 \
  --deadline-seconds 1800 --soft-deadline-seconds 1440 --resume
```

모든 base review를 먼저 완료한 뒤 global priority로 최대 2편만 refinement합니다.
결과는 논문별 Track 2 root, atomic JSON/Markdown/completion marker,
`progress.jsonl`, `backend-progress.jsonl`, `summary.json`으로 저장됩니다.

### Hosted Authentication And Isolation

`codex login status`가 `Logged in using ChatGPT`이면 별도 API key가 필요하지 않습니다.
Worker는 `--ephemeral`, isolated temporary directory, read-only sandbox, strict output
schema를 사용합니다. Shell, browser, apps, plugins, computer-use, multi-agent 기능을
CLI level에서 끄고 child environment는 `PATH`, `HOME`, `CODEX_HOME` 등 최소 항목만
허용하므로 provider API key를 상속하지 않습니다. 공개 논문만 전송해야 하며,
confidential submission에는 사용하면 안 됩니다.

기존 JSON-over-stdin `SubprocessBackend`와 `scripts/codex_backend_adapter.py`는
`legacy-v1` regression/compatibility 용도로만 유지합니다.

## Scope And Policy

Seed corpus는 pipeline smoke용이며 benchmark 결과가 아닙니다. ReviewBench의
paper text가 initial submission임을 보장하지 않고 RecSys 완전 사례도 없습니다.
ICML 2026 confidential submission에는 이 system을 사용하면 안 됩니다. 실제
review에서는 reviewer에게 배정된 ICML LLM policy와 confidentiality rule이 이
repository보다 우선합니다. 상세 설계, evaluation protocol, limitation은
`report/report.pdf`와 `report/report.tex`에 정리되어 있습니다.
