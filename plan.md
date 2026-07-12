# Codex Semantic Reranking + 10편/30분 Fast Review Plan

## Summary

- `fast-v1` pipeline을 기본으로 추가합니다.
- 논문당 기본 호출을 `31 → 5회`로 줄입니다.
  - Extraction 2회 병렬
  - Consolidated reviewer 2회 병렬
  - Final Chair 1회
- 10편 기본 호출은 50회이며, conditional author refinement는 최대 2편에만 적용하여 최대 54회로 제한합니다.
- `codex exec`만 생성 backend로 사용하며 local LLM, `--oss`, `--local-provider`, session resume은 사용하지 않습니다.
- 사용자가 선택한 방식에 따라 dense embedding은 도입하지 않고, lexical top-8 memory 후보를 기존 reviewer/author call 안에서 Codex가 semantic reranking합니다. 따라서 별도 retrieval call은 없습니다.
- 기본 병렬도는 `paper_workers=4`, 전체 `codex_concurrency=4`, hard deadline은 1,800초입니다.

## Pipeline 및 Memory 변경

### Batched extraction

기존 19개 call을 전체 paper를 각각 한 번 받는 2개 call로 묶고 병렬 실행합니다.

- `extraction.paper_method` 10개:
  - Paper Summary, Introduction
  - Preliminaries, Framework, Training, Proofs
  - Inference and Application, Method Summary
  - Limitations and Future Works, Conclusion
- `extraction.experiments` 9개:
  - Supplemental Summary, Related Works
  - Datasets, Implementation Details, Evaluation Metrics
  - Quantitative/Qualitative Results, Ablation Study, Results Summary

응답은 `{"items":[{"task_id","answer","sources"}]}` JSON으로 받고, 기존 19개 `task_id`와 provenance는 유지합니다.

- Paper Summary 최대 1,200자, 나머지 answer 최대 700자
- Task당 source 최대 3개
- Python에서 missing/duplicate/unknown task, 길이 초과, 잘못된 source를 검증
- 두 응답을 기존 `ContextPacket`의 19개 evidence로 병합
- Paper text가 240k자를 넘으면 silent truncation하지 않고 preflight에서 명시적으로 실패

### Consolidated reviewers

기존 2개 domain + 6개 criterion reviewer를 다음 2개로 통합하고 병렬 실행합니다.

- `review.technical`
  - routed domain instruction 최대 2개
  - Soundness, Reproducibility, Ethics
- `review.contribution`
  - Presentation, Significance, Originality

각 reviewer는 structured JSON을 반환합니다.

- Strength 최대 3개, weakness 최대 4개, question 최대 2개
- 각 finding: `criterion`, `severity`, `text`, `evidence_ids`
- Finding text 최대 500자, evidence ID 최대 3개
- `memory_candidate_ids_used`와 `unresolved_contradictions` 포함

별도 Synthesizer call은 제거합니다.

### Codex semantic memory reranking

- `reviewer_memory`/`author_memory`를 forum별로 묶고 동일 cue의 중복을 제거합니다.
- Query는 `title + normalized paper text 앞 4,000자`입니다.
- 기존 lexical score로 forum-diverse top-8을 선택하되 minimum threshold 없이 항상 최대 8개를 구성합니다.
- 각 후보에는 다음만 포함합니다.
  - 안정적인 `candidate_id`
  - title 및 abstract cue 최대 400자
  - raw memory에서 생성한 generalized lesson
- Raw human review/rebuttal은 Codex prompt에 절대 포함하지 않습니다.
- 두 reviewer가 후보의 semantic applicability를 독립적으로 판단하고 사용한 ID를 응답에 기록합니다.
- Conditional Author Agent도 같은 방식으로 `author_memory` top-8을 call 내부에서 reranking합니다.
- Unknown candidate ID를 반환하면 response validation 실패로 처리합니다.
- Offline deterministic learning predictor는 기존 lexical retrieval을 유지하고, Codex reranking은 live review path에만 적용합니다.

### Final Chair와 conditional refinement

Final Chair는 두 reviewer JSON, Paper/Method/Results Summary, reviewer가 실제 인용한 evidence만 받습니다.

- Evidence는 최대 16개
- 초과 시 fatal/major weakness → question → strength 순으로 선택
- Chair는 Markdown이 아닌 7개 review field JSON을 반환
- 추가 orchestration field:
  - `needs_refinement`
  - `refinement_reasons`
- Core Python이 `ReviewOutput` 검증 후 canonical Markdown을 렌더링합니다.

Batch의 base review가 모두 완료된 뒤 다음 조건의 논문만 refinement 후보가 됩니다.

- `needs_refinement=true`
- `confidence <= 2`
- reviewer의 unresolved contradiction 존재

우선순위는 severity 내림차순 → confidence 오름차순 → 입력 순서이며 최대 2편만 선택합니다. 24분 이후에는 새 refinement를 시작하지 않습니다.

선택된 논문은 Author Agent와 post-rebuttal Chair를 순차 실행하여 2회가 추가됩니다. Refinement 실패 또는 soft deadline 초과 시 valid한 base review를 최종 결과로 보존하고 실패 이유만 metadata에 기록합니다.

## Codex Backend, 병렬 실행 및 CLI

### Backend

- `ModelRequest`에 optional `output_schema`를 추가합니다.
- first-class `CodexExecBackend`를 추가하고 기존 JSON-over-stdin adapter는 compatibility wrapper로 유지합니다.
- 설치된 `codex-cli 0.144.1`에서 확인된 `--output-schema`를 extraction/reviewer/chair/author 모두에 사용합니다.
- 모든 call은 별도 temporary directory에서 실행합니다.

```text
codex exec
--ephemeral
--sandbox read-only
--skip-git-repo-check
--ignore-user-config
--ignore-rules
--output-schema <stage-schema.json>
--json
--output-last-message <output>
--cd <isolated-temp-dir>
```

Stage timeout 기본값:

- Extraction: 300초
- Reviewer/Author: 180초
- Chair: 120초
- 최대 2 attempts
- Transport, timeout, schema failure에만 1회 재시도
- Retry 대기 중에는 global semaphore를 반납

Cache fingerprint에는 Codex CLI version, 명시적 model, pipeline config, request, output schema, state digest를 모두 포함합니다.

### Batch scheduler

- `paper_workers=4`
- `codex_concurrency=4`
- 논문 내부 extraction/reviewer concurrency는 각각 2
- 실제 `codex exec` child process는 항상 최대 4개
- Base-review 작업을 refinement보다 우선 스케줄링
- 한 논문 실패가 다른 논문을 취소하지 않음
- Hard deadline 도달 시 진행 중 child process를 종료하고 완료 결과는 보존
- 각 논문 결과를 완료 즉시 atomic write
- 실패가 하나라도 남으면 전체 CLI는 non-zero exit code 반환

### Public CLI

```bash
python3 -m ralphton_icml review-batch papers.manifest.json \
  --backend codex \
  --model <explicit-model> \
  --state artifacts/real_seed_v2/best_state.json \
  --output-dir artifacts/batch_review \
  --pipeline fast-v1 \
  --paper-workers 4 \
  --codex-concurrency 4 \
  --author-loop conditional \
  --max-refinements 2 \
  --deadline-seconds 1800 \
  --resume
```

Manifest는 기존 paper JSON 경로를 사용합니다.

```json
{
  "papers": [
    "inputs/paper-01.json",
    "inputs/paper-02.json"
  ]
}
```

- 상대 경로는 manifest 기준으로 해석
- `paper_id` 중복과 unsafe output path는 preflight에서 거부
- Batch 실행은 reproducibility를 위해 `--model`을 필수로 요구
- 기존 `review`에도 `--pipeline fast-v1|legacy-v1`을 추가하고 기본값은 `fast-v1`
- 기존 `--no-author-loop`는 `--author-loop never`의 deprecated alias로 유지

출력은 논문별 JSON/Markdown, `progress.jsonl`, `summary.json`으로 구성합니다. Summary에는 call 수, cache hit, stage p50/p95, request/output bytes, retry, wall time, papers/min, deadline 충족 여부를 기록합니다.

## Test 및 Acceptance Criteria

### Automated tests

- Extraction 두 call이 정확히 19개 task를 중복·누락 없이 생성
- `fast-v1` base path가 정확히 5개 request를 만들고 refinement 시 7개가 되는지 검증
- Sleeping fake backend로 extraction/reviewer가 실제 overlap하며 active call이 4를 넘지 않는지 검증
- Top-8 memory가 forum-diverse하고 deterministic한지 검증
- Reviewer/author memory 격리, same-forum 제외, raw human text 비노출 검증
- Codex가 반환한 unknown memory/evidence ID 거부
- `--output-schema` 전달, JSON validation, Python Markdown rendering 검증
- Stage timeout, retry semaphore 반환, hard/soft deadline 검증
- 한 논문 실패 시 다른 결과 보존, atomic output, resume fingerprint 검증
- Concurrent cache/log write와 stale cache invalidation 검증
- Blind paper payload에 human review/decision이 포함되지 않는지 검증

### 10-paper cold-cache benchmark

동일한 명시적 Codex model과 고정된 public paper 10편으로 수행합니다.

- 10/10 valid `ReviewOutput`
- 전체 wall time ≤ 1,800초
- 기본 call 50회
- Conditional refinement 포함 정상 최대 54회
- Retry 포함 acceptance 상한 60회
- Unrecovered schema failure 0건
- 모든 major/fatal finding의 evidence ID가 실제 `ContextPacket`에 존재
- Representative paper의 serialized request bytes ≤ paper text bytes의 3배
- 전체 request bytes가 legacy pipeline 대비 최소 85% 감소
- Historical labeled dev에서 기존 MAE/Brier non-regression tolerance `0.02` 유지
- Benchmark manifest에 CLI version, model, state/config digest, concurrency, call 수와 stage latency 기록

## Assumptions 및 Compatibility

- 이번 버전은 사용자의 선택대로 dense vector embedding을 구현하지 않고 Codex in-call semantic reranking을 구현합니다. 향후 true embedding이 필요하면 별도 `EmbeddingBackend` 단계로 추가합니다.
- Local generative model은 사용하지 않으며 모든 생성과 semantic 판단은 hosted `codex exec`가 담당합니다.
- `codex exec resume`은 논문 간 context contamination을 피하기 위해 사용하지 않습니다.
- Existing `legacy-v1` pipeline과 `SubprocessBackend`는 regression 비교와 compatibility를 위해 유지합니다.
- Orchestrator, prompt, team manifest가 바뀌므로 기존 `best_state.json`은 incompatible해집니다. 구현 후 `run-seed`를 다시 실행해 `real_seed_v2` state를 생성합니다.
- Dense vector를 저장하지 않으므로 `LearningState` JSON schema는 v1을 유지하고 migration은 수행하지 않습니다.
- 30분 SLA는 로그인 완료, input/state 준비 완료, cold response cache, `codex_concurrency=4`, 명시된 model 조건을 기준으로 평가합니다.
