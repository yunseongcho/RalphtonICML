"""
Unified Prompt Templates for Paper + Supplemental Material Analysis (Korean)
All prompts include supplemental material references with skip guards.
"""

from __future__ import annotations

# -------------------------
# 0) Common Header & Factory Function
# -------------------------

COMMON_HEADER = """[작성 규칙 리마인더]
1. **포맷 엄수:** 반드시 **ANSWER**와 **SOURCES** 두 섹션으로만 구성하십시오.
2. **근거 원칙:** 답변의 모든 근거는 **첨부 PDF**에서 찾아야 합니다.
3. **인용 표기:** **ANSWER** 섹션에서 외부 문헌은 반드시 **논문 제목(Title)** 로만 지칭하십시오. (예: [[Attention Is All You Need]])
4. **수식 설명:** LaTeX 수식 뒤에는 반드시 그 **직관적인 의미(Why/What)** 를 풀어서 설명하십시오.
5. **위치 표기:** 핵심 내용 끝에는 논문 내 위치를 표기하십시오. (예: (p. 5, Fig. 2))"""

COMMON_HEADER_SHORT = "[리마인더] ANSWER+SOURCES 포맷 엄수. 인용은 논문 Title로만."

BACKGROUND_KNOWLEDGE_ALLOW = """[예외적 허용: 배경지식]
- 이 질문에 한해, 이해를 돕기 위해 **일반적인 AI 배경지식(표준적인 정의, 널리 알려진 개념)** 사용을 허용합니다.
- 단, 논문의 독창적인 주장과 섞이지 않도록 "일반적으로", "배경지식에 따르면" 등의 표현을 사용하여 명확히 구분하십시오."""


NO_INFO_MSG = "제공된 파일에 해당 정보가 없습니다."

SUPPLE_SKIP = "해당 내용이 없다면 이 부분은 생략하십시오."

ANTI_RECITATION = "주의: 논문 본문의 문장을 그대로 옮기지 말고, 반드시 자신의 언어(한국어)로 재구성하여 설명하십시오."

# 각 연구 요약 축약 문장 수: 일반 논문 Related Works = 1-2문장, 서베이 Methods Survey = 2-3문장
ANTI_RECITATION_RELATED = """주의: 논문 본문의 문장을 그대로 옮기지 말고, 반드시 자신의 언어(한국어)로 재구성하여 설명하십시오. 각 연구의 요약은 저자의 관점에서 1-2문장으로 축약하십시오."""

ANTI_RECITATION_SURVEY = """주의: 논문 본문의 문장을 그대로 옮기지 말고, 반드시 자신의 언어(한국어)로 재구성하여 설명하십시오. 각 연구의 요약은 저자의 관점에서 2-3문장으로 축약하십시오."""


def make_prompt(
    scope: str,
    questions: str,
    allow_background: bool = False,
    short_header: bool = False,
    anti_recitation: str | None = None,
) -> str:
    """
    Constructs a consistent prompt with a common header, scope context, and specific questions.
    Use short_header=True for all turns after the first to reduce token overhead.
    Use anti_recitation to prepend an anti-RECITATION guard before the questions.
    """
    header = COMMON_HEADER_SHORT if short_header else COMMON_HEADER
    parts = [header]

    if allow_background:
        parts.append(BACKGROUND_KNOWLEDGE_ALLOW)

    if anti_recitation:
        parts.append(anti_recitation)

    parts.append(f"[검토 범위]\n- {scope}")

    parts.append("[질문]\n" + questions.replace("\n>", "\n").strip())

    return "\n\n".join(parts)


# -------------------------
# 1) Unified Questions Dict
#    항상 보충자료 참조 포함. 보충자료가 없으면 모델이 해당 부분을 생략.
# -------------------------

questions = {
    "Main Paper": {
        "Paper Summary": [
            """논문의 전체 내용을 논문의 섹션 제목(Section Titles)을 따라 구성하되, 신입 대학원생이 이해하기 쉽게 체계적으로 요약해 주십시오. 이 요약은 전체 논문의 큰 그림을 그려주는 것이 목적입니다. 각 섹션의 세부 내용은 이후 질문에서 다룰 예정이므로, 여기서는 핵심만 간결하게 다루십시오.
>
>1. **전체 흐름:** 각 섹션이 유기적으로 어떻게 연결되는지 이야기하듯 설명하십시오.
>2. **섹션별 맞춤 요약:** 각 섹션의 **성격에 맞춰** 다음 내용 중 **해당하는 요소**를 중심으로 요약하십시오:
>   - **문제 제기/동기 (Introduction 등):** Why(문제)와 What(해결책 직관) 중심
>   - **방법론 (Methodology):** What(핵심 아이디어)과 How(동작 원리/수식) 중심
>   - **실험 (Experiments):** How(설정)와 Result(결과/해석) 중심
>   - **기타 섹션:** 해당 섹션의 핵심 목적과 내용 중심
>3. **핵심 자료:** 설명을 뒷받침하는 **핵심 수식(Key Equations)**과 **핵심 참고문헌(Key References)**을 반드시 포함하십시오.
>4. **Core References:** 'SOURCES' 섹션에 이 논문에서 가장 중요하게 인용된 **핵심 문헌**의 서지 정보를 중요도 순으로 최대 10개까지 나열하십시오."""
        ],
        "Supplemental Summary": [
            """보충자료(Supplemental Material)가 포함되어 있다면, 본문에서 다루지 못한 추가 정보들을 신입 대학원생이 이해하기 쉽게 요약해 주십시오. 보충자료가 없거나 별도 섹션으로 구분되지 않는다면 이 부분은 생략하십시오.
>
>1. **자료 구성의 목적:** 저자들이 보충자료를 통해 본문의 어떤 부분을 보완하고자 했는지(예: 이론 증명, 추가 실험, 재현성 등) 설명하십시오.
>2. **핵심 내용 요약:** 각 섹션의 핵심 아이디어를 요약하되, 본문을 읽은 독자가 **'이 내용은 본문의 이해를 위해 중요하구나'**라고 느낄 수 있도록 직관적인 의미를 설명하십시오."""
        ],
        "Introduction": [
            """1. **핵심 과제 (Core Task):** 이 논문이 풀고자 하는 과제를 정의하고, **입력(Input), 출력(Output), 목표(Goal)**, 그리고 이 문제의 중요성을 명확히 하십시오.
>2. **기존의 한계:** 이전에 사용되던 방법(Previous Methods)들이 가진 구체적인 문제점은 무엇입니까? 해당 논문들의 **Title**을 명시하며 한계점을 지적하십시오.
>3. **해결책의 직관:** 저자들은 이 문제를 해결하기 위해 어떤 독창적인 아이디어(Key Idea)를 제안했습니까? 수식보다는 **직관적인 해결책** 위주로 먼저 설명하십시오.
>4. **기여점 (Contributions):** 이 논문이 학계에 기여한 바(이론적 성취, 성능 향상, 새로운 구조 제안 등)를 구체적으로 나열하십시오.""",
        ],
        "Related Works": [
            """1. **연구 흐름 및 분류:** 저자들의 분류 기준에 따라 **PDF에 명시된 모든** 관련 연구들을 누락없이 나열하고 범주화하십시오. 각 연구의 핵심 아이디어와 저자가 지적한 한계점을 요약하십시오.
>2. **차별점 (Differentiation):** 이 논문의 접근 방식이 앞선 연구들과 근본적으로 어떻게 다르며, 어떤 구체적인 부분을 개선했는지 논리적으로 비교하십시오.""",
        ],
    },
    "Methodology": {
        "Preliminaries": [
            """1. **핵심 용어 및 기호:** 논문을 이해하는 데 필수적인 용어와 기호(Notation)를 정의하고 명확히 설명하십시오.
>2. **수학적/이론적 배경:** 이 방법론의 기초가 되는 이론이나 수식 전개가 있다면 단계별로 친절하게 설명하십시오. (이 부분은 일반적인 AI 지식을 활용하여 이해를 도우십시오.)
>3. **기반 연구 (Prior Work):** 이 논문이 직접적으로 기초하고 있는 필수 선행 연구(Baseline 등)가 있다면, **논문 Title**을 명시하고 핵심 내용을 요약하십시오.
>4. **연결 고리:** 위 개념들이 이후 모델 설명과 어떻게 연결되는지(Roadmap) 보여주십시오.""",
        ],
        "Framework": [
            """1. **전체 구조 (Overview):** 전체 프레임워크가 어떤 **모듈(Module)**들로 구성되어 있는지, 각 모듈의 **입출력**은 무엇인지, PDF에서 확인되는 범위 내에서 누락없이 설명하십시오.
>2. **모듈별 상세 분석:**
>   - 각 모듈(Module)의 내부 동작 원리와 역할을 설명하십시오.
>   - 각 모듈(Module)의 정확한 **베이스라인(Baseline)이나 백본(Backbone)**이 있다면 Title로 명시하고, 이를 선택한 이유를 설명하십시오.
>   - 저자들이 제안한 **구조적 개선점**이 무엇인지 강조하십시오.
>3. **데이터 흐름 (Data Flow):** 입력 데이터가 모델을 통과하여 최종 출력이 되기까지의 과정을 수식과 함께 단계별로(Step-by-step) 추적하십시오.
>4. **시각 자료 가이드:** 프레임워크 그림(Figure)이 있다면 번호를 명시하고, 그림의 어떤 부분을 주목해서 봐야 하는지 설명하십시오.
>5. **[보충자료 참조] 세부 모듈 분석:** 보충자료에 본문보다 더 상세한 블록 다이어그램이나 레이어 구성이 있다면, 각 컴포넌트의 구체적인 역할과 입출력 차원 등을 설명하십시오. 해당 내용이 없다면 이 부분은 생략하십시오.
>6. **[보충자료 참조] 설계 근거:** 보충자료에서 세부적인 구조나 수치를 선택한 이유를 설명하고 있다면 정리하십시오. 해당 내용이 없다면 이 부분은 생략하십시오.
>7. **[보충자료 참조] 본문과의 연결:** 보충자료의 세부 내용이 본문의 Methodology와 어떻게 매칭되는지 설명하십시오. 해당 내용이 없다면 이 부분은 생략하십시오."""
        ],
        "Training": [
            """(참고: 구체적인 하이퍼파라미터 수치는 Implementation Details에서 다룹니다. 여기서는 학습 전략의 설계 의도에 집중하십시오.)
>
>1. **학습 프로세스:** 학습이 어떤 단계(Phase)로 진행됩니까? (예: Pre-training, Fine-tuning 등). 각 단계가 왜 필요한지 설명하십시오.
>2. **손실 함수 (Loss Function):**
>   - 각 단계에서 어떤 손실 함수가 사용되며, **어떤 출력(Output)에 적용**됩니까?
>   - 수식(LaTeX)을 적고, **'이 수식이 어떤 모듈을 어떤 방향으로 최적화하는지'** 그 목적과 대상 모듈을 상세히 설명하십시오.
>3. **최적화 설정:** 사용된 Optimizer, Learning Rate Schedule 등 구체적인 하이퍼파라미터 설정과 그 이유를 정리하십시오.
>4. **특수 기법:** Curriculum Learning, Multi-task Learning 등 특별한 기법이 적용되었다면 그 목적과 방법을 설명하십시오.
>5. **[보충자료 참조] 손실 함수 유도:** 보충자료에 손실 함수의 수학적 유도 과정이 있다면 단계별로 설명하고, 직관적으로 어떤 모듈을 어떤 방향으로 최적화하는지 설명하십시오. 해당 내용이 없다면 이 부분은 생략하십시오.
>6. **[보충자료 참조] 추가 학습 전략:** 보충자료에 본문에 없는 상세한 학습 스케줄, 가중치 초기화, 미세 조정 전략 등이 있다면 설명하십시오. 해당 내용이 없다면 이 부분은 생략하십시오.""",
        ],
        "Proofs": [
            """보충자료에 수학적 정리나 증명이 포함되어 있다면 다음에 답변하십시오. 해당 내용이 없다면 이 부분은 생략하십시오.
>
>1. **증명의 목적:** 이 수식 증명이나 유도 과정이 본문의 어떤 핵심 주장을 뒷받침하기 위해 필요한지 설명하십시오.
>2. **단계별 해설:** 증명 과정을 단계별로 나누어 설명하되, **이 수식이 직관적으로 무엇을 최적화/제약/모델링하는지** 풀어서 설명하십시오.
>3. **결론의 의미:** 증명의 결과가 최종적으로 모델이나 이론에 시사하는 바를 요약하십시오."""
        ],
        "Inference and Application": [
            """1. **추론 과정 (Inference):** 학습 시와 달리 추론 단계에서 달라지는 점이 있다면 설명하고, 입력부터 출력까지의 데이터 흐름을 단계별로 기술하십시오.
>2. **사용 사례 (Use Case):** 논문에서 제안하는 실제 적용 시나리오가 있다면 모든 예시를 상세히 설명하십시오. 논문에 구체적인 사용 사례가 명시되어 있지 않다면, 이 부분은 제외하십시오.
>3. **실용적 이점:** 저자가 강조하는 실용적 장점(예: 실시간 처리, 메모리 효율성, 확장성)을 요약하십시오.
>4. **[보충자료 참조] 상세 추론 파이프라인:** 보충자료에 실제 배포나 테스트 단계에서의 데이터 흐름이 더 자세히 묘사되어 있다면 단계별로 설명하십시오. 해당 내용이 없다면 이 부분은 생략하십시오.
>5. **[보충자료 참조] 추가 사용 사례:** 보충자료에 본문에 없는 추가적인 데모나 적용 분야가 있다면 설명하십시오. 해당 내용이 없다면 이 부분은 생략하십시오.""",
        ],
        "Method Summary": [
            """지금까지 분석한 Methodology를 PDF 원문을 기반으로 종합하여 요약해 주십시오.
>
>- **프레임워크 구조:** 구성 요소와 역할
>- **데이터 흐름:** 입력에서 출력까지의 파이프라인
>- **핵심 메커니즘:** 학습 전략과 손실 함수의 핵심
>- **활용:** 추론 절차 및 잠재적 응용 분야
>
>독자가 '이 모델은 이렇게 동작하는구나'라고 머릿속에 그릴 수 있도록 체계적으로 정리하십시오."""
        ],
    },
    "Experiments": {
        "Datasets": [
            """1. **데이터셋 스펙:** 실험에 사용된 데이터셋의 이름(Title), 레이블 유형, 크기, 주요 특징을 **PDF에서 확인되는 범위 내에서 가능한 한** 설명하십시오.
>2. **데이터 분할:** Train/Validation/Test 셋이 어떻게 구성되었는지 설명하십시오.
>3. **수집 및 전처리:** (해당하는 경우) 저자가 직접 데이터를 수집했거나 특별한 전처리를 수행했다면 그 과정을 설명하십시오.
>4. **역할:** 각 데이터셋이 실험 설정 내에서 어떤 용도(학습/평가/적용)로 활용되었는지 명확히 하십시오.
>5. **[보충자료 참조] 추가 데이터 통계:** 보충자료에 데이터의 클래스 분포, 이미지 크기, 라이선스, 윤리적 고려사항 등 추가적인 정보가 있다면 정리하십시오. 해당 내용이 없다면 이 부분은 생략하십시오.
>6. **[보충자료 참조] 수집 절차:** 보충자료에 데이터 수집 절차나 동의(Consent) 등에 대한 설명이 있다면 요약하십시오. 해당 내용이 없다면 이 부분은 생략하십시오.""",
        ],
        "Implementation Details": [
            """(참고: 손실 함수의 설계 의도와 최적화 전략의 이론적 배경은 이미 Training 섹션에서 다루었습니다. 여기서는 재현에 필요한 구체적 수치와 환경에 집중하십시오.)
>
>1. **구현 세부 사항:** 실험의 재현성을 위해 학습률, 배치 크기, 에포크 수 등 PDF에 명시된 모든 하이퍼파라미터 설정을 깊이 있게 설명하십시오.
>2. **컴퓨팅 환경:** 학습에 사용된 GPU 종류/개수 및 대략적인 학습 시간을 명시하십시오.
>3. **재현성 가이드:** 오픈 소스 코드 제공 여부나 저자가 제시한 재현 팁이 있다면 요약하십시오.
>4. **[보충자료 참조] 추가 하이퍼파라미터:** 보충자료에 재현에 핵심적인 매직 넘버(Learning rate, decay, thresholds 등)가 추가로 있다면 정리하십시오. (저자가 강조하거나 결과에 민감하다고 언급한 값 우선) 해당 내용이 없다면 이 부분은 생략하십시오.
>5. **[보충자료 참조] 재현 팁:** 보충자료에 저자들이 언급한 '학습 성공을 위한 팁'이나 주의사항이 있다면 강조하십시오. 해당 내용이 없다면 이 부분은 생략하십시오."""
        ],
        "Evaluation Metrics": [
            """PDF에 명시된 모든 평가 지표를 다음 형식으로 각각 설명하십시오:
>1. **지표 이름 및 정의:** 해당 지표가 무엇을 측정하는지 설명하십시오.
>2. **수식:** LaTeX로 수식을 적고, 직관적으로 해석하십시오.
>3. **해석:** 이 지표가 높다(또는 낮다)는 것이 모델의 어떤 능력을 의미합니까?
>4. **논문 내 역할:** 이 지표가 이 논문의 실험에서 어떤 측면을 평가하기 위해 사용됩니까?"""
        ],
        "Quantitative Results": [
            """표의 모든 숫자를 나열하지 말고, 핵심적인 비교 결과와 경향(Trends)을 중심으로 설명하십시오.
>
>1. **표 분석 (Reference Tables):** 정량적 결과를 보여주는 표(Table)를 지목하고, 제안 모델이 비교군 대비 **수치적으로 얼마나 우수한지**, 그리고 그 표가 무엇을 비교하고 있는지 설명하십시오.
>2. **저자의 해석:** 저자들은 이 수치적 결과를 통해 모델의 강점과 약점을 어떻게 해석하고 있습니까?
>3. **[보충자료 참조] 추가 실험 결과:** 보충자료에 본문의 결과와 비교했을 때 추가된 표들이 보여주는 새로운 사실이나 더 세밀한 비교 결과가 있다면 설명하십시오. 추가된 도표의 위치를 명시하십시오. 해당 내용이 없다면 이 부분은 생략하십시오.
>4. **[보충자료 참조] 다양한 지표:** 보충자료에서 본문 외에 다른 평가 지표를 사용했다면, 그 의미와 결과를 해석하십시오. 해당 내용이 없다면 이 부분은 생략하십시오."""
        ],
        "Qualitative Results": [
            """1. **시각적 결과 분석:** PDF에 명시된 모든 결과 이미지나 그래프(Figure)를 나열하고, 각 그림이 무엇을 나타내는지 상세히 설명하십시오. (Ablation 제외)
>2. **비교 해석:** 제안 모델의 결과물이 비교 모델보다 **시각적으로/질적으로 어떤 차별점**을 보이는지 설명하십시오.
>3. **실패 사례 (Failure Case):** 논문에 언급된 실패 사례나 엣지 케이스가 있다면, 그 예시를 설명하고 잠재적인 원인을 논의하십시오.
>4. **[보충자료 참조] 추가 시각화 분석:** 보충자료에 본문의 예시 외에 더 다양한 샘플이 있다면, 관찰 가능한 패턴(성공/실패/아티팩트/일관성 등)을 요약하고 저자들의 해석을 설명하십시오. 해당 내용이 없다면 이 부분은 생략하십시오.
>5. **[보충자료 참조] 심층 실패 분석:** 보충자료에 본문보다 더 자세한 실패 사례 분석이 있다면, 모델이 어떤 케이스에서 어려움을 겪는지 구체적으로 설명하십시오. 해당 내용이 없다면 이 부분은 생략하십시오."""
        ],
        "Ablation Study": [
            """1. **소거 연구 목록:** PDF에 명시된 모든 소거 연구(모듈 제거, 변경 등)를 나열하고, 각 실험의 목적을 설명하십시오.
>2. **참조 표/그림:** 각 실험 결과를 보여주는 표나 그림 번호를 명시하십시오.
>3. **결과 해석:** 저자들에 따르면, **성능에 가장 결정적인 영향을 미치는 컴포넌트**는 무엇이며, 각 요소의 유무가 결과에 어떤 변화를 줍니까?
>4. **[보충자료 참조] 추가 소거 연구:** 보충자료에 본문 외에 수행된 추가적인 민감도 분석이나 모듈 테스트가 있다면, 그 목적과 결과를 설명하십시오. 관련 표/그림의 위치를 명시하십시오. 해당 내용이 없다면 이 부분은 생략하십시오.
>5. **[보충자료 참조] 타당성 보강:** 보충자료의 추가 실험들이 모델 설계의 타당성을 어떻게 보강하고 있는지 저자의 해석을 설명하십시오. 해당 내용이 없다면 이 부분은 생략하십시오."""
        ],
        "Results Summary": [
            """실험 결과를 PDF 원문을 기반으로 종합하여 평가해 주십시오.
>
>1. **결과 요약:** 정량적/정성적 결과와 Ablation Study를 종합했을 때, 이 모델의 확실한 강점과 약점은 무엇입니까?
>2. **결론:** 실험 결과가 논문의 핵심 주장과 기여점을 충분히 뒷받침하고 있습니까?"""
        ],
    },
    "Conclusion": {
        "Limitations and Future Works": [
            """1. **한계점 (Limitations):** 이 모델이 해결하지 못한 문제나 일반화(Generalization) 이슈, 또는 저자가 명시한 제약 사항은 무엇입니까?
>2. **향후 연구 (Future Works):** 저자들이 제안하는 연구 방향이나 잠재적 확장 가능성은 무엇입니까?"""
        ],
        "Conclusion": [
            """1. **주요 주장:** 저자들이 가장 강조하는 핵심 발견이나 주장은 무엇입니까?
>2. **뒷받침 근거:** 저자들은 자신의 주장을 정당화하기 위해 어떤 방법론적 강점이나 실험 결과를 인용하고 있습니까?
>3. **총평:** 이 연구가 해당 AI 분야에서 어떤 기여를 했으며, 어떤 의미를 갖는지 마무리 설명하십시오. 총평은 저자의 주장과 실험적 근거에 기반하여 이 연구의 객관적인 위치와 기여를 설명하십시오. 주관적 평가는 배제하십시오."""
        ],
    },
}

# -------------------------
# 2) Unified Prompts Dict
#    questions와 동일한 구조. make_prompt()로 각 질문을 wrapping.
#    Paper Summary(Turn 1)만 full COMMON_HEADER, 이후는 COMMON_HEADER_SHORT 사용.
# -------------------------

prompts = {
    "Main Paper": {
        "Paper Summary": [
            (
                make_prompt(
                    scope="첨부된 논문 전체 (보충자료 포함)",
                    questions=questions["Main Paper"]["Paper Summary"][0],
                    short_header=False,
                ),
                "B",
            )
        ],
        "Supplemental Summary": [
            (
                make_prompt(
                    scope="보충자료(Supplemental Material) 전체",
                    questions=questions["Main Paper"]["Supplemental Summary"][0],
                    short_header=True,
                ),
                "B",
            )
        ],
        "Introduction": [
            (
                make_prompt(
                    scope="Introduction (및 관련 References)",
                    questions=questions["Main Paper"]["Introduction"][0],
                    short_header=True,
                ),
                "B",
            )
        ],
        "Related Works": [
            (
                make_prompt(
                    scope="Related Work(s) (및 관련 References)",
                    questions=questions["Main Paper"]["Related Works"][0],
                    short_header=True,
                    anti_recitation=ANTI_RECITATION_RELATED,
                ),
                "C",
            )
        ],
    },
    "Methodology": {
        "Preliminaries": [
            (
                make_prompt(
                    scope="Methodology 중 배경/정의/표기(Preliminaries) 및 관련 References",
                    allow_background=True,
                    questions=questions["Methodology"]["Preliminaries"][0],
                    short_header=True,
                ),
                "B",
            )
        ],
        "Framework": [
            (
                make_prompt(
                    scope="Methodology 중 프레임워크/아키텍처 설명 (보충자료 포함)",
                    questions=questions["Methodology"]["Framework"][0],
                    short_header=True,
                ),
                "A",
            )
        ],
        "Training": [
            (
                make_prompt(
                    scope="Methodology 중 학습(Training) 과정 (보충자료 포함)",
                    questions=questions["Methodology"]["Training"][0],
                    short_header=True,
                ),
                "A",
            )
        ],
        "Proofs": [
            (
                make_prompt(
                    scope="보충자료 내 증명(Proofs)/유도(Derivations)",
                    questions=questions["Methodology"]["Proofs"][0],
                    short_header=True,
                ),
                "A",
            )
        ],
        "Inference and Application": [
            (
                make_prompt(
                    scope="Methodology 및 Inference/Application 관련 섹션 (보충자료 포함)",
                    questions=questions["Methodology"]["Inference and Application"][0],
                    short_header=True,
                ),
                "B",
            )
        ],
        "Method Summary": [
            (
                make_prompt(
                    scope="Methodology 전체 (보충자료 포함)",
                    questions=questions["Methodology"]["Method Summary"][0],
                    short_header=True,
                ),
                "B",
            )
        ],
    },
    "Experiments": {
        "Datasets": [
            (
                make_prompt(
                    scope="Experiments/Results 중 데이터셋 설명 (보충자료 포함)",
                    questions=questions["Experiments"]["Datasets"][0],
                    short_header=True,
                ),
                "D",
            )
        ],
        "Implementation Details": [
            (
                make_prompt(
                    scope="Experiments/Results 중 구현/설정 (보충자료 포함)",
                    questions=questions["Experiments"]["Implementation Details"][0],
                    short_header=True,
                ),
                "D",
            )
        ],
        "Evaluation Metrics": [
            (
                make_prompt(
                    scope="Experiments/Results 중 평가 지표",
                    questions=questions["Experiments"]["Evaluation Metrics"][0],
                    allow_background=True,
                    short_header=True,
                ),
                "B",
            )
        ],
        "Quantitative Results": [
            (
                make_prompt(
                    scope="Experiments/Results 중 정량적 결과 (보충자료 포함)",
                    questions=questions["Experiments"]["Quantitative Results"][0],
                    short_header=True,
                    anti_recitation=ANTI_RECITATION,
                ),
                "C",
            )
        ],
        "Qualitative Results": [
            (
                make_prompt(
                    scope="Experiments/Results 중 정성적 결과 (보충자료 포함)",
                    questions=questions["Experiments"]["Qualitative Results"][0],
                    short_header=True,
                    anti_recitation=ANTI_RECITATION,
                ),
                "D",
            )
        ],
        "Ablation Study": [
            (
                make_prompt(
                    scope="Experiments/Results 중 Ablation Study (보충자료 포함)",
                    questions=questions["Experiments"]["Ablation Study"][0],
                    short_header=True,
                ),
                "B",
            )
        ],
        "Results Summary": [
            (
                make_prompt(
                    scope="Experiments/Results 전체 (보충자료 포함)",
                    questions=questions["Experiments"]["Results Summary"][0],
                    short_header=True,
                ),
                "B",
            )
        ],
    },
    "Conclusion": {
        "Limitations and Future Works": [
            (
                make_prompt(
                    scope="Conclusion/Limitations/Future Works",
                    questions=questions["Conclusion"]["Limitations and Future Works"][0],
                    short_header=True,
                ),
                "D",
            )
        ],
        "Conclusion": [
            (
                make_prompt(
                    scope="Conclusion",
                    questions=questions["Conclusion"]["Conclusion"][0],
                    short_header=True,
                ),
                "B",
            )
        ],
    },
}
