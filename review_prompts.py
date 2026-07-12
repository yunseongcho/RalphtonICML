"""
Prompt Templates for Review/Survey Paper Analysis (Korean)
리뷰/서베이 논문 전용 프롬프트. 분류 체계(Taxonomy), 방법론 조사, 비교 분석 중심.
RECITATION 위험이 일반 논문 대비 높으므로, 다수 섹션에 anti_recitation guard 적용.

용어 차이:
  일반 논문(prompts.py): [보충자료 참조] = 별도 supplemental PDF 파일 참조
  리뷰 논문(review_prompts.py): [부록 참조] = PDF 내 Appendix 섹션 참조
Supplemental Summary 미포함 이유: 서베이 논문은 별도 supplemental 파일이 드물며,
  부록 내용은 각 섹션의 [부록 참조] 하위 질문으로 유기적 통합.
"""

from __future__ import annotations

from prompts import ANTI_RECITATION, ANTI_RECITATION_SURVEY, make_prompt

# -------------------------
# 0) Review-Specific Constants
# -------------------------

REVIEW_NO_SECTION_MSG = "PDF에 해당 섹션이 별도로 구분되어 있지 않다면 이 부분은 생략하십시오."

# -------------------------
# 1) Review Questions Dict
#    리뷰 논문 특성상 존재하지 않을 수 있는 섹션에는 skip guard 포함.
#    섹션 간 범위 중복을 방지하는 (참고: ...) 노트 포함.
# -------------------------

review_questions = {
    "Survey Overview": {
        "Paper Summary": [
            """이 서베이/리뷰 논문의 전체 내용을 논문의 섹션 제목(Section Titles)을 따라 구성하되, 신입 대학원생이 이해하기 쉽게 체계적으로 요약해 주십시오. 이 요약은 서베이 전체의 큰 그림을 그려주는 것이 목적입니다. 각 섹션의 세부 내용은 이후 질문에서 다룰 예정이므로, 여기서는 핵심만 간결하게 다루십시오.
>
>1. **서베이의 범위:** 이 논문이 다루는 연구 분야와 그 경계(Scope)를 명확히 정의하십시오. 어떤 주제를 포함하고 어떤 주제를 제외하는지 설명하십시오.
>2. **전체 흐름:** 각 섹션이 유기적으로 어떻게 연결되는지 이야기하듯 설명하십시오. (예: 배경 → 분류 체계 → 방법론 조사 → 비교 → 응용 → 미래 방향)
>3. **섹션별 맞춤 요약:** 각 섹션의 **성격에 맞춰** 핵심 내용을 요약하십시오:
>   - **서론/동기:** Why(왜 이 서베이가 필요한지)와 What(다루는 범위) 중심
>   - **분류 체계:** 저자들이 제안하는 분류 기준과 카테고리 구조 중심
>   - **방법론 조사:** 각 카테고리의 대표적 접근법과 발전 흐름 중심
>   - **비교 분석:** 벤치마크 결과와 방법론 간 장단점 비교 중심
>4. **핵심 통계:** 이 서베이가 조사한 논문 수, 다루는 시간 범위, 주요 벤치마크 등 규모를 나타내는 정보를 **PDF에서 확인되는 범위 내에서** 포함하십시오.
>5. **Core References:** 'SOURCES' 섹션에 이 서베이에서 가장 중요하게 다루는 **핵심 문헌**의 서지 정보를 중요도 순으로 최대 10개까지 나열하십시오."""
        ],
        "Introduction": [
            """(참고: 분류 체계의 구체적인 구조는 Taxonomy Overview에서 다룹니다. 여기서는 서베이의 동기와 범위에 집중하십시오.)
>
>1. **서베이의 동기 (Motivation):** 왜 이 시점에서 이 분야의 서베이가 필요합니까? 해당 분야의 급격한 발전, 기존 서베이의 부재 또는 한계, 실용적 수요 등 저자들이 밝힌 동기를 설명하십시오.
>2. **연구 질문 (Research Questions):** 이 서베이가 답하고자 하는 핵심 연구 질문이 명시되어 있다면 나열하십시오. 명시적인 연구 질문이 없다면 이 부분은 생략하십시오.
>3. **범위 및 선별 기준 (Scope & Selection Criteria):** 조사 대상 논문의 선별 기준(시간 범위, 학회/저널, 키워드, 포함/제외 조건 등)이 있다면 설명하십시오. 구체적인 선별 방법론(예: PRISMA)이 명시되어 있다면 함께 기술하십시오. 해당 내용이 없다면 이 부분은 생략하십시오.
>4. **기여점 (Contributions):** 이 서베이가 기존 서베이 대비 어떤 독창적인 기여(새로운 분류 체계, 포괄적 벤치마크 비교, 새로운 관점 등)를 하는지 구체적으로 나열하십시오.
>5. **기존 서베이와의 차별점:** 같은 분야의 이전 서베이 논문들이 언급되어 있다면, **Title** 을 명시하고 본 서베이의 차별화 포인트를 간략히 언급하십시오. (상세 비교 및 Gap Analysis는 Related Surveys에서 다룹니다.) 해당 내용이 없다면 이 부분은 생략하십시오."""
        ],
        "Background": [
            f"""(참고: 개별 방법론의 상세한 설명은 이후 Methods Survey에서 다룹니다. 여기서는 서베이 전반을 이해하기 위한 기초 개념에 집중하십시오.)
>
>1. **핵심 용어 및 정의:** 이 서베이를 이해하는 데 필수적인 용어, 기호(Notation), 문제 정의(Problem Definition)를 **PDF에 명시된 범위 내에서** 명확히 설명하십시오.
>2. **기초 이론:** 서베이 대상 분야의 기초가 되는 이론이나 수학적 배경이 있다면 단계별로 친절하게 설명하십시오. (이 부분은 일반적인 AI 지식을 활용하여 이해를 도우십시오.)
>3. **핵심 기반 기술:** 서베이 대상 방법론들이 공통적으로 기반하고 있는 핵심 기술(예: Transformer, GAN, Diffusion 등)이 있다면, 그 개념을 요약하십시오.
>4. **평가 패러다임:** 이 분야에서 일반적으로 사용되는 평가 방법론, 주요 벤치마크 데이터셋, 표준 평가 지표가 있다면 개괄하십시오. (구체적인 벤치마크 비교 결과는 Comparative Analysis에서 다룹니다.)
>5. **연결 고리:** 위 배경지식이 이후 분류 체계 및 방법론 조사와 어떻게 연결되는지(Roadmap) 보여주십시오.
>
>{REVIEW_NO_SECTION_MSG}"""
        ],
        "Related Surveys": [
            f"""1. **기존 서베이 정리:** 이 논문에서 언급하는 같은 분야 또는 인접 분야의 기존 서베이/리뷰 논문들을 **Title** 을 명시하여 **PDF에서 확인되는 범위 내에서 누락없이** 나열하십시오.
>2. **각 서베이의 초점:** 각 기존 서베이가 다루는 범위, 분류 기준, 시간 범위를 1-2문장으로 요약하십시오.
>3. **비교 표:** 기존 서베이들과의 비교 표(Table)가 있다면 번호를 명시하고, 어떤 기준으로 비교하고 있는지 설명하십시오. 해당 내용이 없다면 이 부분은 생략하십시오.
>4. **차별점 (Gap Analysis):** 기존 서베이들이 놓치고 있는 부분(Gap)은 무엇이며, 본 서베이가 이를 어떻게 보완하는지 논리적으로 비교하십시오.
>5. **포지셔닝:** 본 서베이가 기존 서베이들과의 관계에서 어떤 고유한 위치를 차지하는지 정리하십시오.
>
>{REVIEW_NO_SECTION_MSG}"""
        ],
    },
    "Taxonomy & Methods": {
        "Taxonomy Overview": [
            """(참고: 각 카테고리에 속하는 개별 방법론의 상세한 설명은 이후 Methods Survey에서 다룹니다. 여기서는 분류 체계의 **구조와 기준**에 집중하십시오.)
>
>1. **분류 체계 (Taxonomy):** 저자들이 제안하는 분류 체계의 전체 구조를 **최상위 카테고리부터 최하위 카테고리까지** PDF에서 확인되는 범위 내에서 누락없이 계층적으로 설명하십시오. 각 레벨의 분류 기준이 무엇인지 명시하십시오.
>2. **분류 기준의 근거:** 저자들이 이 분류 체계를 선택한 이유는 무엇입니까? 기술적 원리, 응용 분야, 시간순, 또는 다른 기준에 따른 것인지 설명하십시오.
>3. **카테고리 간 관계:** 각 카테고리가 서로 배타적(Mutually Exclusive)입니까, 아니면 중복 가능합니까? 카테고리 간의 관계(유사성, 차이점, 계층 구조)를 설명하십시오.
>4. **시각 자료 가이드:** 분류 체계를 보여주는 Figure나 Table이 있다면 번호를 명시하고, 그 자료를 어떻게 읽어야 하는지 안내하십시오. 해당 내용이 없다면 이 부분은 생략하십시오.
>5. **발전 흐름:** 각 카테고리 내에서 방법론이 시간 순으로 어떻게 발전해 왔는지 큰 흐름을 개괄하십시오. 타임라인 Figure가 있다면 번호를 명시하십시오."""
        ],
        "Methods Survey": [
            """(참고: 방법론 간 정량적 성능 비교는 Comparative Results에서 다룹니다. 여기서는 각 방법론의 **아이디어와 접근 방식**에 집중하십시오.)
>
>분류 체계의 **전반부 카테고리들** (논문의 서술 순서 기준 앞쪽 절반)에 대해 다음을 수행하십시오. 카테고리가 3개 이하인 경우 모든 카테고리를 다루십시오.
>
>각 카테고리에 대해:
>1. **카테고리 정의:** 이 카테고리에 속하는 방법론의 공통적인 특징과 핵심 아이디어를 설명하십시오.
>2. **대표 방법론:** 해당 카테고리에 속한 방법론 중 가장 영향력 있는 것을 선별(최소 3편, 전체의 약 30% 이내)하여 **Title** 을 명시하고, 각각의 핵심 기여와 접근 방식을 요약하십시오. 선별 기준은 인용 빈도·후속 연구 파급력·접근 방식의 대표성이며, 카테고리 내 방법론이 5편 이하인 경우 전수 요약해도 무방합니다.
>3. **나머지 방법론 & 경향:** 선별되지 않은 방법론은 개별 나열 없이 공통 접근 방식·시기별 흐름 등 경향 수준으로 1-2문장 내 압축하십시오.
>4. **발전 계보:** 위에서 선별한 핵심 방법론을 중심으로 발전 관계를 시간순으로 추적하십시오.
>5. **장단점:** 이 카테고리에 속하는 방법론들의 공통적인 강점과 한계를 정리하십시오.
>6. **핵심 기법:** 해당 카테고리에서 핵심적인 수식이나 알고리즘이 PDF에 있다면 LaTeX로 적고 직관적으로 설명하십시오. 해당 내용이 없다면 이 부분은 생략하십시오.
>7. **[부록 참조] 확장 방법론 목록:** 부록(Appendix)에 본문보다 더 많은 방법론 목록이나 상세 비교 표가 있다면 추가로 정리하십시오. 해당 내용이 없다면 이 부분은 생략하십시오."""
        ],
        "Methods Survey (cont.)": [
            """(참고: 방법론 간 정량적 성능 비교는 Comparative Results에서 다룹니다. 여기서는 각 방법론의 **아이디어와 접근 방식**에 집중하십시오.)
>
>분류 체계의 **후반부 카테고리들** (논문의 서술 순서 기준 뒤쪽 절반)에 대해 다음을 수행하십시오. 전반부에서 모든 카테고리를 다루었다면, ANSWER 섹션에 "전반부에서 모든 카테고리를 다루었습니다."라고만 작성하고, SOURCES 섹션은 비워 두십시오.
>
>각 카테고리에 대해:
>1. **카테고리 정의:** 이 카테고리에 속하는 방법론의 공통적인 특징과 핵심 아이디어를 설명하십시오.
>2. **대표 방법론:** 해당 카테고리에 속한 방법론 중 가장 영향력 있는 것을 선별(최소 3편, 전체의 약 30% 이내)하여 **Title** 을 명시하고, 각각의 핵심 기여와 접근 방식을 요약하십시오. 선별 기준은 인용 빈도·후속 연구 파급력·접근 방식의 대표성이며, 카테고리 내 방법론이 5편 이하인 경우 전수 요약해도 무방합니다.
>3. **나머지 방법론 & 경향:** 선별되지 않은 방법론은 개별 나열 없이 공통 접근 방식·시기별 흐름 등 경향 수준으로 1-2문장 내 압축하십시오.
>4. **발전 계보:** 위에서 선별한 핵심 방법론을 중심으로 발전 관계를 시간순으로 추적하십시오.
>5. **장단점:** 이 카테고리에 속하는 방법론들의 공통적인 강점과 한계를 정리하십시오.
>6. **핵심 기법:** 해당 카테고리에서 핵심적인 수식이나 알고리즘이 PDF에 있다면 LaTeX로 적고 직관적으로 설명하십시오. 해당 내용이 없다면 이 부분은 생략하십시오.
>7. **[부록 참조] 확장 방법론 목록:** 부록(Appendix)에 본문보다 더 많은 방법론 목록이나 상세 비교 표가 있다면 추가로 정리하십시오. 해당 내용이 없다면 이 부분은 생략하십시오."""
        ],
        "Taxonomy Summary": [
            """지금까지 분석한 분류 체계와 방법론 조사를 PDF 원문을 기반으로 종합하여 요약해 주십시오.
>
>1. **분류 체계 요약:** 전체 분류 구조를 한눈에 파악할 수 있도록 간결하게 정리하십시오.
>2. **카테고리별 핵심:** 각 카테고리의 핵심 접근법과 대표 연구를 **Title** 을 명시하여 1-2문장으로 압축하십시오.
>3. **패러다임 변화:** 시간에 따른 연구 패러다임의 변화(예: 전통적 방법 → 딥러닝 → 최신 트렌드)를 큰 그림으로 설명하십시오.
>4. **크로스 카테고리 분석:** 카테고리 간 공통점, 차이점, 그리고 최근 카테고리 간 융합 추세가 있다면 설명하십시오. 해당 내용이 없다면 이 부분은 생략하십시오.
>
>독자가 '이 분야의 방법론들은 이렇게 분류되고 발전해 왔구나'라고 머릿속에 그릴 수 있도록 체계적으로 정리하십시오."""
        ],
    },
    "Comparative Analysis": {
        "Benchmarks & Datasets": [
            f"""(참고: 벤치마크의 근본적 한계와 개선 방향은 Challenges & Open Problems에서 다룹니다. 여기서는 벤치마크의 정의와 특성에 집중하십시오.)
>
>1. **주요 벤치마크:** 이 서베이에서 방법론 비교에 사용되는 벤치마크 데이터셋의 이름(Title), 규모, 주요 특징을 **PDF에서 확인되는 범위 내에서 가능한 한** 설명하십시오.
>2. **평가 지표:** 비교에 사용된 평가 지표(Metrics)를 나열하고, 각 지표가 무엇을 측정하는지 설명하십시오. 수식이 PDF에 있다면 LaTeX로 적고 직관적으로 해석하십시오.
>3. **벤치마크 특성:** 각 벤치마크가 어떤 측면(정확도, 효율성, 일반화 등)을 테스트하기 위해 설계되었는지 설명하십시오.
>4. **벤치마크의 한계:** 저자들이 기존 벤치마크의 한계점이나 편향(Bias)을 지적하고 있다면 정리하십시오. 해당 내용이 없다면 이 부분은 생략하십시오.
>5. **[부록 참조] 추가 벤치마크 정보:** 부록에 벤치마크의 상세 통계, 샘플 예시, 또는 추가 데이터셋이 있다면 정리하십시오. 해당 내용이 없다면 이 부분은 생략하십시오.
>
>{REVIEW_NO_SECTION_MSG}"""
        ],
        "Comparative Results": [
            f"""표의 모든 숫자를 나열하지 말고, 핵심적인 비교 결과와 경향(Trends)을 중심으로 설명하십시오.
>
>1. **비교 표 분석:** 방법론 비교 표(Table)가 있다면 번호를 지목하고, **어떤 카테고리의 방법론이 어떤 벤치마크에서 우수한지** 핵심적인 경향을 설명하십시오.
>2. **성능 추이:** 시간 순으로 성능이 어떻게 향상되어 왔는지(State-of-the-Art의 변천)를 설명하십시오. 해당 내용이 없다면 이 부분은 생략하십시오.
>3. **트레이드오프:** 정확도 vs. 효율성, 일반화 vs. 특수화 등 방법론 간 트레이드오프(Trade-off)가 있다면 분석하십시오.
>4. **저자의 해석:** 저자들은 비교 결과를 통해 어떤 결론을 도출하고 있습니까? 특정 접근법이 우월한 이유를 어떻게 설명합니까?
>5. **핵심 발견:** 비교 분석에서 도출된 가장 중요한 발견(Findings)이나 의외의 결과(Surprising Results)가 있다면 강조하십시오.
>6. **[부록 참조] 추가 비교 결과:** 부록에 본문보다 더 많은 벤치마크 결과나 세밀한 비교 표가 있다면 핵심 내용을 추가로 설명하십시오. 해당 내용이 없다면 이 부분은 생략하십시오.
>
>{REVIEW_NO_SECTION_MSG}"""
        ],
        "Applications": [
            f"""1. **응용 분야:** 서베이 대상 방법론들이 활용되는 실제 응용 분야(Application Domains)를 **PDF에서 확인되는 범위 내에서** 모두 나열하고, 각 분야에서의 활용 방식을 설명하십시오.
>2. **분야별 요구사항:** 각 응용 분야가 방법론에 요구하는 특수한 조건(실시간 처리, 정확도, 데이터 제약 등)이 있다면 설명하십시오. 해당 내용이 없다면 이 부분은 생략하십시오.
>3. **성공 사례:** 특정 방법론이 실제 응용에서 성공적으로 적용된 사례가 언급되어 있다면 요약하십시오. 해당 내용이 없다면 이 부분은 생략하십시오.
>4. **응용과 방법론의 연결:** 어떤 카테고리의 방법론이 어떤 응용 분야에 적합한지, 그 이유와 함께 매핑하십시오.
>
>{REVIEW_NO_SECTION_MSG}"""
        ],
        "Comparative Summary": [
            """지금까지 분석한 벤치마크, 비교 결과, 응용 분야를 PDF 원문을 기반으로 종합하여 평가해 주십시오.
>
>1. **방법론-상황 매핑:** 비교 분석 결과를 종합하여, 어떤 카테고리의 방법론이 어떤 벤치마크/응용 상황에서 유리한지 정리하십시오.
>2. **트레이드오프 종합:** 정확도, 효율성, 일반화 등 주요 축에서의 방법론 간 트레이드오프를 요약하십시오.
>3. **실용적 가이드:** 연구자가 특정 목적에 맞는 방법론을 선택할 때 참고할 수 있는 지침을 저자의 분석에 기반하여 제시하십시오.
>
>독자가 '내 상황에는 어떤 접근법이 적합한가'를 판단할 수 있도록 체계적으로 정리하십시오."""
        ],
    },
    "Conclusion": {
        "Challenges & Open Problems": [
            """(참고: 향후 연구 방향과 유망한 트렌드는 Future Directions에서 다룹니다. 여기서는 현재의 한계점과 미해결 문제에 집중하십시오.)
>
>1. **현재의 도전 과제:** 저자들이 지적하는 이 분야의 현재 주요 도전 과제(Challenges)를 **PDF에서 확인되는 범위 내에서 누락없이** 나열하고, 각 도전 과제가 왜 어려운지 설명하십시오.
>2. **미해결 문제 (Open Problems):** 아직 해결되지 않은 근본적인 연구 문제가 있다면 정리하십시오. 해당 내용이 없다면 이 부분은 생략하십시오.
>3. **기술적 병목:** 현재 방법론들의 공통적인 기술적 한계(예: 데이터 의존성, 계산 비용, 일반화 실패 등)를 분석하십시오.
>4. **데이터 및 평가의 한계:** 현재 벤치마크나 평가 방법론의 한계점이 언급되어 있다면 설명하십시오. 해당 내용이 없다면 이 부분은 생략하십시오."""
        ],
        "Future Directions": [
            f"""(참고: 현재의 한계점과 도전 과제는 이미 Challenges & Open Problems에서 다루었습니다. 여기서는 **앞으로의 방향**에 집중하십시오.)
>
>1. **연구 트렌드:** 저자들이 파악한 이 분야의 최근 연구 트렌드(Emerging Trends)는 무엇입니까?
>2. **유망한 방향:** 저자들이 제안하는 향후 유망한 연구 방향(Future Directions)을 **PDF에서 확인되는 범위 내에서** 모두 나열하고, 각 방향의 잠재적 가치를 설명하십시오.
>3. **융합 가능성:** 다른 분야와의 융합이나 새로운 패러다임의 가능성이 제시되어 있다면 설명하십시오. 해당 내용이 없다면 이 부분은 생략하십시오.
>4. **실용화 전망:** 연구에서 실용화로의 전환을 위해 필요한 요소가 언급되어 있다면 정리하십시오. 해당 내용이 없다면 이 부분은 생략하십시오.
>
>{REVIEW_NO_SECTION_MSG}"""
        ],
        "Conclusion": [
            """1. **주요 발견:** 이 서베이를 통해 도출된 가장 중요한 발견이나 통찰(Key Findings)은 무엇입니까?
>2. **분야의 현재 상태:** 저자들은 이 연구 분야의 현재 성숙도(Maturity)를 어떻게 평가하고 있습니까? 해당 내용이 없다면 이 부분은 생략하십시오.
>3. **핵심 메시지:** 저자들이 독자에게 전달하고자 하는 핵심 메시지는 무엇입니까?
>4. **총평:** 이 서베이가 해당 AI 분야의 연구자들에게 어떤 가치를 제공하며, 어떤 의미를 갖는지 마무리 설명하십시오. 총평은 저자의 주장과 서베이의 객관적 범위에 기반하여 작성하십시오. 주관적 평가는 배제하십시오."""
        ],
    },
}

# -------------------------
# 2) Review Prompts Dict
#    review_questions와 동일한 구조. make_prompt()로 각 질문을 wrapping.
#    Paper Summary(Turn 1)만 full COMMON_HEADER, 이후는 COMMON_HEADER_SHORT 사용.
#    anti_recitation 파라미터로 RECITATION guard를 체계적으로 주입.
# -------------------------

review_prompts = {
    "Survey Overview": {
        "Paper Summary": [
            (
                make_prompt(
                    scope="첨부된 서베이 논문 전체",
                    questions=review_questions["Survey Overview"]["Paper Summary"][0],
                    short_header=False,
                    anti_recitation=ANTI_RECITATION,
                ),
                "B",
            )
        ],
        "Introduction": [
            (
                make_prompt(
                    scope="Introduction / Motivation / Scope 관련 섹션",
                    questions=review_questions["Survey Overview"]["Introduction"][0],
                    short_header=True,
                    anti_recitation=ANTI_RECITATION,
                ),
                "B",
            )
        ],
        "Background": [
            (
                make_prompt(
                    scope="Background / Preliminaries / Problem Definition 섹션",
                    questions=review_questions["Survey Overview"]["Background"][0],
                    allow_background=True,
                    short_header=True,
                ),
                "B",
            )
        ],
        "Related Surveys": [
            (
                make_prompt(
                    scope="Related Surveys / 기존 서베이 비교 섹션 (및 관련 References)",
                    questions=review_questions["Survey Overview"]["Related Surveys"][0],
                    short_header=True,
                    anti_recitation=ANTI_RECITATION_SURVEY,
                ),
                "C",
            )
        ],
    },
    "Taxonomy & Methods": {
        "Taxonomy Overview": [
            (
                make_prompt(
                    scope="분류 체계(Taxonomy/Classification) 전체 구조",
                    questions=review_questions["Taxonomy & Methods"]["Taxonomy Overview"][0],
                    short_header=True,
                    anti_recitation=ANTI_RECITATION,
                ),
                "A",
            )
        ],
        "Methods Survey": [
            (
                make_prompt(
                    scope="분류 체계의 전반부 카테고리별 방법론 조사 (부록 포함)",
                    questions=review_questions["Taxonomy & Methods"]["Methods Survey"][0],
                    short_header=True,
                    anti_recitation=ANTI_RECITATION_SURVEY,
                ),
                "C",
            )
        ],
        "Methods Survey (cont.)": [
            (
                make_prompt(
                    scope="분류 체계의 후반부 카테고리별 방법론 조사 (부록 포함)",
                    questions=review_questions["Taxonomy & Methods"]["Methods Survey (cont.)"][0],
                    short_header=True,
                    anti_recitation=ANTI_RECITATION_SURVEY,
                ),
                "C",
            )
        ],
        "Taxonomy Summary": [
            (
                make_prompt(
                    scope="분류 체계 및 방법론 조사 전체",
                    questions=review_questions["Taxonomy & Methods"]["Taxonomy Summary"][0],
                    short_header=True,
                ),
                "B",
            )
        ],
    },
    "Comparative Analysis": {
        "Benchmarks & Datasets": [
            (
                make_prompt(
                    scope="벤치마크 데이터셋 및 평가 지표 관련 섹션 (부록 포함)",
                    questions=review_questions["Comparative Analysis"]["Benchmarks & Datasets"][0],
                    allow_background=True,
                    short_header=True,
                    anti_recitation=ANTI_RECITATION,
                ),
                "B",
            )
        ],
        "Comparative Results": [
            (
                make_prompt(
                    scope="방법론 비교 분석 / 벤치마크 결과 섹션 (부록 포함)",
                    questions=review_questions["Comparative Analysis"]["Comparative Results"][0],
                    short_header=True,
                    anti_recitation=ANTI_RECITATION,
                ),
                "C",
            )
        ],
        "Applications": [
            (
                make_prompt(
                    scope="응용 분야(Applications) 관련 섹션",
                    questions=review_questions["Comparative Analysis"]["Applications"][0],
                    short_header=True,
                    anti_recitation=ANTI_RECITATION,
                ),
                "B",
            )
        ],
        "Comparative Summary": [
            (
                make_prompt(
                    scope="Comparative Analysis 전체 (벤치마크, 비교 결과, 응용 분야)",
                    questions=review_questions["Comparative Analysis"]["Comparative Summary"][0],
                    short_header=True,
                ),
                "B",
            )
        ],
    },
    "Conclusion": {
        "Challenges & Open Problems": [
            (
                make_prompt(
                    scope="Challenges / Open Problems / Limitations 관련 섹션",
                    questions=review_questions["Conclusion"]["Challenges & Open Problems"][0],
                    short_header=True,
                    anti_recitation=ANTI_RECITATION,
                ),
                "B",
            )
        ],
        "Future Directions": [
            (
                make_prompt(
                    scope="Future Directions / Emerging Trends 관련 섹션",
                    questions=review_questions["Conclusion"]["Future Directions"][0],
                    short_header=True,
                    anti_recitation=ANTI_RECITATION,
                ),
                "B",
            )
        ],
        "Conclusion": [
            (
                make_prompt(
                    scope="Conclusion / Summary 관련 섹션",
                    questions=review_questions["Conclusion"]["Conclusion"][0],
                    short_header=True,
                ),
                "B",
            )
        ],
    },
}
