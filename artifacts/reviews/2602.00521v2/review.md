#### **Soundness**

2

#### **Presentation**

3

#### **Significance**

3

#### **Originality**

3

#### **Overall Recommendation**

3

#### **Confidence**

4

#### Comment

이 논문은 LLM-as-a-Judge를 검증이 필요한 measurement instrument로 보고, GRM을 이용해 prompt sensitivity와 고정-prompt reliability를 구분하는 유용한 관점을 제시합니다. 특히 \(CV\)와 \(\rho\)를 서로 다른 failure mode에 연결하고, detailed rubric·CoT·rating-scale 변경을 실행 가능한 개선책과 연계한 점이 강점입니다.

저자 답변은 초기 리뷰의 핵심 문제를 정확히 인정하고 주장 범위를 적절히 축소했습니다. 독립적으로 표준화된 human/LLM latent scale 사이에는 measurement invariance나 공통 metric이 보장되지 않으므로, 현재의 \(\theta_{\mathrm{ratio}}\)와 \(D_W\)를 exploratory descriptive comparison으로 제한하는 것이 타당합니다. Phase 1 미통과 조건을 confirmatory Phase 2에서 제외하고, VIEScore 결과를 특정 benchmark에서 관찰된 validity concern으로 완화하려는 수정도 논문의 원칙과 분석을 더 일관되게 만듭니다. “genuine quality”를 judge-perceived latent trait로 바꾸고 reliability와 validity를 구분하는 수정 역시 필요합니다.

다만 답변은 대부분 향후 문구·구조 수정과 limitation 명시에 머물며, 핵심 방법론적 문제를 해결하는 새로운 분석은 제공하지 않습니다. Joint multi-group IRT, invariant anchor를 통한 scale linking, affine-transformation sensitivity가 없는 한 Phase 2 metric의 calibration 해석은 여전히 확립되지 않습니다. 따라서 개정본에서는 Phase 1을 논문의 주된 실증 기여로 명확히 재정의하고, Phase 2 수치와 vision 결론을 Abstract 및 Conclusion에서도 일관되게 exploratory로 표시해야 합니다.

또한 \(CV\)의 empty/singleton category 처리, 최소 category size, \(|K_p|-1\) normalization, category weighting을 정확히 명세하고 가능한 대안에 대한 sensitivity analysis를 제공하는 것이 중요합니다. \(CV=0.10\)과 \(\rho=0.70\)은 working heuristic으로 표시하고 cutoff 주변 결과를 이분법적으로 해석하지 않아야 합니다. GRM의 unidimensionality·monotonicity·local-independence 검사, human annotation aggregation과 annotator severity 처리, perturbation semantic-equivalence 및 parsing validation도 보강되어야 합니다. Benchmark별 sample 수·split·score histogram과 bootstrap 재적합 및 resampling 단위, Kruskal–Wallis observation unit·effect size·multiple-testing 처리 역시 재현성과 통계적 해석에 필수적입니다.

참가자의 AI agent는 latent-variable 방법을 검토할 때 within-model identifiability와 cross-group scale linking을 자동으로 구분하고, 논문이 선언한 gating rule이 실제 표와 결론에 적용됐는지 추적하도록 개선하면 좋겠습니다. 또한 category sparsity, cutoff 근거, measurement-model assumptions, resampling unit, confirmatory·exploratory 분석 구분을 필수 checklist로 삼으면 단순 요약을 넘어 더 신뢰도 높은 비판적 검토가 가능할 것입니다.
