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

이 논문은 LLM-as-a-Judge를 단순한 agreement generator가 아니라 검증이 필요한 measurement instrument로 보는 중요한 관점을 제시합니다. Prompt variant별 특성과 sample별 latent quality를 GRM으로 분리하고, intrinsic consistency를 human alignment보다 먼저 확인하자는 구성은 명확하고 실용적입니다. 특히 CV와 \(\rho\)가 prompt sensitivity와 고정-prompt discrimination failure를 구분한다는 결과, detailed rubric/CoT와 rating-scale 변경이 서로 다른 지표에 영향을 준다는 ablation, 기존 \(\omega\), correlation, \(\kappa\), \(\alpha\)가 놓치는 failure mode를 보여준 비교는 강점입니다. 방법과 prompt의 구현 세부사항도 비교적 충실합니다.

그러나 Phase 2의 핵심 결과는 현재 형태로 충분히 정당화되지 않습니다. Human과 LLM에 GRM을 독립적으로 fitting하고 각각 \(\theta\sim\mathcal N(0,1)\)로 고정하는 것은 각 모델 내부의 identifiability만 해결하며, 두 latent scale이 동일한 construct와 metric을 공유한다는 measurement invariance를 보장하지 않습니다. 따라서 \(\theta_{\mathrm{ratio}}\)와 \(D_W\)가 실제 calibration 차이를 나타내는지, 독립적인 scale fitting에서 생긴 차이인지 불분명합니다. Joint multi-group IRT, anchor-based scale linking, 또는 scale transformation에 대한 simulation/sensitivity analysis가 필요합니다.

또한 “Phase 1을 통과한 경우에만 Phase 2를 해석한다”는 원칙과 실제 분석이 일치하지 않습니다. 높은 CV를 보인 VIEScore 조건까지 사용해 construct-validity gap을 논의하고 있으므로, Phase 1 통과 조합만으로 Phase 2 결과와 주요 결론을 다시 제시해야 합니다. 미통과 조건의 결과는 exploratory analysis로 분리하는 것이 적절합니다.

추가로 다음 사항을 보강해 주시기 바랍니다.

- CV에서 빈 범주, singleton, prompt별 상이한 \(K_p\)를 처리하는 규칙과 \(|K_p|-1\) normalization의 근거를 명시하고, category-size weighting 등 대안 정의에 대한 sensitivity analysis를 제공해 주십시오.
- \(CV=0.10\), \(\rho=0.70\)을 universal cutoff가 아닌 heuristic으로 명확히 표시하고, rating-scale 및 dataset별 threshold sensitivity를 보고해 주십시오.
- \(\theta\)는 외부적으로 참인 “genuine quality”가 아니라 judge가 인식한 latent quality이므로 용어를 완화하고 reliability와 validity를 분명히 구분해 주십시오.
- GRM의 unidimensionality, monotonicity, local independence를 posterior predictive checks와 residual-dependence 분석으로 검증해 주십시오. Human rating의 aggregation 방법과 annotator severity/disagreement 처리도 필요합니다.
- Prompt perturbation의 semantic equivalence와 parsing validity를 human 또는 독립 검사로 확인하고, invalid perturbation을 제외한 결과를 보고해 주십시오.
- Benchmark별 sample 수, split, score histogram, category별 표본 수, human inter-rater reliability를 공개해 주십시오. Bootstrap의 resampling unit과 GRM 재적합 여부, Kruskal–Wallis 검정 단위 및 multiple-testing 처리도 명확히 해야 합니다.
- Vision의 construct-validity gap은 현재 단일 benchmark에 근거하므로 탐색적 가설로 표현하고, 추가 benchmark와 실제 image-level failure analysis로 검증해 주십시오.

참가자의 AI agent도 단순 요약을 넘어 방법론의 식별 가정과 논문 내부의 원칙–실험 불일치를 우선 점검하도록 개선할 수 있습니다. 구체적으로 latent-variable metric을 검토할 때 “정규화되었다”는 설명만 수용하지 말고 cross-group scale linking과 measurement invariance를 확인하고, 제안된 gating rule이 모든 결과 표에 실제 적용됐는지 자동 점검하며, dataset 통계·범주 희소성·cutoff 근거·통계 검정 단위를 필수 review checklist로 포함하는 것이 좋겠습니다.
