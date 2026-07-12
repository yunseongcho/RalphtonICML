"""Declarative reviewer-team composition and deterministic routing."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import re
from typing import Optional, Tuple

from .schema import ContextPacket, ContextTask, ExtractionOutput, ReviewOutput


class AgentRole(str, Enum):
    DOMAIN_EXPERT = "domain_expert"
    CRITERION_EXPERT = "criterion_expert"
    AUTHOR = "author"
    SYNTHESIZER = "synthesizer"
    CHAIR = "chair"


class Domain(str, Enum):
    CV = "CV"
    CORE_ML = "Core ML"
    NLP = "NLP"
    RECSYS = "RecSys"
    GENERAL = "General"


class Criterion(str, Enum):
    SOUNDNESS = "Soundness"
    PRESENTATION = "Presentation"
    SIGNIFICANCE = "Significance"
    ORIGINALITY = "Originality"
    REPRODUCIBILITY = "Reproducibility"
    ETHICS = "Ethics"


class PipelineStage(str, Enum):
    EXTRACTION = "extraction"
    DOMAIN_REVIEW = "domain_review"
    CRITERION_REVIEW = "criterion_review"
    AUTHOR_REBUTTAL = "author_rebuttal"
    SYNTHESIS = "synthesis"
    FINAL_REVIEW = "final_review"


class OutputContract(str, Enum):
    ANSWER_SOURCES = "answer_sources"
    CRITIQUE = "critique"
    REBUTTAL = "rebuttal"
    SYNTHESIS = "synthesis"
    REVIEW_FORM = "review_form"


@dataclass(frozen=True)
class StageContract:
    stage: PipelineStage
    input_type: type
    output_type: type
    output_contract: OutputContract


EXTRACTION_STAGE_CONTRACT = StageContract(
    stage=PipelineStage.EXTRACTION,
    input_type=ContextTask,
    output_type=ExtractionOutput,
    output_contract=OutputContract.ANSWER_SOURCES,
)
FINAL_REVIEW_STAGE_CONTRACT = StageContract(
    stage=PipelineStage.FINAL_REVIEW,
    input_type=ContextPacket,
    output_type=ReviewOutput,
    output_contract=OutputContract.REVIEW_FORM,
)


@dataclass(frozen=True)
class AgentSpec:
    agent_id: str
    display_name: str
    role: AgentRole
    stage: PipelineStage
    output_contract: OutputContract
    instruction: str
    domain: Optional[Domain] = None
    criterion: Optional[Criterion] = None
    keywords: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for name in ("agent_id", "display_name", "instruction"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError("{} must be a non-empty string".format(name))
            object.__setattr__(self, name, value.strip())
        if not isinstance(self.role, AgentRole):
            raise ValueError("role must be an AgentRole")
        if not isinstance(self.stage, PipelineStage):
            raise ValueError("stage must be a PipelineStage")
        if not isinstance(self.output_contract, OutputContract):
            raise ValueError("output_contract must be an OutputContract")
        if self.domain is not None and not isinstance(self.domain, Domain):
            raise ValueError("domain must be a Domain or None")
        if self.criterion is not None and not isinstance(self.criterion, Criterion):
            raise ValueError("criterion must be a Criterion or None")
        canonical_keywords = tuple(
            keyword.strip().casefold()
            for keyword in self.keywords
            if isinstance(keyword, str) and keyword.strip()
        )
        if len(canonical_keywords) != len(self.keywords):
            raise ValueError("keywords must contain only non-empty strings")
        object.__setattr__(self, "keywords", canonical_keywords)
        if self.role is AgentRole.DOMAIN_EXPERT and self.domain is None:
            raise ValueError("domain experts require a domain")
        if self.role is AgentRole.CRITERION_EXPERT and self.criterion is None:
            raise ValueError("criterion experts require a criterion")

    @property
    def name(self) -> str:
        return self.display_name


DOMAIN_EXPERTS = (
    AgentSpec(
        agent_id="domain.cv",
        display_name="CV Expert",
        role=AgentRole.DOMAIN_EXPERT,
        stage=PipelineStage.DOMAIN_REVIEW,
        output_contract=OutputContract.CRITIQUE,
        domain=Domain.CV,
        instruction=(
            "Evaluate vision assumptions, data, perceptual metrics, architectures, "
            "and qualitative evidence against the shared context."
        ),
        keywords=(
            "computer vision", "vision", "image", "video", "visual", "camera",
            "object detection", "segmentation", "diffusion", "face", "3d",
            "컴퓨터 비전", "이미지", "영상", "객체 탐지", "분할", "얼굴",
        ),
    ),
    AgentSpec(
        agent_id="domain.core_ml",
        display_name="Core ML Expert",
        role=AgentRole.DOMAIN_EXPERT,
        stage=PipelineStage.DOMAIN_REVIEW,
        output_contract=OutputContract.CRITIQUE,
        domain=Domain.CORE_ML,
        instruction=(
            "Evaluate learning theory, optimization, objectives, generalization, "
            "statistical claims, and architecture-independent ML methodology."
        ),
        keywords=(
            "core ml", "machine learning", "learning theory", "generalization",
            "optimization", "optimizer", "loss function", "representation learning",
            "neural network", "gradient", "theorem", "proof", "머신러닝",
            "최적화", "일반화", "학습 이론", "손실 함수", "정리", "증명",
        ),
    ),
    AgentSpec(
        agent_id="domain.nlp",
        display_name="NLP Expert",
        role=AgentRole.DOMAIN_EXPERT,
        stage=PipelineStage.DOMAIN_REVIEW,
        output_contract=OutputContract.CRITIQUE,
        domain=Domain.NLP,
        instruction=(
            "Evaluate language data, tokenization, generation, reasoning, retrieval, "
            "and language-model evaluation practices."
        ),
        keywords=(
            "nlp", "natural language", "language model", "large language model",
            "llm", "text", "token", "translation", "question answering", "prompt",
            "자연어", "언어 모델", "대규모 언어 모델", "텍스트", "토큰", "번역",
        ),
    ),
    AgentSpec(
        agent_id="domain.recsys",
        display_name="RecSys Expert",
        role=AgentRole.DOMAIN_EXPERT,
        stage=PipelineStage.DOMAIN_REVIEW,
        output_contract=OutputContract.CRITIQUE,
        domain=Domain.RECSYS,
        instruction=(
            "Evaluate recommendation objectives, ranking, retrieval, feedback bias, "
            "offline metrics, and user-item experimental protocols."
        ),
        keywords=(
            "recsys", "recommendation", "recommender", "collaborative filtering",
            "user item", "click through", "ranking", "recommendation system",
            "추천", "추천 시스템", "협업 필터링", "사용자 아이템", "클릭률", "랭킹",
        ),
    ),
    AgentSpec(
        agent_id="domain.general",
        display_name="General ML Expert",
        role=AgentRole.DOMAIN_EXPERT,
        stage=PipelineStage.DOMAIN_REVIEW,
        output_contract=OutputContract.CRITIQUE,
        domain=Domain.GENERAL,
        instruction=(
            "Evaluate cross-domain ML claims and identify when additional specialist "
            "coverage is needed."
        ),
    ),
)


def _criterion_spec(criterion: Criterion, instruction: str) -> AgentSpec:
    return AgentSpec(
        agent_id="criterion.{}".format(criterion.value.casefold()),
        display_name="{} Expert".format(criterion.value),
        role=AgentRole.CRITERION_EXPERT,
        stage=PipelineStage.CRITERION_REVIEW,
        output_contract=OutputContract.CRITIQUE,
        criterion=criterion,
        instruction=instruction,
    )


CRITERION_EXPERTS = (
    _criterion_spec(
        Criterion.SOUNDNESS,
        "Check whether technical, experimental, and methodological claims are supported.",
    ),
    _criterion_spec(
        Criterion.PRESENTATION,
        "Check clarity, organization, writing, and contextualization against prior work.",
    ),
    _criterion_spec(
        Criterion.SIGNIFICANCE,
        "Assess the likely importance and breadth of the contribution to its research area.",
    ),
    _criterion_spec(
        Criterion.ORIGINALITY,
        "Assess novelty relative to the cited and shared prior-work evidence.",
    ),
    _criterion_spec(
        Criterion.REPRODUCIBILITY,
        "Audit data, code, hyperparameters, compute, evaluation, and replication details.",
    ),
    _criterion_spec(
        Criterion.ETHICS,
        "Audit harms, bias, privacy, consent, licenses, misuse, and unaddressed impacts.",
    ),
)


AUTHOR_AGENT = AgentSpec(
    agent_id="author",
    display_name="Author Agent",
    role=AgentRole.AUTHOR,
    stage=PipelineStage.AUTHOR_REBUTTAL,
    output_contract=OutputContract.REBUTTAL,
    instruction=(
        "Answer reviewer critiques using paper evidence, concede unsupported claims, and "
        "propose concrete revisions without introducing unverifiable results."
    ),
)
SYNTHESIZER_AGENT = AgentSpec(
    agent_id="review.synthesizer",
    display_name="Review Synthesizer",
    role=AgentRole.SYNTHESIZER,
    stage=PipelineStage.SYNTHESIS,
    output_contract=OutputContract.SYNTHESIS,
    instruction=(
        "Merge domain and criterion critiques, remove duplicates, preserve disagreements, "
        "and tie every material claim to shared evidence."
    ),
)
CHAIR_AGENT = AgentSpec(
    agent_id="review.chair",
    display_name="Reviewer Chair",
    role=AgentRole.CHAIR,
    stage=PipelineStage.FINAL_REVIEW,
    output_contract=OutputContract.REVIEW_FORM,
    instruction=(
        "Calibrate the evidence and synthesis into exactly one validated ReviewOutput "
        "matching reviewer_instruction.md."
    ),
)

# Explicit aliases make the public role names discoverable without guessing.
AUTHOR_AGENT_SPEC = AUTHOR_AGENT
SYNTHESIZER_AGENT_SPEC = SYNTHESIZER_AGENT
CHAIR_AGENT_SPEC = CHAIR_AGENT


def _keyword_count(text: str, keyword: str) -> int:
    normalized_text = re.sub(r"[-_/]+", " ", text.casefold())
    normalized_keyword = re.sub(r"[-_/]+", " ", keyword.casefold())
    pattern = r"(?<!\w){}(?!\w)".format(re.escape(normalized_keyword))
    return len(re.findall(pattern, normalized_text))


def route_domain_experts(
    text: str, max_experts: Optional[int] = None
) -> Tuple[AgentSpec, ...]:
    """Route paper text to scored domain experts, with General as fallback."""

    return _route_specs(DOMAIN_EXPERTS, text, max_experts)


def _route_specs(
    specs: Tuple[AgentSpec, ...],
    text: str,
    max_experts: Optional[int],
) -> Tuple[AgentSpec, ...]:

    if not isinstance(text, str):
        raise TypeError("text must be a string")
    if max_experts is not None and (
        type(max_experts) is not int or max_experts < 1
    ):
        raise ValueError("max_experts must be a positive integer or None")
    scored = []
    for position, spec in enumerate(specs):
        if spec.domain is Domain.GENERAL:
            continue
        score = sum(_keyword_count(text, keyword) for keyword in spec.keywords)
        if score:
            scored.append((score, position, spec))
    if not scored:
        general = next((spec for spec in specs if spec.domain is Domain.GENERAL), None)
        return (general,) if general is not None else ()
    scored.sort(key=lambda item: (-item[0], item[1]))
    routed = tuple(item[2] for item in scored)
    return routed if max_experts is None else routed[:max_experts]


def route_domain_expert(text: str) -> AgentSpec:
    """Return the highest-scoring domain expert."""

    return route_domain_experts(text, max_experts=1)[0]


@dataclass(frozen=True)
class ReviewerTeamSpec:
    domain_experts: Tuple[AgentSpec, ...]
    criterion_experts: Tuple[AgentSpec, ...]
    author: AgentSpec
    synthesizer: AgentSpec
    chair: AgentSpec

    @property
    def reviewer_agents(self) -> Tuple[AgentSpec, ...]:
        return self.domain_experts + self.criterion_experts + (self.synthesizer, self.chair)

    @property
    def all_agents(self) -> Tuple[AgentSpec, ...]:
        return self.reviewer_agents + (self.author,)

    def route_domains(
        self, text: str, max_experts: Optional[int] = None
    ) -> Tuple[AgentSpec, ...]:
        return _route_specs(self.domain_experts, text, max_experts)


DEFAULT_REVIEWER_TEAM = ReviewerTeamSpec(
    domain_experts=DOMAIN_EXPERTS,
    criterion_experts=CRITERION_EXPERTS,
    author=AUTHOR_AGENT,
    synthesizer=SYNTHESIZER_AGENT,
    chair=CHAIR_AGENT,
)
DEFAULT_TEAM = DEFAULT_REVIEWER_TEAM


__all__ = [
    "AUTHOR_AGENT",
    "AUTHOR_AGENT_SPEC",
    "AgentRole",
    "AgentSpec",
    "CHAIR_AGENT",
    "CHAIR_AGENT_SPEC",
    "CRITERION_EXPERTS",
    "Criterion",
    "DEFAULT_REVIEWER_TEAM",
    "DEFAULT_TEAM",
    "DOMAIN_EXPERTS",
    "Domain",
    "EXTRACTION_STAGE_CONTRACT",
    "FINAL_REVIEW_STAGE_CONTRACT",
    "OutputContract",
    "PipelineStage",
    "ReviewerTeamSpec",
    "SYNTHESIZER_AGENT",
    "SYNTHESIZER_AGENT_SPEC",
    "StageContract",
    "route_domain_expert",
    "route_domain_experts",
]
