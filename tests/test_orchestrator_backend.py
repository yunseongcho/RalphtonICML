import json
import sys
import unittest

from ralphton_icml.backend import BackendError, ModelRequest, ReplayBackend, SubprocessBackend
from ralphton_icml.instruction import (
    InstructionContractError,
    load_reviewer_instruction,
    validate_reviewer_instruction,
)
from ralphton_icml.learning import LearningState, MemoryItem
from ralphton_icml.orchestrator import PaperInput, ReviewerOrchestrator


VALID_REVIEW = """#### **Soundness**

3

#### **Presentation**

3

#### **Significance**

3

#### **Originality**

3

#### **Overall Recommendation**

4

#### **Confidence**

4

#### Comment

The claims are mostly supported; add the missing controlled ablation.
"""


def replay_fallback(request):
    if request.stage == "extraction":
        task = request.payload["extraction_task"]["task_id"]
        return "ANSWER\nEvidence for {} (p. 1).\nSOURCES\n- paper, p. 1".format(task)
    if request.stage in {"domain_review", "criterion_review"}:
        return "strengths: grounded\nweaknesses: limited ablation\nevidence_ids: supplied"
    if request.stage == "synthesis":
        return "The evidence supports a technically useful paper with a missing ablation."
    if request.stage == "final_review":
        return VALID_REVIEW
    if request.stage == "author_rebuttal":
        return "We clarify the controlled comparison using the evidence on p. 1."
    raise AssertionError(request.stage)


class BackendTest(unittest.TestCase):
    def test_subprocess_json_protocol(self):
        program = (
            "import json,sys; d=json.load(sys.stdin); "
            "print(json.dumps({'text': d['agent_id'] + ':' + d['stage']}))"
        )
        backend = SubprocessBackend((sys.executable, "-c", program), timeout=5)
        response = backend.complete(
            ModelRequest("r", "agent", "stage", "system", {"x": 1})
        )
        self.assertEqual(response, "agent:stage")

    def test_subprocess_failure_is_reported(self):
        backend = SubprocessBackend((sys.executable, "-c", "raise SystemExit(7)"))
        with self.assertRaises(BackendError):
            backend.complete(ModelRequest("r", "a", "s", "i", {}))


class InstructionTest(unittest.TestCase):
    def test_repository_instruction_matches_machine_contract(self):
        text = load_reviewer_instruction()
        self.assertIn("## Paper Review Criteria", text)
        self.assertIn("human / AI reviewers", text)

    def test_changed_range_is_rejected(self):
        text = load_reviewer_instruction().replace("- 4: excellent", "- 5: excellent", 1)
        with self.assertRaises(InstructionContractError):
            validate_reviewer_instruction(text)


class OrchestratorTest(unittest.TestCase):
    def test_end_to_end_contract_and_information_boundaries(self):
        backend = ReplayBackend({}, fallback=replay_fallback)
        run = ReviewerOrchestrator(backend, max_workers=4).review(
            PaperInput(
                paper_id="paper-1",
                title="Image segmentation with a neural network",
                text="We study computer vision and provide experiments.",
            )
        )
        self.assertEqual(len(run.context.evidence), 19)
        self.assertIn("domain.cv", run.domain_critiques)
        self.assertEqual(len(run.criterion_critiques), 6)
        self.assertEqual(run.initial_review.overall_recommendation, 4)
        self.assertEqual(run.final_review.overall_recommendation, 4)

        extraction_requests = [r for r in backend.requests if r.stage == "extraction"]
        self.assertEqual(len(extraction_requests), 19)
        self.assertTrue(all("decision" not in json.dumps(r.payload).casefold() for r in extraction_requests))
        initial_chair = next(
            r for r in backend.requests
            if r.stage == "final_review" and "initial_review" not in r.payload
        )
        self.assertNotIn("author_rebuttal", initial_chair.payload)
        author = next(r for r in backend.requests if r.stage == "author_rebuttal")
        self.assertIn("initial_review", author.payload)
        self.assertNotIn("decision", author.payload)
        chair = next(r for r in backend.requests if r.stage == "final_review")
        self.assertIn("Paper Review Criteria", chair.system)
        self.assertIn("AI agent and the paper", chair.system)

    def test_learned_state_is_role_separated_in_live_agent_payloads(self):
        state = LearningState(
            version=1,
            reviewer_memory=(MemoryItem(
                "reviewer", "train-review", "Check the Markov game regret proof.",
                "optimistic Markov game regret proof regularized leader",
            ),),
            author_memory=(MemoryItem(
                "author", "train-author", "Answer the Markov regret concern with a lemma.",
                "optimistic Markov game regret proof regularized leader",
            ),),
        )
        backend = ReplayBackend({}, fallback=replay_fallback)
        ReviewerOrchestrator(backend, learning_state=state).review(
            PaperInput(
                "new-paper",
                "Optimistic Markov Game Regret",
                "We prove a regret lemma for an optimistic regularized leader in Markov games.",
            )
        )
        reviewer_requests = [
            request for request in backend.requests
            if request.stage in {"domain_review", "criterion_review", "synthesis"}
        ]
        self.assertTrue(reviewer_requests)
        self.assertTrue(all("reviewer_memory" in request.payload for request in reviewer_requests))
        self.assertTrue(all("author_memory" not in request.payload for request in reviewer_requests))
        author = next(request for request in backend.requests if request.stage == "author_rebuttal")
        self.assertIn("author_memory", author.payload)
        self.assertNotIn("reviewer_memory", author.payload)
        chairs = [request for request in backend.requests if request.stage == "final_review"]
        self.assertTrue(all(request.payload["learned_reviewer_state"]["version"] == 1 for request in chairs))


if __name__ == "__main__":
    unittest.main()
