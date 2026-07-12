import json
from pathlib import Path
import tempfile
import threading
import unittest

from ralphton_icml.backend import ModelRequest
from ralphton_icml.fast import (
    FastReviewerOrchestrator,
    fast_run_as_dict,
    retrieve_live_candidates,
)
from ralphton_icml.learning import LearningState, MemoryItem
from ralphton_icml.orchestrator import PaperInput
from ralphton_icml.schema import ReviewOutput
from ralphton_icml.track2 import create_track2_bundle


class StructuredFakeBackend:
    def __init__(self, needs_refinement=True, contradiction=False):
        self.needs_refinement = needs_refinement
        self.contradiction = contradiction
        self.requests = []
        self.active = 0
        self.max_active = 0
        self.lock = threading.Lock()
        self.barriers = {
            "extraction": threading.Barrier(2),
            "consolidated_review": threading.Barrier(2),
        }

    def complete(self, request: ModelRequest) -> str:
        self.assert_request(request)
        with self.lock:
            self.requests.append(request)
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        try:
            barrier = self.barriers.get(request.stage)
            if barrier is not None:
                barrier.wait(timeout=3)
            return json.dumps(self.response(request), ensure_ascii=False)
        finally:
            with self.lock:
                self.active -= 1

    @staticmethod
    def assert_request(request):
        if request.output_schema is None:
            raise AssertionError("fast-v1 request must include output_schema")

    def response(self, request):
        if request.stage == "extraction":
            return {
                "items": [
                    {
                        "task_id": task["task_id"],
                        "answer": "Grounded evidence for {}.".format(task["task_id"]),
                        "sources": ["paper, p. 1"],
                    }
                    for task in request.payload["tasks"]
                ]
            }
        if request.stage == "consolidated_review":
            criteria = request.payload["criteria"]
            evidence_id = request.payload["context"]["evidence"][0]["evidence_id"]
            return {
                "strengths": [
                    {
                        "criterion": criteria[0],
                        "severity": "positive",
                        "text": "The problem is clearly motivated.",
                        "evidence_ids": [evidence_id],
                    }
                ],
                "weaknesses": [
                    {
                        "criterion": criteria[0],
                        "severity": "major",
                        "text": "A central controlled comparison is missing.",
                        "evidence_ids": [evidence_id],
                    }
                ],
                "questions": [],
                "memory_candidate_ids_used": [],
                "unresolved_contradictions": (
                    [
                        {
                            "text": "{} values conflict.".format(criteria[0]),
                            "evidence_ids": [evidence_id],
                        }
                    ]
                    if self.contradiction
                    else []
                ),
            }
        if request.stage == "final_review":
            post = "simulated_author_response" in request.payload
            needs = self.needs_refinement and not post
            return {
                "soundness": 2,
                "presentation": 3,
                "significance": 3,
                "originality": 3,
                "overall_recommendation": 2,
                "confidence": 4,
                "summary": "The paper addresses an important problem.",
                "strengths": ["The motivation is clear."],
                "weaknesses": ["The central comparison is not controlled."],
                "questions_for_authors": ["Can the claim be tested with existing data?"],
                "contribution": "The framing is useful but validation is incomplete.",
                "ethics_and_limitations": "No direct harm; evidence is limited.",
                "ai_agent_improvements": ["Check controlled comparisons automatically."],
                "needs_refinement": needs,
                "refinement_reasons": (
                    ["A major soundness issue remains."] if needs else []
                ),
            }
        if request.stage == "author_rebuttal":
            ids = [item["finding_id"] for item in request.payload["findings"]]
            contradiction_ids = [
                item["contradiction_id"]
                for item in request.payload["unresolved_contradictions"]
            ]
            return {
                "response": "This simulation concedes the unsupported claim.",
                "addressed_finding_ids": ids[:1],
                "addressed_contradiction_ids": contradiction_ids,
                "memory_candidate_ids_used": [],
            }
        raise AssertionError(request.stage)


class FastPipelineTest(unittest.TestCase):
    def make_bundle(self, directory, with_evidence=False):
        root = Path(directory)
        (root / "inputs").mkdir()
        (root / "evidence").mkdir()
        (root / "outputs").mkdir()
        paper = root / "inputs" / "paper.md"
        paper.write_text(
            "# Frozen Track 1 Paper\n\nWe propose a method and evaluate it.\n",
            encoding="utf-8",
        )
        evidence = ()
        if with_evidence:
            result = root / "evidence" / "results.json"
            result.write_text('{"accuracy": 0.9}\n', encoding="utf-8")
            evidence = (result,)
        return create_track2_bundle(
            root,
            paper,
            evidence_paths=evidence,
            agent_name="Fast Track 2 Reviewer",
            agent_version="fast-v1",
        )

    def test_base_is_exactly_five_calls_and_parallel(self):
        with tempfile.TemporaryDirectory() as directory:
            bundle = self.make_bundle(directory, with_evidence=True)
            backend = StructuredFakeBackend()
            run = FastReviewerOrchestrator(backend, attempts=1).run_base(bundle)
            counts = {}
            for request in backend.requests:
                counts[request.stage] = counts.get(request.stage, 0) + 1
            self.assertEqual(
                counts,
                {"extraction": 2, "consolidated_review": 2, "final_review": 1},
            )
            self.assertEqual(run.logical_call_count, 5)
            self.assertEqual(len(run.context), 19)
            self.assertEqual(set(run.critiques), {"review.technical", "review.contribution"})
            self.assertEqual(backend.max_active, 2)
            self.assertIsInstance(run.effective_review, ReviewOutput)
            self.assertIn("### Summary", run.effective_review.comment)
            serialized = fast_run_as_dict(run)
            self.assertNotIn("accuracy", json.dumps(serialized))
            self.assertEqual(len(serialized["context"]["evidence"]), 19)
            self.assertTrue(serialized["chair_selected_evidence_ids"])
            extraction_requests = [
                request for request in backend.requests if request.stage == "extraction"
            ]
            self.assertTrue(
                all("provided_evidence" not in request.payload for request in extraction_requests)
            )
            self.assertTrue(
                all(
                    item["provenance"]["source_type"] == "track2-paper"
                    for item in serialized["context"]["evidence"]
                )
            )

    def test_conditional_refinement_adds_exactly_two_calls(self):
        with tempfile.TemporaryDirectory() as directory:
            bundle = self.make_bundle(directory)
            backend = StructuredFakeBackend(needs_refinement=True)
            run = FastReviewerOrchestrator(backend, attempts=1).review(
                bundle, author_loop="conditional"
            )
            self.assertEqual(run.logical_call_count, 7)
            self.assertEqual(run.refinement_status, "completed")
            self.assertIsNotNone(run.author_response)
            self.assertIsNotNone(run.final_chair)
            self.assertEqual(
                sum(request.stage == "author_rebuttal" for request in backend.requests),
                1,
            )
            self.assertEqual(
                sum(request.stage == "final_review" for request in backend.requests),
                2,
            )
            final_request = [
                request for request in backend.requests if request.stage == "final_review"
            ][-1]
            self.assertIn("internal Track 2 simulation", final_request.system)

    def test_contradiction_trigger_is_visible_to_refinement_author(self):
        with tempfile.TemporaryDirectory() as directory:
            bundle = self.make_bundle(directory)
            backend = StructuredFakeBackend(
                needs_refinement=False, contradiction=True
            )
            run = FastReviewerOrchestrator(backend, attempts=1).review(
                bundle, author_loop="conditional"
            )
            self.assertEqual(run.logical_call_count, 7)
            author_request = next(
                request
                for request in backend.requests
                if request.stage == "author_rebuttal"
            )
            contradictions = author_request.payload["unresolved_contradictions"]
            self.assertEqual(len(contradictions), 2)
            self.assertTrue(all(item["contradiction_id"] for item in contradictions))
            self.assertEqual(
                {item["reviewer_id"] for item in contradictions},
                {"review.technical", "review.contribution"},
            )
            self.assertEqual(
                set(run.author_response.addressed_contradiction_ids),
                {item["contradiction_id"] for item in contradictions},
            )

    def test_conditional_refinement_skips_clear_high_confidence_case(self):
        with tempfile.TemporaryDirectory() as directory:
            bundle = self.make_bundle(directory)
            backend = StructuredFakeBackend(needs_refinement=False)
            run = FastReviewerOrchestrator(backend, attempts=1).review(
                bundle, author_loop="conditional"
            )
            self.assertEqual(run.logical_call_count, 5)
            self.assertEqual(run.refinement_status, "not_needed")

    def test_live_memory_is_forum_diverse_and_raw_text_is_not_sent(self):
        paper = PaperInput("new", "Vision Language Evaluation", "vision language evaluation")
        state = LearningState(
            reviewer_memory=(
                MemoryItem("reviewer", "f1", "RAW SECRET ONE", "Vision Language\nmetric audit"),
                MemoryItem("reviewer", "f1", "RAW SECRET TWO", "Different cue\nmetric audit"),
                MemoryItem("reviewer", "f2", "RAW SECRET THREE", "Vision Language\nmetric audit"),
                MemoryItem("reviewer", "f3", "Check ablations.", "Ablation Study\nvision"),
            )
        )
        candidates = retrieve_live_candidates(state, "reviewer", paper, limit=8)
        self.assertEqual(len(candidates), 3)
        payload = json.dumps([item.as_dict() for item in candidates])
        self.assertNotIn("RAW SECRET", payload)
        self.assertEqual(len({item.candidate_id for item in candidates}), 3)


if __name__ == "__main__":
    unittest.main()
