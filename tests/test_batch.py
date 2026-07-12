import json
import os
from pathlib import Path
import tempfile
import threading
import time
import unittest

from ralphton_icml.backend import BackendError, ModelRequest
from ralphton_icml.batch import (
    BatchManifestError,
    BatchReviewScheduler,
    prepare_batch,
)


class BatchFakeBackend:
    def __init__(
        self,
        expected_base=1,
        fail_papers=(),
        confidence=None,
        severity=None,
        require_paper_overlap=False,
        block=False,
    ):
        self.expected_base = expected_base
        self.fail_papers = set(fail_papers)
        self.confidence = dict(confidence or {})
        self.severity = dict(severity or {})
        self.require_paper_overlap = require_paper_overlap
        self.block = block
        self.requests = []
        self.active = 0
        self.max_active = 0
        self.base_chairs = set()
        self.author_papers = []
        self.author_before_all_base = False
        self.extraction_papers = set()
        self.paper_overlap = threading.Event()
        self.cancelled = threading.Event()
        self.lock = threading.Lock()

    @staticmethod
    def paper_id(request):
        paper = request.payload.get("paper", {})
        return paper.get("paper_id", "")

    def complete(self, request: ModelRequest) -> str:
        paper_id = self.paper_id(request)
        with self.lock:
            self.requests.append(request)
            self.active += 1
            self.max_active = max(self.max_active, self.active)
            if request.stage == "extraction":
                self.extraction_papers.add(paper_id)
                if len(self.extraction_papers) >= 2:
                    self.paper_overlap.set()
            if request.stage == "author_rebuttal":
                self.author_papers.append(paper_id)
                if len(self.base_chairs) < self.expected_base:
                    self.author_before_all_base = True
        try:
            if self.block:
                self.cancelled.wait(timeout=3.0)
                raise BackendError("cancelled blocked fake call")
            if self.require_paper_overlap and request.stage == "extraction":
                if not self.paper_overlap.wait(timeout=2.0):
                    raise BackendError("papers did not overlap")
            if paper_id in self.fail_papers and request.stage == "extraction":
                raise BackendError("injected paper failure")
            response = self.response(request, paper_id)
            if request.stage == "final_review" and "simulated_author_response" not in request.payload:
                with self.lock:
                    self.base_chairs.add(paper_id)
            return json.dumps(response, ensure_ascii=False)
        finally:
            with self.lock:
                self.active -= 1

    def response(self, request, paper_id):
        if request.stage == "extraction":
            return {
                "items": [
                    {
                        "task_id": task["task_id"],
                        "answer": "Evidence for {}.".format(task["task_id"]),
                        "sources": ["paper, p. 1"],
                    }
                    for task in request.payload["tasks"]
                ]
            }
        if request.stage == "consolidated_review":
            evidence_id = request.payload["context"]["evidence"][0]["evidence_id"]
            criterion = request.payload["criteria"][0]
            return {
                "strengths": [
                    {
                        "criterion": criterion,
                        "severity": "positive",
                        "text": "The framing is clear.",
                        "evidence_ids": [evidence_id],
                    }
                ],
                "weaknesses": [
                    {
                        "criterion": criterion,
                        "severity": self.severity.get(paper_id, "major"),
                        "text": "A controlled comparison is missing.",
                        "evidence_ids": [evidence_id],
                    }
                ],
                "questions": [],
                "memory_candidate_ids_used": [],
                "unresolved_contradictions": [],
            }
        if request.stage == "final_review":
            post = "simulated_author_response" in request.payload
            return {
                "soundness": 2,
                "presentation": 3,
                "significance": 3,
                "originality": 3,
                "overall_recommendation": 2,
                "confidence": self.confidence.get(paper_id, 3),
                "summary": "The paper studies a useful problem.",
                "strengths": ["The motivation is clear."],
                "weaknesses": ["Validation is incomplete."],
                "questions_for_authors": ["Can existing evidence resolve this issue?"],
                "contribution": "The framing is useful.",
                "ethics_and_limitations": "Evidence remains limited.",
                "ai_agent_improvements": ["Check controlled comparisons."],
                "needs_refinement": not post,
                "refinement_reasons": ([] if post else ["A major issue remains."]),
            }
        if request.stage == "author_rebuttal":
            findings = request.payload["findings"]
            return {
                "response": "The simulated author concedes the unsupported claim.",
                "addressed_finding_ids": [findings[0]["finding_id"]] if findings else [],
                "addressed_contradiction_ids": [
                    item["contradiction_id"]
                    for item in request.payload["unresolved_contradictions"]
                ],
                "memory_candidate_ids_used": [],
            }
        raise AssertionError(request.stage)

    def cancel_all(self):
        self.cancelled.set()


class BatchTest(unittest.TestCase):
    def make_manifest(self, root, count=1, object_first=False):
        root = Path(root)
        inputs = root / "inputs"
        evidence = root / "evidence"
        inputs.mkdir()
        evidence.mkdir()
        papers = []
        for index in range(count):
            paper = inputs / "p{}.md".format(index)
            paper.write_text(
                "# Paper {}\n\nWe propose a method and report experiments.\n".format(index),
                encoding="utf-8",
            )
            if index == 0 and object_first:
                result = evidence / "results.json"
                result.write_text('{"score": 0.9}\n', encoding="utf-8")
                papers.append(
                    {
                        "paper": "inputs/{}".format(paper.name),
                        "evidence": ["evidence/results.json"],
                        "evidence_ids": ["evidence.results"],
                        "agent_name": "Batch Reviewer",
                        "agent_version": "fast-v1",
                        "result_filename": "custom-review.md",
                    }
                )
            else:
                papers.append("inputs/{}".format(paper.name))
        manifest = root / "papers.json"
        manifest.write_text(json.dumps({"papers": papers}), encoding="utf-8")
        return manifest

    def test_manifest_materialization_base_first_and_priority(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = self.make_manifest(root, count=3, object_first=True)
            output = root / "batch-output"
            prepared = prepare_batch(manifest, output)
            self.assertEqual(len(prepared.papers), 3)
            first = prepared.papers[0]
            self.assertEqual(first.bundle.evidence[0].evidence_id, "evidence.results")
            self.assertEqual(
                first.bundle.paper.path.read_bytes(),
                (root / "inputs" / "p0.md").read_bytes(),
            )
            self.assertTrue(first.review_agent_path.is_file())
            self.assertEqual(first.result_markdown_path.name, "custom-review.md")

            backend = BatchFakeBackend(
                expected_base=3,
                confidence={"p0": 4, "p1": 1, "p2": 2},
                severity={"p0": "major", "p1": "major", "p2": "fatal"},
                require_paper_overlap=True,
            )
            result = BatchReviewScheduler(
                backend,
                attempts=1,
                paper_workers=3,
                max_refinements=2,
            ).run(manifest, output)

            self.assertEqual(result.exit_code, 0)
            self.assertEqual(result.summary["calls"], 19)
            self.assertTrue(backend.paper_overlap.is_set())
            self.assertFalse(backend.author_before_all_base)
            self.assertEqual(set(backend.author_papers), {"p1", "p2"})
            statuses = {item.paper_id: item.status for item in result.outcomes}
            self.assertEqual(statuses["p0"], "not_selected")
            self.assertEqual(statuses["p1"], "completed")
            self.assertEqual(statuses["p2"], "completed")
            for paper in prepared.papers:
                self.assertTrue(paper.result_json_path.is_file())
                self.assertTrue(paper.result_markdown_path.is_file())
                self.assertTrue(paper.completion_path.is_file())
                self.assertEqual(list(paper.root.rglob("*.tmp")), [])

    def test_resume_zero_calls_and_stale_or_corrupt_reruns(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = self.make_manifest(root)
            output = root / "out"
            first_backend = BatchFakeBackend()
            first = BatchReviewScheduler(
                first_backend, attempts=1, author_loop="never"
            ).run(manifest, output)
            self.assertEqual(first.summary["calls"], 5)

            resumed_backend = BatchFakeBackend()
            resumed = BatchReviewScheduler(
                resumed_backend, attempts=1, author_loop="never"
            ).run(manifest, output, resume=True)
            self.assertEqual(len(resumed_backend.requests), 0)
            self.assertEqual(resumed.summary["papers_resumed"], 1)

            result_json = output / "p0" / "outputs" / "review-result.json"
            result_json.write_text("corrupt", encoding="utf-8")
            corrupt_backend = BatchFakeBackend()
            repaired = BatchReviewScheduler(
                corrupt_backend, attempts=1, author_loop="never"
            ).run(manifest, output, resume=True)
            self.assertEqual(repaired.summary["calls"], 5)

            (root / "inputs" / "p0.md").write_text(
                "# Paper 0\n\nChanged immutable input.\n", encoding="utf-8"
            )
            stale_backend = BatchFakeBackend()
            stale = BatchReviewScheduler(
                stale_backend, attempts=1, author_loop="never"
            ).run(manifest, output, resume=True)
            self.assertEqual(stale.summary["calls"], 5)

    def test_one_paper_failure_preserves_other_result(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = self.make_manifest(root, count=2)
            backend = BatchFakeBackend(fail_papers={"p0"})
            result = BatchReviewScheduler(
                backend, attempts=1, author_loop="never", paper_workers=2
            ).run(manifest, root / "out")
            statuses = {item.paper_id: item.status for item in result.outcomes}
            self.assertEqual(statuses, {"p0": "failed", "p1": "disabled"})
            self.assertEqual(result.exit_code, 1)
            self.assertTrue(
                (root / "out" / "p1" / "outputs" / "review-result.complete.json").is_file()
            )
            self.assertFalse(
                (root / "out" / "p0" / "outputs" / "review-result.complete.json").exists()
            )

    def test_preflight_rejects_duplicate_traversal_and_symlink_escape(self):
        with tempfile.TemporaryDirectory() as directory, tempfile.TemporaryDirectory() as outside:
            root = Path(directory)
            (root / "inputs").mkdir()
            paper = root / "inputs" / "paper.json"
            paper.write_text(
                json.dumps({"paper_id": "duplicate", "title": "T", "text": "Body"}),
                encoding="utf-8",
            )
            second = root / "inputs" / "second.json"
            second.write_bytes(paper.read_bytes())
            duplicate = root / "duplicate.json"
            duplicate.write_text(
                json.dumps({"papers": ["inputs/paper.json", "inputs/second.json"]}),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(BatchManifestError, "duplicate paper_id"):
                prepare_batch(duplicate, root / "out-duplicate")

            traversal = root / "traversal.json"
            traversal.write_text(json.dumps({"papers": ["../outside.md"]}), encoding="utf-8")
            with self.assertRaisesRegex(BatchManifestError, "traversal"):
                prepare_batch(traversal, root / "out-traversal")

            outside_paper = Path(outside) / "outside.md"
            outside_paper.write_text("# Outside\n\nBody\n", encoding="utf-8")
            link = root / "inputs" / "link.md"
            link.symlink_to(outside_paper)
            escaped = root / "escaped.json"
            escaped.write_text(json.dumps({"papers": ["inputs/link.md"]}), encoding="utf-8")
            with self.assertRaisesRegex(BatchManifestError, "escapes"):
                prepare_batch(escaped, root / "out-escaped")

            unsafe_result = root / "unsafe-result.json"
            unsafe_result.write_text(
                json.dumps(
                    {"papers": [{"paper": "inputs/paper.json", "result_filename": "../bad.md"}]}
                ),
                encoding="utf-8",
            )
            with self.assertRaises(BatchManifestError):
                prepare_batch(unsafe_result, root / "out-result")

            output = root / "out-symlink"
            output.mkdir()
            (output / "duplicate").symlink_to(Path(outside), target_is_directory=True)
            single = root / "single.json"
            single.write_text(json.dumps({"papers": ["inputs/paper.json"]}), encoding="utf-8")
            with self.assertRaisesRegex(BatchManifestError, "escapes|symlink"):
                prepare_batch(single, output)

    def test_soft_deadline_preserves_base_and_hard_deadline_cancels(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = self.make_manifest(root)
            soft_backend = BatchFakeBackend()
            soft = BatchReviewScheduler(
                soft_backend,
                attempts=1,
                max_refinements=1,
                deadline_seconds=10,
                soft_deadline_seconds=0,
            ).run(manifest, root / "soft")
            self.assertEqual(soft.summary["calls"], 5)
            self.assertEqual(soft.outcomes[0].status, "soft_deadline")
            self.assertTrue(soft.outcomes[0].has_valid_review)

            blocked_backend = BatchFakeBackend(block=True)
            began = time.monotonic()
            hard = BatchReviewScheduler(
                blocked_backend,
                attempts=1,
                deadline_seconds=0.15,
                soft_deadline_seconds=0.1,
            ).run(manifest, root / "hard")
            self.assertLess(time.monotonic() - began, 2.0)
            self.assertTrue(blocked_backend.cancelled.is_set())
            self.assertTrue(hard.summary["hard_deadline_reached"])
            self.assertEqual(hard.exit_code, 1)
            self.assertEqual(hard.outcomes[0].status, "hard_deadline")


if __name__ == "__main__":
    unittest.main()
