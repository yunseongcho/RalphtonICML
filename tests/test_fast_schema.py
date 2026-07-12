import json
import unittest

from ralphton_icml.context import CONTEXT_TASKS
from ralphton_icml.fast_schema import (
    AuthorRefinementOutput,
    BatchedExtractionOutput,
    ChairOutput,
    ConsolidatedReviewOutput,
    FastContractError,
    TECHNICAL_CRITERIA,
    chair_json_schema,
)


def extraction_payload(tasks):
    return {
        "items": [
            {
                "task_id": task.task_id,
                "answer": "Evidence for {}".format(task.item),
                "sources": ["paper, p. {}".format(task.ordinal + 1)],
            }
            for task in tasks
        ]
    }


def finding(criterion="Soundness", evidence_id="evidence-1", severity="major"):
    return {
        "criterion": criterion,
        "severity": severity,
        "text": "The controlled comparison is missing.",
        "evidence_ids": [evidence_id],
    }


class FastExtractionContractTest(unittest.TestCase):
    def test_exact_task_set_is_restored_to_prompt_order(self):
        tasks = CONTEXT_TASKS[:10]
        payload = extraction_payload(tuple(reversed(tasks)))
        parsed = BatchedExtractionOutput.from_response(json.dumps(payload), tasks)
        self.assertEqual(
            [item.task_id for item in parsed.items],
            [task.task_id for task in tasks],
        )

    def test_missing_duplicate_unknown_and_length_are_rejected(self):
        tasks = CONTEXT_TASKS[:2]
        missing = extraction_payload(tasks)
        missing["items"].pop()
        with self.assertRaisesRegex(FastContractError, "expected 2"):
            BatchedExtractionOutput.from_response(missing, tasks)

        duplicate = extraction_payload(tasks)
        duplicate["items"][1]["task_id"] = tasks[0].task_id
        with self.assertRaisesRegex(FastContractError, "duplicate"):
            BatchedExtractionOutput.from_response(duplicate, tasks)

        unknown = extraction_payload(tasks)
        unknown["items"][1]["task_id"] = "unknown/task"
        with self.assertRaisesRegex(FastContractError, "unknown"):
            BatchedExtractionOutput.from_response(unknown, tasks)

        too_long = extraction_payload((CONTEXT_TASKS[0],))
        too_long["items"][0]["answer"] = "x" * 1201
        with self.assertRaisesRegex(FastContractError, "1200"):
            BatchedExtractionOutput.from_response(too_long, (CONTEXT_TASKS[0],))


class FastReviewerContractTest(unittest.TestCase):
    def valid_payload(self):
        return {
            "strengths": [finding(severity="positive")],
            "weaknesses": [finding()],
            "questions": [],
            "memory_candidate_ids_used": ["memory-1"],
            "unresolved_contradictions": [
                {"text": "Table values disagree.", "evidence_ids": ["evidence-1"]}
            ],
        }

    def test_known_ids_and_caps_are_enforced(self):
        parsed = ConsolidatedReviewOutput.from_response(
            self.valid_payload(),
            TECHNICAL_CRITERIA,
            ("evidence-1",),
            ("memory-1",),
        )
        self.assertEqual(len(parsed.weaknesses), 1)
        self.assertTrue(parsed.weaknesses[0].finding_id.startswith("finding."))

        unknown_evidence = self.valid_payload()
        unknown_evidence["weaknesses"][0]["evidence_ids"] = ["missing"]
        with self.assertRaisesRegex(FastContractError, "unknown IDs"):
            ConsolidatedReviewOutput.from_response(
                unknown_evidence,
                TECHNICAL_CRITERIA,
                ("evidence-1",),
                ("memory-1",),
            )

        unknown_memory = self.valid_payload()
        unknown_memory["memory_candidate_ids_used"] = ["missing"]
        with self.assertRaisesRegex(FastContractError, "unknown IDs"):
            ConsolidatedReviewOutput.from_response(
                unknown_memory,
                TECHNICAL_CRITERIA,
                ("evidence-1",),
                ("memory-1",),
            )


class FastChairContractTest(unittest.TestCase):
    def valid_payload(self):
        return {
            "soundness": 2,
            "presentation": 3,
            "significance": 3,
            "originality": 3,
            "overall_recommendation": 2,
            "confidence": 4,
            "summary": "The paper studies an important evaluation problem.",
            "strengths": ["The problem is relevant."],
            "weaknesses": ["The main metric is not validated."],
            "questions_for_authors": ["Does the conclusion survive permutation?"],
            "contribution": "The framing is useful but the validation is incomplete.",
            "ethics_and_limitations": "No direct harm; external validity is limited.",
            "ai_agent_improvements": ["Check permutation invariance automatically."],
            "needs_refinement": True,
            "refinement_reasons": ["A central soundness question is unresolved."],
        }

    def test_track2_sections_render_inside_strict_review_comment(self):
        output = ChairOutput.from_response(self.valid_payload())
        markdown = output.review.to_markdown()
        for heading in (
            "### Summary",
            "### Strengths",
            "### Weaknesses",
            "### Questions for the Authors",
            "### Contribution",
            "### Ethics and Limitations",
            "### AI Agent Improvements",
        ):
            self.assertIn(heading, markdown)
        schema = chair_json_schema()["properties"]
        self.assertEqual(schema["strengths"]["minItems"], 1)
        self.assertEqual(schema["weaknesses"]["minItems"], 1)
        self.assertEqual(schema["ai_agent_improvements"]["minItems"], 1)

    def test_bool_score_and_inconsistent_refinement_are_rejected(self):
        invalid = self.valid_payload()
        invalid["soundness"] = True
        with self.assertRaises(ValueError):
            ChairOutput.from_response(invalid)

        invalid = self.valid_payload()
        invalid["needs_refinement"] = False
        with self.assertRaisesRegex(FastContractError, "must be empty"):
            ChairOutput.from_response(invalid)

    def test_author_ids_are_validated(self):
        output = AuthorRefinementOutput.from_response(
            {
                "response": "We concede the unsupported claim.",
                "addressed_finding_ids": ["finding-1"],
                "addressed_contradiction_ids": ["contradiction-1"],
                "memory_candidate_ids_used": ["memory-1"],
            },
            ("finding-1",),
            ("contradiction-1",),
            ("memory-1",),
        )
        self.assertEqual(output.addressed_finding_ids, ("finding-1",))
        self.assertEqual(
            output.addressed_contradiction_ids, ("contradiction-1",)
        )


if __name__ == "__main__":
    unittest.main()
