from dataclasses import FrozenInstanceError
import json
from pathlib import Path
import tempfile
import unittest

from ralphton_icml.cli import _load_compatible_learning_state
from ralphton_icml.context import (
    CONTEXT_TASKS,
    ExtractionParseError,
    SharedContextStore,
    parse_extraction_output,
)
from ralphton_icml.schema import (
    ContextPacket,
    Evidence,
    ExtractionOutput,
    Provenance,
    ReviewOutput,
    ReviewParseError,
    ReviewValidationError,
)
from ralphton_icml.learning import LearningState, dump_learning_state
from ralphton_icml.team import (
    AUTHOR_AGENT,
    CHAIR_AGENT,
    CRITERION_EXPERTS,
    DEFAULT_REVIEWER_TEAM,
    DOMAIN_EXPERTS,
    Domain,
    EXTRACTION_STAGE_CONTRACT,
    FINAL_REVIEW_STAGE_CONTRACT,
    OutputContract,
    route_domain_expert,
    route_domain_experts,
)


class ReviewOutputTest(unittest.TestCase):
    def make_review(self, **overrides):
        values = {
            "soundness": 4,
            "presentation": 3,
            "significance": 2,
            "originality": 1,
            "overall_recommendation": 5,
            "confidence": 4,
            "comment": "The evidence is clear. Add the missing ablation.",
        }
        values.update(overrides)
        return ReviewOutput(**values)

    def test_markdown_round_trip_uses_exact_form_headings(self):
        review = self.make_review()
        markdown = review.to_markdown()
        self.assertEqual(review, ReviewOutput.from_markdown(markdown))
        self.assertEqual(
            [line for line in markdown.splitlines() if line.startswith("####")],
            [
                "#### **Soundness**",
                "#### **Presentation**",
                "#### **Significance**",
                "#### **Originality**",
                "#### **Overall Recommendation**",
                "#### **Confidence**",
                "#### Comment",
            ],
        )

    def test_strict_score_ranges_and_types(self):
        for field, bad_value in (
            ("soundness", 0),
            ("presentation", 5),
            ("significance", True),
            ("originality", "4"),
            ("overall_recommendation", 7),
            ("confidence", 0),
        ):
            with self.subTest(field=field):
                with self.assertRaises(ReviewValidationError):
                    self.make_review(**{field: bad_value})

    def test_comment_is_required_and_canonicalized(self):
        with self.assertRaises(ReviewValidationError):
            self.make_review(comment=" \n ")
        self.assertEqual(self.make_review(comment="  useful  ").comment, "useful")

    def test_parser_rejects_extraction_or_noncanonical_form(self):
        with self.assertRaises(ReviewParseError):
            ReviewOutput.from_markdown("ANSWER\nreview\nSOURCES\np. 1")
        markdown = self.make_review().to_markdown().replace(
            "#### **Soundness**", "## Soundness", 1
        )
        with self.assertRaises(ReviewParseError):
            ReviewOutput.from_markdown(markdown)
        out_of_range = self.make_review().to_markdown().replace(
            "#### **Soundness**\n\n4", "#### **Soundness**\n\n5", 1
        )
        with self.assertRaises(ReviewParseError):
            ReviewOutput.from_markdown(out_of_range)


class ContextTest(unittest.TestCase):
    def setUp(self):
        self.task = CONTEXT_TASKS[0]
        self.provenance = Provenance(
            paper_id="paper-1",
            document_id="main.pdf",
            agent_id="extractor-1",
            source_uri="file:///papers/main.pdf",
        )

    def test_all_19_general_prompt_leaves_are_exposed(self):
        self.assertEqual(len(CONTEXT_TASKS), 19)
        self.assertEqual(len({task.task_id for task in CONTEXT_TASKS}), 19)
        self.assertEqual(CONTEXT_TASKS[0].task_id, "Main Paper/Paper Summary")
        self.assertEqual(CONTEXT_TASKS[-1].task_id, "Conclusion/Conclusion")
        self.assertTrue(all(task.tier in "ABCD" for task in CONTEXT_TASKS))
        with self.assertRaises(FrozenInstanceError):
            CONTEXT_TASKS[0].tier = "D"

    def test_answer_sources_parser(self):
        parsed = parse_extraction_output(
            "## ANSWER\nA supported claim (p. 2).\n\n"
            "## SOURCES\n- Main paper, p. 2\n- Supplemental, p. 4\n"
        )
        self.assertEqual(parsed.answer, "A supported claim (p. 2).")
        self.assertEqual(
            parsed.sources, ("Main paper, p. 2", "Supplemental, p. 4")
        )
        bold_colon = parse_extraction_output(
            "**ANSWER:**\nSupported.\n**SOURCES:**\n- p. 3"
        )
        self.assertEqual(bold_colon.sources, ("p. 3",))
        with self.assertRaises(ExtractionParseError):
            parse_extraction_output("SOURCES\np. 1\nANSWER\nclaim")

    def test_shared_store_is_append_only_versioned_and_deduplicated(self):
        store = SharedContextStore()
        packet = store.merge_extraction(
            self.task,
            "ANSWER\nClaim one.\nSOURCES\n- p. 1",
            self.provenance,
        )
        self.assertIsInstance(packet, ContextPacket)
        self.assertEqual((packet.revision, len(packet)), (1, 1))
        same = store.merge_extraction(
            self.task,
            ExtractionOutput("Claim one.", ("p. 1",)),
            self.provenance,
        )
        self.assertEqual((same.revision, len(same)), (1, 1))

        updated_provenance = Provenance(
            paper_id="paper-1",
            document_id="main.pdf",
            agent_id="extractor-1",
            iteration=1,
        )
        updated = store.merge_extraction(
            self.task,
            "**ANSWER**\nClaim two.\n**SOURCES**\n- p. 2",
            updated_provenance,
        )
        self.assertEqual((updated.revision, len(updated)), (2, 2))
        self.assertEqual(updated.latest_for_task(self.task.task_id).answer, "Claim two.")
        self.assertEqual((packet.revision, len(packet)), (1, 1))
        with self.assertRaises(FrozenInstanceError):
            updated.evidence[0].answer = "mutated"

    def test_merge_many_rejects_in_transaction_id_collision(self):
        first = Evidence(
            task_id=self.task.task_id,
            answer="First",
            sources=(),
            provenance=self.provenance,
            evidence_id="manually-assigned-id",
        )
        second = Evidence(
            task_id=self.task.task_id,
            answer="Different content",
            sources=(),
            provenance=self.provenance,
            evidence_id="manually-assigned-id",
        )
        with self.assertRaises(ValueError):
            store = SharedContextStore()
            store.merge_many((first, second))
        self.assertEqual(store.paper_ids(), ())


class TeamTest(unittest.TestCase):
    def test_required_team_members_and_contract_separation(self):
        self.assertEqual(
            {spec.domain for spec in DOMAIN_EXPERTS},
            {Domain.CV, Domain.CORE_ML, Domain.NLP, Domain.RECSYS, Domain.GENERAL},
        )
        self.assertEqual(len(CRITERION_EXPERTS), 6)
        self.assertEqual(DEFAULT_REVIEWER_TEAM.author, AUTHOR_AGENT)
        self.assertEqual(DEFAULT_REVIEWER_TEAM.chair, CHAIR_AGENT)
        self.assertIs(EXTRACTION_STAGE_CONTRACT.output_type, ExtractionOutput)
        self.assertIs(FINAL_REVIEW_STAGE_CONTRACT.output_type, ReviewOutput)
        self.assertEqual(
            EXTRACTION_STAGE_CONTRACT.output_contract,
            OutputContract.ANSWER_SOURCES,
        )
        self.assertEqual(
            FINAL_REVIEW_STAGE_CONTRACT.output_contract,
            OutputContract.REVIEW_FORM,
        )

    def test_keyword_routing_and_general_fallback(self):
        self.assertEqual(
            route_domain_expert("We study image segmentation on video.").domain,
            Domain.CV,
        )
        self.assertEqual(
            route_domain_expert("Collaborative filtering for recommendation.").domain,
            Domain.RECSYS,
        )
        routed = route_domain_experts(
            "A language model recommends text to users through collaborative filtering."
        )
        self.assertIn(Domain.NLP, {spec.domain for spec in routed})
        self.assertIn(Domain.RECSYS, {spec.domain for spec in routed})
        self.assertEqual(
            DEFAULT_REVIEWER_TEAM.route_domains(
                "image image image and a language model", max_experts=2
            )[0].domain,
            Domain.CV,
        )
        self.assertEqual(route_domain_expert("A specialized scientific study.").domain, Domain.GENERAL)


class CliStateCompatibilityTest(unittest.TestCase):
    ROOT = Path(__file__).resolve().parent.parent

    def test_current_artifact_state_matches_source_manifest(self):
        state = _load_compatible_learning_state(
            str(self.ROOT / "artifacts" / "real_seed_v1" / "best_state.json"),
            self.ROOT,
        )
        self.assertEqual(state.version, 1)

        preview = (
            self.ROOT / "artifacts" / "real_seed_v1" / "dev_review_preview.md"
        ).read_text(encoding="utf-8")
        self.assertEqual(ReviewOutput.from_markdown(preview).to_markdown(), preview)

    def test_stale_or_unsigned_state_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            stale_path = Path(directory) / "stale.json"
            dump_learning_state(
                LearningState(prompt_manifest_digest="0" * 64), stale_path
            )
            with self.assertRaisesRegex(ValueError, "prompt manifest mismatch"):
                _load_compatible_learning_state(str(stale_path), self.ROOT)

            unsigned_path = Path(directory) / "unsigned.json"
            payload = json.loads(stale_path.read_text(encoding="utf-8"))
            payload.pop("digest")
            unsigned_path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "integrity digest"):
                _load_compatible_learning_state(str(unsigned_path), self.ROOT)


if __name__ == "__main__":
    unittest.main()
