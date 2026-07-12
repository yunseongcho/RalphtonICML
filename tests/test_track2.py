import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest import mock

from ralphton_icml.track2 import (
    TRACK2_COMMENT_SECTIONS,
    TRACK2_MAX_EVIDENCE_BYTES,
    TRACK2_OUTPUT_CONTRACT,
    FrozenInputFile,
    ProvidedEvidence,
    Track2AgentManifest,
    Track2InputError,
    Track2IntegrityError,
    create_track2_bundle,
    load_track2_bundle,
    render_review_agent,
    write_review_agent,
)


class Track2InputTest(unittest.TestCase):
    def _layout(self, root: Path) -> None:
        (root / "inputs").mkdir()
        (root / "evidence").mkdir()
        (root / "outputs").mkdir()

    def test_markdown_round_trip_without_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._layout(root)
            paper = root / "inputs" / "paper.md"
            paper.write_text("# Frozen Paper\n\nA supported claim.\n", encoding="utf-8")

            bundle = create_track2_bundle(
                root,
                "inputs/paper.md",
                agent_name="Ralphton Track 2 Reviewer",
                agent_version="fast-v1",
            )
            self.assertFalse(bundle.has_evidence)
            self.assertEqual(bundle.title, "Frozen Paper")
            self.assertEqual(bundle.agent_manifest.output_contract, TRACK2_OUTPUT_CONTRACT)
            self.assertEqual(
                bundle.agent_manifest.comment_sections, TRACK2_COMMENT_SECTIONS
            )
            rendered = render_review_agent(bundle)
            self.assertIn("evidence-insufficient", rendered)
            self.assertIn("do not modify the paper", rendered)
            self.assertIn("Do not invent", rendered)
            self.assertIn("Inside the `Comment` field", rendered)

            agent_path = write_review_agent(bundle)
            loaded = load_track2_bundle(agent_path)
            self.assertEqual(loaded.bundle_digest, bundle.bundle_digest)
            self.assertEqual(loaded.as_paper_mapping(), bundle.as_paper_mapping())
            loaded.verify_frozen_inputs()

    def test_json_paper_compatibility_and_utf8_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._layout(root)
            paper = root / "inputs" / "paper.json"
            paper.write_text(
                json.dumps(
                    {
                        "forum_id": "forum-1",
                        "paper": {
                            "title": "JSON Paper",
                            "text": "Method and results.",
                            "document_id": "submission-v1",
                        },
                    }
                ),
                encoding="utf-8",
            )
            evidence = root / "evidence" / "results.json"
            evidence.write_text(
                json.dumps({"accuracy": 0.91}, ensure_ascii=False), encoding="utf-8"
            )
            bundle = create_track2_bundle(
                root,
                paper,
                evidence_paths=(evidence,),
                evidence_ids=("experiment.results",),
                agent_name="Evidence Reviewer",
                result_filename="paper-review.md",
            )
            self.assertEqual(bundle.paper_id, "forum-1")
            self.assertEqual(bundle.document_id, "submission-v1")
            self.assertEqual(len(bundle.evidence), 1)
            item = bundle.evidence[0]
            self.assertIsInstance(item, ProvidedEvidence)
            self.assertEqual(item.evidence_id, "experiment.results")
            self.assertEqual(item.content_sha256, item.sha256)
            self.assertEqual(
                bundle.result_path,
                (root / "outputs" / "paper-review.md").resolve(),
            )
            self.assertTrue(item.available)
            self.assertEqual(item.text, item.content)
            self.assertEqual(item.as_payload()["evidence_id"], "experiment.results")
            self.assertEqual(item.as_payload()["content"], item.content)
            self.assertEqual(bundle.evidence_files, (item,))
            self.assertEqual(bundle.frozen_evidence_files, (item.file,))
            self.assertEqual(bundle.paper_input.paper_id, "forum-1")

            loaded = load_track2_bundle(write_review_agent(bundle))
            self.assertEqual(loaded.evidence[0].content, item.content)
            self.assertEqual(loaded.bundle_digest, bundle.bundle_digest)

    def test_frozen_input_detects_mutation_at_execution_gate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._layout(root)
            paper = root / "inputs" / "paper.md"
            paper.write_text("# Original\n", encoding="utf-8")
            evidence = root / "evidence" / "run.log"
            evidence.write_text("loss=0.25\n", encoding="utf-8")
            bundle = create_track2_bundle(
                root,
                paper,
                evidence_paths=(evidence,),
                agent_name="Integrity Reviewer",
            )
            bundle.verify_frozen_inputs()
            bundle.verify_unchanged()
            evidence.write_text("loss=0.01\n", encoding="utf-8")
            with self.assertRaisesRegex(Track2IntegrityError, "frozen input changed"):
                bundle.revalidate()

    def test_manifest_file_is_also_frozen_after_load(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._layout(root)
            paper = root / "inputs" / "paper.md"
            paper.write_text("# Paper\n", encoding="utf-8")
            bundle = create_track2_bundle(root, paper, agent_name="Reviewer")
            path = write_review_agent(bundle)
            loaded = load_track2_bundle(path)
            path.write_text(path.read_text(encoding="utf-8") + "\nchanged\n", encoding="utf-8")
            with self.assertRaisesRegex(Track2IntegrityError, "frozen input changed"):
                loaded.verify_frozen_inputs()

    def test_unsafe_result_and_input_paths_are_rejected(self):
        with self.assertRaisesRegex(Track2InputError, "unsafe path|outputs/<filename>"):
            Track2AgentManifest("Reviewer", "v1", "../review.md")
        with self.assertRaisesRegex(Track2InputError, "safe Markdown"):
            Track2AgentManifest("Reviewer", "v1", "outputs/bad name.md")

        with tempfile.TemporaryDirectory() as directory, tempfile.TemporaryDirectory() as outside:
            root = Path(directory)
            self._layout(root)
            foreign = Path(outside) / "paper.md"
            foreign.write_text("# Outside\n", encoding="utf-8")
            with self.assertRaisesRegex(Track2InputError, "must stay inside"):
                create_track2_bundle(root, foreign, agent_name="Reviewer")

    def test_pdf_records_layout_extractor_version_and_text_digest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._layout(root)
            paper = root / "inputs" / "paper.pdf"
            paper.write_bytes(b"%PDF-fake-for-unit-test")

            version = subprocess.CompletedProcess(
                ["pdftotext", "-v"], 0, stdout=b"", stderr=b"pdftotext version 25.01\n"
            )
            extraction = subprocess.CompletedProcess(
                ["pdftotext", "-layout", str(paper), "-"],
                0,
                stdout=b"PDF title\n\nExtracted body.\n",
                stderr=b"",
            )
            with mock.patch(
                "ralphton_icml.track2.subprocess.run",
                side_effect=(version, extraction, version, extraction),
            ) as run:
                bundle = create_track2_bundle(root, paper, agent_name="PDF Reviewer")
                self.assertEqual(bundle.paper_extractor, "pdftotext -layout")
                self.assertEqual(bundle.paper_extractor_version, "pdftotext version 25.01")
                self.assertEqual(len(bundle.paper_text_sha256), 64)
                self.assertEqual(bundle.title, "PDF title")
                self.assertTrue(bundle.paper_id.startswith("paper-"))
                bundle.verify_frozen_inputs()
            self.assertEqual(run.call_args_list[1].args[0][1], "-layout")

    def test_total_evidence_size_is_bounded_before_model_work(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._layout(root)
            paper = root / "inputs" / "paper.md"
            paper.write_text("# Paper\n", encoding="utf-8")
            evidence = root / "evidence" / "oversized.txt"
            evidence.write_text("x" * (TRACK2_MAX_EVIDENCE_BYTES + 1), encoding="utf-8")
            with self.assertRaisesRegex(Track2InputError, "evidence exceeds"):
                create_track2_bundle(
                    root,
                    paper,
                    evidence_paths=(evidence,),
                    agent_name="Reviewer",
                )

    def test_bundle_digest_is_portable_and_evidence_order_independent(self):
        digests = []
        for reverse in (False, True):
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                self._layout(root)
                (root / "inputs" / "paper.md").write_text("# Same\n", encoding="utf-8")
                first = root / "evidence" / "a.txt"
                second = root / "evidence" / "b.txt"
                first.write_text("a\n", encoding="utf-8")
                second.write_text("b\n", encoding="utf-8")
                pairs = [(first, "evidence.a"), (second, "evidence.b")]
                if reverse:
                    pairs.reverse()
                bundle = create_track2_bundle(
                    root,
                    "inputs/paper.md",
                    evidence_paths=tuple(pair[0] for pair in pairs),
                    evidence_ids=tuple(pair[1] for pair in pairs),
                    agent_name="Portable Reviewer",
                )
                digests.append(bundle.bundle_digest)
        self.assertEqual(digests[0], digests[1])

    def test_path_loader_accepts_track2_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._layout(root)
            (root / "inputs" / "paper.md").write_text("# Batch Entry\n", encoding="utf-8")
            bundle = create_track2_bundle(root, "inputs/paper.md", agent_name="Batch Reviewer")
            write_review_agent(bundle)
            loaded = load_track2_bundle(root)
            self.assertEqual(loaded.paper_id, "paper")

    def test_invalid_json_evidence_and_invalid_id_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._layout(root)
            (root / "inputs" / "paper.md").write_text("# Paper\n", encoding="utf-8")
            bad = root / "evidence" / "bad.json"
            bad.write_text("{not-json", encoding="utf-8")
            with self.assertRaisesRegex(Track2InputError, "JSON evidence is invalid"):
                create_track2_bundle(
                    root,
                    "inputs/paper.md",
                    evidence_paths=(bad,),
                    agent_name="Reviewer",
                )
            bad.write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(Track2InputError, "evidence_id must match"):
                create_track2_bundle(
                    root,
                    "inputs/paper.md",
                    evidence_paths=(bad,),
                    evidence_ids=("bad id",),
                    agent_name="Reviewer",
                )

    def test_frozen_input_class_verifies_raw_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "value.txt"
            path.write_text("value", encoding="utf-8")
            frozen = FrozenInputFile.snapshot(path)
            frozen.verify()
            path.write_text("other", encoding="utf-8")
            with self.assertRaises(Track2IntegrityError):
                frozen.verify()


if __name__ == "__main__":
    unittest.main()
