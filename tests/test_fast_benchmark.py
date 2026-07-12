import hashlib
import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_ROOT = ROOT / "data" / "benchmark_fast_v1"
PAPER_KEYS = {
    "schema_version",
    "paper_id",
    "title",
    "text",
    "document_id",
    "source_uri",
    "version_status",
    "license",
}
FORBIDDEN_KEYS = {
    "reviews",
    "review_content",
    "rebuttal_dialogues",
    "metareview",
    "decision",
    "rating",
    "initial_score",
    "final_score",
}


def sha256_file(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class FastBenchmarkInputTest(unittest.TestCase):
    def test_ten_manifested_inputs_are_public_paper_only_and_hash_frozen(self):
        manifest_path = BENCHMARK_ROOT / "papers.manifest.json"
        manifest = json.loads(
            manifest_path.read_text(encoding="utf-8")
        )
        self.assertEqual(set(manifest), {"papers"})
        self.assertEqual(len(manifest["papers"]), 10)
        self.assertEqual(len(set(manifest["papers"])), 10)
        provenance = json.loads(
            (BENCHMARK_ROOT / "input_provenance.json").read_text(encoding="utf-8")
        )
        self.assertEqual(provenance["benchmark"], "track2-fast-v1-public-paper-only-10")
        self.assertEqual(
            sha256_file(manifest_path), provenance["batch_manifest_sha256"]
        )
        self.assertEqual(len(provenance["paper_files"]), 10)
        self.assertEqual(
            sha256_file(ROOT / provenance["source"]["path"]),
            provenance["source"]["sha256"],
        )

        records = {item["path"]: item for item in provenance["paper_files"]}
        for relative in manifest["papers"]:
            self.assertIn(relative, records)
            path = BENCHMARK_ROOT / relative
            paper = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(set(paper), PAPER_KEYS)
            self.assertTrue(FORBIDDEN_KEYS.isdisjoint(paper))
            self.assertEqual(sha256_file(path), records[relative]["sha256"])
            self.assertEqual(
                hashlib.sha256(paper["text"].encode("utf-8")).hexdigest(),
                records[relative]["text_sha256"],
            )
            self.assertEqual(len(paper["text"]), records[relative]["text_characters"])


if __name__ == "__main__":
    unittest.main()
