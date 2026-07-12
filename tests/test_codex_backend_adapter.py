import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest import mock


def _load_adapter():
    path = Path(__file__).parents[1] / "scripts" / "codex_backend_adapter.py"
    spec = importlib.util.spec_from_file_location("codex_backend_adapter", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


adapter = _load_adapter()


class CodexBackendAdapterTest(unittest.TestCase):
    def test_compact_final_review_is_rendered_canonically(self):
        compact = """#### **Soundness**
2
#### **Presentation**
3
#### **Significance**
3
#### **Originality**
3
#### **Overall Recommendation**
2
#### **Confidence**
4
#### Comment
The method needs a controlled validation study.
"""
        rendered = adapter._canonicalize_final_review(compact)
        self.assertIn("#### **Soundness**\n\n2\n\n", rendered)
        self.assertIn("#### Comment\n\nThe method", rendered)

    def test_semantic_markdown_headings_are_canonicalized(self):
        response = """## Soundness
2
### **Presentation**
3
# Significance
3
###### Originality
3
## Overall Recommendation
2
#### Confidence
4
## **Comment**
The central metric needs sample-level validation.
"""
        rendered = adapter._canonicalize_final_review(response)
        headings = [line for line in rendered.splitlines() if line.startswith("####")]
        self.assertEqual(
            headings,
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

    def test_decorated_score_is_rejected(self):
        invalid = """#### **Soundness**

2: fair
"""
        with self.assertRaisesRegex(ValueError, "bare ASCII integer"):
            adapter._canonicalize_final_review(invalid)

    def test_cache_key_covers_full_request(self):
        left = {
            "request_id": "same",
            "agent_id": "review.chair",
            "stage": "final_review",
            "system": "rules",
            "payload": {"paper": "one"},
        }
        right = dict(left, payload={"paper": "two"})
        self.assertNotEqual(
            adapter._request_fingerprint(left),
            adapter._request_fingerprint(right),
        )

    def test_cache_round_trip(self):
        request = {
            "request_id": "request",
            "agent_id": "extractor.0",
            "stage": "extraction",
            "system": "rules",
            "payload": {"paper": "public"},
        }
        with tempfile.TemporaryDirectory() as directory:
            with mock.patch.dict("os.environ", {"CODEX_BACKEND_CACHE": directory}):
                adapter._write_cache(request, "ANSWER\ntext\nSOURCES\np. 1")
                self.assertEqual(
                    adapter._read_cache(request),
                    "ANSWER\ntext\nSOURCES\np. 1",
                )


if __name__ == "__main__":
    unittest.main()
