from pathlib import Path
import unittest

from ralphton_icml.seed import (
    load_seed_cases,
    paper_only_signals,
    seed_case_to_example,
    split_seed_examples,
)


SEED = Path(__file__).parent.parent / "data" / "real" / "seed_cases.jsonl"


class RealSeedTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cases = load_seed_cases(SEED)

    def test_twenty_public_complete_cases_and_no_author_identity(self):
        self.assertEqual(len(self.cases), 20)
        for case in self.cases:
            self.assertNotIn("authors", case["paper"])
            self.assertTrue(case["paper"]["text"])
            self.assertTrue(case["reviews"])
            self.assertTrue(case["rebuttal_dialogues"])
            self.assertTrue(case["decision"])

    def test_paper_signals_do_not_change_with_supervision(self):
        case = dict(self.cases[0])
        expected = paper_only_signals(case)
        case["reviews"] = [{"review_content": "changed", "initial_score": {"rating": 1}}]
        case["decision"] = "Reject"
        case["rebuttal_dialogues"] = [{"messages": [{"role": "user", "content": "changed"}]}]
        self.assertEqual(paper_only_signals(case), expected)

    def test_target_adapter_and_forum_split(self):
        example = seed_case_to_example(self.cases[0])
        self.assertTrue(example.target_scores)
        self.assertTrue(example.reviewer_lessons)
        self.assertTrue(example.author_lessons)
        train, dev, test, split = split_seed_examples(self.cases)
        self.assertEqual((len(train), len(dev), len(test)), (16, 2, 2))
        self.assertFalse(set(split.train).intersection(split.dev + split.test))


if __name__ == "__main__":
    unittest.main()
