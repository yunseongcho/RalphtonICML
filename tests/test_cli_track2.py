import unittest

from ralphton_icml.cli import build_parser


class Track2CliTest(unittest.TestCase):
    def test_single_review_defaults_to_fast_codex(self):
        args = build_parser().parse_args(
            ["review", "track2/inputs/paper.pdf", "--output", "run.json"]
        )
        self.assertEqual(args.pipeline, "fast-v1")
        self.assertEqual(args.backend, "codex")
        self.assertEqual(args.author_loop, "conditional")
        self.assertEqual(args.attempts, 2)

    def test_batch_contract_exposes_bounded_scheduler_options(self):
        args = build_parser().parse_args(
            [
                "review-batch",
                "papers.manifest.json",
                "--model",
                "gpt-5.6-sol",
                "--output-dir",
                "artifacts/batch",
                "--resume",
            ]
        )
        self.assertEqual(args.pipeline, "fast-v1")
        self.assertEqual(args.backend, "codex")
        self.assertEqual(args.paper_workers, 4)
        self.assertEqual(args.codex_concurrency, 4)
        self.assertEqual(args.max_refinements, 2)
        self.assertEqual(args.deadline_seconds, 1800.0)
        self.assertEqual(args.soft_deadline_seconds, 1440.0)
        self.assertTrue(args.resume)


if __name__ == "__main__":
    unittest.main()
