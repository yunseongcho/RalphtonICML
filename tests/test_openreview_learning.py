from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
import urllib.parse
from pathlib import Path

from ralphton_icml.backend import ReplayBackend
from ralphton_icml.learning import (
    DEFAULT_PROMPT_MANIFEST_DIGEST,
    GENERIC_REVIEW_COMMENT,
    DataLeakageError,
    EvaluationResult,
    LearningConfig,
    LearningExample,
    LearningState,
    MemoryItem,
    PredictionInput,
    author_memory_context,
    behavioral_delta,
    compute_prompt_manifest_digest,
    deserialize_learning_state,
    dump_learning_state,
    evaluate_predictions,
    is_non_regression,
    load_learning_state,
    load_seed_examples,
    predict_many,
    propose_update,
    retrieve_memory,
    run_learning_loop,
    seed_case_to_learning_example,
    serialize_learning_state,
    state_delta,
)
from ralphton_icml.orchestrator import PaperInput, ReviewerOrchestrator
from ralphton_icml.openreview import (
    OpenReviewClient,
    SnapshotIntegrityError,
    classify_note,
    deterministic_forum_split,
    load_raw_snapshot,
    normalize_forum,
    normalize_score,
    snapshot_forum,
    unwrap_content_value,
    write_raw_snapshot,
)


FIXTURE = Path(__file__).parent / "fixtures" / "openreview_forum.json"
REAL_SEED = Path(__file__).parents[1] / "data" / "real" / "seed_cases.jsonl"


class _FakeResponse:
    def __init__(self, value):
        self._payload = json.dumps(value).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False

    def read(self):
        return self._payload


class OpenReviewTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.raw = json.loads(FIXTURE.read_text(encoding="utf-8"))

    def test_content_unwrap_and_score_normalization(self):
        self.assertEqual(unwrap_content_value({"value": [{"value": "x"}]}), ["x"])
        self.assertEqual(normalize_score({"value": "6: Strong Accept"}), 6.0)
        self.assertEqual(normalize_score("weak reject"), 3.0)
        self.assertIsNone(normalize_score("not provided"))
        self.assertIsNone(normalize_score(float("nan")))

    def test_forum_graph_normalization_and_completeness(self):
        forum = normalize_forum(self.raw)
        self.assertEqual(forum.forum_id, "forum-001")
        self.assertIsNotNone(forum.paper)
        self.assertEqual(forum.paper.title, "A Calibrated Multi-Agent Reviewer")
        self.assertEqual(forum.paper.authors, ("Anonymous Author 1", "Anonymous Author 2"))
        self.assertEqual(len(forum.reviews), 2)
        self.assertEqual(forum.reviews[0].overall_recommendation, 5.0)
        self.assertEqual(forum.reviews[1].soundness, 3.0)
        self.assertEqual(len(forum.rebuttals), 1)
        self.assertIn("patience sensitivity", forum.rebuttals[0].text)
        self.assertTrue(forum.decision.accepted)
        self.assertEqual(forum.completeness.fraction, 1.0)
        self.assertEqual(forum.completeness.missing, ())
        self.assertEqual(forum.unclassified_note_ids, ("public-comment-001",))

    def test_missing_data_is_explicit(self):
        root_only = {"forum_id": "forum-001", "notes": [self.raw["notes"][0]]}
        forum = normalize_forum(root_only)
        self.assertEqual(forum.completeness.fraction, 0.2)
        self.assertEqual(
            set(forum.completeness.missing),
            {"reviews", "review_scores", "rebuttal", "decision"},
        )

    def test_invitation_and_signature_classification(self):
        review = {
            "id": "r",
            "invitation": "Venue/Submission1/-/Official_Review",
            "signatures": ["Venue/Submission1/Reviewer_A"],
            "content": {"rating": {"value": "4: accept"}},
        }
        response = {
            "id": "a",
            "invitation": "Venue/Submission1/-/Author_Response",
            "signatures": ["Venue/Submission1/Authors"],
            "content": {"comment": {"value": "response"}},
        }
        reviewer_discussion = {
            "id": "discussion",
            "invitation": "Venue/Submission1/-/Official_Comment",
            "signatures": ["Venue/Submission1/Reviewer_A"],
            "content": {"comment": {"value": "Thanks for the clarification."}},
        }
        self.assertEqual(classify_note(review), "review")
        self.assertEqual(classify_note(response), "rebuttal")
        self.assertEqual(classify_note(reviewer_discussion), "other")

    def test_api2_client_paginates_and_fetches_missing_root(self):
        replies = [
            {"id": "r%d" % index, "forum": "forum-x", "content": {"rating": "3"}}
            for index in range(5)
        ]
        requested = []

        def opener(request, timeout):
            self.assertEqual(timeout, 2.0)
            parsed = urllib.parse.urlparse(request.full_url)
            query = urllib.parse.parse_qs(parsed.query)
            requested.append(query)
            if "id" in query:
                return _FakeResponse({"notes": [{"id": "forum-x", "content": {"title": "T"}}], "count": 1})
            offset = int(query["offset"][0])
            limit = int(query["limit"][0])
            return _FakeResponse({"notes": replies[offset:offset + limit], "count": len(replies)})

        client = OpenReviewClient(base_url="https://example.invalid", timeout=2, page_size=2, opener=opener)
        graph = client.fetch_forum("forum-x")
        self.assertEqual(len(graph["notes"]), 6)
        self.assertEqual(graph["notes"][0]["id"], "forum-x")
        self.assertEqual(sum("forum" in query for query in requested), 3)
        self.assertEqual(sum("id" in query for query in requested), 1)

    def test_content_addressed_snapshot_is_reproducible(self):
        with tempfile.TemporaryDirectory() as directory:
            first = write_raw_snapshot(
                self.raw,
                directory,
                retrieved_at="2026-01-01T00:00:00+00:00",
            )
            second = write_raw_snapshot(
                self.raw,
                directory,
                retrieved_at="2030-01-01T00:00:00+00:00",
            )
            self.assertEqual(first.sha256, second.sha256)
            self.assertEqual(first.snapshot_path, second.snapshot_path)
            self.assertEqual(second.retrieved_at, "2026-01-01T00:00:00+00:00")
            raw_bytes = Path(first.snapshot_path).read_bytes()
            self.assertEqual(hashlib.sha256(raw_bytes).hexdigest(), first.sha256)
            self.assertEqual(load_raw_snapshot(first.snapshot_path), self.raw)
            manifests = list(Path(directory).glob("*.manifest.json"))
            snapshots = [path for path in Path(directory).glob("*.json") if ".manifest." not in path.name]
            self.assertEqual(len(manifests), 1)
            self.assertEqual(len(snapshots), 1)
            Path(first.snapshot_path).write_bytes(b"{}")
            with self.assertRaises(SnapshotIntegrityError):
                load_raw_snapshot(first.snapshot_path)

    def test_snapshot_forum_uses_client_source(self):
        class Client:
            base_url = "https://example.invalid"

            def fetch_forum(inner_self, forum_id):
                self.assertEqual(forum_id, "forum-001")
                return self.raw

        with tempfile.TemporaryDirectory() as directory:
            snapshot = snapshot_forum(Client(), "forum-001", directory)
            manifest = json.loads(Path(snapshot.manifest_path).read_text(encoding="utf-8"))
            self.assertEqual(manifest["source_url"], "https://example.invalid")

    def test_forum_split_is_deterministic_and_disjoint(self):
        forum_ids = ["forum-%02d" % index for index in range(20)] + ["forum-03"]
        first = deterministic_forum_split(forum_ids, seed="seed")
        second = deterministic_forum_split(reversed(forum_ids), seed="seed")
        self.assertEqual(first, second)
        self.assertEqual((len(first.train), len(first.dev), len(first.test)), (16, 2, 2))
        self.assertFalse(set(first.train).intersection(first.dev))
        self.assertFalse(set(first.train).intersection(first.test))
        self.assertFalse(set(first.dev).intersection(first.test))
        self.assertEqual(len(set(first.train + first.dev + first.test)), 20)


def _example(
    forum_id,
    signal,
    target,
    accepted,
    lessons=True,
):
    return LearningExample(
        forum_id=forum_id,
        paper_signals=tuple((name, signal) for name in (
            "soundness", "presentation", "significance", "originality"
        )),
        target_scores=(("overall_recommendation", target),),
        accepted=accepted,
        reviewer_lessons=("Check unsupported claims for %s." % forum_id,) if lessons else (),
        author_lessons=("Answer the main concern for %s." % forum_id,) if lessons else (),
        retrieval_text="calibration evidence %s" % forum_id,
    )


class LearningTests(unittest.TestCase):
    def test_forum_record_conversion_keeps_targets_out_of_inputs(self):
        raw = json.loads(FIXTURE.read_text(encoding="utf-8"))
        forum = normalize_forum(raw)
        signals = tuple((name, 2.5) for name in (
            "soundness", "presentation", "significance", "originality"
        ))
        example = LearningExample.from_forum_record(forum, signals)
        self.assertEqual(example.paper_signals, tuple(sorted(signals)))
        self.assertIn(("overall_recommendation", 4.5), example.target_scores)
        self.assertTrue(example.accepted)
        self.assertTrue(example.reviewer_lessons)
        self.assertTrue(example.author_lessons)

    def test_evaluation_reports_mae_brier_and_schema_coverage(self):
        example = _example("dev-1", 3.0, 4.0, True, lessons=False)
        prediction = {
            "soundness": 3,
            "presentation": 3,
            "significance": 3,
            "originality": 3,
            "overall_recommendation": 4,
            "confidence": 3,
            "comment": "Grounded critique.",
            "accept_probability": 0.8,
        }
        result = evaluate_predictions({"dev-1": prediction}, (example,))
        self.assertEqual(result.field_coverage, 1.0)
        self.assertEqual(result.complete_coverage, 1.0)
        self.assertEqual(result.mae, 0.0)
        self.assertAlmostEqual(result.brier, 0.04)

        missing = evaluate_predictions({"dev-1": {"soundness": 9}}, (example,))
        self.assertEqual(missing.field_coverage, 0.0)
        self.assertEqual(missing.complete_coverage, 0.0)
        self.assertIsNone(missing.mae)

    def test_behavioral_and_state_delta_ignore_version_only_changes(self):
        output = {
            "x": {
                "soundness": 3,
                "presentation": 3,
                "significance": 3,
                "originality": 3,
                "overall_recommendation": 4,
                "confidence": 3,
                "comment": "same",
                "accept_probability": 0.6,
            }
        }
        self.assertEqual(behavioral_delta(output, output), 0.0)
        left = LearningState()
        right = LearningState(version=1, parent_digest=left.digest)
        self.assertEqual(state_delta(left, right), 0.0)

        changed_manifest = LearningState(prompt_manifest_digest=compute_prompt_manifest_digest({
            "prompts": "v2", "schema": "v1", "team": "v1"
        }))
        self.assertAlmostEqual(state_delta(left, changed_manifest), 0.2)

    def test_prompt_manifest_is_preserved_and_state_json_is_integrity_checked(self):
        manifest = compute_prompt_manifest_digest({
            "prompts.py": "sha-a", "reviewer_schema": "sha-b", "team": "sha-c"
        })
        state = LearningState(
            prompt_manifest_digest=manifest,
            reviewer_memory=(MemoryItem("reviewer", "train", "lesson", "cue"),),
        )
        candidate = propose_update(
            state,
            (_example("train-2", 2.5, 3.5, False, lessons=False),),
            LearningConfig(learning_rate=0.0),
        )
        self.assertEqual(candidate.prompt_manifest_digest, manifest)
        encoded = serialize_learning_state(candidate)
        self.assertEqual(deserialize_learning_state(encoded), candidate)
        tampered = json.loads(encoded)
        tampered["calibration_bias"] = 1.0
        with self.assertRaises(ValueError):
            deserialize_learning_state(json.dumps(tampered))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "state.json"
            dump_learning_state(candidate, path)
            self.assertEqual(load_learning_state(path), candidate)
        self.assertEqual(len(DEFAULT_PROMPT_MANIFEST_DIGEST), 64)

    def test_bounded_memory_is_idempotent_and_forum_stratified(self):
        examples = tuple(
            LearningExample(
                forum_id=forum_id,
                reviewer_lessons=tuple(
                    "{} reviewer lesson {}".format(forum_id, index)
                    for index in range(3)
                ),
                retrieval_text="{} retrieval cue".format(forum_id),
            )
            for forum_id in ("forum-c", "forum-a", "forum-b")
        )
        config = LearningConfig(
            learning_rate=0.0,
            decision_calibration_weight=0.0,
            memory_limit=5,
        )

        first = propose_update(LearningState(), examples, config)
        reordered = propose_update(LearningState(), tuple(reversed(examples)), config)
        replayed = propose_update(first, examples, config)

        self.assertEqual(len(first.reviewer_memory), 5)
        self.assertEqual(
            {item.forum_id for item in first.reviewer_memory},
            {example.forum_id for example in examples},
        )
        self.assertEqual(first.reviewer_memory, reordered.reviewer_memory)
        self.assertEqual(first.reviewer_memory, replayed.reviewer_memory)
        self.assertEqual(state_delta(first, replayed), 0.0)

    def test_train_decision_explicitly_calibrates_acceptance_bias(self):
        accepted = LearningExample(
            forum_id="accepted",
            paper_signals=tuple((name, 2.5) for name in (
                "soundness", "presentation", "significance", "originality"
            )),
            accepted=True,
        )
        rejected = LearningExample(
            forum_id="rejected",
            paper_signals=accepted.paper_signals,
            accepted=False,
        )
        config = LearningConfig(
            learning_rate=0.5,
            decision_calibration_weight=1.0,
        )
        accepted_state = propose_update(LearningState(), (accepted,), config)
        rejected_state = propose_update(LearningState(), (rejected,), config)
        self.assertGreater(accepted_state.calibration_bias, 0.0)
        self.assertLess(rejected_state.calibration_bias, 0.0)
        no_decision_update = propose_update(
            LearningState(),
            (accepted,),
            LearningConfig(learning_rate=0.5, decision_calibration_weight=0.0),
        )
        self.assertEqual(no_decision_update.calibration_bias, 0.0)

    def test_predictor_receives_only_paper_view(self):
        example = _example("sealed", 2.5, 5.0, True)
        observed = []

        def predictor(state, view):
            observed.append(view)
            self.assertIsInstance(view, PredictionInput)
            self.assertFalse(hasattr(view, "target_scores"))
            self.assertFalse(hasattr(view, "accepted"))
            return {
                "soundness": 3,
                "presentation": 3,
                "significance": 3,
                "originality": 3,
                "overall_recommendation": 4,
                "confidence": 3,
                "comment": "paper-only",
                "accept_probability": 0.5,
            }

        predict_many(LearningState(), (example,), predictor)
        self.assertEqual(len(observed), 1)

    def test_unrelated_retrieval_falls_back_without_leaking_review_text(self):
        unrelated_text = "Neural collapse improves image classification geometry."
        state = LearningState(reviewer_memory=(MemoryItem(
            "reviewer",
            "neural-collapse-forum",
            unrelated_text,
            "vision features class means covariance",
        ),))
        markov_game = LearningExample(
            forum_id="markov-game-forum",
            paper_signals=tuple((name, 2.5) for name in (
                "soundness", "presentation", "significance", "originality"
            )),
            retrieval_text="multi-agent Markov game equilibrium policy regret",
        )
        prediction = predict_many(state, (markov_game,))["markov-game-forum"]
        self.assertEqual(prediction["comment"], GENERIC_REVIEW_COMMENT)
        self.assertNotIn("Neural collapse", prediction["comment"])
        self.assertFalse(prediction["_retrieval"]["matched"])
        self.assertIsNone(prediction["_retrieval"]["source_forum"])
        self.assertEqual(
            retrieve_memory(state, "reviewer", markov_game.retrieval_text), ()
        )

    def test_current_forum_memory_is_excluded_and_live_guidance_is_generalized(self):
        raw_review = "This specific FooNet baseline is invalid and Table 7 is wrong."
        raw_rebuttal = "We reran FooNet and obtained 91.7 percent in Table 7."
        state = LearningState(
            reviewer_memory=(MemoryItem(
                "reviewer", "same-forum", raw_review, "same title exact topic baseline"
            ),),
            author_memory=(MemoryItem(
                "author", "same-forum", raw_rebuttal, "same title exact topic baseline"
            ),),
        )
        orchestrator = ReviewerOrchestrator(
            ReplayBackend({}, fallback=lambda _request: "unused"),
            learning_state=state,
        )
        same_paper = PaperInput("same-forum", "same title", "exact topic baseline")
        self.assertEqual(orchestrator._reviewer_memory_payload(same_paper), ())
        self.assertEqual(orchestrator._author_memory_payload(same_paper), ())

        new_paper = PaperInput("new-forum", "same title", "exact topic baseline")
        reviewer_payload = orchestrator._reviewer_memory_payload(new_paper)
        author_payload = orchestrator._author_memory_payload(new_paper)
        self.assertTrue(reviewer_payload)
        self.assertNotIn("source_forum", reviewer_payload[0])
        self.assertNotIn("FooNet", reviewer_payload[0]["lesson"])
        self.assertNotIn("91.7", " ".join(author_payload))
        self.assertIn("current paper", reviewer_payload[0]["lesson"])
        self.assertIn("current paper", author_payload[0])

    def test_generic_ml_vocabulary_does_not_trigger_retrieval(self):
        state = LearningState(reviewer_memory=(MemoryItem(
            "reviewer",
            "collapse",
            "A review about imbalanced classification geometry.",
            "A neural learning model studies imbalanced classification and collapse.",
        ),))
        query = "A neural learning model studies optimistic Markov games and regret."
        self.assertEqual(retrieve_memory(state, "reviewer", query), ())

    def test_seed_case_conversion_and_author_memory_retrieval(self):
        case = {
            "forum_id": "seed-1",
            "conference_year_track": "ICLR 2023 Conference",
            "decision": "Accept: poster",
            "paper": {"title": "Title", "abstract": "Abstract", "keywords": ["vision"]},
            "reviews": [{
                "review_content": "Check the baseline.",
                "final_score": {
                    "rating": "8: accept, good paper",
                    "confidence": "4: confident",
                    "aspect_score": "correctness: 3: good\ncontribution: 4: high\ntechnical_novelty_and_significance: 4: high\nempirical_novelty_and_significance: 2: fair",
                },
            }],
            "metareview": "The concern was resolved.",
            "rebuttal_dialogues": [{"messages": [
                {"role": "user", "content": "We added the baseline."},
                {"role": "assistant", "content": "Thank you."},
            ]}],
        }
        example = seed_case_to_learning_example(case)
        targets = dict(example.target_scores)
        self.assertEqual(targets["overall_recommendation"], 5.0)
        self.assertEqual(targets["soundness"], 3.0)
        self.assertAlmostEqual(targets["significance"], 10.0 / 3.0)
        self.assertTrue(example.accepted)
        self.assertEqual(example.author_lessons, ("We added the baseline.",))
        state = propose_update(
            LearningState(),
            (example,),
            LearningConfig(learning_rate=0.0),
        )
        held_out_view = PredictionInput(
            "new-paper",
            example.paper_signals,
            example.retrieval_text,
        )
        context = author_memory_context(state, held_out_view, limit=1)
        self.assertEqual(len(context), 1)
        self.assertIn("current paper", context[0])
        self.assertNotIn("We added the baseline.", context[0])

    @unittest.skipUnless(REAL_SEED.exists(), "real seed corpus is not present")
    def test_real_seed_jsonl_loads_as_learning_examples(self):
        examples = load_seed_examples(REAL_SEED)
        self.assertGreaterEqual(len(examples), 3)
        self.assertEqual(len({example.forum_id for example in examples}), len(examples))
        self.assertTrue(all(example.paper_signals for example in examples))
        self.assertTrue(any(dict(example.target_scores).get("overall_recommendation") for example in examples))
        self.assertTrue(any(example.author_lessons for example in examples))

    def test_non_regression_checks_each_metric(self):
        reference = EvaluationResult(2, 1.0, 1.0, 0.5, 0.1, (), 0.2, 0.8)
        better = EvaluationResult(2, 1.0, 1.0, 0.4, 0.08, (), 0.18, 0.82)
        brier_regression = EvaluationResult(2, 1.0, 1.0, 0.4, 0.08, (), 0.3, 0.78)
        self.assertTrue(is_non_regression(better, reference))
        self.assertFalse(is_non_regression(brier_regression, reference))
        self.assertTrue(is_non_regression(brier_regression, reference, tolerance=0.11))

    def test_learning_loop_updates_only_train_memory_and_restores_best(self):
        train = (
            _example("train-low", 1.8, 2.5, False),
            _example("train-high", 3.2, 5.0, True),
        )
        dev = (_example("dev", 2.8, 4.5, True, lessons=False),)
        test = (_example("test", 3.0, 4.8, True, lessons=False),)
        baseline = evaluate_predictions(predict_many(LearningState(), dev), dev)
        run = run_learning_loop(
            train,
            dev,
            test,
            config=LearningConfig(
                min_iterations=2,
                max_iterations=20,
                patience=3,
                epsilon_quality=1e-3,
                epsilon_behavior=2e-3,
                epsilon_state=2e-3,
                non_regression_tolerance=0.02,
                learning_rate=0.4,
                memory_limit=16,
            ),
        )
        self.assertGreater(run.state.version, 0)
        self.assertGreaterEqual(run.dev_evaluation.utility, baseline.utility)
        self.assertEqual(run.dev_evaluation.utility, max(
            [baseline.utility] + [item.evaluation.utility for item in run.history]
        ))
        memory_forums = {
            item.forum_id for item in run.state.reviewer_memory + run.state.author_memory
        }
        self.assertEqual(memory_forums, {"train-low", "train-high"})
        self.assertNotIn("dev", memory_forums)
        self.assertNotIn("test", memory_forums)
        self.assertEqual(run.test_evaluation.n_examples, 1)
        self.assertLessEqual(len(run.history), 20)

    def test_plateau_converges_after_minimum_and_patience(self):
        train = (_example("train", 2.5, 3.5, False, lessons=False),)
        dev = (_example("dev", 2.5, 3.5, False, lessons=False),)
        run = run_learning_loop(
            train,
            dev,
            (),
            config=LearningConfig(
                min_iterations=2,
                max_iterations=10,
                patience=2,
                learning_rate=0.0,
            ),
        )
        self.assertEqual(run.stop_reason, "converged")
        self.assertEqual(len(run.history), 3)
        self.assertEqual(run.history[-1].plateau_count, 2)

    def test_forum_overlap_is_rejected_before_learning(self):
        shared = _example("shared", 2.5, 3.5, False)
        with self.assertRaises(DataLeakageError):
            run_learning_loop((shared,), (shared,), ())


if __name__ == "__main__":
    unittest.main()
