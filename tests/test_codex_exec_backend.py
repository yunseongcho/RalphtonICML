from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import tempfile
import threading
import time
import unittest

from ralphton_icml.backend import BackendError, ModelRequest
from ralphton_icml.codex_backend import CodexExecBackend


OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {"ok": {"type": "boolean"}},
    "required": ["ok"],
    "additionalProperties": False,
}


def request(identifier="request", stage="extraction", schema=OUTPUT_SCHEMA):
    return ModelRequest(
        request_id=identifier,
        agent_id="agent." + identifier,
        stage=stage,
        system="Return only the requested structured result.",
        payload={"paper": identifier},
        output_schema=schema,
    )


def make_fake_codex(directory):
    path = Path(directory) / "fake-codex"
    path.write_text(
        """#!/usr/bin/env python3
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

args = sys.argv[1:]
if args == ["--version"]:
    print("fake-codex 1.0")
    raise SystemExit(0)

if os.environ.get("FAKE_CODEX_IGNORE_TERM", ""):
    signal.signal(signal.SIGTERM, signal.SIG_IGN)

def option(name):
    return args[args.index(name) + 1]

started = os.environ.get("FAKE_CODEX_STARTED", "")
if started:
    Path(started).write_text(str(os.getpid()), encoding="utf-8")

child_path = os.environ.get("FAKE_CODEX_CHILD", "")
if child_path:
    child = subprocess.Popen([
        sys.executable,
        "-c",
        "import signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(30)",
    ])
    Path(child_path).write_text(str(child.pid), encoding="utf-8")

prompt = sys.stdin.read()
schema = json.loads(Path(option("--output-schema")).read_text(encoding="utf-8"))
workdir = Path(option("--cd"))
log = os.environ.get("FAKE_CODEX_LOG", "")
if log:
    Path(log).write_text(json.dumps({
        "args": args,
        "agents": (workdir / "AGENTS.md").read_text(encoding="utf-8"),
        "cwd": os.getcwd(),
        "inherited_openai_api_key": os.environ.get("OPENAI_API_KEY"),
        "prompt": prompt,
        "schema": schema,
    }, sort_keys=True), encoding="utf-8")

delay = float(os.environ.get("FAKE_CODEX_SLEEP", "0"))
if delay:
    time.sleep(delay)
response = os.environ.get("FAKE_CODEX_RESPONSE", '{"ok":true}')
Path(option("--output-last-message")).write_text(response, encoding="utf-8")
""",
        encoding="utf-8",
    )
    path.chmod(0o755)
    return path


class TrackingBackend(CodexExecBackend):
    def __init__(self, **kwargs):
        super().__init__(model="test-model", cli_version="fake 1", **kwargs)
        self.active = 0
        self.max_active = 0
        self.guard = threading.Lock()
        self.overlap = threading.Event()

    def _run_codex(self, model_request, fingerprint, timeout):
        del model_request, fingerprint, timeout
        with self.guard:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
            if self.active >= 2:
                self.overlap.set()
        if not self.overlap.wait(timeout=2.0):
            raise AssertionError("two calls did not overlap")
        time.sleep(0.02)
        with self.guard:
            self.active -= 1
        return '{"ok":true}'


class CountingCacheBackend(CodexExecBackend):
    def __init__(self, **kwargs):
        super().__init__(model="test-model", cli_version="fake 1", **kwargs)
        self.call_count = 0
        self.guard = threading.Lock()
        self.entered = threading.Event()
        self.release = threading.Event()

    def _run_codex(self, model_request, fingerprint, timeout):
        del model_request, fingerprint, timeout
        with self.guard:
            self.call_count += 1
        self.entered.set()
        if not self.release.wait(timeout=2.0):
            raise AssertionError("cache test was not released")
        return '{"ok":true}'


class CodexExecBackendTest(unittest.TestCase):
    def test_model_request_schema_is_backward_compatible_and_optional(self):
        legacy = ModelRequest("r", "a", "s", "system", {"x": 1})
        self.assertNotIn("output_schema", legacy.as_dict())
        structured = request()
        self.assertEqual(structured.as_dict()["output_schema"], OUTPUT_SCHEMA)

    def test_explicit_model_and_full_fingerprint_inputs(self):
        with self.assertRaisesRegex(ValueError, "explicit"):
            CodexExecBackend("", cli_version="fake 1")

        base = CodexExecBackend(
            "model-a",
            cli_version="cli-1",
            pipeline_digest="pipeline-1",
            state_digest="state-1",
        )
        equivalent = request(schema={
            "required": ["ok"],
            "properties": {"ok": {"type": "boolean"}},
            "additionalProperties": False,
            "type": "object",
        })
        self.assertEqual(
            base.request_fingerprint(request()),
            base.request_fingerprint(equivalent),
        )

        variants = [
            (CodexExecBackend("model-b", cli_version="cli-1", pipeline_digest="pipeline-1", state_digest="state-1"), request()),
            (CodexExecBackend("model-a", cli_version="cli-2", pipeline_digest="pipeline-1", state_digest="state-1"), request()),
            (CodexExecBackend("model-a", cli_version="cli-1", pipeline_digest="pipeline-2", state_digest="state-1"), request()),
            (CodexExecBackend("model-a", cli_version="cli-1", pipeline_digest="pipeline-1", state_digest="state-2"), request()),
            (base, request(schema={"type": "object"})),
            (base, ModelRequest("request", "agent.request", "extraction", "different", {"paper": "request"}, OUTPUT_SCHEMA)),
        ]
        fingerprint = base.request_fingerprint(request())
        for backend, changed_request in variants:
            self.assertNotEqual(fingerprint, backend.request_fingerprint(changed_request))

    def test_fake_executable_receives_schema_and_isolated_flags(self):
        with tempfile.TemporaryDirectory() as directory:
            fake = make_fake_codex(directory)
            log = Path(directory) / "call.json"
            progress = Path(directory) / "progress.jsonl"
            original_key = os.environ.get("OPENAI_API_KEY")
            os.environ["OPENAI_API_KEY"] = "must-not-reach-child"
            try:
                backend = CodexExecBackend(
                    "explicit-model",
                    executable=str(fake),
                    progress_path=progress,
                    environment={"FAKE_CODEX_LOG": str(log)},
                )
            finally:
                if original_key is None:
                    os.environ.pop("OPENAI_API_KEY", None)
                else:
                    os.environ["OPENAI_API_KEY"] = original_key
            self.assertEqual(backend.cli_version, "fake-codex 1.0")
            response = backend.complete(request())

            self.assertEqual(json.loads(response), {"ok": True})
            call = json.loads(log.read_text(encoding="utf-8"))
            self.assertEqual(call["schema"], OUTPUT_SCHEMA)
            self.assertIn("Return only the requested", call["agents"])
            self.assertIn("REQUEST_JSON", call["prompt"])
            self.assertIn("--ephemeral", call["args"])
            self.assertIn("read-only", call["args"])
            self.assertIn("--output-schema", call["args"])
            self.assertIn("--json", call["args"])
            disabled = [
                call["args"][index + 1]
                for index, value in enumerate(call["args"])
                if value == "--disable"
            ]
            for feature in ("shell_tool", "browser_use", "apps", "plugins"):
                self.assertIn(feature, disabled)
            self.assertIsNone(call["inherited_openai_api_key"])
            self.assertEqual(
                call["args"][call["args"].index("--model") + 1],
                "explicit-model",
            )
            self.assertEqual(
                Path(call["cwd"]).resolve(),
                Path(call["args"][call["args"].index("--cd") + 1]).resolve(),
            )
            events = [
                json.loads(line)["event"]
                for line in progress.read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(events, ["start", "success"])
            metrics = backend.snapshot_metrics()
            self.assertEqual(metrics["requests"], 1)
            self.assertEqual(metrics["calls"], 1)
            self.assertEqual(metrics["successes"], 1)
            self.assertGreater(metrics["request_bytes"], 0)
            self.assertGreater(metrics["output_bytes"], 0)

    def test_schema_is_required_and_invalid_json_is_rejected(self):
        backend = CodexExecBackend("model", cli_version="fake 1")
        with self.assertRaisesRegex(BackendError, "output_schema"):
            backend.complete(ModelRequest("r", "a", "s", "system", {}))

        with tempfile.TemporaryDirectory() as directory:
            fake = make_fake_codex(directory)
            malformed = CodexExecBackend(
                "model",
                executable=str(fake),
                cli_version="fake 1",
                environment={"FAKE_CODEX_RESPONSE": "not-json"},
            )
            with self.assertRaisesRegex(BackendError, "not valid JSON"):
                malformed.complete(request())

    def test_shared_semaphore_caps_concurrency_and_allows_overlap(self):
        backend = TrackingBackend(concurrency=2)
        requests = [request("r{}".format(index)) for index in range(4)]
        with ThreadPoolExecutor(max_workers=4) as executor:
            results = list(executor.map(backend.complete, requests))
        self.assertTrue(all(json.loads(value) == {"ok": True} for value in results))
        self.assertEqual(backend.max_active, 2)
        self.assertEqual(backend.snapshot_metrics()["calls"], 4)

    def test_cache_single_flight_is_atomic_and_reused(self):
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory) / "cache"
            backend = CountingCacheBackend(cache_dir=cache, concurrency=4)
            item = request("same")
            with ThreadPoolExecutor(max_workers=2) as executor:
                first = executor.submit(backend.complete, item)
                second = executor.submit(backend.complete, item)
                self.assertTrue(backend.entered.wait(timeout=1.0))
                backend.release.set()
                values = [first.result(timeout=2.0), second.result(timeout=2.0)]
            self.assertEqual([json.loads(value) for value in values], [{"ok": True}] * 2)
            self.assertEqual(backend.call_count, 1)
            metrics = backend.snapshot_metrics()
            self.assertEqual(metrics["calls"], 1)
            self.assertEqual(metrics["cache_hits"], 1)
            cache_files = list(cache.glob("*.json"))
            self.assertEqual(len(cache_files), 1)
            json.loads(cache_files[0].read_text(encoding="utf-8"))
            self.assertEqual(list(cache.glob("*.tmp")), [])

            reused = CountingCacheBackend(cache_dir=cache)
            reused.release.set()
            self.assertEqual(json.loads(reused.complete(item)), {"ok": True})
            self.assertEqual(reused.call_count, 0)

            cache_files[0].write_text("corrupt", encoding="utf-8")
            repaired = CountingCacheBackend(cache_dir=cache)
            repaired.release.set()
            self.assertEqual(json.loads(repaired.complete(item)), {"ok": True})
            self.assertEqual(repaired.call_count, 1)
            self.assertEqual(repaired.snapshot_metrics()["cache_invalidations"], 1)
            json.loads(cache_files[0].read_text(encoding="utf-8"))

    def test_stage_timeout_kills_process_group(self):
        with tempfile.TemporaryDirectory() as directory:
            fake = make_fake_codex(directory)
            started = Path(directory) / "started"
            child = Path(directory) / "child"
            progress = Path(directory) / "progress.jsonl"
            backend = CodexExecBackend(
                "model",
                executable=str(fake),
                cli_version="fake 1",
                stage_timeouts={"extraction": 0.5},
                progress_path=progress,
                kill_grace=0.05,
                environment={
                    "FAKE_CODEX_STARTED": str(started),
                    "FAKE_CODEX_CHILD": str(child),
                    "FAKE_CODEX_SLEEP": "5",
                },
            )
            began = time.monotonic()
            with self.assertRaisesRegex(BackendError, "exceeded"):
                backend.complete(request())
            self.assertLess(time.monotonic() - began, 2.0)
            process_id = int(started.read_text(encoding="utf-8"))
            child_id = int(child.read_text(encoding="utf-8"))
            with self.assertRaises(ProcessLookupError):
                os.kill(process_id, 0)
            limit = time.monotonic() + 1.0
            while time.monotonic() < limit:
                try:
                    os.kill(child_id, 0)
                except ProcessLookupError:
                    break
                time.sleep(0.01)
            with self.assertRaises(ProcessLookupError):
                os.kill(child_id, 0)
            events = [
                json.loads(line)["event"]
                for line in progress.read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(events, ["start", "timeout"])
            self.assertEqual(backend.snapshot_metrics()["timeouts"], 1)

    def test_hard_deadline_and_cancel_all_stop_new_work(self):
        expired = TrackingBackend(hard_deadline=time.monotonic() - 1.0)
        with self.assertRaisesRegex(BackendError, "deadline"):
            expired.complete(request())
        self.assertEqual(expired.snapshot_metrics()["calls"], 0)

        with tempfile.TemporaryDirectory() as directory:
            fake = make_fake_codex(directory)
            started = Path(directory) / "started"
            backend = CodexExecBackend(
                "model",
                executable=str(fake),
                cli_version="fake 1",
                kill_grace=0.05,
                environment={
                    "FAKE_CODEX_STARTED": str(started),
                    "FAKE_CODEX_SLEEP": "5",
                },
            )
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(backend.complete, request())
                limit = time.monotonic() + 2.0
                while not started.exists() and time.monotonic() < limit:
                    time.sleep(0.01)
                self.assertTrue(started.exists())
                backend.cancel_all()
                with self.assertRaisesRegex(BackendError, "cancelled"):
                    future.result(timeout=2.0)
            with self.assertRaisesRegex(BackendError, "cancelled"):
                backend.complete(request("later"))


if __name__ == "__main__":
    unittest.main()
