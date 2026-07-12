"""Provider-independent model backend contracts.

The reviewer pipeline never imports a hosted-model SDK.  A production model can
be connected through :class:`SubprocessBackend`: the command receives one JSON
request on stdin and must emit the response text on stdout.  This keeps model,
credential, and deployment choices outside the research pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import subprocess
from typing import Any, Callable, Dict, Mapping, Optional, Protocol, Sequence, Tuple


class BackendError(RuntimeError):
    """Raised when a model backend fails or violates its transport contract."""


@dataclass(frozen=True)
class ModelRequest:
    request_id: str
    agent_id: str
    stage: str
    system: str
    payload: Mapping[str, Any]
    output_schema: Optional[Mapping[str, Any]] = None

    def as_dict(self) -> Dict[str, Any]:
        value = {
            "request_id": self.request_id,
            "agent_id": self.agent_id,
            "stage": self.stage,
            "system": self.system,
            "payload": dict(self.payload),
        }
        if self.output_schema is not None:
            value["output_schema"] = dict(self.output_schema)
        return value


class ModelBackend(Protocol):
    """Minimal interface implemented by all model providers."""

    def complete(self, request: ModelRequest) -> str:
        ...


class SubprocessBackend:
    """Invoke one isolated command per request using JSON over stdin.

    The command may emit either plain response text or a JSON object containing
    a string field named ``text``.  Shell expansion is deliberately disabled.
    """

    def __init__(
        self,
        command: Sequence[str],
        timeout: float = 300.0,
        environment: Optional[Mapping[str, str]] = None,
    ) -> None:
        command_tuple = tuple(str(part) for part in command)
        if not command_tuple or any(not part for part in command_tuple):
            raise ValueError("command must contain at least one non-empty argument")
        if timeout <= 0:
            raise ValueError("timeout must be positive")
        self.command = command_tuple
        self.timeout = float(timeout)
        self.environment = None if environment is None else dict(environment)

    def complete(self, request: ModelRequest) -> str:
        if not isinstance(request, ModelRequest):
            raise TypeError("request must be a ModelRequest")
        encoded = json.dumps(
            request.as_dict(), ensure_ascii=False, sort_keys=True
        ).encode("utf-8")
        try:
            completed = subprocess.run(
                self.command,
                input=encoded,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
                timeout=self.timeout,
                env=self.environment,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise BackendError("backend command failed: {}".format(exc)) from exc
        if completed.returncode != 0:
            stderr = completed.stderr.decode("utf-8", "replace").strip()
            raise BackendError(
                "backend exited with code {}: {}".format(
                    completed.returncode, stderr[:1000]
                )
            )
        try:
            output = completed.stdout.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise BackendError("backend stdout must be UTF-8") from exc
        stripped = output.strip()
        if not stripped:
            raise BackendError("backend returned an empty response")
        if stripped.startswith("{"):
            try:
                decoded = json.loads(stripped)
            except json.JSONDecodeError:
                decoded = None
            if isinstance(decoded, Mapping) and isinstance(decoded.get("text"), str):
                stripped = decoded["text"].strip()
        if not stripped:
            raise BackendError("backend response text is empty")
        return stripped


class ReplayBackend:
    """Deterministic backend for tests and cached experiment replay."""

    def __init__(
        self,
        responses: Mapping[
            Tuple[str, str], Sequence[str]
        ],
        fallback: Optional[Callable[[ModelRequest], str]] = None,
    ) -> None:
        self._responses = {
            key: list(values) for key, values in responses.items()
        }
        self._fallback = fallback
        self.requests = []

    def complete(self, request: ModelRequest) -> str:
        self.requests.append(request)
        key = (request.stage, request.agent_id)
        queue = self._responses.get(key)
        if queue:
            return queue.pop(0)
        if self._fallback is not None:
            response = self._fallback(request)
            if not isinstance(response, str) or not response.strip():
                raise BackendError("replay fallback returned an empty response")
            return response.strip()
        raise BackendError("no replay response for {!r}".format(key))


__all__ = [
    "BackendError",
    "ModelBackend",
    "ModelRequest",
    "ReplayBackend",
    "SubprocessBackend",
]
