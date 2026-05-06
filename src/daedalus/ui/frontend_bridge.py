"""JSON-RPC bridge for the TypeScript frontend.

When ``--frontend-mode`` is active, all console output is suppressed and
structured messages are emitted as newline-delimited JSON to stdout.

Protocol:
  - Notifications (server → client): {"jsonrpc": "2.0", "method": ..., "params": ...}
  - Requests (server → client): {"jsonrpc": "2.0", "id": N, "method": ..., "params": ...}
  - Responses (client → server): {"jsonrpc": "2.0", "id": N, "result": ...}
"""

from __future__ import annotations

import json
import sys
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class FrontendBridge:
    """Bidirectional JSON-RPC bridge over stdin/stdout."""

    def __init__(self) -> None:
        self._write_lock = threading.Lock()
        self._read_lock = threading.Lock()
        self._next_id = 1
        self._pending: dict[int, threading.Event] = {}
        self._responses: dict[int, dict[str, Any]] = {}
        self._reader_thread: threading.Thread | None = None
        self._running = False
        self._on_command: Callable[[str, dict[str, Any]], None] | None = None
        self._events_file: Any | None = None
        self._run_events_file: Any | None = None
        self._events_lock = threading.Lock()

    def start(self, on_command: Callable[[str, dict[str, Any]], None] | None = None) -> None:
        """Start the background reader thread for incoming commands."""
        self._on_command = on_command
        self._running = True
        self._reader_thread = threading.Thread(target=self._read_loop, daemon=True)
        self._reader_thread.start()

    def set_trace_dir(self, trace_dir: Path) -> None:
        """Enable recording bridge events to bridge_events.jsonl in the given directory."""
        trace_dir.mkdir(parents=True, exist_ok=True)
        self._events_file = open(trace_dir / "bridge_events.jsonl", "a")

    def set_run_trace_dir(self, run_dir: Path) -> None:
        """Set per-run trace directory so bridge events are also saved inside it."""
        with self._events_lock:
            if self._run_events_file:
                self._run_events_file.close()
            run_dir.mkdir(parents=True, exist_ok=True)
            self._run_events_file = open(run_dir / "bridge_events.jsonl", "a")

    def stop(self) -> None:
        self._running = False
        if self._events_file:
            self._events_file.close()
            self._events_file = None
        if self._run_events_file:
            self._run_events_file.close()
            self._run_events_file = None

    def _read_loop(self) -> None:
        """Read lines from stdin, dispatch responses and commands."""
        while self._running:
            try:
                line = sys.stdin.readline()
                if not line:
                    break
                line = line.strip()
                if not line:
                    continue
                msg = json.loads(line)
                self._handle_incoming(msg)
            except (json.JSONDecodeError, OSError):
                continue

    def _handle_incoming(self, msg: dict[str, Any]) -> None:
        # Response to a request we sent
        if "id" in msg and ("result" in msg or "error" in msg):
            msg_id = msg["id"]
            if msg_id in self._pending:
                self._responses[msg_id] = msg
                self._pending[msg_id].set()
            return

        # Command from frontend (notification)
        if "method" in msg and "id" not in msg:
            method = msg["method"]
            params = msg.get("params", {})
            if self._on_command:
                self._on_command(method, params)

    # ------------------------------------------------------------------
    # Sending
    # ------------------------------------------------------------------

    def notify(self, method: str, params: dict[str, Any] | None = None) -> None:
        """Send a notification (no response expected)."""
        msg = {"jsonrpc": "2.0", "method": method, "params": params or {}}
        self._write(msg)
        self._record_event(method, params or {})

    def request(self, method: str, params: dict[str, Any] | None = None, timeout: float = 300.0) -> Any:
        """Send a request and block until the frontend responds."""
        msg_id = self._next_id
        self._next_id += 1
        event = threading.Event()
        self._pending[msg_id] = event

        msg = {"jsonrpc": "2.0", "id": msg_id, "method": method, "params": params or {}}
        self._write(msg)

        if not event.wait(timeout=timeout):
            self._pending.pop(msg_id, None)
            raise TimeoutError(f"Frontend did not respond to {method} within {timeout}s")

        self._pending.pop(msg_id, None)
        response = self._responses.pop(msg_id, {})
        if "error" in response:
            raise RuntimeError(f"Frontend error: {response['error'].get('message', 'unknown')}")
        return response.get("result")

    def _write(self, msg: dict[str, Any]) -> None:
        with self._write_lock:
            sys.stdout.write(json.dumps(msg) + "\n")
            sys.stdout.flush()

    def _record_event(self, method: str, params: dict[str, Any]) -> None:
        """Append a notification to the bridge events log file(s)."""
        if not self._events_file and not self._run_events_file:
            return
        record = {"method": method, "params": params, "ts": _now_iso()}
        line = json.dumps(record) + "\n"
        with self._events_lock:
            try:
                if self._events_file:
                    self._events_file.write(line)
                    self._events_file.flush()
                if self._run_events_file:
                    self._run_events_file.write(line)
                    self._run_events_file.flush()
            except (OSError, ValueError):
                pass

    # ------------------------------------------------------------------
    # Convenience event emitters
    # ------------------------------------------------------------------

    def emit_event(self, kind: str, data: dict[str, Any] | None = None, level: str = "info") -> None:
        """Emit a trace event to the frontend."""
        self.notify("event", {
            "kind": kind,
            "ts": _now_iso(),
            "level": level,
            "data": data or {},
        })

    def emit_phase(self, phase: str, status: str, summary: str | None = None) -> None:
        """Emit a phase change event."""
        self.emit_event("phase_changed", {
            "phase": phase,
            "status": status,
            "summary": summary,
        })

    def emit_thinking(self, text: str) -> None:
        """Stream thinking text to the frontend."""
        self.notify("thinking", {"text": text})

    def emit_thinking_clear(self) -> None:
        """Clear the thinking display."""
        self.notify("thinking_clear")

    def emit_explorer_progress(self, current: int, total: int) -> None:
        self.emit_event("explorer_progress", {"current": current, "total": total})

    def emit_context_usage(self, used_tokens: int, max_tokens: int) -> None:
        self.emit_event("context_usage", {"used_tokens": used_tokens, "max_tokens": max_tokens})

    def emit_executor_progress(self, current: int, total: int) -> None:
        self.emit_event("executor_progress", {"current": current, "total": total})

    # ------------------------------------------------------------------
    # Confirmation requests
    # ------------------------------------------------------------------

    def confirm_program(self, program_data: dict[str, Any]) -> dict[str, Any]:
        """Ask the frontend to approve/deny a program. Returns {decision, comments?}."""
        return self.request("confirm_program", {"program": program_data})

    def confirm_criteria(self, criteria_data: dict[str, Any]) -> dict[str, Any]:
        """Ask the frontend to approve/deny success criteria."""
        return self.request("confirm_criteria", criteria_data)

    def confirm_skills(self, skills_data: list[dict[str, Any]]) -> dict[str, Any]:
        """Ask the frontend to approve/deny proposed skills."""
        return self.request("confirm_skills", {"skills": skills_data})

    # ------------------------------------------------------------------
    # Status callback (compatible with overlay.update_status signature)
    # ------------------------------------------------------------------

    def update_status(self, status_text: str) -> None:
        """Status callback suitable for passing to executor."""
        self.emit_event("status_update", {"text": status_text})
