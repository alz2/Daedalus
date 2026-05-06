"""Python program executor (DSL v2).

Executes LLM-generated Python code that calls skills as functions via a
sandboxed namespace. Skills are exposed as ``ctx.<skill_id>(**kwargs)``
methods on a context proxy object.

Safety: the code is AST-linted before execution (reusing the safety module)
with relaxed rules -- loops and conditionals are allowed, but subprocess,
eval, exec, os.system, and network access remain banned.
"""

from __future__ import annotations

import ast
import contextlib
import math
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from daedalus.core.context import ExecutionContext, TaskState, compute_coordinate_scale
from daedalus.core.errors import (
    BackendError,
    DaedalusError,
    ProgramValidationError,
    SkillNotFoundError,
    TimeoutError as DaedalusTimeoutError,
    UserAbortError,
)
from daedalus.core.registry import Registry, get_registry
from daedalus.core.store import RunStore
from daedalus.evaluator.criteria import GoalVerdict, SuccessCriteria
from daedalus.evaluator.evaluator import Evaluator
from daedalus.executor.daemons import DaemonHandle, DaemonSpec, start_daemons, stop_daemons
from daedalus.executor.dsl import PythonProgram
from daedalus.executor.runner import RunResult, StepResult
from daedalus.tracing.recorder import TraceRecorder


_ALLOWED_BUILTINS = {
    "abs", "all", "any", "bool", "dict", "divmod", "enumerate",
    "filter", "float", "frozenset", "hasattr", "int", "isinstance",
    "len", "list", "map", "max", "min", "pow", "print", "range",
    "repr", "reversed", "round", "set", "sorted", "str", "sum",
    "tuple", "type", "zip",
}

_ALLOWED_MODULES = {
    "math", "re", "json", "itertools", "functools", "collections",
    "string", "textwrap", "copy", "operator", "heapq", "bisect",
    "statistics", "random", "time",
}


def lint_plan_code(source: str) -> list[str]:
    """AST-lint generated plan code. Returns a list of error messages."""
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        return [f"SyntaxError at line {exc.lineno}: {exc.msg}"]

    errors: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names = ", ".join(a.name for a in node.names)
            errors.append(
                f"line {node.lineno}: import statement not allowed "
                f"({names}). Allowed modules are pre-imported in the "
                f"sandbox — use them directly without import."
            )
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            errors.append(
                f"line {node.lineno}: import statement not allowed "
                f"(from {mod}). Allowed modules are pre-imported in "
                f"the sandbox — use them directly without import."
            )
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                name = node.func.id
                if name in ("eval", "exec", "compile", "__import__", "open", "globals", "locals"):
                    errors.append(
                        f"line {node.lineno}: call to {name!r} is banned in plans"
                    )
    return errors


class SkillProxy:
    """Proxy object that exposes skills as callable methods (ctx.skill_id(...))."""

    def __init__(
        self,
        execution_ctx: ExecutionContext,
        registry: Registry,
        tracer: TraceRecorder,
        abort_event: threading.Event,
        step_timeout_s: float = 30.0,
        event_callback: Callable[[str, dict[str, Any]], None] | None = None,
    ) -> None:
        self._ctx = execution_ctx
        self._registry = registry
        self._tracer = tracer
        self._abort_event = abort_event
        self._step_timeout_s = step_timeout_s
        self._step_idx = 0
        self._results: list[StepResult] = []
        self._event_callback = event_callback
        self.store = execution_ctx.store
        self.state = execution_ctx.task_state

    def __getattr__(self, name: str) -> Callable[..., Any]:
        try:
            entry = self._registry.get(name)
        except SkillNotFoundError:
            raise AttributeError(f"no skill named {name!r}") from None

        def _invoke(**kwargs: Any) -> Any:
            if self._abort_event.is_set():
                raise UserAbortError("aborted by user")

            idx = self._step_idx
            self._step_idx += 1

            inputs_model = entry.cls.Inputs.model_validate(kwargs)
            sensitive = entry.cls.SPEC.sensitive_inputs
            redacted = _redact_sensitive(kwargs, sensitive)

            self._tracer.skill_started(
                skill_id=entry.id,
                version=entry.version.raw,
                step_idx=idx,
                inputs=redacted,
                content_hash=entry.content_hash,
            )
            if self._event_callback:
                self._event_callback("skill_started", {
                    "step_idx": idx,
                    "skill_id": entry.id,
                    "inputs": redacted,
                })

            t0 = time.perf_counter()
            try:
                instance = entry.cls()
                output = instance.run(inputs_model, self._ctx)
            except Exception as exc:
                dur = (time.perf_counter() - t0) * 1000
                self._tracer.skill_error(
                    skill_id=entry.id,
                    step_idx=idx,
                    error_type=type(exc).__name__,
                    message=str(exc),
                    duration_ms=dur,
                )
                self._results.append(StepResult(
                    skill_id=entry.id, step_idx=idx, status="failed",
                    duration_ms=dur, error=f"{type(exc).__name__}: {exc}",
                ))
                if self._event_callback:
                    self._event_callback("skill_error", {
                        "step_idx": idx,
                        "skill_id": entry.id,
                        "message": f"{type(exc).__name__}: {exc}",
                    })
                raise

            dur = (time.perf_counter() - t0) * 1000
            out_dict = output.model_dump(mode="json") if hasattr(output, "model_dump") else dict(output)

            self._tracer.skill_finished(
                skill_id=entry.id,
                step_idx=idx,
                outputs=_redact_for_trace(out_dict),
                duration_ms=dur,
            )
            self._results.append(StepResult(
                skill_id=entry.id, step_idx=idx, status="success",
                duration_ms=dur, output=out_dict,
            ))
            if self._event_callback:
                event_data: dict[str, Any] = {
                    "step_idx": idx,
                    "skill_id": entry.id,
                    "outputs": _redact_for_trace(out_dict),
                    "duration_ms": dur,
                }
                image_path = out_dict.get("image_path")
                if image_path and isinstance(image_path, str):
                    event_data["image_path"] = image_path
                self._event_callback("skill_finished", event_data)
            return output

        return _invoke


class PythonProgramExecutor:
    """Execute a PythonProgram (v2) against a backend."""

    def __init__(
        self,
        backend: Any,
        registry: Registry | None = None,
        traces_root: Path | None = None,
        tasks_db: Path | None = None,
        llm: Any = None,
        config: dict[str, Any] | None = None,
        abort_event: threading.Event | None = None,
        step_timeout_s: float = 30.0,
        program_timeout_s: float = 300.0,
        status_callback: Callable[[str], None] | None = None,
        event_callback: Callable[[str, dict[str, Any]], None] | None = None,
    ) -> None:
        self._backend = backend
        self._registry = registry if registry is not None else get_registry()
        self._traces_root = traces_root or Path("traces")
        self._tasks_db = tasks_db or Path("tasks.db")
        self._llm = llm
        self._config = config or {}
        self._abort_event = abort_event or threading.Event()
        self._step_timeout_s = step_timeout_s
        self._program_timeout_s = program_timeout_s
        self._status_callback = status_callback
        self._event_callback = event_callback

    def run(
        self,
        program: PythonProgram,
        *,
        program_ref: str | None = None,
        success_criteria: SuccessCriteria | None = None,
    ) -> RunResult:
        lint_errors = lint_plan_code(program.code)
        if lint_errors:
            raise ProgramValidationError(
                "Plan code failed safety lint:\n" + "\n".join(f"  - {e}" for e in lint_errors)
            )

        tracer = TraceRecorder(
            traces_root=self._traces_root,
            db_path=self._tasks_db,
            task_name=program.name,
            program_ref=program_ref,
        )
        if self._llm is not None:
            self._llm.set_tracer(tracer)
        task_state = TaskState(self._tasks_db, tracer.task_id)
        store = RunStore(self._tasks_db, tracer.task_id)

        result = RunResult(
            task_id=tracer.task_id,
            program_name=program.name,
            status="running",
            started_at=time.time(),
        )

        tracer.start()
        tracer.emit(
            "program_started",
            {
                "name": program.name,
                "description": program.description,
                "dsl_version": 2,
                "skills": program.referenced_skill_ids(),
            },
        )

        if self._event_callback:
            self._event_callback("run_trace_dir", {
                "task_id": tracer.task_id,
                "trace_dir": str(tracer.task_dir),
            })

        connected = False
        daemons: list[DaemonHandle] = []
        try:
            try:
                self._backend.connect()
                connected = True
            except Exception as exc:
                raise BackendError(f"backend.connect failed: {exc}") from exc

            ctx = ExecutionContext(
                task_id=tracer.task_id,
                backend=self._backend,
                task_state=task_state,
                tracer=tracer,
                store=store,
                llm=self._llm,
                config=self._config,
                abort_event=self._abort_event,
                coordinate_scale=compute_coordinate_scale(self._backend.size[0]) if hasattr(self._backend, "size") else 1.0,
            )

            if program.daemons:
                daemon_specs = [
                    DaemonSpec(skill=d.skill, version=d.version, inputs=d.inputs)
                    for d in program.daemons
                ]
                daemons = start_daemons(daemon_specs, ctx, registry=self._registry)

            proxy = SkillProxy(
                execution_ctx=ctx,
                registry=self._registry,
                tracer=tracer,
                abort_event=self._abort_event,
                step_timeout_s=self._step_timeout_s,
                event_callback=self._event_callback,
            )

            sandbox_globals = self._build_sandbox(proxy)

            func_code = f"def __plan__(ctx):\n"
            for line in program.code.splitlines():
                func_code += f"    {line}\n"

            exec(compile(func_code, "<plan>", "exec"), sandbox_globals)  # noqa: S102
            plan_fn = sandbox_globals["__plan__"]

            with ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(plan_fn, proxy)
                try:
                    future.result(timeout=self._program_timeout_s)
                except FuturesTimeoutError:
                    self._abort_event.set()
                    raise DaedalusTimeoutError(
                        f"plan timed out after {self._program_timeout_s:.0f}s"
                    )

            result.steps = proxy._results
            result.status = "success"
            tracer.emit("program_finished", {"status": "success"})

            if success_criteria is not None:
                evaluator = Evaluator(llm=self._llm)
                verdict = evaluator.evaluate(success_criteria, ctx, tracer)
                result.goal_verdict = verdict
                if verdict.achieved:
                    result.status = "goal_achieved"
                else:
                    result.status = "goal_not_achieved"
                tracer.emit(
                    "program_finished",
                    {"status": result.status, "goal_achieved": verdict.achieved},
                )

            tracer.finish(
                "success" if result.status in ("success", "goal_achieved") else "failed"
            )
        except UserAbortError:
            result.steps = proxy._results if "proxy" in dir() else []
            result.status = "aborted"
            tracer.finish("aborted")
            tracer.emit("program_finished", {"status": "aborted"}, level="warn")
        except DaedalusError as exc:
            result.steps = proxy._results if "proxy" in dir() else []
            result.status = "failed"
            result.error_message = str(exc)
            tracer.finish("failed", notes=str(exc))
            tracer.emit(
                "program_finished",
                {"status": "failed", "error": str(exc), "error_type": type(exc).__name__},
                level="error",
            )
        except Exception as exc:
            result.steps = proxy._results if "proxy" in dir() else []
            result.status = "failed"
            result.error_message = str(exc)
            tracer.finish("failed", notes=str(exc))
            tracer.emit(
                "program_finished",
                {"status": "failed", "error": str(exc), "error_type": type(exc).__name__},
                level="error",
            )
        finally:
            if daemons:
                stop_daemons(daemons, ctx)
            result.finished_at = time.time()
            if connected:
                with contextlib.suppress(Exception):
                    self._backend.disconnect()

        return result

    def _build_sandbox(self, proxy: SkillProxy) -> dict[str, Any]:
        """Build a restricted globals dict for plan execution."""
        import builtins as _builtins

        safe_builtins = {
            name: getattr(_builtins, name)
            for name in _ALLOWED_BUILTINS
            if hasattr(_builtins, name)
        }
        safe_builtins["True"] = True
        safe_builtins["False"] = False
        safe_builtins["None"] = None

        sandbox: dict[str, Any] = {"__builtins__": safe_builtins}

        import collections
        import copy
        import functools
        import heapq
        import itertools
        import json
        import operator
        import random
        import re
        import statistics
        import string
        import textwrap

        sandbox["math"] = math
        sandbox["re"] = re
        sandbox["json"] = json
        sandbox["itertools"] = itertools
        sandbox["functools"] = functools
        sandbox["collections"] = collections
        sandbox["string"] = string
        sandbox["textwrap"] = textwrap
        sandbox["copy"] = copy
        sandbox["operator"] = operator
        sandbox["heapq"] = heapq
        sandbox["time"] = time
        sandbox["random"] = random
        sandbox["statistics"] = statistics

        return sandbox


def _redact_sensitive(d: dict[str, Any], sensitive: list[str]) -> dict[str, Any]:
    if not sensitive:
        return d
    out = dict(d)
    for key in sensitive:
        if key in out:
            out[key] = "<REDACTED>"
    return out


_MAX_INLINE_VALUE_LEN = 4096


def _redact_for_trace(d: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in d.items():
        if isinstance(v, str) and len(v) > _MAX_INLINE_VALUE_LEN:
            out[k] = f"<{len(v)} bytes elided>"
        elif isinstance(v, dict):
            out[k] = _redact_for_trace(v)
        else:
            out[k] = v
    return out
