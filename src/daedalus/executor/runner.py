"""Sequential program executor (Phase 0).

Walks the program, validates each step's inputs against the skill's Pydantic
``Inputs``, runs the skill, validates outputs against ``Outputs``, and pushes
structured events into the trace recorder. On any failure it raises and the
trace is closed with status ``failed``.

The executor is intentionally synchronous and single-threaded for Phase 0;
``executor/daemons.py`` will add concurrency / daemon lifecycle in Phase 2.
"""

from __future__ import annotations

import contextlib
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
    PreconditionError,
    ProgramValidationError,
    SkillNotFoundError,
    TimeoutError as DaedalusTimeoutError,
    UserAbortError,
)
from daedalus.core.registry import Registry, get_registry
from daedalus.core.spec import check_preconditions
from daedalus.core.store import RunStore
from daedalus.evaluator.criteria import GoalVerdict, SuccessCriteria
from daedalus.evaluator.evaluator import Evaluator
from daedalus.executor.daemons import DaemonHandle, DaemonSpec, start_daemons, stop_daemons
from daedalus.executor.dsl import Program, resolve_inputs, validate_program_against_registry
from daedalus.tracing.recorder import TraceRecorder


@dataclass
class StepResult:
    skill_id: str
    step_idx: int
    status: str  # "success" | "failed" | "skipped"
    duration_ms: float
    output: dict[str, Any] | None = None
    error: str | None = None


@dataclass
class RunResult:
    task_id: str
    program_name: str
    status: str  # "goal_achieved" | "goal_not_achieved" | "success" | "failed" | "aborted"
    steps: list[StepResult] = field(default_factory=list)
    started_at: float = 0.0
    finished_at: float = 0.0
    goal_verdict: GoalVerdict | None = None
    error_message: str | None = None

    @property
    def duration_s(self) -> float:
        return max(0.0, self.finished_at - self.started_at)


class SequentialExecutor:
    """Run a :class:`Program` end-to-end against a single backend."""

    def __init__(
        self,
        backend,  # type: ignore[no-untyped-def]  # backends/protocol.RemoteDesktop
        registry: Registry | None = None,
        traces_root: Path | None = None,
        tasks_db: Path | None = None,
        llm=None,  # type: ignore[no-untyped-def]
        config: dict[str, Any] | None = None,
        abort_event: threading.Event | None = None,
        step_timeout_s: float = 30.0,
        status_callback: Callable[[str], None] | None = None,
        event_callback: Callable[[str, dict[str, Any]], None] | None = None,
    ) -> None:
        self._backend = backend
        self._registry = registry if registry is not None else get_registry()
        self._traces_root = traces_root or Path("traces")
        self._tasks_db = tasks_db or Path("tasks.db")
        self._llm = llm
        self._config = config or {}
        self._abort_event = abort_event
        self._step_timeout_s = step_timeout_s
        self._status_callback = status_callback
        self._event_callback = event_callback

    def run(
        self,
        program: Program,
        *,
        program_ref: str | None = None,
        success_criteria: SuccessCriteria | None = None,
    ) -> RunResult:
        validate_program_against_registry(program, self._registry)

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
                "step_count": program.step_count,
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
                abort_event=self._abort_event or threading.Event(),
                coordinate_scale=compute_coordinate_scale(self._backend.size[0]) if hasattr(self._backend, "size") else 1.0,
            )

            if program.daemons:
                daemon_specs = [
                    DaemonSpec(skill=d.skill, version=d.version, inputs=d.inputs)
                    for d in program.daemons
                ]
                daemons = start_daemons(daemon_specs, ctx, registry=self._registry)

            saved_outputs: dict[str, dict[str, Any]] = {}

            for i, step in enumerate(program.steps):
                if ctx.aborted():
                    raise UserAbortError("aborted by user")

                step_result = self._run_step(i, step, ctx, tracer, saved_outputs)
                result.steps.append(step_result)
                if self._status_callback:
                    self._status_callback(f"step {i}: {step.skill} -> {step_result.status}")
                if step_result.status == "failed":
                    raise DaedalusError(f"step {i} ({step.skill}) failed: {step_result.error}")

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
            result.status = "aborted"
            tracer.finish("aborted")
            tracer.emit("program_finished", {"status": "aborted"}, level="warn")
        except DaedalusError as exc:
            result.status = "failed"
            result.error_message = str(exc)
            tracer.finish("failed", notes=str(exc))
            tracer.emit(
                "program_finished",
                {"status": "failed", "error": str(exc), "error_type": type(exc).__name__},
                level="error",
            )
        except Exception as exc:
            result.status = "failed"
            result.error_message = str(exc)
            tracer.finish("failed", notes=str(exc))
            tracer.emit(
                "program_finished",
                {"status": "failed", "error": str(exc), "error_type": type(exc).__name__},
                level="error",
            )
            raise
        finally:
            if daemons:
                stop_daemons(daemons, ctx)
            result.finished_at = time.time()
            if connected:
                with contextlib.suppress(Exception):
                    self._backend.disconnect()

        return result

    # ------------------------------------------------------------------

    def _run_step(
        self,
        idx: int,
        step,  # type: ignore[no-untyped-def]  # ProgramStep
        ctx: ExecutionContext,
        tracer: TraceRecorder,
        saved_outputs: dict[str, dict[str, Any]],
    ) -> StepResult:
        try:
            entry = self._registry.get(step.skill, version_constraint=step.version)
        except SkillNotFoundError as exc:
            raise ProgramValidationError(str(exc)) from exc

        raw_inputs = resolve_inputs(step.inputs, saved_outputs, store=ctx.store)
        skill_cls = entry.cls
        inputs = skill_cls.Inputs.model_validate(raw_inputs)

        failed_pre = check_preconditions(entry.cls.SPEC.preconditions, ctx)
        if failed_pre:
            tracer.skill_error(
                skill_id=entry.id,
                step_idx=idx,
                error_type="PreconditionError",
                message=f"precondition {failed_pre!r} not met",
                duration_ms=0,
            )
            return StepResult(
                skill_id=entry.id,
                step_idx=idx,
                status="failed",
                duration_ms=0,
                error=f"PreconditionError: {failed_pre!r} not met",
            )

        sensitive = entry.cls.SPEC.sensitive_inputs
        redacted_inputs = _redact_sensitive_inputs(_safe_dump(inputs), sensitive)
        tracer.skill_started(
            skill_id=entry.id,
            version=entry.version.raw,
            step_idx=idx,
            inputs=redacted_inputs,
            content_hash=entry.content_hash,
        )
        if self._event_callback:
            self._event_callback("skill_started", {
                "step_idx": idx,
                "skill_id": entry.id,
                "inputs": redacted_inputs,
            })
        if self._status_callback:
            self._status_callback(f"step {idx}: {step.skill}")
        t0 = time.perf_counter()
        try:
            instance = skill_cls()
            timeout_s = (step.max_duration_ms / 1000.0) if step.max_duration_ms else self._step_timeout_s
            with ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(instance.run, inputs, ctx)
                try:
                    output = future.result(timeout=timeout_s)
                except FuturesTimeoutError:
                    ctx.abort_event.set()
                    dur = (time.perf_counter() - t0) * 1000
                    msg = f"step {idx} ({entry.id}) timed out after {timeout_s:.1f}s"
                    tracer.skill_error(
                        skill_id=entry.id,
                        step_idx=idx,
                        error_type="DaedalusTimeoutError",
                        message=msg,
                        duration_ms=dur,
                    )
                    return StepResult(
                        skill_id=entry.id,
                        step_idx=idx,
                        status="failed",
                        duration_ms=dur,
                        error=f"DaedalusTimeoutError: {msg}",
                    )
        except Exception as exc:
            dur = (time.perf_counter() - t0) * 1000
            tracer.skill_error(
                skill_id=entry.id,
                step_idx=idx,
                error_type=type(exc).__name__,
                message=str(exc),
                duration_ms=dur,
            )
            return StepResult(
                skill_id=entry.id,
                step_idx=idx,
                status="failed",
                duration_ms=dur,
                error=f"{type(exc).__name__}: {exc}",
            )

        dur = (time.perf_counter() - t0) * 1000
        try:
            output = skill_cls.Outputs.model_validate(_safe_dump(output))
        except Exception as exc:
            tracer.skill_error(
                skill_id=entry.id,
                step_idx=idx,
                error_type="OutputValidationError",
                message=str(exc),
                duration_ms=dur,
            )
            return StepResult(
                skill_id=entry.id,
                step_idx=idx,
                status="failed",
                duration_ms=dur,
                error=f"output validation failed: {exc}",
            )

        out_dict = _safe_dump(output)
        if step.save_as:
            ctx.task_state.set(step.save_as, out_dict)
            saved_outputs[step.save_as] = out_dict

        tracer.skill_finished(
            skill_id=entry.id,
            step_idx=idx,
            outputs=_redact_for_trace(out_dict),
            duration_ms=dur,
        )
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
        return StepResult(
            skill_id=entry.id,
            step_idx=idx,
            status="success",
            duration_ms=dur,
            output=out_dict,
        )


def _redact_sensitive_inputs(d: dict[str, Any], sensitive: list[str]) -> dict[str, Any]:
    if not sensitive:
        return d
    out = dict(d)
    for key in sensitive:
        if key in out:
            out[key] = "<REDACTED>"
    return out


def _safe_dump(model_or_dict: Any) -> dict[str, Any]:
    if hasattr(model_or_dict, "model_dump"):
        return model_or_dict.model_dump(mode="json")
    if isinstance(model_or_dict, dict):
        return model_or_dict
    raise TypeError(f"cannot dump {type(model_or_dict).__name__} to dict")


# Trace files can balloon if every screenshot's base64 is logged. We trim
# any value over 4 KiB and replace with a marker; the actual screenshot is
# still saved separately by the tracer.
_MAX_INLINE_VALUE_LEN = 4096


def _redact_for_trace(d: dict[str, Any], sensitive_keys: list[str] | None = None) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in d.items():
        if sensitive_keys and k in sensitive_keys:
            out[k] = "<REDACTED>"
        elif isinstance(v, str) and len(v) > _MAX_INLINE_VALUE_LEN:
            out[k] = f"<{len(v)} bytes elided>"
        elif isinstance(v, dict):
            out[k] = _redact_for_trace(v, sensitive_keys)
        else:
            out[k] = v
    return out
