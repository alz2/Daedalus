"""Explorer: a freeform tool-calling agent for environment discovery.

The explorer runs before the planner. It can invoke any registered skill
directly as a tool call, request new skills to be implemented, and ultimately
produce structured observations that the planner uses as context for its first
plan.
"""

from __future__ import annotations

import json
import logging
import threading
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from daedalus.backends.protocol import RemoteDesktop
from daedalus.core.context import ExecutionContext, TaskState, compute_coordinate_scale
from daedalus.core.errors import DaedalusError, SkillNotFoundError
from daedalus.core.registry import Registry, get_registry
from daedalus.core.store import RunStore
from daedalus.implementor.implementor import ImplementorRequest, SyntheticSkillImplementor
from daedalus.library.librarian import Librarian
from daedalus.llm.context import estimate_token_count, get_context_config, prune_old_images, summarize_and_compact
from daedalus.llm.gateway import LLMCall, LLMGateway, LLMRole, ToolCall
from daedalus.tracing.recorder import TraceRecorder

log = logging.getLogger(__name__)

DEFAULT_MAX_ITERATIONS = 20

# Maximum image payload size (bytes) before converting PNG → JPEG for the LLM.
# Bedrock/Anthropic payload limit is ~5MB; we convert above 3.5MB to stay safe.
_MAX_PNG_BYTES_FOR_LLM = 3_500_000


def _encode_image_for_llm(path: Path) -> tuple[str, str]:
    """Encode an image for LLM consumption, converting large PNGs to JPEG.

    Returns (base64_str, mime_type).
    On-disk file is never modified — conversion is in-memory only.
    """
    import base64
    import io

    raw = path.read_bytes()
    if len(raw) <= _MAX_PNG_BYTES_FOR_LLM:
        return base64.b64encode(raw).decode("ascii"), "image/png"

    from PIL import Image

    img = Image.open(io.BytesIO(raw))
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=85)
    jpeg_bytes = buf.getvalue()
    log.debug(
        "image %s converted PNG(%dKB)->JPEG(%dKB) for LLM",
        path.name, len(raw) // 1024, len(jpeg_bytes) // 1024,
    )
    return base64.b64encode(jpeg_bytes).decode("ascii"), "image/jpeg"


class ExplorerError(DaedalusError):
    pass


@dataclass
class ExploreResult:
    """Output of the explore phase, consumed by the planner."""

    observations: str
    new_skills: list[str] = field(default_factory=list)
    tool_calls_count: int = 0
    raw_messages: list[dict[str, Any]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Special tool definitions (non-skill tools)
# ---------------------------------------------------------------------------

_TOOL_IMPLEMENT_SKILL = {
    "type": "function",
    "function": {
        "name": "implement_skill",
        "description": (
            "Request implementation of a new skill. The system will use an LLM to "
            "generate, test, and publish the skill. Returns the tool signature "
            "(input/output schema) on success, or error details on failure. "
            "After a successful implementation, the new skill becomes available "
            "as a tool call in subsequent turns."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "skill_name": {
                    "type": "string",
                    "description": "Snake_case name for the new skill (e.g. 'extract_grid_state').",
                },
                "description": {
                    "type": "string",
                    "description": "Detailed description of what the skill should do, including inputs/outputs.",
                },
            },
            "required": ["skill_name", "description"],
        },
    },
}

_TOOL_EXPLORE_DONE = {
    "type": "function",
    "function": {
        "name": "explore_done",
        "description": (
            "Signal that exploration is complete. Provide your observations about "
            "the environment, problem structure, rules, layout, and any other "
            "information the planner will need to produce a good plan."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "observations": {
                    "type": "string",
                    "description": (
                        "Structured observations from exploration. Include: "
                        "environment state, rules discovered, problem constraints, "
                        "layout information, and recommended approach."
                    ),
                },
            },
            "required": ["observations"],
        },
    },
}


def _skill_to_tool_def(entry) -> dict[str, Any]:
    """Convert a registered skill into an OpenAI function-calling tool definition."""
    spec = entry.cls.SPEC
    input_schema = entry.cls.Inputs.model_json_schema()

    # Clean up the schema for the tool definition: strip $defs and title at the
    # top level so it's a simple properties/required object for the LLM.
    params: dict[str, Any] = {"type": "object"}
    if "properties" in input_schema:
        params["properties"] = input_schema["properties"]
    else:
        params["properties"] = {}
    if "required" in input_schema:
        params["required"] = input_schema["required"]
    if "$defs" in input_schema:
        params["$defs"] = input_schema["$defs"]

    return {
        "type": "function",
        "function": {
            "name": entry.id,
            "description": spec.description,
            "parameters": params,
        },
    }


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

EXPLORER_SYSTEM_PROMPT = """\
You are the Explorer agent for Daedalus, a computer-control system. Your job is \
to explore and understand the current environment BEFORE any plan is made.

YOUR GOAL:
Discover HOW to interact with the environment so a downstream planner can \
produce a working plan. Focus on mechanics, not on solving the problem yourself.

STRATEGY:
1. Start by calling view_screen() to see the current state.
2. Test interactions: click buttons, try hotkeys, discover navigation methods.
3. If you need a specialized capability that doesn't exist, use implement_skill to create it.
4. When you understand the mechanics, call explore_done with structured observations.

WHAT TO DISCOVER:
- What application/interface is open and its general purpose
- Available controls: buttons, menus, keyboard shortcuts, hotkeys
- How to navigate (scroll, switch views, open panels)
- The spatial layout (where things are, coordinate ranges)
- Any relevant rules or constraints for interaction
- Current state summary (what's on screen, what stage we're at)

GUIDELINES:
- Be ACTION-ORIENTED: test tools and discover interactions, don't just look
- Be EFFICIENT: read something once with vision_query, trust the result, move on
- DO NOT endlessly verify or re-read data — one clean reading is sufficient
- DO NOT try to solve the problem — just understand how to interact with it
- Prefer discovering hotkeys/shortcuts over coordinate-based clicking
- If you create new skills, test them once to confirm they work
- Move quickly: gather what the planner needs and call explore_done

ANTI-PATTERNS TO AVOID:
- Reading the same data multiple times to "verify" it
- Asking follow-up questions about data you already read correctly
- Spending iterations double-checking your own observations
- Getting stuck perfecting details instead of exploring broadly
- Reporting pixel coordinates as reliable interaction targets — coordinates
  from vision_query or manual estimation are INACCURATE and shift between runs.
  Only coordinates from locate_element are reliable.

LLM COORDINATE SCALING (IMPORTANT):
Screenshots are automatically downscaled to match LLM vision processing
resolution. All coordinates across skills (view_screen, locate_element,
click_element, mouse) use this consistent downscaled coordinate space.
The mouse skill automatically scales coordinates back up when interacting
with the actual screen.

This means:
- You do NOT need to worry about coordinate scaling manually.
- Coordinates from view_screen, locate_element, and your own visual
  estimation all live in the same space.
- Just use coordinates as you see them — the system handles the rest.

COORDINATE WARNING:
Do NOT report hardcoded pixel coordinates in your observations as a way to
interact with the UI. Coordinates are fragile — window position, zoom level,
and dynamic content make them unreliable. Instead, report:
- Keyboard shortcuts and hotkeys (always reliable)
- Element descriptions that locate_element/click_element can find
- Relative spatial relationships ("the button is below the header")

Call explore_done when you understand the interaction mechanics well enough \
for the planner to write a working plan. Make sure to provide the planner with \
robust instructions for how to solve the problem that would work even if \
the environment or UI positions change.
"""


EXPLORER_SOLVE_SYSTEM_PROMPT = """\
You are the Explorer agent for Daedalus, a computer-control system. Your job is \
to DIRECTLY SOLVE the given task by interacting with the environment.

YOUR GOAL:
Accomplish the user's task yourself using the available tools. You are the sole \
actor — there is no downstream planner. Solve the problem step by step.

STRATEGY:
1. Start by calling view_screen() to see the current state.
2. Interact with the environment: click buttons, type text, use hotkeys, navigate.
3. If you need a specialized capability that doesn't exist, use implement_skill to create it.
4. When you have completed the task, call explore_done with a summary of what you did.

GUIDELINES:
- Be ACTION-ORIENTED: take concrete steps to accomplish the goal
- Be EFFICIENT: don't over-verify, trust your observations and move forward
- DO NOT just observe — actually solve the problem
- If you create new skills, test them once to confirm they work
- Call explore_done when the task is complete with a summary of actions taken

ANTI-PATTERNS TO AVOID:
- Reading the same data multiple times to "verify" it
- Getting stuck perfecting details instead of making progress
- Reporting back without actually attempting to solve the task
- Spending iterations double-checking your own observations

LLM COORDINATE SCALING (IMPORTANT):
Screenshots are automatically downscaled to match LLM vision processing
resolution. All coordinates across skills (view_screen, locate_element,
click_element, mouse) use this consistent downscaled coordinate space.
The mouse skill automatically scales coordinates back up when interacting
with the actual screen.

This means:
- You do NOT need to worry about coordinate scaling manually.
- Coordinates from view_screen, locate_element, and your own visual
  estimation all live in the same space.
- Just use coordinates as you see them — the system handles the rest.

COORDINATE WARNING:
Do NOT report hardcoded pixel coordinates in your observations as a way to
interact with the UI. Coordinates are fragile — window position, zoom level,
and dynamic content make them unreliable. Instead, use:
- Keyboard shortcuts and hotkeys (always reliable)
- locate_element/click_element to find elements dynamically
- Relative spatial relationships ("the button is below the header")

Call explore_done when the task is complete. Provide a summary of what you \
accomplished and any relevant observations.
"""


# ---------------------------------------------------------------------------
# Explorer class
# ---------------------------------------------------------------------------


class Explorer:
    """Freeform tool-calling agent that explores the environment before planning."""

    def __init__(
        self,
        gateway: LLMGateway,
        librarian: Librarian,
        implementor: SyntheticSkillImplementor,
        skills_dir: Path,
        *,
        registry: Registry | None = None,
        traces_root: Path | None = None,
        tasks_db: Path | None = None,
        max_iterations: int = DEFAULT_MAX_ITERATIONS,
        verbose: bool = False,
    ) -> None:
        self._gateway = gateway
        self._librarian = librarian
        self._implementor = implementor
        self._skills_dir = skills_dir
        self._registry = registry or get_registry()
        self._traces_root = traces_root or Path("traces")
        self._tasks_db = tasks_db or self._traces_root / "tasks.db"
        self._max_iterations = max_iterations
        self._verbose = verbose

    def _build_tools(self) -> list[dict[str, Any]]:
        """Build the full tool list: all registered skills + special tools."""
        tools: list[dict[str, Any]] = []
        for entry in self._registry:
            if entry.cls.SPEC.kind == "daemon":
                continue
            tools.append(_skill_to_tool_def(entry))
        tools.append(_TOOL_IMPLEMENT_SKILL)
        tools.append(_TOOL_EXPLORE_DONE)
        return tools

    def explore(
        self,
        goal: str,
        backend: RemoteDesktop,
        *,
        abort_event: threading.Event | None = None,
        progress_callback: Callable[[int, int, str], None] | None = None,
        stream_callback: Callable[[str], None] | None = None,
        tool_callback: Callable[[str, str, dict[str, Any], str | None], None] | None = None,
        context_usage_callback: Callable[[int, int], None] | None = None,
        solve_mode: bool = False,
    ) -> ExploreResult:
        """Run the exploration loop. Returns observations for the planner.
        
        progress_callback receives (iteration, max_iterations, description).
        stream_callback receives streamed text tokens from the LLM.
        tool_callback receives (tool_name, tool_id, arguments, result_or_None).
          Called with result=None at start, then again with result at end.
        context_usage_callback receives (used_tokens, max_tokens).
        If solve_mode is True, the explorer attempts to solve the task directly
        rather than just gathering information for a downstream planner.
        """
        abort = abort_event or threading.Event()
        task_id = f"explore-{uuid.uuid4().hex[:8]}"

        self._traces_root.mkdir(parents=True, exist_ok=True)
        self._tasks_db.parent.mkdir(parents=True, exist_ok=True)

        tracer = TraceRecorder(
            traces_root=self._traces_root,
            db_path=self._tasks_db,
            task_name="explore",
        )
        self._gateway.set_tracer(tracer)
        state = TaskState(self._tasks_db, task_id)
        store = RunStore(self._tasks_db, task_id)
        screen_w, screen_h = backend.size if hasattr(backend, "size") else (1728, 1117)
        ctx = ExecutionContext(
            task_id=task_id,
            backend=backend,
            task_state=state,
            tracer=tracer,
            store=store,
            llm=self._gateway,
            abort_event=abort,
            coordinate_scale=compute_coordinate_scale(screen_w),
        )

        tools = self._build_tools()
        if self._verbose:
            tool_names = [t["function"]["name"] for t in tools]
            log.info("explorer tools: %s", ", ".join(tool_names))

        system_prompt = EXPLORER_SOLVE_SYSTEM_PROMPT if solve_mode else EXPLORER_SYSTEM_PROMPT
        user_msg = (
            f"Goal: {goal}\n\nSolve this task directly."
            if solve_mode
            else f"Goal: {goal}\n\nExplore the environment to understand the problem."
        )
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_msg},
        ]

        new_skills: list[str] = []
        total_tool_calls = 0
        observations: str | None = None

        for iteration in range(self._max_iterations):
            if abort.is_set():
                log.warning("explorer aborted by user")
                break

            if progress_callback:
                progress_callback(iteration + 1, self._max_iterations, f"iteration {iteration + 1}/{self._max_iterations}")

            if self._verbose:
                log.info("explorer iteration %d/%d", iteration + 1, self._max_iterations)

            summarize_and_compact(messages, self._gateway)
            prune_old_images(messages)

            if context_usage_callback:
                used = estimate_token_count(messages)
                max_tokens = get_context_config().max_context_tokens
                context_usage_callback(used, max_tokens)

            response = self._gateway.complete(
                LLMCall(
                    role=LLMRole.EXPLORER,
                    messages=messages,
                    tools=tools,
                ),
                stream_callback=stream_callback,
            )

            if self._verbose and response.content:
                log.info("explorer thinking: %s", response.content[:500])

            if not response.tool_calls:
                if self._verbose:
                    log.info("explorer returned text (no tool calls) — treating as implicit explore_done")
                if response.content.strip():
                    observations = response.content.strip()
                break

            # Append the assistant message with tool calls to conversation history.
            assistant_msg: dict[str, Any] = {"role": "assistant", "content": response.content or None}
            assistant_msg["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.name, "arguments": json.dumps(tc.arguments)},
                }
                for tc in response.tool_calls
            ]
            messages.append(assistant_msg)

            for tc in response.tool_calls:
                total_tool_calls += 1

                if progress_callback:
                    progress_callback(iteration + 1, self._max_iterations, f"calling {tc.name}")

                if self._verbose:
                    args_preview = json.dumps(tc.arguments, default=str)
                    if len(args_preview) > 200:
                        args_preview = args_preview[:200] + "..."
                    log.info("explorer tool_call [%d]: %s(%s)", total_tool_calls, tc.name, args_preview)

                tracer.emit("tool_call", {
                    "iteration": iteration + 1,
                    "call_index": total_tool_calls,
                    "tool": tc.name,
                    "arguments": tc.arguments,
                })

                if tool_callback:
                    tool_callback(tc.name, tc.id, tc.arguments, None)

                result_content, done = self._dispatch(tc, ctx, new_skills)

                # Emit full tool result to trace (summarize image content)
                image_path_for_cb: str | None = getattr(self, "_last_image_path", None)
                if isinstance(result_content, str):
                    trace_result = result_content
                else:
                    trace_result = "[multimodal content with image]"

                if tool_callback:
                    tool_callback(tc.name, tc.id, tc.arguments, trace_result, image_path_for_cb)
                tracer.emit("tool_result", {
                    "iteration": iteration + 1,
                    "call_index": total_tool_calls,
                    "tool": tc.name,
                    "result": trace_result,
                    "done": done,
                })

                if self._verbose:
                    if isinstance(result_content, str):
                        result_preview = result_content[:300] if len(result_content) > 300 else result_content
                    else:
                        result_preview = "[multimodal: image + metadata]"
                    log.info("explorer tool_result [%d]: %s", total_tool_calls, result_preview)

                # Build the tool result message. If the result contains image
                # content blocks, use the multimodal content format.
                if isinstance(result_content, list):
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": result_content,
                    })
                else:
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": result_content,
                    })

                if done:
                    observations = tc.arguments.get("observations", result_content if isinstance(result_content, str) else "")
                    if self._verbose:
                        log.info("explorer called explore_done")
                    tools = self._build_tools()
                    break
            else:
                if new_skills:
                    tools = self._build_tools()
                continue
            break  # explore_done was called

        tracer.finish("success" if observations else "incomplete")

        if observations is None:
            # Max iterations reached without explore_done — send one final
            # message asking the explorer to consolidate its findings so far.
            if not abort.is_set():
                if self._verbose:
                    log.info("explorer hit max iterations — requesting final observations")
                # Insert an assistant message to maintain strict role alternation
                # (the last message is a tool result; Bedrock requires assistant
                # between tool and user).
                messages.append({
                    "role": "assistant",
                    "content": "I have reached the maximum number of exploration steps.",
                })
                messages.append({
                    "role": "user",
                    "content": (
                        "You have reached the maximum number of exploration steps. "
                        "Please call explore_done NOW with a comprehensive summary of "
                        "everything you have learned so far. Include all observations "
                        "about the environment, layout coordinates, rules, and any other "
                        "information that would help the planner succeed."
                    ),
                })
                try:
                    summarize_and_compact(messages, self._gateway)
                    prune_old_images(messages)
                    final_response = self._gateway.complete(
                        LLMCall(
                            role=LLMRole.EXPLORER,
                            messages=messages,
                            tools=tools,
                        ),
                        stream_callback=stream_callback,
                    )
                    # Check if it called explore_done
                    if final_response.tool_calls:
                        for tc in final_response.tool_calls:
                            if tc.name == "explore_done":
                                observations = tc.arguments.get("observations", "")
                                if self._verbose:
                                    log.info("explorer provided final observations via explore_done")
                                break
                    # If it just returned text instead
                    if observations is None and final_response.content and final_response.content.strip():
                        observations = final_response.content.strip()
                        if self._verbose:
                            log.info("explorer provided final observations as text")
                except Exception as exc:
                    log.warning("final exploration summary request failed: %s", exc)

            if observations is None:
                observations = (
                    f"Explorer reached max iterations ({self._max_iterations}) without "
                    "calling explore_done. The environment may need more investigation."
                )

        if self._verbose:
            log.info("explorer finished: %d tool calls, %d new skills", total_tool_calls, len(new_skills))
            log.info("explorer observations: %s", observations[:500])

        return ExploreResult(
            observations=observations,
            new_skills=new_skills,
            tool_calls_count=total_tool_calls,
            raw_messages=messages,
        )

    # ------------------------------------------------------------------
    # Tool dispatch
    # ------------------------------------------------------------------

    def _dispatch(
        self,
        tc: ToolCall,
        ctx: ExecutionContext,
        new_skills: list[str],
    ) -> tuple[str | list[dict[str, Any]], bool]:
        """Dispatch a tool call. Returns (result_content, is_done).

        result_content is either a plain string or a list of content blocks
        (for multimodal responses containing images).
        """
        self._last_image_path = None
        try:
            if tc.name == "explore_done":
                return self._handle_explore_done(tc.arguments), True
            elif tc.name == "implement_skill":
                result = self._handle_implement_skill(tc.arguments, new_skills)
                return result, False
            else:
                # All other tool names are skill IDs.
                return self._handle_skill_call(tc.name, tc.arguments, ctx), False
        except Exception as exc:
            log.warning("explorer tool %s failed: %s", tc.name, exc)
            return json.dumps({"error": f"{type(exc).__name__}: {exc}"}), False

    def _handle_explore_done(self, args: dict[str, Any]) -> str:
        return args.get("observations", "")

    def _handle_skill_call(self, skill_id: str, kwargs: dict[str, Any], ctx: ExecutionContext) -> str | list[dict[str, Any]]:
        """Execute a skill directly by ID with the provided arguments.

        Returns either a plain string or a multimodal content list (with image
        blocks) when the skill output contains base64 image data.
        """
        try:
            entry = self._registry.get(skill_id)
        except SkillNotFoundError:
            return json.dumps({"error": f"skill {skill_id!r} not found"})

        try:
            inputs_model = entry.cls.Inputs.model_validate(kwargs)
            instance = entry.cls()
            output = instance.run(inputs_model, ctx)
            out_dict = output.model_dump(mode="json") if hasattr(output, "model_dump") else dict(output)

            # If the output contains an image_path, read it and return as a
            # multimodal content block so the LLM can see it directly.
            image_path = out_dict.get("image_path")
            if image_path and isinstance(image_path, str):
                self._last_image_path = image_path
                from pathlib import Path as _Path
                img_file = _Path(image_path)
                if img_file.exists():
                    import base64 as _b64
                    image_b64, mime = _encode_image_for_llm(img_file)
                    metadata = json.dumps({k: v for k, v in out_dict.items() if k != "image_path"}, default=str)
                    content_parts: list[dict[str, Any]] = [
                        {"type": "text", "text": metadata},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:{mime};base64,{image_b64}"},
                        },
                    ]
                    return content_parts

            # Legacy: if the output contains a base64 image directly.
            image_b64 = out_dict.pop("image_b64", None)
            if image_b64 and isinstance(image_b64, str) and len(image_b64) > 100:
                metadata = json.dumps(out_dict, default=str)
                content_parts = [
                    {"type": "text", "text": metadata},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{image_b64}"},
                    },
                ]
                return content_parts

            # Truncate very large outputs for the conversation.
            result_str = json.dumps(out_dict, default=str)
            if len(result_str) > 8000:
                for key in list(out_dict.keys()):
                    val = out_dict[key]
                    if isinstance(val, str) and len(val) > 2000:
                        out_dict[key] = val[:200] + f"... [truncated, {len(val)} chars total]"
                result_str = json.dumps(out_dict, default=str)

            return result_str
        except Exception as exc:
            return json.dumps({"error": f"{type(exc).__name__}: {exc}"})

    def _handle_implement_skill(self, args: dict[str, Any], new_skills: list[str]) -> str:
        skill_name = args.get("skill_name", "")
        description = args.get("description", "")

        if not skill_name or not description:
            return json.dumps({"error": "skill_name and description are both required"})

        if skill_name in self._registry:
            entry = self._registry.get(skill_name)
            return json.dumps({
                "status": "already_exists",
                "inputs": entry.cls.Inputs.model_json_schema(),
                "outputs": entry.cls.Outputs.model_json_schema(),
            })

        request = ImplementorRequest(
            proposed_id=skill_name,
            description=description,
            rationale="Requested by explorer during exploration phase",
            side_effects=["screen_capture", "screen_input", "llm_call"],
        )

        try:
            result = self._implementor.synthesize(request)
        except Exception as exc:
            return json.dumps({"status": "failed", "error": f"Implementor error: {exc}"})

        if result.ok and result.bundle is not None:
            try:
                self._implementor.publish(result.bundle)
                self._librarian.reindex()
                new_skills.append(skill_name)

                entry = self._registry.get(skill_name)
                return json.dumps({
                    "status": "success",
                    "skill_id": skill_name,
                    "inputs": entry.cls.Inputs.model_json_schema(),
                    "outputs": entry.cls.Outputs.model_json_schema(),
                })
            except Exception as exc:
                return json.dumps({"status": "failed", "error": f"Publish error: {exc}"})
        else:
            errors = result.test_failures + [str(v) for v in result.violations]
            return json.dumps({
                "status": "failed",
                "errors": errors,
                "notes": result.notes,
            })
