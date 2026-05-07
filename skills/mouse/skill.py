"""mouse: move, click, or drag the mouse."""

from __future__ import annotations

import math
import time
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from daedalus.backends.protocol import Button
from daedalus.core import AtomicSkill, ExecutionContext, SkillSpec, register
from daedalus.core.spec import SkillExample, SkillVersion


class MouseInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action: Literal["move", "click", "drag"] = Field(
        description="'move' repositions cursor, 'click' clicks at position, 'drag' drags from (x,y) to (x2,y2)."
    )
    x: int = Field(ge=0, le=4095, description="Target x coordinate (or drag start x).")
    y: int = Field(ge=0, le=4095, description="Target y coordinate (or drag start y).")
    x2: int | None = Field(default=None, ge=0, le=4095, description="Drag end x (required for drag).")
    y2: int | None = Field(default=None, ge=0, le=4095, description="Drag end y (required for drag).")
    button: Button = Field(default=Button.LEFT, description="Mouse button: left, middle, or right.")
    double: bool = Field(default=False, description="If true, double-click (only for action=click).")
    speed: float = Field(
        default=800.0,
        gt=0,
        description="Drag speed in pixels per second. Lower values = slower, more reliable drags.",
    )

    @model_validator(mode="after")
    def _validate_drag_coords(self) -> "MouseInput":
        if self.action == "drag":
            if self.x2 is None or self.y2 is None:
                raise ValueError("x2 and y2 are required for drag action")
        return self


class MouseOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action: Literal["move", "click", "drag"]
    position: tuple[int, int] | None = Field(default=None, description="Final cursor position for move/click.")
    start: tuple[int, int] | None = Field(default=None, description="Drag start position.")
    end: tuple[int, int] | None = Field(default=None, description="Drag end position.")
    button: Button | None = Field(default=None)
    double: bool | None = Field(default=None)


@register
class Mouse(AtomicSkill):
    SPEC = SkillSpec(
        id="mouse",
        version=SkillVersion(raw="0.2.0"),
        kind="atomic",
        description=(
            "General-purpose mouse skill: move without clicking, click "
            "(left/right/double), or drag from point A to point B with "
            "configurable speed (linear interpolation for reliable drags)."
        ),
        side_effects=["screen_input"],
        preconditions=["backend.connected", "0 <= x < screen.width", "0 <= y < screen.height"],
        postconditions=["cursor.position == (x, y) or cursor.position == (x2, y2) for drag"],
        examples=[
            SkillExample(
                inputs={"action": "move", "x": 400, "y": 300},
                expected={"action": "move", "position": [400, 300]},
            ),
            SkillExample(
                inputs={"action": "click", "x": 100, "y": 200},
                expected={"action": "click", "position": [100, 200], "button": "left", "double": False},
            ),
            SkillExample(
                inputs={"action": "drag", "x": 100, "y": 100, "x2": 500, "y2": 300},
                expected={"action": "drag", "start": [100, 100], "end": [500, 300], "button": "left"},
            ),
        ],
        tests=["basic.json"],
        tags=["mouse", "input", "core"],
    )
    Inputs = MouseInput
    Outputs = MouseOutput

    def run(self, inputs: MouseInput, ctx: ExecutionContext) -> MouseOutput:  # type: ignore[override]
        s = ctx.coordinate_scale
        x = int(inputs.x * s)
        y = int(inputs.y * s)

        if inputs.action == "move":
            ctx.backend.move(x, y)
            return MouseOutput(action="move", position=(x, y))

        if inputs.action == "click":
            ctx.backend.click(x, y, button=inputs.button, double=inputs.double)
            return MouseOutput(
                action="click",
                position=(x, y),
                button=inputs.button,
                double=inputs.double,
            )

        # drag with linear interpolation
        assert inputs.x2 is not None and inputs.y2 is not None
        x2 = int(inputs.x2 * s)
        y2 = int(inputs.y2 * s)

        dx = x2 - x
        dy = y2 - y
        distance = math.hypot(dx, dy)
        speed = inputs.speed * s

        # Compute intermediate steps: ~60 updates/sec, at least 5 steps
        duration = max(distance / speed, 0.05)
        num_steps = max(5, int(duration * 60))
        step_delay = duration / num_steps

        backend = ctx.backend
        backend.move(x, y)
        time.sleep(0.02)
        backend.mouse_down(button=inputs.button)
        time.sleep(0.02)

        for i in range(1, num_steps + 1):
            t = i / num_steps
            ix = int(x + dx * t)
            iy = int(y + dy * t)
            backend.move(ix, iy)
            time.sleep(step_delay)

        backend.mouse_up(button=inputs.button)

        return MouseOutput(
            action="drag",
            start=(x, y),
            end=(x2, y2),
            button=inputs.button,
        )
