"""Deterministic in-memory backend for unit tests and skill development.

The mock backend records every method call (so tests can assert against the
event log) and returns canned screenshots from a fixture directory. This lets
us:

- run skills in CI without any display server or VNC server,
- snapshot expected behaviour as JSON,
- replay traces against the same fixtures.
"""

from __future__ import annotations

import time
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from PIL import Image

from daedalus.backends.protocol import Button, Rect, Screenshot


@dataclass
class MockEvent:
    op: str
    args: dict[str, Any] = field(default_factory=dict)
    ts: float = field(default_factory=time.time)


class MockBackend:
    """In-memory ``RemoteDesktop`` implementation.

    Parameters
    ----------
    width, height:
        Logical screen size. Defaults to 1920x1080 (Phase 0 baseline).
    fixtures_dir:
        Optional directory of PNG screenshots returned in order by
        :meth:`screenshot`. If not supplied, a procedurally generated
        single-color frame is returned.
    """

    def __init__(
        self,
        width: int = 1920,
        height: int = 1080,
        fixtures_dir: Path | None = None,
        fixture_sequence: Iterable[str] | None = None,
    ) -> None:
        self._width = width
        self._height = height
        self._fixtures_dir = fixtures_dir
        self._fixture_sequence = list(fixture_sequence) if fixture_sequence else []
        self._fixture_idx = 0
        self._connected = False
        self._cursor: tuple[int, int] = (0, 0)
        self.events: list[MockEvent] = []

    # -- Lifecycle -----------------------------------------------------------------

    @property
    def size(self) -> tuple[int, int]:
        return (self._width, self._height)

    @property
    def is_connected(self) -> bool:
        return self._connected

    def connect(self) -> None:
        self._connected = True
        self.events.append(MockEvent(op="connect"))

    def disconnect(self) -> None:
        self._connected = False
        self.events.append(MockEvent(op="disconnect"))

    # -- Capture -------------------------------------------------------------------

    def screenshot(self, region: Rect | None = None) -> Screenshot:
        self._require_connected("screenshot")
        if self._fixtures_dir and self._fixture_sequence:
            name = self._fixture_sequence[
                min(self._fixture_idx, len(self._fixture_sequence) - 1)
            ]
            self._fixture_idx += 1
            img = Image.open(self._fixtures_dir / name).convert("RGBA")
        else:
            # Plain mid-gray frame so tests have something deterministic.
            img = Image.new("RGBA", (self._width, self._height), (128, 128, 128, 255))

        if region is not None:
            img = img.crop((region.x, region.y, region.x + region.width, region.y + region.height))
            w, h = region.width, region.height
        else:
            w, h = img.size

        self.events.append(
            MockEvent(
                op="screenshot",
                args={"region": region.__dict__ if region else None, "size": (w, h)},
            )
        )
        return Screenshot(image=img, width=w, height=h, captured_at=time.time())

    # -- Mouse ---------------------------------------------------------------------

    def move(self, x: int, y: int) -> None:
        self._require_connected("move")
        self._check_in_bounds(x, y)
        self._cursor = (x, y)
        self.events.append(MockEvent(op="move", args={"x": x, "y": y}))

    def click(
        self,
        x: int,
        y: int,
        button: Button = Button.LEFT,
        double: bool = False,
    ) -> None:
        self._require_connected("click")
        self._check_in_bounds(x, y)
        self._cursor = (x, y)
        self.events.append(
            MockEvent(
                op="click",
                args={"x": x, "y": y, "button": button.value, "double": double},
            )
        )

    def drag(
        self,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        button: Button = Button.LEFT,
    ) -> None:
        self._require_connected("drag")
        self._check_in_bounds(x1, y1)
        self._check_in_bounds(x2, y2)
        self._cursor = (x2, y2)
        self.events.append(
            MockEvent(
                op="drag",
                args={"x1": x1, "y1": y1, "x2": x2, "y2": y2, "button": button.value},
            )
        )

    def mouse_down(self, button: Button = Button.LEFT) -> None:
        self._require_connected("mouse_down")
        self.events.append(MockEvent(op="mouse_down", args={"button": button.value}))

    def mouse_up(self, button: Button = Button.LEFT) -> None:
        self._require_connected("mouse_up")
        self.events.append(MockEvent(op="mouse_up", args={"button": button.value}))

    def scroll(self, dx: int, dy: int) -> None:
        self._require_connected("scroll")
        self.events.append(MockEvent(op="scroll", args={"dx": dx, "dy": dy}))

    # -- Keyboard ------------------------------------------------------------------

    def write(self, text: str) -> None:
        self._require_connected("write")
        self.events.append(MockEvent(op="write", args={"text": text}))

    def press(self, *keys: str) -> None:
        self._require_connected("press")
        self.events.append(MockEvent(op="press", args={"keys": list(keys)}))

    # -- Internal helpers ----------------------------------------------------------

    def _require_connected(self, op: str) -> None:
        if not self._connected:
            raise RuntimeError(f"MockBackend.{op}: not connected")

    def _check_in_bounds(self, x: int, y: int) -> None:
        if not (0 <= x < self._width and 0 <= y < self._height):
            raise ValueError(
                f"({x},{y}) outside MockBackend bounds {self._width}x{self._height}"
            )

    # -- Test helpers --------------------------------------------------------------

    def cursor(self) -> tuple[int, int]:
        return self._cursor

    def event_ops(self) -> list[str]:
        return [e.op for e in self.events]

    def reset(self) -> None:
        self.events.clear()
        self._fixture_idx = 0
        self._cursor = (0, 0)
